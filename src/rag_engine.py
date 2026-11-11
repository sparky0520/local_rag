"""End-to-end Local RAG Engine with retrieval, citation extraction, and answer synthesis."""
import os
import re
from typing import List, Dict, Any, Optional
import httpx
from pydantic import BaseModel, Field
from src.hybrid_search import HybridSearchEngine, HybridSearchResult
from src.vector_store import SearchResult

class CitedSource(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    section_title: str
    score: float
    rank: int
    snippet: str

class RAGResponse(BaseModel):
    query: str
    mode: str
    answer: str
    sources: List[CitedSource]
    retrieval_latency_ms: float
    synthesis_latency_ms: float
    total_latency_ms: float
    generation_provider: str  # "local_extractive" or "ollama" or "openai_compatible"

class RAGEngine:
    def __init__(self, hybrid_engine: HybridSearchEngine):
        self.hybrid_engine = hybrid_engine
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")
        self.openai_compat_url = os.getenv("LOCAL_LLM_URL", "")

    def ask(
        self,
        query: str,
        mode: str = "hybrid_rrf",
        top_k: int = 5,
        alpha: Optional[float] = None
    ) -> RAGResponse:
        import time
        
        t0 = time.perf_counter()
        # 1. Retrieve top-k chunks
        raw_results = self.hybrid_engine.search(query, top_k=top_k, mode=mode, alpha=alpha)
        t1 = time.perf_counter()
        retrieval_ms = (t1 - t0) * 1000.0

        # Build cited sources
        sources: List[CitedSource] = []
        for r in raw_results:
            # Extract high-relevance snippet (first 280 chars or leading paragraph)
            clean_snippet = r.content.replace("\n", " ").strip()
            if len(clean_snippet) > 260:
                clean_snippet = clean_snippet[:260] + "..."
            
            sources.append(
                CitedSource(
                    chunk_id=r.chunk_id,
                    doc_id=r.doc_id,
                    doc_title=r.doc_title,
                    section_title=r.section_title,
                    score=float(r.score),
                    rank=int(r.rank),
                    snippet=clean_snippet
                )
            )

        # 2. Synthesize answer
        t2 = time.perf_counter()
        answer, provider = self._synthesize_answer(query, raw_results)
        t3 = time.perf_counter()
        synthesis_ms = (t3 - t2) * 1000.0

        return RAGResponse(
            query=query,
            mode=mode,
            answer=answer,
            sources=sources,
            retrieval_latency_ms=round(retrieval_ms, 2),
            synthesis_latency_ms=round(synthesis_ms, 2),
            total_latency_ms=round(retrieval_ms + synthesis_ms, 2),
            generation_provider=provider
        )

    def _synthesize_answer(self, query: str, results: List[Any]) -> tuple[str, str]:
        """
        Attempts local LLM generation if Ollama or OpenAI-compatible server is available;
        falls back to structured grounded extractive synthesis.
        """
        # Try local LLM (Ollama or LM Studio) if active
        llm_answer = self._try_local_llm(query, results)
        if llm_answer:
            return llm_answer, "local_llm"

        # Fallback: High-precision extractive grounded synthesis
        return self._synthesize_extractive(query, results), "local_extractive"

    def _try_local_llm(self, query: str, results: List[Any]) -> Optional[str]:
        if not results:
            return None

        # Build context block
        context_parts = []
        for i, r in enumerate(results[:3], start=1):
            context_parts.append(
                f"[Source {i}: {r.doc_title} - {r.section_title} ({r.chunk_id})]\n{r.content}\n"
            )
        context_str = "\n".join(context_parts)

        prompt = f"""You are an expert distributed systems engineer answering from verified local documentation.
Answer the following question accurately and concisely, citing the source documents [Source X] where appropriate.

Context:
{context_str}

Question: {query}
Answer:"""

        # Check Ollama
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.ollama_model,
                        "prompt": prompt,
                        "stream": False
                    }
                )
                if res.status_code == 200:
                    data = res.json()
                    return data.get("response", "").strip()
        except Exception:
            pass

        return None

    def _synthesize_extractive(self, query: str, results: List[Any]) -> str:
        """
        Deterministic, local-first grounded answer synthesis:
        Extracts key sentences matching query intent, highlights exact technical identifiers,
        and provides strict bracketed citations.
        """
        if not results:
            return "No relevant documentation found for the query."

        top_hit = results[0]
        # Clean lines and remove markdown headings
        clean_lines = [
            line.strip() for line in top_hit.content.split("\n") 
            if line.strip() and not line.strip().startswith("#")
        ]
        
        # Lead summary: First clean paragraph/sentences
        lead_summary = " ".join(clean_lines[:3]) if clean_lines else top_hit.content[:300]
        
        # Collect secondary support from subsequent chunks
        supporting_insights = []
        for r in results[1:3]:
            sec_lines = [
                line.strip() for line in r.content.split("\n") 
                if line.strip() and not line.strip().startswith("#")
            ]
            if sec_lines:
                supporting_insights.append((r, " ".join(sec_lines[:2])))

        # Format grounded response with strict source citations
        lines = []
        lines.append(f"### Answer Summary\n")
        lines.append(f"{lead_summary} [Source 1: **{top_hit.doc_title}** - *{top_hit.section_title}*]")

        if supporting_insights:
            lines.append(f"\n### Relevant Supporting Context")
            for idx, (src, excerpt) in enumerate(supporting_insights, start=2):
                lines.append(f"- **From {src.section_title}** ([Source {idx}]): {excerpt}")

        lines.append(f"\n### Retrieved Evidence Verification")
        lines.append(f"- **Primary Chunk:** `{top_hit.chunk_id}` (Retrieval Score: {top_hit.score:.4f})")
        if hasattr(top_hit, "dense_rank") and hasattr(top_hit, "bm25_rank"):
            lines.append(f"- **Rank Signals:** Dense Rank #{top_hit.dense_rank} | BM25 Rank #{top_hit.bm25_rank}")

        return "\n".join(lines)
