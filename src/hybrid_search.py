"""Hybrid search combining dense semantic embeddings and BM25 keyword search."""
from typing import List, Dict, Any, Optional
import numpy as np
from pydantic import BaseModel
from src.vector_store import VectorStore, SearchResult
from src.keyword_search import KeywordSearchEngine
from src.config import settings

class HybridSearchResult(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    section_title: str
    content: str
    score: float
    rank: int
    dense_score: float
    dense_rank: int
    bm25_score: float
    bm25_rank: int
    fusion_method: str
    metadata: Dict[str, Any] = {}

def min_max_normalize(scores_dict: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalize a dictionary of {id: score} to [0.0, 1.0]."""
    if not scores_dict:
        return {}
    vals = list(scores_dict.values())
    min_v, max_v = min(vals), max(vals)
    if max_v - min_v < 1e-9:
        return {k: 1.0 for k in scores_dict}
    return {k: (v - min_v) / (max_v - min_v) for k, v in scores_dict.items()}

class HybridSearchEngine:
    def __init__(self, vector_store: VectorStore, keyword_engine: KeywordSearchEngine):
        self.vector_store = vector_store
        self.keyword_engine = keyword_engine

    def search_rrf(self, query: str, top_k: int = 5, rrf_k: Optional[int] = None) -> List[HybridSearchResult]:
        """
        Reciprocal Rank Fusion (RRF):
        score(d) = 1 / (rrf_k + rank_dense(d)) + 1 / (rrf_k + rank_bm25(d))
        """
        k_val = rrf_k if rrf_k is not None else settings.rrf_k
        all_chunks = self.vector_store.get_all_chunks()
        if not all_chunks:
            return []

        # 1. Dense search across all chunks
        total_chunks = len(all_chunks)
        dense_results = self.vector_store.search(query, top_k=total_chunks)
        dense_rank_map = {r.chunk_id: (r.rank, r.score) for r in dense_results}

        # 2. BM25 search across all chunks
        bm25_results = self.keyword_engine.search(query, top_k=total_chunks)
        bm25_rank_map = {r.chunk_id: (r.rank, r.score) for r in bm25_results}

        # 3. Compute RRF scores
        rrf_scores: Dict[str, float] = {}
        for chunk in all_chunks:
            cid = chunk.chunk_id
            d_rank, _ = dense_rank_map.get(cid, (total_chunks + 1, 0.0))
            b_rank, _ = bm25_rank_map.get(cid, (total_chunks + 1, 0.0))

            rrf = (1.0 / (k_val + d_rank)) + (1.0 / (k_val + b_rank))
            rrf_scores[cid] = rrf

        # Sort descending by RRF score
        sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        top_cids = sorted_cids[:top_k]

        results: List[HybridSearchResult] = []
        for rank, cid in enumerate(top_cids, start=1):
            chunk = self.vector_store.get_chunk(cid)
            if not chunk:
                continue
            d_rank, d_score = dense_rank_map.get(cid, (total_chunks + 1, 0.0))
            b_rank, b_score = bm25_rank_map.get(cid, (total_chunks + 1, 0.0))

            results.append(
                HybridSearchResult(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    doc_title=chunk.doc_title,
                    section_title=chunk.section_title,
                    content=chunk.content,
                    score=float(rrf_scores[cid]),
                    rank=rank,
                    dense_score=float(d_score),
                    dense_rank=int(d_rank),
                    bm25_score=float(b_score),
                    bm25_rank=int(b_rank),
                    fusion_method="rrf",
                    metadata=chunk.metadata
                )
            )

        return results

    def search_linear(self, query: str, top_k: int = 5, alpha: Optional[float] = None) -> List[HybridSearchResult]:
        """
        Convex Score Combination:
        score(d) = alpha * norm_dense(d) + (1 - alpha) * norm_bm25(d)
        """
        alpha_val = alpha if alpha is not None else settings.default_alpha
        alpha_val = max(0.0, min(1.0, alpha_val))

        all_chunks = self.vector_store.get_all_chunks()
        if not all_chunks:
            return []

        total_chunks = len(all_chunks)
        dense_results = self.vector_store.search(query, top_k=total_chunks)
        dense_scores_raw = {r.chunk_id: r.score for r in dense_results}
        dense_rank_map = {r.chunk_id: r.rank for r in dense_results}

        bm25_results = self.keyword_engine.search(query, top_k=total_chunks)
        bm25_scores_raw = {r.chunk_id: r.score for r in bm25_results}
        bm25_rank_map = {r.chunk_id: r.rank for r in bm25_results}

        norm_dense = min_max_normalize(dense_scores_raw)
        norm_bm25 = min_max_normalize(bm25_scores_raw)

        combined_scores: Dict[str, float] = {}
        for chunk in all_chunks:
            cid = chunk.chunk_id
            s_dense = norm_dense.get(cid, 0.0)
            s_bm25 = norm_bm25.get(cid, 0.0)
            combined_scores[cid] = (alpha_val * s_dense) + ((1.0 - alpha_val) * s_bm25)

        sorted_cids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)
        top_cids = sorted_cids[:top_k]

        results: List[HybridSearchResult] = []
        for rank, cid in enumerate(top_cids, start=1):
            chunk = self.vector_store.get_chunk(cid)
            if not chunk:
                continue
            d_score = dense_scores_raw.get(cid, 0.0)
            d_rank = dense_rank_map.get(cid, total_chunks + 1)
            b_score = bm25_scores_raw.get(cid, 0.0)
            b_rank = bm25_rank_map.get(cid, total_chunks + 1)

            results.append(
                HybridSearchResult(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    doc_title=chunk.doc_title,
                    section_title=chunk.section_title,
                    content=chunk.content,
                    score=float(combined_scores[cid]),
                    rank=rank,
                    dense_score=float(d_score),
                    dense_rank=int(d_rank),
                    bm25_score=float(b_score),
                    bm25_rank=int(b_rank),
                    fusion_method="linear",
                    metadata=chunk.metadata
                )
            )

        return results

    def search(self, query: str, top_k: int = 5, mode: str = "hybrid_rrf", alpha: Optional[float] = None) -> List[Any]:
        """Unified search interface supporting dense, bm25, hybrid_rrf, hybrid_linear."""
        if mode == "dense":
            return self.vector_store.search(query, top_k=top_k)
        elif mode == "bm25":
            return self.keyword_engine.search(query, top_k=top_k)
        elif mode == "hybrid_linear":
            return self.search_linear(query, top_k=top_k, alpha=alpha)
        else: # default hybrid_rrf
            return self.search_rrf(query, top_k=top_k)
