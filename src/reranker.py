"""Cross-Encoder Re-Ranking Engine for Local-First RAG."""
from typing import List, Optional, Any
import numpy as np
import torch
from sentence_transformers import CrossEncoder
from pydantic import BaseModel
from src.config import settings

class RerankResult(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    section_title: str
    content: str
    score: float
    rank: int
    initial_rank: int
    initial_score: float
    metadata: dict = {}

def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))

class CrossEncoderReranker:
    _instance: Optional["CrossEncoderReranker"] = None

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[CrossEncoderReranker] Loading '{model_name}' on device '{self.device}'...")
        self.model = CrossEncoder(model_name, device=self.device)

    @classmethod
    def get_instance(cls, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> "CrossEncoderReranker":
        if cls._instance is None:
            cls._instance = cls(model_name)
        return cls._instance

    def rerank(self, query: str, candidates: List[Any], top_k: int = 5) -> List[RerankResult]:
        """Re-ranks a list of candidate search results using cross-attention."""
        if not candidates:
            return []

        pairs = [(query, f"{c.doc_title} - {c.section_title}: {c.content}") for c in candidates]
        raw_scores = self.model.predict(pairs)

        # Pair candidates with their cross-encoder score
        scored_candidates = []
        for orig_idx, (cand, raw_sc) in enumerate(zip(candidates, raw_scores), start=1):
            norm_score = sigmoid(float(raw_sc))
            scored_candidates.append((cand, norm_score, orig_idx, float(cand.score)))

        # Sort descending by cross-encoder score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        k = min(top_k, len(scored_candidates))
        results: List[RerankResult] = []

        for rank, (cand, score, orig_rank, orig_score) in enumerate(scored_candidates[:k], start=1):
            results.append(
                RerankResult(
                    chunk_id=cand.chunk_id,
                    doc_id=cand.doc_id,
                    doc_title=cand.doc_title,
                    section_title=cand.section_title,
                    content=cand.content,
                    score=score,
                    rank=rank,
                    initial_rank=orig_rank,
                    initial_score=orig_score,
                    metadata=getattr(cand, "metadata", {})
                )
            )

        return results
