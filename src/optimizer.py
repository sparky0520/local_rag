"""Hyperparameter optimization for Hybrid Search parameters (Alpha and RRF k)."""
from typing import List, Dict, Any, Tuple
import numpy as np
from pydantic import BaseModel
from src.evaluation import RAGEvaluator

class TuningPoint(BaseModel):
    param_name: str
    param_value: float
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    ndcg_at_5: float

class OptimizationReport(BaseModel):
    best_alpha: float
    best_alpha_mrr: float
    best_alpha_recall_1: float
    alpha_curve: List[TuningPoint]
    best_rrf_k: int
    best_rrf_k_mrr: float
    rrf_k_curve: List[TuningPoint]

class HyperparameterOptimizer:
    def __init__(self, evaluator: RAGEvaluator):
        self.evaluator = evaluator
        self.queries = evaluator.queries

    def tune_alpha(self, steps: int = 21) -> Tuple[float, List[TuningPoint]]:
        """Sweeps alpha from 0.0 to 1.0 (e.g. 0.0, 0.05, 0.10, ..., 1.0)."""
        alphas = np.linspace(0.0, 1.0, steps)
        points: List[TuningPoint] = []
        best_alpha = 0.5
        best_mrr = -1.0

        for a in alphas:
            a_val = round(float(a), 3)
            recalls_1 = []
            recalls_3 = []
            recalls_5 = []
            mrrs = []
            ndcgs = []

            for q in self.queries:
                expected_set = set(q.expected_chunks)
                results = self.evaluator.hybrid_engine.search_linear(q.query, top_k=5, alpha=a_val)
                cids = [r.chunk_id for r in results]

                # Recall@1
                r1 = 1.0 if cids and cids[0] in expected_set else 0.0
                recalls_1.append(r1)

                # Recall@3
                r3 = len(expected_set.intersection(cids[:3])) / len(expected_set)
                recalls_3.append(r3)

                # Recall@5
                r5 = len(expected_set.intersection(cids[:5])) / len(expected_set)
                recalls_5.append(r5)

                # MRR
                mrr = 0.0
                for rank, cid in enumerate(cids, start=1):
                    if cid in expected_set:
                        mrr = 1.0 / rank
                        break
                mrrs.append(mrr)

                # NDCG@5
                from src.evaluation import compute_ndcg
                ndcgs.append(compute_ndcg(cids, expected_set, 5))

            mean_r1 = float(np.mean(recalls_1))
            mean_r3 = float(np.mean(recalls_3))
            mean_r5 = float(np.mean(recalls_5))
            mean_mrr = float(np.mean(mrrs))
            mean_ndcg = float(np.mean(ndcgs))

            pt = TuningPoint(
                param_name="alpha",
                param_value=a_val,
                recall_at_1=mean_r1,
                recall_at_3=mean_r3,
                recall_at_5=mean_r5,
                mrr=mean_mrr,
                ndcg_at_5=mean_ndcg
            )
            points.append(pt)

            if mean_mrr > best_mrr or (mean_mrr == best_mrr and mean_r1 > points[0].recall_at_1):
                best_mrr = mean_mrr
                best_alpha = a_val

        return best_alpha, points

    def tune_rrf_k(self, k_values: List[int] = [10, 20, 40, 60, 80, 100]) -> Tuple[int, List[TuningPoint]]:
        """Sweeps RRF k smoothing constant."""
        points: List[TuningPoint] = []
        best_k = 60
        best_mrr = -1.0

        for k in k_values:
            recalls_1 = []
            recalls_3 = []
            recalls_5 = []
            mrrs = []
            ndcgs = []

            for q in self.queries:
                expected_set = set(q.expected_chunks)
                results = self.evaluator.hybrid_engine.search_rrf(q.query, top_k=5, rrf_k=k)
                cids = [r.chunk_id for r in results]

                r1 = 1.0 if cids and cids[0] in expected_set else 0.0
                recalls_1.append(r1)

                r3 = len(expected_set.intersection(cids[:3])) / len(expected_set)
                recalls_3.append(r3)

                r5 = len(expected_set.intersection(cids[:5])) / len(expected_set)
                recalls_5.append(r5)

                mrr = 0.0
                for rank, cid in enumerate(cids, start=1):
                    if cid in expected_set:
                        mrr = 1.0 / rank
                        break
                mrrs.append(mrr)

                from src.evaluation import compute_ndcg
                ndcgs.append(compute_ndcg(cids, expected_set, 5))

            mean_r1 = float(np.mean(recalls_1))
            mean_r3 = float(np.mean(recalls_3))
            mean_r5 = float(np.mean(recalls_5))
            mean_mrr = float(np.mean(mrrs))
            mean_ndcg = float(np.mean(ndcgs))

            pt = TuningPoint(
                param_name="rrf_k",
                param_value=float(k),
                recall_at_1=mean_r1,
                recall_at_3=mean_r3,
                recall_at_5=mean_r5,
                mrr=mean_mrr,
                ndcg_at_5=mean_ndcg
            )
            points.append(pt)

            if mean_mrr > best_mrr:
                best_mrr = mean_mrr
                best_k = k

        return best_k, points

    def run_full_tuning(self) -> OptimizationReport:
        best_alpha, alpha_pts = self.tune_alpha()
        best_k, k_pts = self.tune_rrf_k()

        best_alpha_pt = next(p for p in alpha_pts if p.param_value == best_alpha)
        best_k_pt = next(p for p in k_pts if int(p.param_value) == best_k)

        return OptimizationReport(
            best_alpha=best_alpha,
            best_alpha_mrr=best_alpha_pt.mrr,
            best_alpha_recall_1=best_alpha_pt.recall_at_1,
            alpha_curve=alpha_pts,
            best_rrf_k=best_k,
            best_rrf_k_mrr=best_k_pt.mrr,
            rrf_k_curve=k_pts
        )

