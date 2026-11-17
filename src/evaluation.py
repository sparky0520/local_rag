"""Evaluation engine measuring Recall@k, MRR, NDCG, and MAP across search modes."""
import json
import time
import math
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from pydantic import BaseModel, Field
from tabulate import tabulate
from src.config import settings
from src.corpus_loader import CorpusLoader
from src.vector_store import VectorStore
from src.keyword_search import KeywordSearchEngine
from src.hybrid_search import HybridSearchEngine

class EvalQuery(BaseModel):
    query_id: str
    query: str
    category: str
    expected_chunks: List[str]
    rationale: Optional[str] = None

class ModeMetrics(BaseModel):
    mode: str
    display_name: str
    recall_at_k: Dict[int, float]
    precision_at_k: Dict[int, float]
    mrr: float
    map_score: float
    ndcg_at_k: Dict[int, float]
    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_p99: float
    category_recall_at_5: Dict[str, float] = Field(default_factory=dict)

class QueryResultDetail(BaseModel):
    query_id: str
    query: str
    category: str
    expected_chunks: List[str]
    ranks: Dict[str, Optional[int]] = {}  # mode -> first relevant rank
    hits_at_k: Dict[str, Dict[int, bool]] = {} # mode -> {k: True/False}

class EvaluationReport(BaseModel):
    timestamp: float
    total_queries: int
    corpus_chunks: int
    k_values: List[int]
    modes: Dict[str, ModeMetrics]
    query_details: List[QueryResultDetail]
    summary_markdown: str = ""

def compute_dcg(relevances: List[int], k: int) -> float:
    """Computes Discounted Cumulative Gain at k."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        if rel > 0:
            dcg += (rel) / math.log2(i + 1)
    return dcg

def compute_ndcg(retrieved_ids: List[str], expected_ids: set, k: int) -> float:
    """Computes Normalized Discounted Cumulative Gain at k."""
    actual_relevances = [1 if cid in expected_ids else 0 for cid in retrieved_ids[:k]]
    actual_dcg = compute_dcg(actual_relevances, k)
    
    # Ideal DCG: all relevant items ranked at top
    ideal_relevances = [1] * min(k, len(expected_ids))
    ideal_dcg = compute_dcg(ideal_relevances, k)
    
    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg

class RAGEvaluator:
    def __init__(
        self,
        hybrid_engine: Optional[HybridSearchEngine] = None,
        test_set_path: Optional[Path] = None,
        k_list: Optional[List[int]] = None
    ):
        self.test_set_path = Path(test_set_path or settings.test_set_path)
        self.k_list = k_list or settings.eval_k_list
        self.max_k = max(self.k_list)
        
        if hybrid_engine is None:
            loader = CorpusLoader(settings.corpus_dir)
            chunks = loader.load_corpus()
            vector_store = VectorStore()
            vector_store.index_chunks(chunks)
            keyword_engine = KeywordSearchEngine(chunks)
            self.hybrid_engine = HybridSearchEngine(vector_store, keyword_engine)
        else:
            self.hybrid_engine = hybrid_engine

        self.queries = self._load_test_set()

    def _load_test_set(self) -> List[EvalQuery]:
        if not self.test_set_path.exists():
            raise FileNotFoundError(f"Test set not found at {self.test_set_path}")
        with open(self.test_set_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [EvalQuery(**q) for q in data]

    def run_evaluation(self, alpha: Optional[float] = None) -> EvaluationReport:
        """
        Runs comprehensive benchmark across:
        - BM25
        - Dense Vector
        - Hybrid RRF
        - Hybrid Linear
        """
        alpha_val = alpha if alpha is not None else settings.default_alpha
        modes_to_test = [
            ("bm25", "BM25 (Keyword Only)", {}),
            ("dense", "Dense Embeddings Only", {}),
            ("hybrid_rrf", "Hybrid (RRF, k=60)", {}),
            ("hybrid_linear", f"Hybrid (Linear, α={alpha_val})", {"alpha": alpha_val}),
            ("hybrid_rerank", "Hybrid + Cross-Encoder Rerank", {"alpha": alpha_val}),
        ]

        mode_metrics: Dict[str, ModeMetrics] = {}
        all_query_details: List[QueryResultDetail] = [
            QueryResultDetail(
                query_id=q.query_id,
                query=q.query,
                category=q.category,
                expected_chunks=q.expected_chunks
            ) for q in self.queries
        ]
        query_map = {q.query_id: detail for detail, q in zip(all_query_details, self.queries)}

        for mode_key, display_name, extra_args in modes_to_test:
            recalls = {k: [] for k in self.k_list}
            precisions = {k: [] for k in self.k_list}
            ndcgs = {k: [] for k in self.k_list}
            mrr_scores = []
            map_scores = []
            latencies = []
            cat_recalls_at_5: Dict[str, List[float]] = {}

            for q in self.queries:
                expected_set = set(q.expected_chunks)
                
                # Measure latency
                t0 = time.perf_counter()
                results = self.hybrid_engine.search(
                    q.query,
                    top_k=self.max_k,
                    mode=mode_key,
                    **extra_args
                )
                t1 = time.perf_counter()
                latency_ms = (t1 - t0) * 1000.0
                latencies.append(latency_ms)

                retrieved_cids = [r.chunk_id for r in results]

                # First relevant rank for MRR
                first_rel_rank = None
                for rank, cid in enumerate(retrieved_cids, start=1):
                    if cid in expected_set:
                        first_rel_rank = rank
                        break

                query_map[q.query_id].ranks[mode_key] = first_rel_rank
                if mode_key not in query_map[q.query_id].hits_at_k:
                    query_map[q.query_id].hits_at_k[mode_key] = {}

                # Compute Recall@k and Precision@k
                for k in self.k_list:
                    top_k_cids = retrieved_cids[:k]
                    hits = len(expected_set.intersection(top_k_cids))
                    rec = hits / len(expected_set) if expected_set else 0.0
                    prec = hits / k
                    recalls[k].append(rec)
                    precisions[k].append(prec)
                    query_map[q.query_id].hits_at_k[mode_key][k] = (hits > 0)

                    # Track category recall@5
                    if k == 5:
                        cat_recalls_at_5.setdefault(q.category, []).append(rec)

                # NDCG@k
                for k in self.k_list:
                    ndcg_val = compute_ndcg(retrieved_cids, expected_set, k)
                    ndcgs[k].append(ndcg_val)

                # MRR
                if first_rel_rank is not None:
                    mrr_scores.append(1.0 / first_rel_rank)
                else:
                    mrr_scores.append(0.0)

                # Average Precision (AP) for MAP
                hits_count = 0
                running_prec_sum = 0.0
                for rank, cid in enumerate(retrieved_cids, start=1):
                    if cid in expected_set:
                        hits_count += 1
                        running_prec_sum += (hits_count / rank)
                ap = running_prec_sum / min(len(expected_set), self.max_k) if expected_set else 0.0
                map_scores.append(ap)

            # Summarize metrics
            mean_recalls = {k: float(np.mean(recalls[k])) for k in self.k_list}
            mean_precisions = {k: float(np.mean(precisions[k])) for k in self.k_list}
            mean_ndcgs = {k: float(np.mean(ndcgs[k])) for k in self.k_list}
            mean_mrr = float(np.mean(mrr_scores))
            mean_map = float(np.mean(map_scores))
            
            cat_summary = {
                cat: float(np.mean(scores)) for cat, scores in cat_recalls_at_5.items()
            }

            lat_sorted = sorted(latencies)
            mode_metrics[mode_key] = ModeMetrics(
                mode=mode_key,
                display_name=display_name,
                recall_at_k=mean_recalls,
                precision_at_k=mean_precisions,
                mrr=mean_mrr,
                map_score=mean_map,
                ndcg_at_k=mean_ndcgs,
                latency_ms_mean=float(np.mean(latencies)),
                latency_ms_p50=float(np.percentile(lat_sorted, 50)),
                latency_ms_p95=float(np.percentile(lat_sorted, 95)),
                latency_ms_p99=float(np.percentile(lat_sorted, 99)),
                category_recall_at_5=cat_summary
            )

        report = EvaluationReport(
            timestamp=time.time(),
            total_queries=len(self.queries),
            corpus_chunks=len(self.hybrid_engine.vector_store.get_all_chunks()),
            k_values=self.k_list,
            modes=mode_metrics,
            query_details=all_query_details
        )
        report.summary_markdown = self.format_markdown_report(report)
        return report

    @staticmethod
    def format_markdown_report(report: EvaluationReport) -> str:
        """Formats the evaluation results as a clean GitHub-style Markdown table."""
        headers = ["Search Mode", "Recall@1", "Recall@3", "Recall@5", "Recall@10", "MRR", "MAP", "NDCG@5", "P50 Latency (ms)"]
        rows = []
        for m_key, m in report.modes.items():
            r1 = f"{m.recall_at_k.get(1, 0.0) * 100:.1f}%"
            r3 = f"{m.recall_at_k.get(3, 0.0) * 100:.1f}%"
            r5 = f"{m.recall_at_k.get(5, 0.0) * 100:.1f}%"
            r10 = f"{m.recall_at_k.get(10, 0.0) * 100:.1f}%"
            mrr = f"{m.mrr:.3f}"
            map_s = f"{m.map_score:.3f}"
            ndcg5 = f"{m.ndcg_at_k.get(5, 0.0):.3f}"
            p50 = f"{m.latency_ms_p50:.2f}"
            rows.append([m.display_name, r1, r3, r5, r10, mrr, map_s, ndcg5, p50])

        table_str = tabulate(rows, headers=headers, tablefmt="github")
        
        # Category breakdown table
        cat_headers = ["Search Mode", "Keyword Exact (R@5)", "Semantic Concept (R@5)", "Hybrid Balanced (R@5)"]
        cat_rows = []
        for m_key, m in report.modes.items():
            kw = f"{m.category_recall_at_5.get('keyword_exact', 0.0) * 100:.1f}%"
            sem = f"{m.category_recall_at_5.get('semantic_concept', 0.0) * 100:.1f}%"
            hyb = f"{m.category_recall_at_5.get('hybrid_balanced', 0.0) * 100:.1f}%"
            cat_rows.append([m.display_name, kw, sem, hyb])
        cat_table_str = tabulate(cat_rows, headers=cat_headers, tablefmt="github")

        md = f"""# 📊 Local-First RAG Quality Measurement Benchmark

**Benchmark Dataset:** {report.total_queries} curated test queries across {report.corpus_chunks} technical specification chunks.
**Local Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (Local inference)
**Keyword Engine:** Okapi BM25 (`rank-bm25`)

### Primary Retrieval Performance (Recall@k & Ranking Metrics)

{table_str}

### Performance Breakdown by Query Archetype (Recall@5)

{cat_table_str}
"""
        return md

