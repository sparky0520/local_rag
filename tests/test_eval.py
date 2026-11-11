"""Unit tests for Evaluation metrics (Recall@k, MRR, NDCG, MAP)."""
import pytest
from src.evaluation import RAGEvaluator, compute_dcg, compute_ndcg

def test_compute_dcg():
    relevances = [1, 0, 1]
    # dcg = 1/log2(2) + 0 + 1/log2(4) = 1.0 + 0 + 0.5 = 1.5
    assert abs(compute_dcg(relevances, 3) - 1.5) < 1e-4

def test_compute_ndcg():
    retrieved = ["docA", "docB", "docC"]
    expected = {"docA"}
    # Ideal: docA at rank 1 -> DCG=1.0, IDCG=1.0 -> NDCG=1.0
    assert compute_ndcg(retrieved, expected, 3) == 1.0

    retrieved2 = ["docB", "docA", "docC"]
    # Actual: docA at rank 2 -> DCG=1/log2(3) ≈ 0.6309, IDCG=1.0 -> NDCG ≈ 0.6309
    ndcg2 = compute_ndcg(retrieved2, expected, 3)
    assert 0.63 < ndcg2 < 0.64

def test_evaluator_run():
    evaluator = RAGEvaluator()
    assert len(evaluator.queries) == 36
    report = evaluator.run_evaluation(alpha=0.5)
    
    assert report.total_queries == 36
    assert report.corpus_chunks == 42
    assert "bm25" in report.modes
    assert "dense" in report.modes
    assert "hybrid_rrf" in report.modes
    assert "hybrid_linear" in report.modes

    # Check Recall@5 is high
    assert report.modes["hybrid_linear"].recall_at_k[5] >= 0.95
    assert report.modes["bm25"].recall_at_k[5] >= 0.90
    assert report.modes["dense"].recall_at_k[5] >= 0.85
