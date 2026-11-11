"""Unit tests for Dense, BM25, and Hybrid search engines."""
import pytest
from pathlib import Path
from src.config import settings
from src.corpus_loader import CorpusLoader
from src.vector_store import VectorStore
from src.keyword_search import KeywordSearchEngine, tokenize
from src.hybrid_search import HybridSearchEngine

@pytest.fixture(scope="module")
def pipeline():
    loader = CorpusLoader(settings.corpus_dir)
    chunks = loader.load_corpus()
    vstore = VectorStore()
    if vstore.count() != len(chunks):
        vstore.index_chunks(chunks)
    kengine = KeywordSearchEngine(chunks)
    hybrid = HybridSearchEngine(vstore, kengine)
    return chunks, vstore, kengine, hybrid

def test_corpus_loaded(pipeline):
    chunks, vstore, kengine, hybrid = pipeline
    assert len(chunks) == 42
    assert vstore.count() == 42

def test_tokenizer():
    tokens = tokenize("RAFT_ELECTION_TIMEOUT_MS and W + R > N in 2PC")
    assert "raft_election_timeout_ms" in tokens
    assert "w" in tokens
    assert "plus" in tokens
    assert "greater_than" in tokens
    assert "2pc" in tokens

def test_dense_search(pipeline):
    _, vstore, _, _ = pipeline
    results = vstore.search("Raft consensus leader election", top_k=3)
    assert len(results) == 3
    assert any("raft" in r.doc_id for r in results)

def test_bm25_search(pipeline):
    _, _, kengine, _ = pipeline
    results = kengine.search("HEARTBEAT_INTERVAL_MS and ELECTION_TIMEOUT_MIN_MS", top_k=3)
    assert len(results) == 3
    assert results[0].chunk_id == "01_raft_consensus#chunk-1"

def test_hybrid_rrf_search(pipeline):
    _, _, _, hybrid = pipeline
    results = hybrid.search_rrf("Two-Phase Commit prepare phase and lock tables", top_k=5)
    assert len(results) == 5
    assert results[0].fusion_method == "rrf"
    assert results[0].dense_rank >= 1
    assert results[0].bm25_rank >= 1

def test_hybrid_linear_search(pipeline):
    _, _, _, hybrid = pipeline
    results = hybrid.search_linear("Consistent hashing with virtual nodes vnodes_per_node", top_k=5, alpha=0.5)
    assert len(results) == 5
    assert results[0].chunk_id == "11_sharding_and_rebalancing#chunk-0"
    assert results[0].fusion_method == "linear"
