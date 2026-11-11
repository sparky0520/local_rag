"""FastAPI backend for Local-First RAG Web UI and Benchmark Dashboard."""
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.config import settings
from src.corpus_loader import CorpusLoader
from src.vector_store import VectorStore
from src.keyword_search import KeywordSearchEngine
from src.hybrid_search import HybridSearchEngine
from src.evaluation import RAGEvaluator, EvaluationReport
from src.rag_engine import RAGEngine

app = FastAPI(
    title="Local-First RAG with Measured Quality",
    description="Hybrid search (BM25 + Local SentenceTransformer embeddings) with measured Recall@k benchmark",
    version="1.0.0"
)

# Shared in-memory pipeline
loader = CorpusLoader(settings.corpus_dir)
chunks = loader.load_corpus()
vstore = VectorStore()
if vstore.count() != len(chunks):
    vstore.index_chunks(chunks)
kengine = KeywordSearchEngine(chunks)
hybrid_engine = HybridSearchEngine(vstore, kengine)
rag_engine = RAGEngine(hybrid_engine)
evaluator = RAGEvaluator(hybrid_engine=hybrid_engine)

# Cached latest evaluation report
_cached_report: Optional[EvaluationReport] = None

class SearchRequest(BaseModel):
    query: str
    mode: str = "hybrid_rrf"
    top_k: int = 5
    alpha: Optional[float] = 0.5

class AskRequest(BaseModel):
    query: str
    mode: str = "hybrid_rrf"
    top_k: int = 5
    alpha: Optional[float] = 0.5

class EvalRunRequest(BaseModel):
    alpha: Optional[float] = 0.5

@app.get("/api/benchmark")
def get_benchmark():
    global _cached_report
    if _cached_report is None:
        if settings.report_output_path.exists():
            try:
                with open(settings.report_output_path, "r", encoding="utf-8") as f:
                    _cached_report = EvaluationReport(**json.load(f))
            except Exception:
                _cached_report = evaluator.run_evaluation()
        else:
            _cached_report = evaluator.run_evaluation()
    return _cached_report

@app.post("/api/benchmark/run")
def trigger_benchmark(req: EvalRunRequest):
    global _cached_report
    _cached_report = evaluator.run_evaluation(alpha=req.alpha)
    # Save to disk
    with open(settings.report_output_path, "w", encoding="utf-8") as f:
        f.write(_cached_report.model_dump_json(indent=2))
    return _cached_report

@app.post("/api/search")
def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    results = hybrid_engine.search(
        req.query,
        top_k=req.top_k,
        mode=req.mode,
        alpha=req.alpha
    )
    return {"results": results, "mode": req.mode, "query": req.query}

@app.post("/api/ask")
def ask(req: AskRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    response = rag_engine.ask(
        req.query,
        mode=req.mode,
        top_k=req.top_k,
        alpha=req.alpha
    )
    return response

@app.get("/api/corpus")
def get_corpus():
    all_chunks = vstore.get_all_chunks()
    return {
        "total_chunks": len(all_chunks),
        "documents_count": len(set(c.doc_id for c in all_chunks)),
        "chunks": [c.model_dump() for c in all_chunks]
    }

@app.get("/api/test-set")
def get_test_set():
    with open(settings.test_set_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {"total_queries": len(data), "queries": data}

# Mount static web UI
static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
def index():
    return FileResponse(static_dir / "index.html")
