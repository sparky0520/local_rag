"""Configuration settings for Local-First RAG."""
import os
from pathlib import Path
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class Settings(BaseModel):
    # Paths
    project_root: Path = PROJECT_ROOT
    corpus_dir: Path = PROJECT_ROOT / "data" / "corpus"
    test_set_path: Path = PROJECT_ROOT / "data" / "test_set.json"
    db_path: Path = PROJECT_ROOT / "data" / "rag.db"
    report_output_path: Path = PROJECT_ROOT / "data" / "evaluation_report.json"

    # Embedding Model Settings
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    batch_size: int = 32

    # Retrieval & Hybrid Search
    default_top_k: int = 5
    rrf_k: int = 60
    default_alpha: float = 0.5  # Convex combination: alpha * dense + (1 - alpha) * bm25

    # Evaluation K values
    eval_k_list: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])

    # Server
    host: str = "127.0.0.1"
    port: int = 8000

settings = Settings()
