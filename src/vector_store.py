"""Local vector store backed by SQLite and numpy vector indexing."""
import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pydantic import BaseModel
from src.corpus_loader import Chunk
from src.embeddings import EmbeddingEngine
from src.config import settings

class SearchResult(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    section_title: str
    content: str
    score: float
    rank: int
    metadata: Dict[str, Any] = {}

class VectorStore:
    def __init__(self, db_path: Optional[Path] = None, embedding_engine: Optional[EmbeddingEngine] = None):
        self.db_path = Path(db_path or settings.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_engine = embedding_engine or EmbeddingEngine.get_instance()
        
        # In-memory cache for fast vector operations
        self._chunk_ids: List[str] = []
        self._chunks_by_id: Dict[str, Chunk] = {}
        self._embedding_matrix: Optional[np.ndarray] = None
        
        self._init_db()
        self._load_cache()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                doc_title TEXT NOT NULL,
                section_title TEXT NOT NULL,
                content TEXT NOT NULL,
                text_for_embedding TEXT NOT NULL,
                word_count INTEGER NOT NULL,
                char_count INTEGER NOT NULL,
                metadata_json TEXT NOT NULL,
                embedding BLOB NOT NULL
            );
            """)
            conn.commit()

    def _load_cache(self):
        with self._get_connection() as conn:
            cursor = conn.execute("""
            SELECT chunk_id, doc_id, doc_title, section_title, content, 
                   text_for_embedding, word_count, char_count, metadata_json, embedding
            FROM chunks
            ORDER BY chunk_id ASC
            """)
            rows = cursor.fetchall()

        if not rows:
            self._chunk_ids = []
            self._chunks_by_id = {}
            self._embedding_matrix = None
            return

        self._chunk_ids = []
        self._chunks_by_id = {}
        embeddings_list = []

        for row in rows:
            c_id = row["chunk_id"]
            self._chunk_ids.append(c_id)
            meta = json.loads(row["metadata_json"])
            chunk = Chunk(
                chunk_id=c_id,
                doc_id=row["doc_id"],
                doc_title=row["doc_title"],
                section_title=row["section_title"],
                content=row["content"],
                text_for_embedding=row["text_for_embedding"],
                word_count=row["word_count"],
                char_count=row["char_count"],
                metadata=meta
            )
            self._chunks_by_id[c_id] = chunk
            emb = np.frombuffer(row["embedding"], dtype=np.float32)
            embeddings_list.append(emb)

        self._embedding_matrix = np.vstack(embeddings_list)

    def count(self) -> int:
        return len(self._chunk_ids)

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        return self._chunks_by_id.get(chunk_id)

    def get_all_chunks(self) -> List[Chunk]:
        return [self._chunks_by_id[cid] for cid in self._chunk_ids]

    def index_chunks(self, chunks: List[Chunk], force_reindex: bool = False):
        """Computes embeddings for chunks and saves them to SQLite."""
        if not chunks:
            return

        if not force_reindex and self.count() == len(chunks):
            # Check if chunk IDs match
            existing_ids = set(self._chunk_ids)
            new_ids = set(c.chunk_id for c in chunks)
            if existing_ids == new_ids:
                print(f"[VectorStore] Already indexed {len(chunks)} chunks in SQLite database. Skipping reindex.")
                return

        print(f"[VectorStore] Indexing {len(chunks)} chunks with local embeddings...")
        texts = [c.text_for_embedding for c in chunks]
        embeddings = self.embedding_engine.encode_batch(texts, batch_size=settings.batch_size, show_progress=True)

        with self._get_connection() as conn:
            conn.execute("DELETE FROM chunks")
            for chunk, emb in zip(chunks, embeddings):
                conn.execute("""
                INSERT INTO chunks (
                    chunk_id, doc_id, doc_title, section_title, content,
                    text_for_embedding, word_count, char_count, metadata_json, embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk.chunk_id,
                    chunk.doc_id,
                    chunk.doc_title,
                    chunk.section_title,
                    chunk.content,
                    chunk.text_for_embedding,
                    chunk.word_count,
                    chunk.char_count,
                    json.dumps(chunk.metadata),
                    emb.tobytes()
                ))
            conn.commit()

        # Reload cache
        self._load_cache()
        print(f"[VectorStore] Successfully stored and cached {len(self._chunk_ids)} chunk embeddings.")

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Dense semantic search using cosine similarity."""
        if self._embedding_matrix is None or len(self._chunk_ids) == 0:
            return []

        query_vec = self.embedding_engine.encode_text(query)
        scores = self.embedding_engine.compute_similarity(query_vec, self._embedding_matrix)

        # Get top-k indices sorted descending
        k = min(top_k, len(self._chunk_ids))
        top_indices = np.argsort(scores)[::-1][:k]

        results: List[SearchResult] = []
        for rank, idx in enumerate(top_indices, start=1):
            c_id = self._chunk_ids[idx]
            chunk = self._chunks_by_id[c_id]
            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    doc_title=chunk.doc_title,
                    section_title=chunk.section_title,
                    content=chunk.content,
                    score=float(scores[idx]),
                    rank=rank,
                    metadata=chunk.metadata
                )
            )

        return results
