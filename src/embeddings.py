"""Local embedding engine powered by SentenceTransformers."""
from typing import List, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from src.config import settings

class EmbeddingEngine:
    _instance: Optional["EmbeddingEngine"] = None

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.embedding_model_name
        self.device = self._detect_device()
        print(f"[EmbeddingEngine] Initializing model '{self.model_name}' on device '{self.device}'...")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        # get_embedding_dimension or get_sentence_embedding_dimension
        if hasattr(self.model, "get_embedding_dimension"):
            self.dimension = self.model.get_embedding_dimension()
        else:
            self.dimension = self.model.get_sentence_embedding_dimension()

    @classmethod
    def get_instance(cls, model_name: Optional[str] = None) -> "EmbeddingEngine":
        if cls._instance is None:
            cls._instance = cls(model_name)
        return cls._instance

    def _detect_device(self) -> str:
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a single text string into a normalized 1D float32 vector."""
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embedding.astype(np.float32)

    def encode_batch(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """Encode a batch of texts into a normalized 2D (N, D) float32 matrix."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress
        )
        return embeddings.astype(np.float32)

    @staticmethod
    def compute_similarity(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity scores between a normalized query vector (D,)
        and a matrix of normalized document vectors (N, D) or single vector (D,).
        Returns a 1D array of scores (N,).
        """
        query_vec = np.asarray(query_vec, dtype=np.float32)
        doc_matrix = np.asarray(doc_matrix, dtype=np.float32)

        if doc_matrix.ndim == 1:
            doc_matrix = doc_matrix.reshape(1, -1)
        if query_vec.ndim == 2:
            query_vec = query_vec.squeeze(0)

        # Dot product of L2-normalized vectors is exact cosine similarity
        scores = np.dot(doc_matrix, query_vec)
        return scores
