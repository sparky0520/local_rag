"""Local keyword search using Okapi BM25."""
import re
from typing import List, Dict, Tuple, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from src.corpus_loader import Chunk
from src.vector_store import SearchResult

# Standard English stopwords (preserving common system keywords like L0, 2Q, 2PC, WAL)
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}

def tokenize(text: str) -> List[str]:
    """
    Robust tokenizer for technical corpus:
    Preserves alphanumeric terms, underscored variables (e.g. HEARTBEAT_INTERVAL_MS),
    hyphenated terms (e.g. Ping-Req), and handles mathematical equations.
    """
    if not text:
        return []
    
    # Replace math operators with whitespace-separated words so symbols are indexed
    cleaned = text.replace("+", " plus ").replace(">", " greater_than ").replace("<", " less_than ")
    
    # Find word tokens including underscores and hyphens
    tokens = re.findall(r"[a-zA-Z0-9_]+(?:-[a-zA-Z0-9_]+)*", cleaned.lower())
    
    # Filter stopwords unless token is short technical term or number
    filtered = [t for t in tokens if t not in STOPWORDS or len(t) <= 2 or t.isdigit()]
    return filtered

class KeywordSearchEngine:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.chunk_ids = [c.chunk_id for c in chunks]
        self._chunks_by_id = {c.chunk_id: c for c in chunks}
        
        # Build tokenized corpus
        self.tokenized_corpus = [
            tokenize(f"{c.doc_title} {c.section_title} {c.content}")
            for c in chunks
        ]
        
        # Initialize BM25
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def get_all_scores(self, query: str) -> Dict[str, float]:
        """Computes BM25 score for every chunk in the corpus."""
        tokens = tokenize(query)
        if not tokens:
            return {cid: 0.0 for cid in self.chunk_ids}
        
        raw_scores = self.bm25.get_scores(tokens)
        return {cid: float(score) for cid, score in zip(self.chunk_ids, raw_scores)}

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Performs BM25 keyword search and returns top-k SearchResults."""
        tokens = tokenize(query)
        if not tokens:
            return []

        raw_scores = self.bm25.get_scores(tokens)
        
        # Filter out 0 score results if possible, or sort all
        ranked_indices = np.argsort(raw_scores)[::-1]
        
        k = min(top_k, len(self.chunks))
        top_indices = ranked_indices[:k]

        results: List[SearchResult] = []
        for rank, idx in enumerate(top_indices, start=1):
            score = float(raw_scores[idx])
            chunk = self.chunks[idx]
            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    doc_title=chunk.doc_title,
                    section_title=chunk.section_title,
                    content=chunk.content,
                    score=score,
                    rank=rank,
                    metadata=chunk.metadata
                )
            )

        return results

