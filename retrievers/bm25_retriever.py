#!/usr/bin/env python3
"""
BM25 Sparse Retriever - Lightweight wrapper for rank-bm25.

Used for lexical/keyword-based retrieval to complement dense vector search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import numpy as np

from rank_bm25 import BM25Okapi


@dataclass
class BM25Document:
    """Document with ID and text for BM25 indexing."""
    doc_id: str
    text: str


class BM25Retriever:
    """
    Lightweight BM25 retriever wrapping rank-bm25.
    
    Usage:
        # From dict
        retriever = BM25Retriever.from_id_to_text({"doc1": "text1", "doc2": "text2"})
        
        # Search - returns list of (doc_id, score) tuples
        results = retriever.retrieve("query text", top_k=10)
    """

    def __init__(self, documents: Sequence[BM25Document]) -> None:
        """Initialize BM25 index from documents."""
        self.documents = list(documents)
        tokenized = [doc.text.lower().split() for doc in self.documents]
        self._bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Retrieve top-k documents matching the query.
        
        Optimized to use numpy argpartition for efficient top-k selection
        without fully sorting all scores (O(n + k log k) vs O(n log n)).
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of (doc_id, score) tuples sorted by relevance
        """
        query_tokens = query.lower().split()
        
        # Get scores for all documents (still O(n) but unavoidable for BM25)
        scores = self._bm25.get_scores(query_tokens)
        scores_array = np.array(scores)
        
        # Use argpartition to get top-k indices without full sort
        # This is O(n) instead of O(n log n) for full sort
        if len(scores_array) <= top_k:
            # If we have fewer docs than top_k, just sort all
            top_indices = np.argsort(scores_array)[::-1]
        else:
            # Get indices of top-k largest scores (negative for descending order)
            top_indices = np.argpartition(-scores_array, top_k)[:top_k]
            # Sort only the top-k indices by score (O(k log k))
            top_indices = top_indices[np.argsort(-scores_array[top_indices])]
        
        # Map indices to (doc_id, score) tuples
        results = [
            (self.documents[idx].doc_id, float(scores_array[idx]))
            for idx in top_indices
        ]
        
        return results

    @classmethod
    def from_id_to_text(cls, id_to_text: Dict[str, str]) -> "BM25Retriever":
        """Create BM25Retriever from a dictionary mapping IDs to text."""
        documents = [
            BM25Document(doc_id=doc_id, text=text) 
            for doc_id, text in id_to_text.items()
        ]
        return cls(documents)
    
    def __len__(self) -> int:
        return len(self.documents)
    
    def __repr__(self) -> str:
        return f"BM25Retriever(n_docs={len(self.documents)})"
