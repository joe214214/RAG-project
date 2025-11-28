#!/usr/bin/env python3
"""
Hybrid Retriever - Combines BM25 (sparse) + Qdrant (dense) retrieval.

Fusion: score = α * BM25 + (1-α) * dense

Benefits:
- BM25: Exact keyword matching, entities, rare terms
- Dense: Semantic similarity, paraphrases, context
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from retrievers.bm25_retriever import BM25Retriever


@dataclass
class HybridResult:
    """Result from hybrid retrieval with component scores."""
    doc_id: str
    hybrid_score: float
    bm25_score: float
    embedding_score: float


class HybridRetriever:
    """
    Combine lexical BM25 and dense Qdrant retrieval via interpolation.
    
    Usage:
        from retrievers import BM25Retriever, HybridRetriever
        from vector_stores.qdrant_store import QdrantVectorStore
        from sentence_transformers import SentenceTransformer
        
        # Build BM25 from corpus
        bm25 = BM25Retriever.from_id_to_text(id_to_text)
        
        # Initialize Qdrant and embedder
        qdrant = QdrantVectorStore(config, collection_config)
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Create hybrid retriever
        hybrid = HybridRetriever(bm25, qdrant, embedder)
        
        # Search
        results = hybrid.retrieve("query", top_k=10, alpha=0.3)
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        qdrant_store,  # QdrantVectorStore
        embedder,      # SentenceTransformer or similar with encode() method
    ) -> None:
        """
        Initialize hybrid retriever.
        
        Args:
            bm25: BM25Retriever instance
            qdrant_store: QdrantVectorStore instance  
            embedder: Model with encode() method for query embedding
        """
        self.bm25 = bm25
        self.qdrant = qdrant_store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        alpha: float = 0.3,
        allowed_ids: Optional[Sequence[str]] = None,
    ) -> List[HybridResult]:
        """
        Perform hybrid search combining BM25 and dense retrieval.
        
        Args:
            query: Search query
            top_k: Number of final results
            alpha: BM25 weight (0-1). 0=dense only, 1=BM25 only. Default 0.3.
            allowed_ids: Optional filter to restrict results to specific doc IDs
            
        Returns:
            List of HybridResult sorted by hybrid score
        """
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")

        # Get BM25 results
        bm25_results = self.bm25.retrieve(query, top_k=max(top_k * 2, 50))
        bm25_scores = dict(bm25_results)

        # Get dense results from Qdrant
        query_vec = self.embedder.encode(query)
        if isinstance(query_vec, list):
            query_vec = np.array(query_vec)
        
        # Qdrant search returns list of (doc_id, score, metadata)
        dense_results = self.qdrant.search(query_vec, top_k=max(top_k * 5, 50))

        # Apply ID filter if provided
        id_filter = set(allowed_ids) if allowed_ids else None

        # Combine scores
        combined_scores: Dict[str, HybridResult] = {}
        
        # Process dense results first
        for doc_id, score, metadata in dense_results:
            if id_filter and doc_id not in id_filter:
                continue
            combined_scores[doc_id] = HybridResult(
                doc_id=doc_id,
                hybrid_score=(1 - alpha) * score,
                bm25_score=0.0,
                embedding_score=score,
            )

        # Add BM25 scores
        for doc_id, score in bm25_scores.items():
            if id_filter and doc_id not in id_filter:
                continue
            if doc_id not in combined_scores:
                combined_scores[doc_id] = HybridResult(
                    doc_id=doc_id,
                    hybrid_score=alpha * score,
                    bm25_score=score,
                    embedding_score=0.0,
                )
            else:
                result = combined_scores[doc_id]
                result.hybrid_score = alpha * score + (1 - alpha) * result.embedding_score
                result.bm25_score = score

        # Sort by hybrid score and return top-k
        ranked = sorted(combined_scores.values(), key=lambda r: r.hybrid_score, reverse=True)
        return ranked[:top_k]
    
    def retrieve_bm25_only(self, query: str, top_k: int = 10) -> List[HybridResult]:
        """Search using BM25 only (for ablation)."""
        return self.retrieve(query, top_k=top_k, alpha=1.0)
    
    def retrieve_dense_only(self, query: str, top_k: int = 10) -> List[HybridResult]:
        """Search using dense retrieval only (for ablation)."""
        return self.retrieve(query, top_k=top_k, alpha=0.0)
