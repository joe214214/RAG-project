from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ml.embeddings import SentenceTransformerEmbedder
from retrievers.bm25_retriever import BM25Retriever
from vector_stores.superdb import FaissVectorStore


@dataclass
class HybridResult:
    doc_id: str
    hybrid_score: float
    bm25_score: float
    embedding_score: float


class HybridRetriever:
    """
    Combine lexical BM25 and dense retrieval via interpolation.
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        vector_store: FaissVectorStore,
        embedder: SentenceTransformerEmbedder,
    ) -> None:
        self.bm25 = bm25
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        alpha: float = 0.3,
        allowed_ids: Optional[Sequence[str]] = None,
    ) -> List[HybridResult]:
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")

        bm25_scores = dict(self.bm25.retrieve(query, top_k=max(top_k * 2, top_k)))

        query_vec = self.embedder.embed_query(query)
        vector_results = self.vector_store.search(query_vec, top_k=max(top_k * 5, 50))

        id_filter = set(allowed_ids) if allowed_ids else None

        combined_scores: Dict[str, HybridResult] = {}
        for doc_id, score in vector_results:
            if id_filter and doc_id not in id_filter:
                continue
            combined_scores[doc_id] = HybridResult(
                doc_id=doc_id,
                hybrid_score=(1 - alpha) * score,
                bm25_score=0.0,
                embedding_score=score,
            )

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

        ranked = sorted(combined_scores.values(), key=lambda r: r.hybrid_score, reverse=True)
        return ranked[:top_k]

