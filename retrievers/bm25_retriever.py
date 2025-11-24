from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from rank_bm25 import BM25Okapi


@dataclass
class BM25Document:
    doc_id: str
    text: str


class BM25Retriever:
    """
    Lightweight BM25 retriever wrapping rank-bm25.
    """

    def __init__(self, documents: Sequence[BM25Document]) -> None:
        self.documents = list(documents)
        tokenized = [doc.text.lower().split() for doc in self.documents]
        self._bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        scores = self._bm25.get_scores(query.lower().split())
        doc_scores = list(zip((doc.doc_id for doc in self.documents), scores))
        doc_scores.sort(key=lambda item: item[1], reverse=True)
        return doc_scores[:top_k]

    @classmethod
    def from_id_to_text(cls, id_to_text: Dict[str, str]) -> "BM25Retriever":
        documents = [BM25Document(doc_id=doc_id, text=text) for doc_id, text in id_to_text.items()]
        return cls(documents)

