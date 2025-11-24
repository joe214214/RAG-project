from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ml.embeddings import SentenceTransformerEmbedder
from vector_stores.finedb import FineChunkVectorStore
from vector_stores.superdb import SuperChunkVectorStore


@dataclass
class RetrievedChunk:
    chunk_id: str
    score: float
    text: str
    level: str  # "super" or "fine"


class HierarchicalRetriever:
    """
    Implements a two-stage retrieval strategy:
        1. Retrieve candidate super chunks.
        2. Search within child fine chunks tied to each super chunk.
    """

    def __init__(
        self,
        super_store: SuperChunkVectorStore,
        fine_store: FineChunkVectorStore,
        embedder: SentenceTransformerEmbedder,
        super_chunk_paths: Dict[str, str],
        fine_chunk_paths: Dict[str, str],
        super_to_fine: Dict[str, List[str]],
    ) -> None:
        self.super_store = super_store
        self.fine_store = fine_store
        self.embedder = embedder
        self.super_chunk_paths = {key: Path(path) for key, path in super_chunk_paths.items()}
        self.fine_chunk_paths = {key: Path(path) for key, path in fine_chunk_paths.items()}
        self.super_to_fine = super_to_fine
        self._cache: Dict[str, str] = {}

    def _read_chunk(self, chunk_id: str, level: str) -> str:
        cache_key = f"{level}:{chunk_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        path_map = self.super_chunk_paths if level == "super" else self.fine_chunk_paths
        path = path_map.get(chunk_id)
        if not path:
            return ""
        text = path.read_text(encoding="utf-8")
        self._cache[cache_key] = text
        return text

    def _filter_vector_results(
        self,
        results: List[tuple[str, float]],
        allowed_ids: Optional[Sequence[str]],
        top_k: int,
    ) -> List[tuple[str, float]]:
        if not allowed_ids:
            return results[:top_k]
        allowed = set(allowed_ids)
        filtered = [item for item in results if item[0] in allowed]
        return filtered[:top_k]

    def retrieve(
        self,
        query: str,
        *,
        top_k_super: int = 5,
        top_k_fine: int = 10,
        use_hierarchical: bool = True,
    ) -> Dict[str, List[RetrievedChunk]]:
        query_vec = self.embedder.embed_query(query)

        super_hits: List[RetrievedChunk] = []
        fine_hits: List[RetrievedChunk] = []

        if use_hierarchical:
            super_results = self.super_store.search(query_vec, top_k_super)
            for chunk_id, score in super_results:
                super_hits.append(
                    RetrievedChunk(
                        chunk_id=chunk_id,
                        score=score,
                        text=self._read_chunk(chunk_id, level="super"),
                        level="super",
                    )
                )
            candidate_fine_ids: List[str] = []
            for chunk in super_hits:
                candidate_fine_ids.extend(self.super_to_fine.get(chunk.chunk_id, []))
            fine_results = self._filter_vector_results(
                self.fine_store.search(query_vec, top_k=max(top_k_fine * 5, 50)),
                allowed_ids=candidate_fine_ids,
                top_k=top_k_fine,
            )
        else:
            fine_results = self.fine_store.search(query_vec, top_k=top_k_fine)

        for chunk_id, score in fine_results:
            fine_hits.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    score=score,
                    text=self._read_chunk(chunk_id, level="fine"),
                    level="fine",
                )
            )

        return {"super_chunks": super_hits, "fine_chunks": fine_hits}
