from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import faiss
import numpy as np


@dataclass
class VectorStoreConfig:
    dim: int
    metric: str = "ip"  # ip (cosine) or l2
    persist_dir: Path | None = None


class FaissVectorStore:
    """
    Minimal FAISS wrapper that stores embeddings + metadata for later retrieval.
    """

    def __init__(self, config: VectorStoreConfig) -> None:
        self.config = config
        self._id_to_index: Dict[str, int] = {}
        self._index: faiss.Index | None = None

    def _build_index(self) -> faiss.Index:
        if self.config.metric == "l2":
            index = faiss.IndexFlatL2(self.config.dim)
        else:
            index = faiss.IndexFlatIP(self.config.dim)
        return index

    def build(self, embeddings: np.ndarray, ids: Sequence[str]) -> None:
        if embeddings.shape[0] != len(ids):
            raise ValueError("Number of embeddings must match ids.")
        if embeddings.shape[1] != self.config.dim:
            raise ValueError(f"Embedding dim mismatch: {embeddings.shape[1]} vs {self.config.dim}")

        self._index = self._build_index()
        # For cosine similarity, FAISS expects normalized vectors.
        if self.config.metric == "ip":
            faiss.normalize_L2(embeddings)

        self._index.add(embeddings.astype("float32"))
        self._id_to_index = {doc_id: idx for idx, doc_id in enumerate(ids)}

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        if self._index is None:
            raise RuntimeError("Index has not been built.")
        query_vec = np.expand_dims(query_vector.astype("float32"), axis=0)
        if self.config.metric == "ip":
            faiss.normalize_L2(query_vec)
        scores, indices = self._index.search(query_vec, top_k)
        inv_map = {idx: doc_id for doc_id, idx in self._id_to_index.items()}

        results: List[Tuple[str, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            doc_id = inv_map.get(idx)
            if doc_id is None:
                continue
            results.append((doc_id, float(score)))
        return results

    def save(self, directory: Path | str) -> None:
        if self._index is None:
            raise RuntimeError("Index has not been built.")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / "vector.index"))
        (directory / "ids.txt").write_text("\n".join(self._id_to_index.keys()), encoding="utf-8")

    def load(self, directory: Path | str) -> None:
        directory = Path(directory)
        index_path = directory / "vector.index"
        ids_path = directory / "ids.txt"
        if not index_path.exists() or not ids_path.exists():
            raise FileNotFoundError(f"Vector store files missing under {directory}")

        self._index = faiss.read_index(str(index_path))
        ids = ids_path.read_text(encoding="utf-8").splitlines()
        self._id_to_index = {doc_id: idx for idx, doc_id in enumerate(ids)}


class SuperChunkVectorStore(FaissVectorStore):
    """
    Concrete vector store for super chunks.
    """

    def __init__(self, dim: int, persist_dir: Path | None = None) -> None:
        super().__init__(VectorStoreConfig(dim=dim, persist_dir=persist_dir))


def build_superdb_from_embeddings(
    embeddings: np.ndarray,
    ids: Sequence[str],
    *,
    persist_dir: Path | None = None,
) -> SuperChunkVectorStore:
    store = SuperChunkVectorStore(dim=embeddings.shape[1], persist_dir=persist_dir)
    store.build(embeddings, ids)
    if persist_dir:
        store.save(persist_dir)
    return store

