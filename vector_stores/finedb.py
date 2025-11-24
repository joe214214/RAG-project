from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .superdb import FaissVectorStore, VectorStoreConfig


class FineChunkVectorStore(FaissVectorStore):
    """
    Vector store dedicated to fine-grained chunks.
    """

    def __init__(self, dim: int, persist_dir: Path | None = None) -> None:
        super().__init__(VectorStoreConfig(dim=dim, persist_dir=persist_dir))


def build_finedb_from_embeddings(
    embeddings: np.ndarray,
    ids: Sequence[str],
    *,
    persist_dir: Path | None = None,
) -> FineChunkVectorStore:
    store = FineChunkVectorStore(dim=embeddings.shape[1], persist_dir=persist_dir)
    store.build(embeddings, ids)
    if persist_dir:
        store.save(persist_dir)
    return store

