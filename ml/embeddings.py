from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 16
    device: str | None = None  # None -> auto


class SentenceTransformerEmbedder:
    """
    Thin wrapper around sentence-transformers to keep the rest of the codebase
    decoupled from specific embedding implementations.
    """

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig()
        self.model = SentenceTransformer(self.config.model_name, device=self.config.device)

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        embeddings = self.model.encode(
            list(texts),
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings

    def embed_file(self, path: Path) -> np.ndarray:
        text = Path(path).read_text(encoding="utf-8")
        return self.embed_texts([text])[0]

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]

