from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass
class ChunkMetadata:
    chunk_id: str
    level: str  # "super" or "fine"
    source_doc: str
    parent_chunk: str | None
    start_token: int
    end_token: int
    path: str


def clean_text(text: str) -> str:
    """
    Perform lightweight cleanup before tokenization.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u00a0", " ")
    lines = [line.strip() for line in normalized.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def tokenize(text: str) -> List[str]:
    """
    Approximate tokenization using whitespace splitting.

    This keeps the implementation dependency-free while producing stable
    chunk boundaries. When migrating to a production tokenizer (e.g., tiktoken),
    replace this function only.
    """
    return text.split()


def sliding_window_chunk(
    tokens: Sequence[str],
    chunk_size: int,
    overlap: int,
) -> Iterable[Tuple[int, int, List[str]]]:
    """
    Yield start/end positions and token slices for the requested chunking scheme.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    start = 0
    num_tokens = len(tokens)
    while start < num_tokens:
        end = min(start + chunk_size, num_tokens)
        yield (start, end, list(tokens[start:end]))
        if end == num_tokens:
            break
        start = end - overlap


def save_chunk(metadata: ChunkMetadata, text: str) -> None:
    """
    Persist the chunk text to disk and update metadata path.
    """
    path = Path(metadata.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_metadata(metadata_list: List[ChunkMetadata], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = [asdict(item) for item in metadata_list]
    output_path.write_text(json.dumps(serialized, indent=2, ensure_ascii=False), encoding="utf-8")


def create_hierarchical_chunks(
    text_dir: Path | str,
    super_chunk_dir: Path | str,
    fine_chunk_dir: Path | str,
    *,
    super_chunk_tokens: int = 2000,
    fine_chunk_tokens: int = 512,
    fine_overlap: int = 64,
) -> Tuple[List[ChunkMetadata], List[ChunkMetadata], Dict[str, List[str]]]:
    """
    Generate hierarchical chunks (super + fine) from plain text documents.

    Returns:
        super_metadata: metadata for every super-chunk.
        fine_metadata: metadata for every fine-chunk.
        parent_mapping: dict mapping super_chunk_id -> list[fine_chunk_id]
    """
    text_dir = Path(text_dir)
    super_chunk_dir = Path(super_chunk_dir)
    fine_chunk_dir = Path(fine_chunk_dir)

    super_metadata: List[ChunkMetadata] = []
    fine_metadata: List[ChunkMetadata] = []
    parent_mapping: Dict[str, List[str]] = {}

    for text_path in sorted(text_dir.glob("*.txt")):
        document_text = clean_text(text_path.read_text(encoding="utf-8"))
        tokens = tokenize(document_text)

        # Super chunking (approx. 2-3 pages => 1800-2500 tokens)
        for super_index, (start, end, token_slice) in enumerate(
            sliding_window_chunk(tokens, super_chunk_tokens, overlap=int(super_chunk_tokens * 0.2))
        ):
            chunk_id = f"{text_path.stem}_super_{super_index}"
            chunk_text = " ".join(token_slice)
            chunk_path = super_chunk_dir / f"{chunk_id}.txt"
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                level="super",
                source_doc=text_path.name,
                parent_chunk=None,
                start_token=start,
                end_token=end,
                path=str(chunk_path),
            )
            save_chunk(metadata, chunk_text)
            super_metadata.append(metadata)

            # Fine chunking inside super chunk
            fine_parent_tokens = tokenize(chunk_text)
            for fine_index, (f_start, f_end, fine_tokens) in enumerate(
                sliding_window_chunk(fine_parent_tokens, fine_chunk_tokens, overlap=fine_overlap)
            ):
                fine_chunk_id = f"{chunk_id}_fine_{fine_index}"
                fine_chunk_path = fine_chunk_dir / f"{fine_chunk_id}.txt"
                fine_metadata_item = ChunkMetadata(
                    chunk_id=fine_chunk_id,
                    level="fine",
                    source_doc=text_path.name,
                    parent_chunk=chunk_id,
                    start_token=start + f_start,
                    end_token=start + f_end,
                    path=str(fine_chunk_path),
                )
                save_chunk(fine_metadata_item, " ".join(fine_tokens))
                fine_metadata.append(fine_metadata_item)
                parent_mapping.setdefault(chunk_id, []).append(fine_chunk_id)

    # Persist metadata for downstream loading
    save_metadata(super_metadata, super_chunk_dir / "metadata.json")
    save_metadata(fine_metadata, fine_chunk_dir / "metadata.json")
    (super_chunk_dir / "parent_mapping.json").write_text(
        json.dumps(parent_mapping, indent=2),
        encoding="utf-8",
    )

    return super_metadata, fine_metadata, parent_mapping

