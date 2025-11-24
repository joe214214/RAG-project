from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from pypdf import PdfReader


@dataclass
class PdfExtractionStats:
    """Simple statistics returned after a batch extraction run."""

    num_files: int
    total_pages: int
    total_chars: int


def normalize_pdf_text(text: str) -> str:
    """
    Apply lightweight cleanup to raw PDF text.

    Steps:
        - Replace Windows line endings.
        - Collapse repeated whitespace.
        - Strip leading/trailing blank space.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in normalized.split("\n")]
    filtered = "\n".join(line for line in lines if line)
    return filtered.strip()


def extract_pdf_to_text(pdf_path: Path, output_path: Path) -> int:
    """
    Extract a single PDF into a UTF-8 text file.

    Returns the number of pages processed.
    """
    reader = PdfReader(str(pdf_path))
    text_parts: List[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text_parts.append(page_text)

    cleaned = normalize_pdf_text("\n".join(text_parts))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(cleaned, encoding="utf-8")
    return len(reader.pages)


def extract_pdfs_to_text(
    pdf_dir: Path | str,
    output_dir: Path | str,
    *,
    overwrite: bool = True,
) -> PdfExtractionStats:
    """
    Convert every PDF under `pdf_dir` into plain text files stored in `output_dir`.

    The function returns aggregate statistics that can feed into downstream
    logging or benchmarking pipelines.
    """
    pdf_dir = Path(pdf_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory does not exist: {pdf_dir}")

    total_pages = 0
    total_chars = 0
    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    for pdf_path in pdf_files:
        output_path = output_dir / f"{pdf_path.stem}.txt"
        if output_path.exists() and not overwrite:
            continue

        pages = extract_pdf_to_text(pdf_path, output_path)
        total_pages += pages
        total_chars += len(output_path.read_text(encoding="utf-8"))

    return PdfExtractionStats(
        num_files=len(pdf_files),
        total_pages=total_pages,
        total_chars=total_chars,
    )


def iter_text_files(text_dir: Path | str) -> Iterable[Path]:
    """
    Yield all text files produced by `extract_pdfs_to_text`.
    """
    text_dir = Path(text_dir)
    yield from sorted(text_dir.glob("*.txt"))

