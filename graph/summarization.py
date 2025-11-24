from __future__ import annotations

from typing import Dict, List


def _split_sentences(text: str) -> List[str]:
    sentences: List[str] = []
    for raw_sentence in text.replace("?", ".").replace("!", ".").split("."):
        sentence = raw_sentence.strip()
        if sentence:
            sentences.append(sentence)
    return sentences


def summarize_paragraph_level(chunks: List[str], sentences_per_chunk: int = 2) -> List[str]:
    """
    Produce paragraph-level summaries by selecting representative sentences.
    """
    summaries: List[str] = []
    for chunk in chunks:
        sentences = _split_sentences(chunk)
        summaries.append(". ".join(sentences[:sentences_per_chunk]))
    return summaries


def summarize_community_level(
    community_to_chunks: Dict[int, List[str]],
    sentences_per_community: int = 3,
) -> Dict[int, str]:
    """
    Aggregate paragraph summaries into community-level narratives.
    """
    community_summaries: Dict[int, str] = {}
    for community_id, chunks in community_to_chunks.items():
        sentences = _split_sentences(" ".join(chunks))
        summary = ". ".join(sentences[:sentences_per_community])
        community_summaries[community_id] = summary
    return community_summaries


def summarize_global(community_summaries: Dict[int, str]) -> str:
    """
    Combine community summaries into a global overview.
    """
    ordered = [community_summaries[key] for key in sorted(community_summaries)]
    return "\n\n".join(ordered)

