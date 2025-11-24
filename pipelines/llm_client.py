from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class LLMConfig:
    """
    Placeholder config for wiring in a real LLM (OpenAI, Azure, etc.).
    """

    provider: str = "stub"
    model_name: str = "gpt-4o-mini"
    max_tokens: int = 512


class LLMClient:
    """
    Minimal interface that can later wrap a true LLM provider.

    The current implementation is intentionally simple and deterministic so
    integration tests do not require external API calls.
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()

    def generate(self, question: str, contexts: List[str]) -> str:
        """
        Combine question + contexts into a pseudo-answer.
        """
        synthesized_context = "\n\n".join(contexts[:3])
        response = (
            f"[Stubbed {self.config.model_name} answer]\n"
            f"Question: {question}\n"
            f"Context snippets:\n{synthesized_context[: self.config.max_tokens * 4]}"
        )
        return response

