from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LABELS = ["oltp", "olap"]


@dataclass
class ClassifierConfig:
    model_name_or_path: str = "distilbert-base-uncased"
    device: str = "cpu"


class QueryClassifier:
    """
    DistilBERT-based classifier for routing queries.
    """

    def __init__(self, config: ClassifierConfig | None = None) -> None:
        self.config = config or ClassifierConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name_or_path,
            num_labels=len(LABELS),
        )
        self.model.to(self.config.device)
        self.model.eval()

    @torch.inference_mode()
    def classify(self, query: str) -> tuple[str, float]:
        tokens = self.tokenizer(query, return_tensors="pt", truncation=True).to(self.config.device)
        logits = self.model(**tokens).logits
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        index = int(probabilities.argmax())
        label = LABELS[index]
        confidence = float(probabilities[index])
        return label, confidence


CLASSIFIER_INSTANCE: QueryClassifier | None = None


def get_classifier() -> QueryClassifier:
    global CLASSIFIER_INSTANCE
    if CLASSIFIER_INSTANCE is None:
        CLASSIFIER_INSTANCE = QueryClassifier()
    return CLASSIFIER_INSTANCE


def classify_query(query_string: str) -> Literal["oltp", "olap"]:
    classifier = get_classifier()
    label, _ = classifier.classify(query_string)
    return label  # type: ignore[return-value]

