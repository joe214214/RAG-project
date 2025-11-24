from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

LABELS = ["oltp", "olap"]


@dataclass
class TrainingConfig:
    model_name: str = "distilbert-base-uncased"
    output_dir: Path = Path("artifacts/classifier")
    batch_size: int = 8
    epochs: int = 1
    learning_rate: float = 5e-5


class QueryDataset(Dataset):
    def __init__(self, queries: List[str], labels: List[int], tokenizer) -> None:
        self.encodings = tokenizer(queries, truncation=True, padding=True, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {key: tensor[idx] for key, tensor in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def generate_dummy_dataset(num_samples: int = 200) -> Tuple[List[str], List[int]]:
    """
    Produce a toy dataset where OLTP queries are short factoids and OLAP queries
    are multi-hop summaries.
    """
    factoids = [
        "What year was Apple founded?",
        "Who is the CEO of Microsoft?",
        "Capital of France?",
        "Stock ticker for Tesla?",
    ]
    analytics = [
        "Summarize the AI investments made by FAANG companies in 2023.",
        "Compare quarterly revenue trends for Apple and Microsoft.",
        "Describe strategies for reducing data center latency across regions.",
        "How have EV subsidies impacted manufacturer margins over the last decade?",
    ]
    queries: List[str] = []
    labels: List[int] = []
    for _ in range(num_samples):
        if random.random() > 0.5:
            queries.append(random.choice(factoids))
            labels.append(0)
        else:
            queries.append(random.choice(analytics))
            labels.append(1)
    return queries, labels


def train_classifier(config: TrainingConfig) -> None:
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=len(LABELS))

    queries, labels = generate_dummy_dataset()
    dataset = QueryDataset(queries, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = len(dataloader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    model.train()
    for epoch in range(config.epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train DistilBERT classifier for OLTP/OLAP routing.")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--output-dir", default="artifacts/classifier")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    args = parser.parse_args()
    return TrainingConfig(
        model_name=args.model_name,
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train_classifier(cfg)

