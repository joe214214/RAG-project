#!/usr/bin/env python3
"""
Classifier Evaluation Script - Mixed Dataset with Edge Cases

Creates a realistic evaluation set by:
1. Mixing MS MARCO (OLTP) and HotpotQA (OLAP) queries
2. Adding edge cases that blur the line
3. Computing proper metrics on mixed data
"""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
import random

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class EvalExample:
    text: str
    true_label: str
    source: str
    difficulty: str


EDGE_CASES = [
    EvalExample("What are the capitals of France and Germany?", "oltp", "edge_case", "hard"),
    EvalExample("Who are the CEOs of Apple and Microsoft?", "oltp", "edge_case", "hard"),
    EvalExample("Is Paris bigger than London?", "olap", "edge_case", "medium"),
    EvalExample("Who is older, Elon Musk or Jeff Bezos?", "olap", "edge_case", "medium"),
    EvalExample("Compare Tesla and Ford profits", "olap", "edge_case", "medium"),
    EvalExample("Summarize Apple's strategy", "olap", "edge_case", "medium"),
    EvalExample("How has Amazon's market share changed?", "olap", "edge_case", "hard"),
    EvalExample("Why is the sky blue?", "oltp", "edge_case", "medium"),
    EvalExample("Where was the founder of Microsoft born?", "olap", "edge_case", "medium"),
    EvalExample("What university did the CEO of Apple attend?", "olap", "edge_case", "medium"),
    EvalExample("What is the average salary at Google?", "olap", "edge_case", "medium"),
    EvalExample("When did Apple become worth $1 trillion?", "oltp", "edge_case", "easy"),
    EvalExample("How has Apple's valuation changed over the past decade?", "olap", "edge_case", "easy"),
]


def load_classifier(model_path: str, device: str = "cuda"):
    print(f"Loading classifier from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return tokenizer, model


def classify_query(query: str, tokenizer, model, device: str = "cuda") -> Tuple[str, float, float]:
    start = time.time()
    with torch.no_grad():
        inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    latency = (time.time() - start) * 1000
    pred_idx = int(probs.argmax())
    labels = ["oltp", "olap"]
    return labels[pred_idx], float(probs[pred_idx]), latency


def load_dataset_samples(n_per_class: int = 50) -> List[EvalExample]:
    examples = []
    try:
        from datasets import load_dataset
        print(f"Loading {n_per_class} MS MARCO samples...")
        msmarco = load_dataset("microsoft/ms_marco", "v1.1", split="validation")
        msmarco_queries = [item["query"] for item in msmarco if item.get("query")][:n_per_class * 2]
        random.shuffle(msmarco_queries)
        for q in msmarco_queries[:n_per_class]:
            examples.append(EvalExample(q, "oltp", "msmarco", "easy"))
        
        print(f"Loading {n_per_class} HotpotQA samples...")
        hotpot = load_dataset("hotpot_qa", "fullwiki", split="validation")
        hotpot_queries = [item["question"] for item in hotpot if item.get("question")][:n_per_class * 2]
        random.shuffle(hotpot_queries)
        for q in hotpot_queries[:n_per_class]:
            examples.append(EvalExample(q, "olap", "hotpotqa", "easy"))
    except Exception as e:
        print(f"Error loading datasets: {e}")
    return examples


def create_mixed_eval_set(n_per_class: int = 50) -> List[EvalExample]:
    examples = load_dataset_samples(n_per_class)
    examples.extend(EDGE_CASES)
    random.shuffle(examples)
    print(f"Total: {len(examples)} examples")
    return examples


def evaluate_classifier(model_path: str, examples: List[EvalExample], device: str = "cuda") -> Dict:
    tokenizer, model = load_classifier(model_path, device)
    
    results, true_labels, pred_labels, latencies = [], [], [], []
    
    for i, ex in enumerate(examples):
        pred, conf, latency = classify_query(ex.text, tokenizer, model, device)
        results.append({
            "text": ex.text, "true_label": ex.true_label, "pred_label": pred,
            "confidence": conf, "correct": pred == ex.true_label,
            "source": ex.source, "difficulty": ex.difficulty, "latency_ms": latency,
        })
        true_labels.append(ex.true_label)
        pred_labels.append(pred)
        latencies.append(latency)
        if (i + 1) % 20 == 0:
            print(f"  Evaluated {i + 1}/{len(examples)}...")
    
    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, labels=["oltp", "olap"], average=None)
    cm = confusion_matrix(true_labels, pred_labels, labels=["oltp", "olap"])
    
    source_acc = {s: sum(r["correct"] for r in results if r["source"] == s) / max(1, sum(1 for r in results if r["source"] == s)) 
                  for s in set(ex.source for ex in examples)}
    diff_acc = {d: sum(r["correct"] for r in results if r["difficulty"] == d) / max(1, sum(1 for r in results if r["difficulty"] == d))
                for d in ["easy", "medium", "hard"]}
    
    return {
        "model": model_path, "num_examples": len(examples),
        "metrics": {"accuracy": accuracy, "oltp_precision": precision[0], "oltp_recall": recall[0], "oltp_f1": f1[0],
                    "olap_precision": precision[1], "olap_recall": recall[1], "olap_f1": f1[1], "macro_f1": (f1[0] + f1[1]) / 2},
        "confusion_matrix": {"TN": int(cm[0][0]), "FP": int(cm[0][1]), "FN": int(cm[1][0]), "TP": int(cm[1][1])},
        "accuracy_by_source": source_acc, "accuracy_by_difficulty": diff_acc,
        "latency": {"avg_ms": sum(latencies) / len(latencies), "p95_ms": sorted(latencies)[int(len(latencies) * 0.95)]},
        "misclassified": [r for r in results if not r["correct"]],
    }


def print_results(results: Dict):
    print("\n" + "=" * 60)
    print(f"CLASSIFIER EVALUATION: {results['model']}")
    print("=" * 60)
    m = results["metrics"]
    print(f"\nAccuracy: {m['accuracy']:.1%} | Macro F1: {m['macro_f1']:.3f}")
    print(f"\n{'Class':<8} {'Prec':<8} {'Recall':<8} {'F1':<8}")
    print("-" * 32)
    print(f"{'OLTP':<8} {m['oltp_precision']:<8.3f} {m['oltp_recall']:<8.3f} {m['oltp_f1']:<8.3f}")
    print(f"{'OLAP':<8} {m['olap_precision']:<8.3f} {m['olap_recall']:<8.3f} {m['olap_f1']:<8.3f}")
    print("\nBy Source:", {k: f"{v:.1%}" for k, v in results["accuracy_by_source"].items()})
    print("By Difficulty:", {k: f"{v:.1%}" for k, v in results["accuracy_by_difficulty"].items()})
    print(f"\nLatency: {results['latency']['avg_ms']:.1f}ms avg")
    if results["misclassified"]:
        print(f"\nMisclassified ({len(results['misclassified'])}):")
        for r in results["misclassified"][:5]:
            print(f"  [{r['source']}] True:{r['true_label']} Pred:{r['pred_label']} - {r['text'][:60]}...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained classifier")
    parser.add_argument("--n-per-class", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    examples = create_mixed_eval_set(args.n_per_class)
    results = evaluate_classifier(args.model, examples, args.device)
    print_results(results)
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
