#!/usr/bin/env python3
"""
Comprehensive Classifier Validation Script

Addresses the 100% accuracy concern by:
1. Testing on fresh holdout data (Natural Questions)
2. Analyzing model confidence distributions
3. Testing adversarial examples
4. Checking for dataset artifacts (keyword shortcuts)
5. Reporting detailed metrics

Can run locally on Windows/Mac before deploying to cluster.
"""

import os
import json
import argparse
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple
import re

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    precision_recall_fscore_support
)

# ============================================================
# MODEL LOADING
# ============================================================

def load_classifier(model_path: str, device: str = "cpu"):
    """Load the trained classifier."""
    print(f"Loading classifier from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return tokenizer, model


def predict_with_confidence(
    tokenizer, 
    model, 
    queries: List[str], 
    device: str = "cpu"
) -> List[Tuple[str, float]]:
    """
    Predict labels with confidence scores.
    Returns: List of (predicted_label, confidence_score)
    """
    results = []
    model.eval()
    
    with torch.no_grad():
        for query in queries:
            inputs = tokenizer(
                query, 
                return_tensors="pt", 
                truncation=True, 
                max_length=128,
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_idx = probs.argmax(dim=-1).item()
            confidence = probs[0, pred_idx].item()
            
            label = "oltp" if pred_idx == 0 else "olap"
            results.append((label, confidence))
    
    return results


# ============================================================
# DATA LOADING - FRESH HOLDOUT
# ============================================================

def load_natural_questions(n_samples: int = 100) -> List[Dict]:
    """
    Load Natural Questions as fresh holdout dataset.
    NQ was NOT used in training, so this tests generalization.
    
    NOTE: Natural Questions is ~95% factoid (OLTP) questions.
    Only a small fraction require multi-step reasoning (OLAP).
    """
    print(f"Loading {n_samples} Natural Questions samples...")
    
    try:
        # Load NQ dataset from HuggingFace
        ds = load_dataset("natural_questions", "default", split="validation", 
                         streaming=True, trust_remote_code=True)
        
        examples = []
        for i, item in enumerate(ds):
            if i >= n_samples:
                break
            
            question = item.get("question", {}).get("text", "")
            if not question:
                continue
            
            # Natural Questions is primarily factoid (OLTP)
            # Only mark as OLAP if clearly analytical
            label = heuristic_label(question)
            
            examples.append({
                "query": question,
                "label": label,
                "source": "natural_questions",
                "difficulty": "easy" if label == "oltp" else "medium"
            })
        
        return examples
        
    except Exception as e:
        print(f"Warning: Could not load NQ dataset: {e}")
        print("Falling back to synthetic holdout...")
        return create_synthetic_holdout(n_samples)


def load_eli5_questions(n_samples: int = 50) -> List[Dict]:
    """
    Load ELI5 (Explain Like I'm 5) as another fresh holdout.
    These are typically OLAP-style (require explanation/synthesis).
    """
    print(f"Loading {n_samples} ELI5 samples...")
    
    try:
        ds = load_dataset("eli5", split="validation_eli5", trust_remote_code=True)
        
        examples = []
        for i, item in enumerate(ds):
            if i >= n_samples:
                break
            
            question = item.get("title", "")
            if not question:
                continue
            
            # ELI5 questions require explanation -> mostly OLAP
            examples.append({
                "query": question,
                "label": "olap",  # ELI5 requires synthesis
                "source": "eli5",
                "difficulty": "medium"
            })
        
        return examples
        
    except Exception as e:
        print(f"Warning: Could not load ELI5 dataset: {e}")
        return []


def create_synthetic_holdout(n_samples: int = 100) -> List[Dict]:
    """Create synthetic holdout examples if real datasets unavailable."""
    
    # OLTP examples (factoid lookups)
    oltp_templates = [
        "What is the capital of {}?",
        "Who wrote {}?",
        "When was {} born?",
        "What year did {} happen?",
        "What is the definition of {}?",
        "Who is the CEO of {}?",
        "What country is {} in?",
        "How tall is {}?",
        "What is {}'s population?",
        "When did {} die?",
    ]
    
    # OLAP examples (analysis/synthesis)
    olap_templates = [
        "Compare {} and {} in terms of performance.",
        "What are the main causes of {}?",
        "Explain the relationship between {} and {}.",
        "How has {} changed over the past decade?",
        "Summarize the key findings about {}.",
        "What factors contribute to {}?",
        "Analyze the impact of {} on {}.",
        "Why did {} happen and what were the consequences?",
        "What are the similarities and differences between {} and {}?",
        "How does {} affect different regions differently?",
    ]
    
    # Fillers for templates
    entities = ["France", "Tesla", "Apple", "Amazon", "climate change", 
                "AI", "renewable energy", "Shakespeare", "Einstein", "Google"]
    
    examples = []
    
    # Generate OLTP examples
    for i in range(n_samples // 2):
        template = oltp_templates[i % len(oltp_templates)]
        entity = entities[i % len(entities)]
        query = template.format(entity)
        examples.append({
            "query": query,
            "label": "oltp",
            "source": "synthetic",
            "difficulty": "easy"
        })
    
    # Generate OLAP examples
    for i in range(n_samples // 2):
        template = olap_templates[i % len(olap_templates)]
        e1 = entities[i % len(entities)]
        e2 = entities[(i + 1) % len(entities)]
        query = template.format(e1, e2) if "{}" in template[template.find("{}") + 2:] else template.format(e1)
        examples.append({
            "query": query,
            "label": "olap",
            "source": "synthetic",
            "difficulty": "medium"
        })
    
    return examples


def heuristic_label(query: str) -> str:
    """
    Heuristic labeling for queries based on linguistic patterns.
    CONSERVATIVE: Only mark as OLAP if clearly analytical/comparative.
    Natural Questions is mostly factoid (OLTP), so default to OLTP.
    """
    query_lower = query.lower()
    
    # Strong OLAP indicators (clearly analytical, NOT just presence of keywords)
    # Must require synthesis, comparison, or multi-step reasoning
    strong_olap_patterns = [
        r"compare\s.+\s(and|with|to)\s",  # "compare X and Y"
        r"contrast\s.+\s(and|with)\s",     # "contrast X and Y"
        r"what are the (main |key )?(differences?|similarities)",
        r"analyze (the |how )",
        r"summarize (the |how |what )",
        r"explain (why|how|the relationship)",
        r"what (factors?|causes?) (lead|contribute|cause)",
        r"(impact|effect|influence) of .+ on",
        r"how has .+ changed",
        r"relationship between .+ and",
        r"pros and cons",
        r"advantages and disadvantages",
        r"should (we|i|you|they)",  # Opinion/reasoning questions
    ]
    
    # Check strong OLAP patterns (require explicit analytical structure)
    for pattern in strong_olap_patterns:
        if re.search(pattern, query_lower):
            return "olap"
    
    # Most questions are factoid (OLTP) - this is the default
    # Natural Questions, MS MARCO are primarily factoid datasets
    return "oltp"


# ============================================================
# ADVERSARIAL EXAMPLES
# ============================================================

def create_adversarial_examples() -> List[Dict]:
    """
    Create adversarial examples that challenge the classifier.
    These are designed to exploit potential shortcuts.
    """
    examples = []
    
    # Type 1: OLAP queries that LOOK like OLTP (short, simple phrasing)
    olap_looking_like_oltp = [
        ("What is inflation?", "olap"),  # Requires explanation
        ("Who benefits from AI?", "olap"),  # Requires analysis
        ("Why is the sky blue?", "olap"),  # Requires explanation
        ("What causes cancer?", "olap"),  # Multi-factor analysis
        ("Is democracy good?", "olap"),  # Requires argumentation
        ("What makes a good leader?", "olap"),  # Analysis/synthesis
        ("Should we use nuclear energy?", "olap"),  # Requires reasoning
        ("What is success?", "olap"),  # Philosophical, requires synthesis
    ]
    
    # Type 2: OLTP queries that LOOK like OLAP (long, complex phrasing)
    oltp_looking_like_olap = [
        ("What is the specific date when the Declaration of Independence was signed?", "oltp"),
        ("Who was the person that first set foot on the moon during the Apollo mission?", "oltp"),
        ("What is the exact height in meters of Mount Everest above sea level?", "oltp"),
        ("In which year did the company Apple Inc. officially go public on the stock market?", "oltp"),
        ("What is the current population count of the city of Tokyo, Japan?", "oltp"),
        ("Who served as the 44th president of the United States of America?", "oltp"),
    ]
    
    # Type 3: Ambiguous queries (genuinely hard to classify)
    ambiguous = [
        ("How did World War 2 start?", "olap"),  # Could be fact or analysis
        ("What happened in 2008?", "olap"),  # Needs context, likely analysis
        ("Is Tesla profitable?", "oltp"),  # Single fact, but could require analysis
        ("Who won the 2020 election?", "oltp"),  # Fact, but politically loaded
        ("What is machine learning used for?", "olap"),  # Enumeration/synthesis
        ("How does Google make money?", "olap"),  # Requires explanation
    ]
    
    # Type 4: Questions with "compare" that are actually simple
    compare_but_simple = [
        ("Is Python faster than Java?", "oltp"),  # Simple comparison
        ("Which is bigger, Earth or Mars?", "oltp"),  # Simple fact
        ("Is gold more expensive than silver?", "oltp"),  # Simple lookup
    ]
    
    for query, label in olap_looking_like_oltp:
        examples.append({
            "query": query,
            "label": label,
            "source": "adversarial",
            "difficulty": "hard",
            "adversarial_type": "olap_looks_oltp"
        })
    
    for query, label in oltp_looking_like_olap:
        examples.append({
            "query": query,
            "label": label,
            "source": "adversarial",
            "difficulty": "hard",
            "adversarial_type": "oltp_looks_olap"
        })
    
    for query, label in ambiguous:
        examples.append({
            "query": query,
            "label": label,
            "source": "adversarial",
            "difficulty": "hard",
            "adversarial_type": "ambiguous"
        })
    
    for query, label in compare_but_simple:
        examples.append({
            "query": query,
            "label": label,
            "source": "adversarial",
            "difficulty": "medium",
            "adversarial_type": "compare_simple"
        })
    
    return examples


# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def analyze_keyword_shortcuts(
    queries: List[str], 
    true_labels: List[str], 
    pred_labels: List[str]
) -> Dict:
    """
    Analyze if the model is using keyword shortcuts.
    """
    olap_keywords = ["compare", "analyze", "explain", "why", "how", 
                     "summarize", "relationship", "impact", "effect"]
    oltp_keywords = ["what is", "who is", "when", "where", "capital", 
                     "population", "definition"]
    
    results = {
        "olap_keyword_present": {"correct": 0, "total": 0},
        "oltp_keyword_present": {"correct": 0, "total": 0},
        "no_keywords": {"correct": 0, "total": 0}
    }
    
    for query, true, pred in zip(queries, true_labels, pred_labels):
        query_lower = query.lower()
        
        has_olap_kw = any(kw in query_lower for kw in olap_keywords)
        has_oltp_kw = any(kw in query_lower for kw in oltp_keywords)
        
        if has_olap_kw:
            results["olap_keyword_present"]["total"] += 1
            if true == pred:
                results["olap_keyword_present"]["correct"] += 1
        elif has_oltp_kw:
            results["oltp_keyword_present"]["total"] += 1
            if true == pred:
                results["oltp_keyword_present"]["correct"] += 1
        else:
            results["no_keywords"]["total"] += 1
            if true == pred:
                results["no_keywords"]["correct"] += 1
    
    # Calculate accuracies
    for key in results:
        total = results[key]["total"]
        if total > 0:
            results[key]["accuracy"] = results[key]["correct"] / total
        else:
            results[key]["accuracy"] = 0.0
    
    return results


def analyze_confidence_distribution(
    confidences: List[float], 
    correct: List[bool]
) -> Dict:
    """
    Analyze confidence score distribution.
    Check if model is overconfident on wrong predictions.
    """
    correct_confs = [c for c, ok in zip(confidences, correct) if ok]
    wrong_confs = [c for c, ok in zip(confidences, correct) if not ok]
    
    results = {
        "correct_predictions": {
            "count": len(correct_confs),
            "mean_confidence": np.mean(correct_confs) if correct_confs else 0,
            "min_confidence": min(correct_confs) if correct_confs else 0,
            "max_confidence": max(correct_confs) if correct_confs else 0,
        },
        "wrong_predictions": {
            "count": len(wrong_confs),
            "mean_confidence": np.mean(wrong_confs) if wrong_confs else 0,
            "min_confidence": min(wrong_confs) if wrong_confs else 0,
            "max_confidence": max(wrong_confs) if wrong_confs else 0,
        },
        "calibration_gap": 0  # Difference between confidence and accuracy
    }
    
    # Calibration: is the model overconfident?
    all_confs = np.array(confidences)
    all_correct = np.array(correct)
    
    # Bin confidences and check calibration
    bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    calibration = []
    
    for low, high in bins:
        mask = (all_confs >= low) & (all_confs < high)
        if mask.sum() > 0:
            bin_accuracy = all_correct[mask].mean()
            bin_confidence = all_confs[mask].mean()
            calibration.append({
                "bin": f"{low:.1f}-{high:.1f}",
                "count": int(mask.sum()),
                "accuracy": float(bin_accuracy),
                "mean_confidence": float(bin_confidence),
                "gap": float(bin_confidence - bin_accuracy)  # Positive = overconfident
            })
    
    results["calibration_bins"] = calibration
    
    # Overall calibration gap
    if calibration:
        results["calibration_gap"] = np.mean([b["gap"] for b in calibration])
    
    return results


def analyze_query_length(
    queries: List[str],
    true_labels: List[str],
    pred_labels: List[str]
) -> Dict:
    """Analyze if query length affects classification."""
    lengths = [len(q.split()) for q in queries]
    correct = [t == p for t, p in zip(true_labels, pred_labels)]
    
    # Bin by length
    short = [(l, c) for l, c in zip(lengths, correct) if l < 8]
    medium = [(l, c) for l, c in zip(lengths, correct) if 8 <= l < 15]
    long_q = [(l, c) for l, c in zip(lengths, correct) if l >= 15]
    
    return {
        "short_queries (<8 words)": {
            "count": len(short),
            "accuracy": np.mean([c for _, c in short]) if short else 0
        },
        "medium_queries (8-14 words)": {
            "count": len(medium),
            "accuracy": np.mean([c for _, c in medium]) if medium else 0
        },
        "long_queries (15+ words)": {
            "count": len(long_q),
            "accuracy": np.mean([c for _, c in long_q]) if long_q else 0
        }
    }


# ============================================================
# HEURISTIC BASELINE (Sanity Check)
# ============================================================

def run_heuristic_baseline(queries: List[str], true_labels: List[str]) -> Dict:
    """
    Run a simple keyword-based heuristic classifier as baseline.
    If the transformer is only slightly better than this, the task may be too easy.
    """
    heuristic_preds = [heuristic_label(q) for q in queries]
    correct = [t == p for t, p in zip(true_labels, heuristic_preds)]
    accuracy = sum(correct) / len(correct)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, heuristic_preds, labels=["oltp", "olap"], zero_division=0
    )
    
    return {
        "accuracy": accuracy,
        "macro_f1": np.mean(f1),
        "per_class": {
            "oltp": {"precision": precision[0], "recall": recall[0], "f1": f1[0]},
            "olap": {"precision": precision[1], "recall": recall[1], "f1": f1[1]},
        }
    }


# ============================================================
# MAIN VALIDATION
# ============================================================

def run_validation(
    model_path: str,
    device: str = "cpu",
    n_nq_samples: int = 50,
    n_eli5_samples: int = 30,
    output_path: str = None
) -> Dict:
    """
    Run comprehensive validation.
    """
    print("=" * 60)
    print("COMPREHENSIVE CLASSIFIER VALIDATION")
    print("=" * 60)
    
    # Load model
    tokenizer, model = load_classifier(model_path, device)
    
    # Collect all test examples
    all_examples = []
    
    # 1. Natural Questions (fresh holdout)
    nq_examples = load_natural_questions(n_nq_samples)
    all_examples.extend(nq_examples)
    
    # 2. ELI5 (fresh holdout - synthesis questions)
    eli5_examples = load_eli5_questions(n_eli5_samples)
    all_examples.extend(eli5_examples)
    
    # 3. Adversarial examples
    adversarial = create_adversarial_examples()
    all_examples.extend(adversarial)
    
    # 4. Synthetic holdout if we don't have enough
    if len(all_examples) < 100:
        synthetic = create_synthetic_holdout(100 - len(all_examples))
        all_examples.extend(synthetic)
    
    print(f"\nTotal validation examples: {len(all_examples)}")
    print(f"  - Natural Questions: {len(nq_examples)}")
    print(f"  - ELI5: {len(eli5_examples)}")
    print(f"  - Adversarial: {len(adversarial)}")
    
    # Extract queries and labels
    queries = [ex["query"] for ex in all_examples]
    true_labels = [ex["label"] for ex in all_examples]
    sources = [ex["source"] for ex in all_examples]
    difficulties = [ex.get("difficulty", "unknown") for ex in all_examples]
    
    # Run predictions
    print("\nRunning predictions...")
    predictions = predict_with_confidence(tokenizer, model, queries, device)
    pred_labels = [p[0] for p in predictions]
    confidences = [p[1] for p in predictions]
    
    # Calculate metrics
    correct = [t == p for t, p in zip(true_labels, pred_labels)]
    accuracy = sum(correct) / len(correct)
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=["oltp", "olap"])
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, pred_labels, labels=["oltp", "olap"], zero_division=0
    )
    
    # Analyze by source
    source_results = {}
    for src in set(sources):
        src_mask = [s == src for s in sources]
        src_correct = [c for c, m in zip(correct, src_mask) if m]
        source_results[src] = {
            "count": sum(src_mask),
            "accuracy": np.mean(src_correct) if src_correct else 0
        }
    
    # Analyze by difficulty
    diff_results = {}
    for diff in set(difficulties):
        diff_mask = [d == diff for d in difficulties]
        diff_correct = [c for c, m in zip(correct, diff_mask) if m]
        diff_results[diff] = {
            "count": sum(diff_mask),
            "accuracy": np.mean(diff_correct) if diff_correct else 0
        }
    
    # Keyword shortcut analysis
    keyword_analysis = analyze_keyword_shortcuts(queries, true_labels, pred_labels)
    
    # Confidence analysis
    confidence_analysis = analyze_confidence_distribution(confidences, correct)
    
    # Query length analysis
    length_analysis = analyze_query_length(queries, true_labels, pred_labels)
    
    # Heuristic baseline comparison
    print("\nRunning heuristic baseline...")
    heuristic_results = run_heuristic_baseline(queries, true_labels)
    
    # Compile results
    results = {
        "model_path": model_path,
        "total_examples": len(all_examples),
        "overall": {
            "accuracy": accuracy,
            "macro_f1": np.mean(f1),
        },
        "per_class": {
            "oltp": {"precision": precision[0], "recall": recall[0], "f1": f1[0], "support": int(support[0])},
            "olap": {"precision": precision[1], "recall": recall[1], "f1": f1[1], "support": int(support[1])},
        },
        "confusion_matrix": {
            "labels": ["oltp", "olap"],
            "matrix": cm.tolist()
        },
        "by_source": source_results,
        "by_difficulty": diff_results,
        "keyword_analysis": keyword_analysis,
        "confidence_analysis": confidence_analysis,
        "length_analysis": length_analysis,
        "heuristic_baseline": heuristic_results,
    }
    
    # Identify misclassified examples
    misclassified = []
    for i, (query, true, pred, conf) in enumerate(zip(queries, true_labels, pred_labels, confidences)):
        if true != pred:
            misclassified.append({
                "query": query[:100] + "..." if len(query) > 100 else query,
                "true_label": true,
                "predicted": pred,
                "confidence": round(conf, 3),
                "source": sources[i],
                "difficulty": difficulties[i]
            })
    
    results["misclassified"] = misclassified[:20]  # Top 20
    
    # Print report
    print_report(results)
    
    # Save if output path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")
    
    return results


def print_report(results: Dict):
    """Print a formatted validation report."""
    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)
    
    print(f"\n📊 OVERALL METRICS")
    print(f"   Accuracy: {results['overall']['accuracy']*100:.1f}%")
    print(f"   Macro F1: {results['overall']['macro_f1']:.3f}")
    
    print(f"\n📈 PER-CLASS METRICS")
    print(f"   {'Class':<8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8}")
    print(f"   {'-'*40}")
    for cls in ["oltp", "olap"]:
        m = results["per_class"][cls]
        print(f"   {cls.upper():<8} {m['precision']:>8.3f} {m['recall']:>8.3f} {m['f1']:>8.3f} {m['support']:>8}")
    
    print(f"\n📋 CONFUSION MATRIX")
    cm = results["confusion_matrix"]["matrix"]
    print(f"              Pred OLTP  Pred OLAP")
    print(f"   True OLTP  {cm[0][0]:>8}  {cm[0][1]:>8}")
    print(f"   True OLAP  {cm[1][0]:>8}  {cm[1][1]:>8}")
    
    print(f"\n🔍 BY SOURCE (Fresh Holdout Test)")
    for src, data in results["by_source"].items():
        status = "✅" if data["accuracy"] >= 0.8 else "⚠️" if data["accuracy"] >= 0.6 else "❌"
        print(f"   {status} {src}: {data['accuracy']*100:.1f}% ({data['count']} examples)")
    
    print(f"\n📊 BY DIFFICULTY")
    for diff, data in results["by_difficulty"].items():
        status = "✅" if data["accuracy"] >= 0.8 else "⚠️" if data["accuracy"] >= 0.6 else "❌"
        print(f"   {status} {diff}: {data['accuracy']*100:.1f}% ({data['count']} examples)")
    
    print(f"\n🔑 KEYWORD SHORTCUT ANALYSIS")
    ka = results["keyword_analysis"]
    for key, data in ka.items():
        if data["total"] > 0:
            print(f"   {key}: {data['accuracy']*100:.1f}% ({data['total']} examples)")
    
    print(f"\n📉 CONFIDENCE ANALYSIS")
    ca = results["confidence_analysis"]
    print(f"   Correct predictions: mean confidence = {ca['correct_predictions']['mean_confidence']:.3f}")
    print(f"   Wrong predictions: mean confidence = {ca['wrong_predictions']['mean_confidence']:.3f}")
    print(f"   Calibration gap: {ca['calibration_gap']:.3f} (positive = overconfident)")
    
    print(f"\n📏 QUERY LENGTH ANALYSIS")
    for key, data in results["length_analysis"].items():
        if data["count"] > 0:
            print(f"   {key}: {data['accuracy']*100:.1f}% ({data['count']} examples)")
    
    # Heuristic baseline comparison
    print(f"\n🎯 HEURISTIC BASELINE COMPARISON")
    hb = results.get("heuristic_baseline", {})
    if hb:
        model_acc = results["overall"]["accuracy"]
        baseline_acc = hb["accuracy"]
        improvement = model_acc - baseline_acc
        
        print(f"   Heuristic baseline: {baseline_acc*100:.1f}% accuracy")
        print(f"   Transformer model:  {model_acc*100:.1f}% accuracy")
        print(f"   Improvement:        {improvement*100:+.1f}%")
        
        if improvement < 0.05:
            print(f"   ⚠️ WARNING: Model is only {improvement*100:.1f}% better than keywords!")
            print(f"      Task may be too easy or model learned keyword shortcuts.")
        elif improvement < 0.15:
            print(f"   🟡 Model shows modest improvement over keywords.")
        else:
            print(f"   ✅ Model significantly outperforms keyword heuristics.")
    
    print(f"\n❌ MISCLASSIFIED EXAMPLES ({len(results['misclassified'])} shown)")
    for m in results["misclassified"][:10]:
        print(f"   [{m['source']}] True:{m['true_label']} Pred:{m['predicted']} Conf:{m['confidence']:.2f}")
        print(f"      {m['query'][:70]}...")
    
    # Final verdict
    print("\n" + "=" * 60)
    print("VALIDATION VERDICT")
    print("=" * 60)
    
    overall_acc = results["overall"]["accuracy"]
    adversarial_acc = results["by_source"].get("adversarial", {}).get("accuracy", 0)
    
    if overall_acc >= 0.90 and adversarial_acc >= 0.70:
        print("✅ PASS: Model generalizes well to fresh data")
    elif overall_acc >= 0.80 and adversarial_acc >= 0.50:
        print("⚠️ CAUTION: Model works but struggles with edge cases")
        print("   Consider: confidence thresholding, more training data")
    else:
        print("❌ CONCERN: Model may have overfit to training distribution")
        print("   Investigate: data leakage, keyword shortcuts, train/test overlap")
    
    # Check for the 100% issue
    nq_acc = results["by_source"].get("natural_questions", {}).get("accuracy", 0)
    if nq_acc == 1.0:
        print("\n🔴 WARNING: 100% on Natural Questions is still suspicious!")
    elif nq_acc < 0.7:
        print(f"\n🟡 Note: Only {nq_acc*100:.1f}% on fresh NQ data - model may not generalize")


def main():
    parser = argparse.ArgumentParser(description="Validate classifier comprehensively")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--n-nq", type=int, default=50, help="Natural Questions samples")
    parser.add_argument("--n-eli5", type=int, default=30, help="ELI5 samples")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    
    args = parser.parse_args()
    
    run_validation(
        model_path=args.model,
        device=args.device,
        n_nq_samples=args.n_nq,
        n_eli5_samples=args.n_eli5,
        output_path=args.output
    )


if __name__ == "__main__":
    main()

