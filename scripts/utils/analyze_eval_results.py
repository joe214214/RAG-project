#!/usr/bin/env python3
"""Analyze evaluation results to find meaningful differentiation."""

import json
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"

def analyze_file(filepath, label):
    """Analyze a single evaluation file."""
    print(f"\n{label}")
    print("=" * 80)
    
    with open(filepath) as f:
        data = json.load(f)
    
    results = []
    for config_data in data['configs']:
        config_name = config_data['config']['name']
        queries = config_data['queries']
        if queries:
            avg_mrr = np.mean([q.get('mrr', 0) for q in queries])
            avg_recall = np.mean([q.get('recall_at_10', 0) for q in queries])
            avg_ndcg = np.mean([q.get('ndcg_at_10', 0) for q in queries])
            avg_latency = np.mean([q.get('latency_ms', 0) for q in queries])
            results.append({
                'name': config_name,
                'mrr': avg_mrr,
                'recall': avg_recall,
                'ndcg': avg_ndcg,
                'latency': avg_latency
            })
            print(f"{config_name:35} MRR={avg_mrr:.3f}  Recall@10={avg_recall:.3f}  NDCG={avg_ndcg:.3f}  Lat={avg_latency:.1f}ms")
    
    # Find best and baseline
    baseline = next((r for r in results if 'baseline' in r['name']), results[0])
    best = max(results, key=lambda x: x['mrr'])
    
    print(f"\nBaseline: {baseline['name']} (MRR={baseline['mrr']:.3f})")
    print(f"Best:     {best['name']} (MRR={best['mrr']:.3f})")
    
    if baseline['mrr'] > 0:
        improvement = (best['mrr'] - baseline['mrr']) / baseline['mrr'] * 100
        print(f"Improvement: +{improvement:.1f}%")
    
    return results


def main():
    print("Analyzing Evaluation Results for Meaningful Differentiation")
    print("=" * 80)
    
    # Analyze different evaluation files
    files = [
        (RESULTS_DIR / "evaluation" / "comprehensive_eval_improved.json", "ID-Based (improved params)"),
        (RESULTS_DIR / "initialfindings" / "comprehensive_eval_ngram.json", "N-gram Similarity"),
        (RESULTS_DIR / "initialfindings" / "comprehensive_eval_hybrid.json", "Hybrid Similarity"),
        (RESULTS_DIR / "initialfindings" / "comprehensive_eval_jaccard.json", "Jaccard Similarity"),
    ]
    
    for filepath, label in files:
        if filepath.exists():
            analyze_file(filepath, label)
        else:
            print(f"\n{label}: File not found ({filepath})")


if __name__ == "__main__":
    main()

