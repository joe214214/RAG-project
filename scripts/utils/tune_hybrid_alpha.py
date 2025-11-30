#!/usr/bin/env python3
"""
Tune Hybrid Retrieval Alpha Parameter

Tests different α values (BM25 vs dense weight) to find optimal balance.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.rag_pipeline import QueryAwareRAGPipeline
from scripts.comprehensive_evaluation import load_queries_with_ground_truth, calculate_correctness_metrics, EvalQueryWithGT


def test_alpha(
    alpha: float,
    queries: List[EvalQueryWithGT],
    qdrant_host: str,
    qdrant_port: int = 6333,
    device: str = "cpu",
    num_queries: int = 20,
) -> Dict:
    """Test a specific alpha value."""
    print(f"\nTesting α = {alpha:.1f}...")
    
    # Initialize pipeline with custom alpha
    pipeline = QueryAwareRAGPipeline(
        router_model_path="models/feature_router.pkl",
        classifier_type="transformer",
        classifier_model="classifier2/microsoft_MiniLM-L12-H384-uncased",
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        device=device,
        use_hybrid=True,
    )
    
    # Temporarily modify hybrid_alpha
    original_alpha = pipeline.hybrid_alpha
    pipeline.hybrid_alpha = alpha
    print(f"  Using α = {alpha:.1f} (BM25 weight)")
    
    all_mrr = []
    all_recall_10 = []
    all_ndcg_10 = []
    
    for i, eval_query in enumerate(queries[:num_queries], 1):
        if i % 5 == 0:
            print(f"  Processing query {i}/{min(num_queries, len(queries))}...")
        
        try:
            # Classify query
            query_type, _ = pipeline.classify_query(eval_query.query)
            
            # Retrieve
            result = pipeline.query(
                query=eval_query.query,
                top_k=10,
                use_reranker=False,  # Test without reranker to isolate alpha effect
                reranker_model=None,
            )
            
            # Calculate correctness
            collection = "oltp_chunks" if query_type == "oltp" else "olap_chunks"
            metrics = calculate_correctness_metrics(
                result.retrieved_chunks,
                eval_query,
                pipeline.qdrant_client,
                collection,
                use_text_first=False,  # Keep ID-first for alpha tuning
                similarity_threshold=0.15,
                similarity_method="hybrid",
            )
            
            all_mrr.append(metrics["mrr"])
            all_recall_10.append(metrics["recall_at_10"])
            all_ndcg_10.append(metrics["ndcg_at_10"])
        
        except Exception as e:
            print(f"  ⚠ Query failed: {e}")
            continue
    
    # Restore original alpha
    pipeline.hybrid_alpha = original_alpha
    
    if not all_mrr:
        return None
    
    return {
        'alpha': alpha,
        'mrr': np.mean(all_mrr),
        'recall_at_10': np.mean(all_recall_10),
        'ndcg_at_10': np.mean(all_ndcg_10),
        'num_queries': len(all_mrr),
    }


def main():
    parser = argparse.ArgumentParser(description="Tune hybrid retrieval alpha parameter")
    parser.add_argument("--qdrant-host", type=str, default="localhost", help="Qdrant host")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant port")
    parser.add_argument("--device", type=str, default="cpu", help="Device for models")
    parser.add_argument("--num-queries", type=int, default=20, help="Number of queries to test per alpha")
    parser.add_argument("--output", type=str, default="results/alpha_tuning.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("HYBRID RETRIEVAL ALPHA TUNING")
    print("=" * 80)
    
    # Load queries with ground truth
    print("\nLoading queries with ground truth...")
    queries_dict = load_queries_with_ground_truth(oltp_limit=10, olap_limit=10)
    all_queries = queries_dict["oltp"] + queries_dict["olap"]
    
    if not all_queries:
        print("❌ Failed to load queries with ground truth")
        return
    
    print(f"Loaded {len(all_queries)} queries")
    
    # Test different alpha values
    alpha_values = [0.3, 0.5, 0.7, 0.9]
    results = []
    
    print(f"\nTesting {len(alpha_values)} alpha values with {args.num_queries} queries each...")
    
    for alpha in alpha_values:
        result = test_alpha(
            alpha=alpha,
            queries=all_queries,
            qdrant_host=args.qdrant_host,
            qdrant_port=args.qdrant_port,
            device=args.device,
            num_queries=args.num_queries,
        )
        if result:
            results.append(result)
            print(f"  α={alpha:.1f}: MRR={result['mrr']:.3f}, R@10={result['recall_at_10']:.3f}, NDCG@10={result['ndcg_at_10']:.3f}")
    
    if not results:
        print("\n❌ No results collected. Check errors above.")
        return
    
    # Find best alpha
    best_mrr = max(results, key=lambda x: x['mrr'])
    best_recall = max(results, key=lambda x: x['recall_at_10'])
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\n{'Alpha':<8} {'MRR':<8} {'Recall@10':<12} {'NDCG@10':<12}")
    print("-" * 80)
    for r in sorted(results, key=lambda x: x['alpha']):
        print(f"{r['alpha']:<8.1f} {r['mrr']:<8.3f} {r['recall_at_10']:<12.3f} {r['ndcg_at_10']:<12.3f}")
    
    print(f"\n🏆 Best MRR: α={best_mrr['alpha']:.1f} (MRR={best_mrr['mrr']:.3f})")
    print(f"🏆 Best Recall@10: α={best_recall['alpha']:.1f} (R@10={best_recall['recall_at_10']:.3f})")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'results': results,
            'best_mrr': best_mrr,
            'best_recall': best_recall,
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {args.output}")
    print("\n⚠️  Note: To use the optimal alpha, update rag_pipeline.py:")
    print(f"   Set hybrid_alpha = {best_mrr['alpha']:.1f} in QueryAwareRAGPipeline.__init__")


if __name__ == "__main__":
    main()

