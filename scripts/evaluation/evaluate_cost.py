#!/usr/bin/env python3
"""Evaluate RAG pipeline with cost tracking - Baseline vs Best comparison."""
import json
import argparse
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.rag_pipeline import QueryAwareRAGPipeline
from scripts.comprehensive_evaluation import load_queries_with_ground_truth

def evaluate_cost(
    qdrant_host: str = "localhost",
    output_file: str = "results/cost_evaluation.json",
    num_queries: int = 50,
    configs: list = None,
):
    """
    Evaluate cost for different pipeline configurations.
    
    Baseline: Feature classifier, dense-only, no reranker (simplest, fastest, cheapest)
    Best: Transformer classifier, hybrid retrieval, reranker (best quality)
    """
    
    # Load queries
    print("Loading queries with ground truth...")
    queries_dict = load_queries_with_ground_truth(oltp_limit=num_queries, olap_limit=num_queries)
    
    # Flatten dictionary into list of (query, query_type, query_data) tuples
    queries = []
    for query_type, query_list in queries_dict.items():
        for query_item in query_list:
            # Extract query text from EvalQueryWithGT object
            query_text = query_item.query if hasattr(query_item, 'query') else str(query_item)
            queries.append({
                "query": query_text,
                "query_type": query_type,
                "query_data": query_item,
            })
    
    # Limit total queries if needed
    queries = queries[:num_queries]
    print(f"Loaded {len(queries)} queries ({sum(1 for q in queries if q['query_type'] == 'oltp')} OLTP, {sum(1 for q in queries if q['query_type'] == 'olap')} OLAP)")
    
    # Default configs: Baseline vs Best
    if configs is None:
        configs = [
            "baseline_feature_dense",  # BASELINE: Feature classifier, dense-only, no reranker
            "transformer_hybrid_rerank",  # BEST: Transformer classifier, hybrid retrieval, reranker
        ]
    
    results = []
    
    for config_name in configs:
        print(f"\n{'='*60}")
        print(f"Evaluating: {config_name}")
        if config_name == "baseline_minimal":
            print("  → MINIMAL BASELINE: No classifier, no hybrid, no reranker")
        elif config_name == "baseline_feature_dense":
            print("  → BASELINE: Feature classifier, dense-only, no reranker")
        elif config_name == "transformer_hybrid_rerank":
            print("  → BEST: Transformer classifier, hybrid retrieval, reranker")
        print(f"{'='*60}")
        
        # Configure pipeline based on config name
        if config_name == "baseline_minimal":
            # MINIMAL BASELINE: No classifier, no hybrid, no reranker
            skip_classifier = True
            use_transformer = False
            use_hybrid = False
            use_reranker = False
            classifier_model = None
        elif config_name == "baseline_feature_dense":
            # BASELINE: Feature classifier, dense-only, no reranker
            skip_classifier = False
            use_transformer = False
            use_hybrid = False
            use_reranker = False
            classifier_model = None
        elif config_name == "transformer_hybrid_rerank":
            # BEST: Full pipeline
            skip_classifier = False
            use_transformer = True
            use_hybrid = True
            use_reranker = True
            classifier_model = "classifier2/microsoft_MiniLM-L12-H384-uncased"
        else:
            # Fallback: parse from name
            skip_classifier = False
            use_transformer = "transformer" in config_name
            use_hybrid = "hybrid" in config_name
            use_reranker = "rerank" in config_name
            classifier_model = "classifier2/microsoft_MiniLM-L12-H384-uncased" if use_transformer else None
        
        pipeline = QueryAwareRAGPipeline(
            qdrant_host=qdrant_host,
            classifier_type="transformer" if use_transformer else "feature",
            classifier_model=classifier_model,
            use_hybrid=use_hybrid,
            use_llm_answer=True,  # Enable answer generation
            llm_model="gpt-4o-mini",
            skip_classifier=skip_classifier,
            default_collection="oltp_chunks",  # Use oltp_chunks for minimal baseline
        )
        
        config_results = {
            "config": config_name,
            "is_minimal_baseline": config_name == "baseline_minimal",
            "is_baseline": config_name == "baseline_feature_dense",
            "is_best": config_name == "transformer_hybrid_rerank",
            "queries": [],
            "total_cost_usd": 0.0,
            "avg_cost_per_query_usd": 0.0,
            "oltp_cost_usd": 0.0,
            "olap_cost_usd": 0.0,
            "oltp_count": 0,
            "olap_count": 0,
            "total_retrieval_latency_ms": 0.0,
            "total_answer_latency_ms": 0.0,
        }
        
        for i, query_item in enumerate(queries):
            query = query_item["query"]
            if (i + 1) % 10 == 0:
                print(f"  Processing query {i+1}/{len(queries)}...")
            
            result = pipeline.query(query, top_k=10, use_reranker=use_reranker)
            
            query_result = {
                "query": query,
                "query_type": result.query_type,
                "retrieval_latency_ms": result.metadata.get("total_latency_ms", 0),
                "answer_cost_usd": result.answer_cost_usd or 0.0,
                "answer_latency_ms": result.answer_latency_ms or 0.0,
                "total_latency_ms": (
                    result.metadata.get("total_latency_ms", 0) + 
                    (result.answer_latency_ms or 0)
                ),
            }
            
            config_results["queries"].append(query_result)
            config_results["total_cost_usd"] += query_result["answer_cost_usd"]
            config_results["total_retrieval_latency_ms"] += query_result["retrieval_latency_ms"]
            config_results["total_answer_latency_ms"] += query_result["answer_latency_ms"]
            
            if result.query_type == "oltp":
                config_results["oltp_cost_usd"] += query_result["answer_cost_usd"]
                config_results["oltp_count"] += 1
            else:
                config_results["olap_cost_usd"] += query_result["answer_cost_usd"]
                config_results["olap_count"] += 1
        
        config_results["avg_cost_per_query_usd"] = (
            config_results["total_cost_usd"] / len(queries)
        )
        config_results["avg_oltp_cost_usd"] = (
            config_results["oltp_cost_usd"] / config_results["oltp_count"]
            if config_results["oltp_count"] > 0 else 0.0
        )
        config_results["avg_olap_cost_usd"] = (
            config_results["olap_cost_usd"] / config_results["olap_count"]
            if config_results["olap_count"] > 0 else 0.0
        )
        config_results["avg_retrieval_latency_ms"] = (
            config_results["total_retrieval_latency_ms"] / len(queries)
        )
        config_results["avg_answer_latency_ms"] = (
            config_results["total_answer_latency_ms"] / len(queries)
        )
        
        # Get LLM stats
        if pipeline.llm_generator:
            llm_stats = pipeline.llm_generator.get_total_stats()
            config_results["llm_stats"] = llm_stats
        
        results.append(config_results)
        
        print(f"\n  Total cost: ${config_results['total_cost_usd']:.4f}")
        print(f"  Avg cost/query: ${config_results['avg_cost_per_query_usd']:.4f}")
        print(f"  OLTP avg cost: ${config_results['avg_oltp_cost_usd']:.4f} ({config_results['oltp_count']} queries)")
        print(f"  OLAP avg cost: ${config_results['avg_olap_cost_usd']:.4f} ({config_results['olap_count']} queries)")
        print(f"  Avg retrieval latency: {config_results['avg_retrieval_latency_ms']:.1f}ms")
        print(f"  Avg answer latency: {config_results['avg_answer_latency_ms']:.1f}ms")
    
    # Calculate comparison metrics
    if len(results) >= 2:
        minimal_baseline = next((r for r in results if r.get("is_minimal_baseline")), None)
        baseline = next((r for r in results if r.get("is_baseline")), None)
        best = next((r for r in results if r.get("is_best")), None)
        
        comparisons = {}
        
        if minimal_baseline and best:
            comparisons["minimal_vs_best"] = {
                "cost_increase": best["avg_cost_per_query_usd"] - minimal_baseline["avg_cost_per_query_usd"],
                "cost_increase_pct": ((best["avg_cost_per_query_usd"] / minimal_baseline["avg_cost_per_query_usd"]) - 1) * 100 if minimal_baseline["avg_cost_per_query_usd"] > 0 else 0,
                "latency_increase_ms": best["avg_retrieval_latency_ms"] - minimal_baseline["avg_retrieval_latency_ms"],
                "latency_increase_pct": ((best["avg_retrieval_latency_ms"] / minimal_baseline["avg_retrieval_latency_ms"]) - 1) * 100 if minimal_baseline["avg_retrieval_latency_ms"] > 0 else 0,
            }
        
        if baseline and best:
            comparisons["baseline_vs_best"] = {
                "cost_increase": best["avg_cost_per_query_usd"] - baseline["avg_cost_per_query_usd"],
                "cost_increase_pct": ((best["avg_cost_per_query_usd"] / baseline["avg_cost_per_query_usd"]) - 1) * 100 if baseline["avg_cost_per_query_usd"] > 0 else 0,
                "latency_increase_ms": best["avg_retrieval_latency_ms"] - baseline["avg_retrieval_latency_ms"],
                "latency_increase_pct": ((best["avg_retrieval_latency_ms"] / baseline["avg_retrieval_latency_ms"]) - 1) * 100 if baseline["avg_retrieval_latency_ms"] > 0 else 0,
            }
        
        if comparisons:
            results.append({"comparisons": comparisons})
            
            print(f"\n{'='*60}")
            print("COMPARISONS")
            print(f"{'='*60}")
            if "minimal_vs_best" in comparisons:
                comp = comparisons["minimal_vs_best"]
                print(f"\nMinimal Baseline vs Best:")
                print(f"  Cost increase: ${comp['cost_increase']:.4f} ({comp['cost_increase_pct']:.1f}%)")
                print(f"  Latency increase: {comp['latency_increase_ms']:.1f}ms ({comp['latency_increase_pct']:.1f}%)")
            if "baseline_vs_best" in comparisons:
                comp = comparisons["baseline_vs_best"]
                print(f"\nBaseline vs Best:")
                print(f"  Cost increase: ${comp['cost_increase']:.4f} ({comp['cost_increase_pct']:.1f}%)")
                print(f"  Latency increase: {comp['latency_increase_ms']:.1f}ms ({comp['latency_increase_pct']:.1f}%)")
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("COST SUMMARY")
    print(f"{'='*60}")
    for r in results:
        # Skip comparison entries and entries without 'config' key
        if "comparison" in r or "comparisons" in r or "config" not in r:
            continue
        print(f"\n{r['config']}:")
        print(f"  Total: ${r['total_cost_usd']:.4f}")
        print(f"  Avg/query: ${r['avg_cost_per_query_usd']:.4f}")
        print(f"  OLTP: ${r['avg_oltp_cost_usd']:.4f} ({r['oltp_count']} queries)")
        print(f"  OLAP: ${r['avg_olap_cost_usd']:.4f} ({r['olap_count']} queries)")
        print(f"  Avg latency: {r['avg_retrieval_latency_ms']:.1f}ms (retrieval) + {r['avg_answer_latency_ms']:.1f}ms (answer)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline cost - Baseline vs Best")
    parser.add_argument("--qdrant-host", default="localhost", help="Qdrant host")
    parser.add_argument("--output", default="results/cost_evaluation.json", help="Output file")
    parser.add_argument("--num-queries", type=int, default=50, help="Number of queries to evaluate")
    parser.add_argument("--configs", nargs="+", help="Configurations to test (default: baseline_feature_dense transformer_hybrid_rerank)")
    args = parser.parse_args()
    
    evaluate_cost(
        qdrant_host=args.qdrant_host,
        output_file=args.output,
        num_queries=args.num_queries,
        configs=args.configs,
    )
