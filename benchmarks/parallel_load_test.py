#!/usr/bin/env python3
"""
Parallel Query Load Testing for RAG System

Tests the system under concurrent query load to measure:
- Throughput (QPS - Queries Per Second)
- Latency (P50, P95, P99)
- System behavior under stress

Usage:
    # Load testing (dense-only, fast QPS):
    python benchmarks/parallel_load_test.py \
        --qdrant-host ecetesla0 \
        --num-queries 1000 \
        --concurrency-levels 1,5,10,20,50 \
        --output results/load_test.json
    
    # Quality evaluation (hybrid retrieval, slower but better recall):
    python benchmarks/parallel_load_test.py \
        --qdrant-host ecetesla0 \
        --use-hybrid \
        --num-queries 100 \
        --concurrency-levels 1,5,10 \
        --output results/load_test_hybrid.json

Performance Trade-offs:
- Dense-only (default): ~80 QPS, ~17ms latency, good for load testing
- Hybrid (--use-hybrid): ~7 QPS, ~200ms latency, better recall for quality evaluation
"""

import argparse
import json
import time
import statistics
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.rag_pipeline import QueryAwareRAGPipeline


@dataclass
class QueryLatency:
    """Latency measurement for a single query."""
    query: str
    latency_ms: float
    query_type: str
    success: bool
    error: Optional[str] = None


@dataclass
class LoadTestResult:
    """Results for a single concurrency level."""
    concurrency: int
    total_queries: int
    successful_queries: int
    failed_queries: int
    duration_seconds: float
    qps: float  # Queries per second
    p50_ms: float
    p95_ms: float
    p99_ms: float
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    latencies: List[float]  # All latency measurements
    p999_ms: Optional[float] = None


def load_test_queries(num_queries: int) -> List[str]:
    """Generate test queries (mix of OLTP and OLAP)."""
    oltp_queries = [
        "What is machine learning?",
        "Who founded Microsoft?",
        "What is the capital of France?",
        "When was Python created?",
        "What is the speed of light?",
        "Who wrote Romeo and Juliet?",
        "What is the largest planet?",
        "What is artificial intelligence?",
        "Who invented the telephone?",
        "What is the chemical formula for water?",
    ]
    
    olap_queries = [
        "Compare machine learning and deep learning",
        "What are the differences between AI and ML?",
        "Explain the relationship between neural networks and deep learning",
        "How do transformers differ from RNNs?",
        "What are the advantages and disadvantages of supervised learning?",
        "Compare supervised and unsupervised learning approaches",
        "How does reinforcement learning relate to other ML paradigms?",
        "What are the key differences between CNNs and RNNs?",
        "Explain the evolution of NLP from rule-based to transformer models",
        "Compare different optimization algorithms used in deep learning",
    ]
    
    all_queries = oltp_queries + olap_queries
    # Repeat queries to reach num_queries
    queries = []
    for i in range(num_queries):
        queries.append(all_queries[i % len(all_queries)])
    
    return queries


def run_single_query(
    pipeline: QueryAwareRAGPipeline,
    query: str,
    use_reranker: bool = False,
    reranker_model: Optional[str] = None,
) -> QueryLatency:
    """Run a single query and measure latency."""
    start_time = time.time()
    try:
        result = pipeline.query(
            query=query,
            top_k=10,
            use_reranker=use_reranker,
            reranker_model=reranker_model,
        )
        latency_ms = (time.time() - start_time) * 1000
        return QueryLatency(
            query=query,
            latency_ms=latency_ms,
            query_type=result.query_type,
            success=True,
        )
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        return QueryLatency(
            query=query,
            latency_ms=latency_ms,
            query_type="unknown",
            success=False,
            error=str(e),
        )


def run_concurrency_test(
    pipeline: QueryAwareRAGPipeline,
    queries: List[str],
    concurrency: int,
    use_reranker: bool = False,
    reranker_model: Optional[str] = None,
) -> LoadTestResult:
    """
    Run queries with specified concurrency level.
    
    Uses ThreadPoolExecutor with context manager for proper resource management.
    Each query is independent, making this suitable for parallel execution.
    
    Args:
        pipeline: Initialized RAG pipeline (should be thread-safe for read operations)
        queries: List of queries to run
        concurrency: Number of concurrent clients
        use_reranker: Whether to use reranking
        reranker_model: Reranker model to use
        
    Returns:
        LoadTestResult with metrics
    """
    print(f"\n{'='*60}")
    print(f"Testing concurrency level: {concurrency}")
    print(f"{'='*60}")
    
    results: List[QueryLatency] = []
    start_time = time.time()
    
    # Use ThreadPoolExecutor with context manager (best practice)
    # This ensures proper cleanup and resource management
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        # Submit all queries - each task is independent
        futures = {
            executor.submit(run_single_query, pipeline, query, use_reranker, reranker_model): query
            for query in queries
        }
        
        # Collect results as they complete using as_completed (best practice)
        # This processes results in completion order, not submission order
        completed = 0
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                # Handle exceptions from worker threads
                # Create a failed result entry
                query = futures.get(future, "unknown")
                results.append(QueryLatency(
                    query=query if isinstance(query, str) else "unknown",
                    latency_ms=0.0,
                    query_type="unknown",
                    success=False,
                    error=str(e),
                ))
            completed += 1
            if completed % max(1, len(queries) // 10) == 0:
                print(f"  Completed {completed}/{len(queries)} queries...")
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Calculate metrics
    latencies = [r.latency_ms for r in results if r.success]
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    
    if not latencies:
        print(f"  ⚠️  All queries failed!")
        return LoadTestResult(
            concurrency=concurrency,
            total_queries=len(queries),
            successful_queries=0,
            failed_queries=len(queries),
            duration_seconds=duration,
            qps=0.0,
            p50_ms=0.0,
            p95_ms=0.0,
            p99_ms=0.0,
            avg_latency_ms=0.0,
            min_latency_ms=0.0,
            max_latency_ms=0.0,
            latencies=[],
        )
    
    latencies_sorted = sorted(latencies)
    qps = len(queries) / duration
    
    # Calculate percentiles
    def percentile(values: List[float], pct: float) -> float:
        """Calculate percentile value."""
        if not values:
            return 0.0
        idx = int(len(values) * pct / 100.0)
        idx = min(idx, len(values) - 1)
        return values[idx]
    
    return LoadTestResult(
        concurrency=concurrency,
        total_queries=len(queries),
        successful_queries=successful,
        failed_queries=failed,
        duration_seconds=duration,
        qps=qps,
        p50_ms=statistics.median(latencies_sorted),
        p95_ms=percentile(latencies_sorted, 95),
        p99_ms=percentile(latencies_sorted, 99),
        p999_ms=percentile(latencies_sorted, 99.9) if len(latencies_sorted) > 0 else None,
        avg_latency_ms=statistics.mean(latencies),
        min_latency_ms=min(latencies),
        max_latency_ms=max(latencies),
        latencies=latencies,
    )


def print_results(results: List[LoadTestResult]):
    """Print load test results in a table."""
    print("\n" + "="*80)
    print("PARALLEL LOAD TEST RESULTS")
    print("="*80)
    print(f"{'Concurrency':<12} {'QPS':<10} {'P50 (ms)':<12} {'P95 (ms)':<12} {'P99 (ms)':<12} {'Avg (ms)':<12} {'Success %':<10}")
    print("-"*80)
    
    for r in results:
        success_rate = (r.successful_queries / r.total_queries * 100) if r.total_queries > 0 else 0
        print(f"{r.concurrency:<12} {r.qps:<10.2f} {r.p50_ms:<12.2f} {r.p95_ms:<12.2f} {r.p99_ms:<12.2f} {r.avg_latency_ms:<12.2f} {success_rate:<10.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Parallel query load testing for RAG system")
    parser.add_argument("--qdrant-host", type=str, default="localhost", help="Qdrant server host")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant server port")
    parser.add_argument("--num-queries", type=int, default=100, help="Total number of queries to run")
    parser.add_argument(
        "--concurrency-levels",
        type=str,
        default="1,5,10,20,50",
        help="Comma-separated list of concurrency levels to test (e.g., '1,5,10,20,50')"
    )
    parser.add_argument("--classifier-type", type=str, default="transformer", choices=["feature", "transformer"], help="Classifier type")
    parser.add_argument("--classifier-model", type=str, default="classifier2/microsoft_MiniLM-L12-H384-uncased", help="Classifier model path (for transformer)")
    parser.add_argument("--use-hybrid", action="store_true", 
                       help="Use hybrid retrieval (BM25 + dense). WARNING: ~10x slower (~7 QPS vs ~80 QPS). Use for quality evaluation, not load testing.")
    parser.add_argument("--use-reranker", action="store_true", help="Use reranker")
    parser.add_argument("--reranker-model", type=str, default=None, help="Reranker model (e.g., 'tinybert', 'minilm')")
    parser.add_argument("--output", type=str, default="results/parallel_load_test.json", help="Output JSON file")
    parser.add_argument("--device", type=str, default="cpu", help="Device for embedding model")
    
    args = parser.parse_args()
    
    # Parse concurrency levels
    concurrency_levels = [int(x.strip()) for x in args.concurrency_levels.split(",")]
    
    # Generate test queries
    print(f"Generating {args.num_queries} test queries...")
    queries = load_test_queries(args.num_queries)
    
    # Initialize pipeline (shared across threads - should be thread-safe)
    # Note: Qdrant client, embedding models, and classifiers are typically thread-safe for read operations
    print(f"\nInitializing RAG pipeline...")
    print(f"  Qdrant: {args.qdrant_host}:{args.qdrant_port}")
    print(f"  Classifier: {args.classifier_type}")
    if args.classifier_model:
        print(f"  Classifier model: {args.classifier_model}")
    print(f"  Hybrid: {args.use_hybrid}")
    if args.use_hybrid:
        print(f"    ⚠️  Hybrid mode enabled - expect ~7 QPS (vs ~80 QPS dense-only)")
        print(f"    → Use for quality evaluation, not load testing")
    else:
        print(f"    ✓ Dense-only mode (default) - optimized for load testing")
    print(f"  Reranker: {args.use_reranker}")
    if args.reranker_model:
        print(f"  Reranker model: {args.reranker_model}")
    
    pipeline = QueryAwareRAGPipeline(
        classifier_type=args.classifier_type,
        classifier_model=args.classifier_model,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        use_hybrid=args.use_hybrid,
        use_query_expansion=False,  # Disable query expansion for load testing (saves ~20-50ms per query)
        device=args.device,
    )
    
    # Warm-up: Run a few queries to initialize models and caches
    print("\nWarming up pipeline...")
    warmup_queries = ["What is machine learning?", "Compare AI and ML"]
    for query in warmup_queries:
        try:
            pipeline.query(query, top_k=5)
        except Exception as e:
            print(f"  Warning: Warm-up query failed: {e}")
    
    print("  Warm-up complete")
    
    # Run tests for each concurrency level
    all_results: List[LoadTestResult] = []
    
    for concurrency in concurrency_levels:
        result = run_concurrency_test(
            pipeline=pipeline,
            queries=queries,
            concurrency=concurrency,
            use_reranker=args.use_reranker,
            reranker_model=args.reranker_model,
        )
        all_results.append(result)
        
        # Print summary for this concurrency level
        print(f"\n  Results:")
        print(f"    QPS: {result.qps:.2f}")
        print(f"    P50: {result.p50_ms:.2f}ms, P95: {result.p95_ms:.2f}ms, P99: {result.p99_ms:.2f}ms")
        success_rate = (result.successful_queries / result.total_queries * 100) if result.total_queries > 0 else 0
        print(f"    Success rate: {result.successful_queries}/{result.total_queries} ({success_rate:.1f}%)")
        
        # Small delay between tests to let system stabilize
        if concurrency != concurrency_levels[-1]:
            time.sleep(2)
    
    # Print summary table
    print_results(all_results)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results_dict = {
        "config": {
            "qdrant_host": args.qdrant_host,
            "qdrant_port": args.qdrant_port,
            "num_queries": args.num_queries,
            "concurrency_levels": concurrency_levels,
            "classifier_type": args.classifier_type,
            "classifier_model": args.classifier_model,
            "use_hybrid": args.use_hybrid,
            "use_reranker": args.use_reranker,
            "reranker_model": args.reranker_model,
            "device": args.device,
        },
        "results": [asdict(r) for r in all_results],
    }
    
    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")
    print(f"\nSummary:")
    print(f"  Tested {len(concurrency_levels)} concurrency levels")
    print(f"  Total queries per level: {args.num_queries}")
    if all_results:
        best_result = max(all_results, key=lambda r: r.qps)
        print(f"  Best QPS: {best_result.qps:.2f} (concurrency={best_result.concurrency})")
        worst_latency = max(all_results, key=lambda r: r.p95_ms)
        print(f"  Worst P95 latency: {worst_latency.p95_ms:.2f}ms (concurrency={worst_latency.concurrency})")


if __name__ == "__main__":
    main()

