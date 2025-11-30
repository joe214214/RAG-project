#!/usr/bin/env python3
"""
MPI-Based Parallel Query Load Testing for RAG System

Distributes query clients across multiple nodes using MPI.
Each MPI rank acts as an independent client, simulating realistic distributed load.

Usage:
    # Run with 4 clients across 2 nodes
    mpirun -np 4 -host ecetesla1,ecetesla2 \
        python benchmarks/mpi_load_test.py \
        --qdrant-host ecetesla0 \
        --num-queries-per-client 100 \
        --output results/mpi_load_test.json
    
    # Run with 8 clients across 4 nodes
    mpirun -np 8 -host ecetesla1,ecetesla2,ecetesla3,ecetesla4 \
        python benchmarks/mpi_load_test.py \
        --qdrant-host ecetesla0 \
        --num-queries-per-client 50 \
        --output results/mpi_load_test_8clients.json
"""

import argparse
import json
import time
import statistics
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import sys

try:
    from mpi4py import MPI
except ImportError:
    print("Error: mpi4py not installed. Install with: pip install mpi4py")
    sys.exit(1)

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
class ClientResult:
    """Results from a single MPI client (rank)."""
    rank: int
    hostname: str
    total_queries: int
    successful_queries: int
    failed_queries: int
    duration_seconds: float
    qps: float  # Queries per second for this client
    p50_ms: float
    p95_ms: float
    p99_ms: float
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    latencies: List[float]


@dataclass
class LoadTestResult:
    """Aggregated results across all clients."""
    total_clients: int
    total_queries: int
    successful_queries: int
    failed_queries: int
    duration_seconds: float
    aggregate_qps: float  # Total QPS across all clients
    p50_ms: float
    p95_ms: float
    p99_ms: float
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    client_results: List[ClientResult]


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


def run_client_load_test(
    rank: int,
    comm: MPI.Comm,
    queries: List[str],
    qdrant_host: str,
    qdrant_port: int,
    classifier_type: str,
    classifier_model: Optional[str],
    use_hybrid: bool,
    use_reranker: bool,
    reranker_model: Optional[str],
    device: str,
) -> ClientResult:
    """
    Run load test on a single MPI client (rank).
    
    Each rank acts as an independent client, running queries concurrently
    with other ranks across the network.
    """
    import socket
    hostname = socket.gethostname()
    
    # Initialize pipeline for this client
    pipeline = QueryAwareRAGPipeline(
        classifier_type=classifier_type,
        classifier_model=classifier_model,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        use_hybrid=use_hybrid,
        device=device,
    )
    
    # Warm-up: Run a few queries to initialize models
    warmup_queries = ["What is machine learning?", "Compare AI and ML"]
    for query in warmup_queries:
        try:
            pipeline.query(query, top_k=5)
        except:
            pass
    
    # Synchronize all clients before starting
    comm.Barrier()
    
    # Run queries and measure latency
    results: List[QueryLatency] = []
    start_time = time.time()
    
    for query in queries:
        result = run_single_query(
            pipeline=pipeline,
            query=query,
            use_reranker=use_reranker,
            reranker_model=reranker_model,
        )
        results.append(result)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Calculate metrics for this client
    latencies = [r.latency_ms for r in results if r.success]
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    
    if not latencies:
        return ClientResult(
            rank=rank,
            hostname=hostname,
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
    
    def percentile(values: List[float], pct: float) -> float:
        """Calculate percentile value."""
        if not values:
            return 0.0
        idx = int(len(values) * pct / 100.0)
        idx = min(idx, len(values) - 1)
        return values[idx]
    
    return ClientResult(
        rank=rank,
        hostname=hostname,
        total_queries=len(queries),
        successful_queries=successful,
        failed_queries=failed,
        duration_seconds=duration,
        qps=qps,
        p50_ms=statistics.median(latencies_sorted),
        p95_ms=percentile(latencies_sorted, 95),
        p99_ms=percentile(latencies_sorted, 99),
        avg_latency_ms=statistics.mean(latencies),
        min_latency_ms=min(latencies),
        max_latency_ms=max(latencies),
        latencies=latencies,
    )


def aggregate_results(client_results: List[ClientResult]) -> LoadTestResult:
    """Aggregate results from all clients."""
    if not client_results:
        raise ValueError("No client results to aggregate")
    
    # Aggregate statistics
    total_queries = sum(r.total_queries for r in client_results)
    successful_queries = sum(r.successful_queries for r in client_results)
    failed_queries = sum(r.failed_queries for r in client_results)
    
    # Duration is max across all clients (they run concurrently)
    duration = max(r.duration_seconds for r in client_results)
    
    # Aggregate QPS is sum of all client QPS
    aggregate_qps = sum(r.qps for r in client_results)
    
    # Combine all latencies for global percentiles
    all_latencies = []
    for r in client_results:
        all_latencies.extend(r.latencies)
    
    if not all_latencies:
        return LoadTestResult(
            total_clients=len(client_results),
            total_queries=total_queries,
            successful_queries=0,
            failed_queries=failed_queries,
            duration_seconds=duration,
            aggregate_qps=0.0,
            p50_ms=0.0,
            p95_ms=0.0,
            p99_ms=0.0,
            avg_latency_ms=0.0,
            min_latency_ms=0.0,
            max_latency_ms=0.0,
            client_results=client_results,
        )
    
    all_latencies_sorted = sorted(all_latencies)
    
    def percentile(values: List[float], pct: float) -> float:
        """Calculate percentile value."""
        if not values:
            return 0.0
        idx = int(len(values) * pct / 100.0)
        idx = min(idx, len(values) - 1)
        return values[idx]
    
    return LoadTestResult(
        total_clients=len(client_results),
        total_queries=total_queries,
        successful_queries=successful_queries,
        failed_queries=failed_queries,
        duration_seconds=duration,
        aggregate_qps=aggregate_qps,
        p50_ms=statistics.median(all_latencies_sorted),
        p95_ms=percentile(all_latencies_sorted, 95),
        p99_ms=percentile(all_latencies_sorted, 99),
        avg_latency_ms=statistics.mean(all_latencies),
        min_latency_ms=min(all_latencies),
        max_latency_ms=max(all_latencies),
        client_results=client_results,
    )


def print_results(result: LoadTestResult):
    """Print load test results."""
    print("\n" + "="*80)
    print("MPI DISTRIBUTED LOAD TEST RESULTS")
    print("="*80)
    print(f"Total Clients: {result.total_clients}")
    print(f"Total Queries: {result.total_queries}")
    print(f"Successful: {result.successful_queries} ({result.successful_queries/result.total_queries*100:.1f}%)")
    print(f"Failed: {result.failed_queries}")
    print(f"\nDuration: {result.duration_seconds:.2f}s")
    print(f"Aggregate QPS: {result.aggregate_qps:.2f} queries/second")
    print(f"\nGlobal Latency Metrics:")
    print(f"  P50: {result.p50_ms:.2f}ms")
    print(f"  P95: {result.p95_ms:.2f}ms")
    print(f"  P99: {result.p99_ms:.2f}ms")
    print(f"  Avg: {result.avg_latency_ms:.2f}ms")
    print(f"  Min: {result.min_latency_ms:.2f}ms")
    print(f"  Max: {result.max_latency_ms:.2f}ms")
    
    print(f"\n{'='*80}")
    print("Per-Client Breakdown:")
    print(f"{'Rank':<6} {'Hostname':<20} {'QPS':<10} {'P50 (ms)':<12} {'P95 (ms)':<12} {'Success %':<10}")
    print("-"*80)
    
    for r in result.client_results:
        success_rate = (r.successful_queries / r.total_queries * 100) if r.total_queries > 0 else 0
        print(f"{r.rank:<6} {r.hostname:<20} {r.qps:<10.2f} {r.p50_ms:<12.2f} {r.p95_ms:<12.2f} {success_rate:<10.1f}%")


def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    parser = argparse.ArgumentParser(description="MPI-based parallel query load testing for RAG system")
    parser.add_argument("--qdrant-host", type=str, default="localhost", help="Qdrant server host")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant server port")
    parser.add_argument("--num-queries-per-client", type=int, default=100, help="Number of queries per MPI client")
    parser.add_argument("--classifier-type", type=str, default="feature", choices=["feature", "transformer"], help="Classifier type")
    parser.add_argument("--classifier-model", type=str, default=None, help="Classifier model path (for transformer)")
    parser.add_argument("--use-hybrid", action="store_true", help="Use hybrid retrieval")
    parser.add_argument("--use-reranker", action="store_true", help="Use reranker")
    parser.add_argument("--reranker-model", type=str, default=None, help="Reranker model (e.g., 'tinybert', 'minilm')")
    parser.add_argument("--output", type=str, default="results/mpi_load_test.json", help="Output JSON file")
    parser.add_argument("--device", type=str, default="cpu", help="Device for embedding model")
    
    args = parser.parse_args()
    
    # Generate queries for this client
    queries = load_test_queries(args.num_queries_per_client)
    
    # Only rank 0 prints initial info
    if rank == 0:
        print(f"\n{'='*80}")
        print("MPI DISTRIBUTED QUERY LOAD TEST")
        print(f"{'='*80}")
        print(f"Total MPI Clients: {size}")
        print(f"Queries per client: {args.num_queries_per_client}")
        print(f"Total queries: {size * args.num_queries_per_client}")
        print(f"Qdrant: {args.qdrant_host}:{args.qdrant_port}")
        print(f"Classifier: {args.classifier_type}")
        if args.classifier_model:
            print(f"Classifier model: {args.classifier_model}")
        print(f"Hybrid: {args.use_hybrid}")
        print(f"Reranker: {args.use_reranker}")
        if args.reranker_model:
            print(f"Reranker model: {args.reranker_model}")
        print(f"\nStarting load test...")
    
    # Run load test on this client
    client_result = run_client_load_test(
        rank=rank,
        comm=comm,
        queries=queries,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        classifier_type=args.classifier_type,
        classifier_model=args.classifier_model,
        use_hybrid=args.use_hybrid,
        use_reranker=args.use_reranker,
        reranker_model=args.reranker_model,
        device=args.device,
    )
    
    # Gather results from all clients to rank 0
    all_results = comm.gather(client_result, root=0)
    
    # Rank 0 aggregates and saves results
    if rank == 0:
        aggregated = aggregate_results(all_results)
        print_results(aggregated)
        
        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results_dict = {
            "config": {
                "total_clients": size,
                "queries_per_client": args.num_queries_per_client,
                "total_queries": size * args.num_queries_per_client,
                "qdrant_host": args.qdrant_host,
                "qdrant_port": args.qdrant_port,
                "classifier_type": args.classifier_type,
                "classifier_model": args.classifier_model,
                "use_hybrid": args.use_hybrid,
                "use_reranker": args.use_reranker,
                "reranker_model": args.reranker_model,
                "device": args.device,
            },
            "aggregated": asdict(aggregated),
        }
        
        with open(output_path, "w") as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\n✅ Results saved to: {output_path}")
        print(f"\nSummary:")
        print(f"  Aggregate QPS: {aggregated.aggregate_qps:.2f} queries/second")
        print(f"  Global P95 latency: {aggregated.p95_ms:.2f}ms")
        print(f"  Success rate: {aggregated.successful_queries}/{aggregated.total_queries} ({aggregated.successful_queries/aggregated.total_queries*100:.1f}%)")


if __name__ == "__main__":
    main()

