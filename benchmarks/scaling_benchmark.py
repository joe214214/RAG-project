#!/usr/bin/env python3
"""
Scaling Benchmark for RAG System

Experiments:
1. Corpus scaling: Measure retrieval latency at different corpus sizes
2. HNSW parameter tuning: ef_search vs recall/latency tradeoff
3. QPS stress test: Maximum queries per second

Usage:
    python benchmarks/scaling_benchmark.py --qdrant-host ecetesla0 --all
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
import statistics

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, HnswConfigDiff, PointStruct
from sentence_transformers import SentenceTransformer


@dataclass
class BenchmarkResult:
    experiment: str
    config: Dict
    num_queries: int
    latencies_ms: List[float]
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    qps: float
    timestamp: str


def get_percentiles(latencies: List[float]) -> Tuple[float, float, float, float]:
    if not latencies:
        return 0, 0, 0, 0
    sorted_lat = sorted(latencies)
    n = len(sorted_lat)
    return sorted_lat[int(n * 0.50)], sorted_lat[int(n * 0.95)], sorted_lat[int(n * 0.99)], statistics.mean(sorted_lat)


class ScalingBenchmark:
    def __init__(self, qdrant_host: str = "localhost", qdrant_port: int = 6333, use_real_collections: bool = False):
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.results: List[BenchmarkResult] = []
        self.use_real_collections = use_real_collections
        
    def _generate_test_vectors(self, n: int, dim: int = 384) -> np.ndarray:
        vectors = np.random.randn(n, dim).astype(np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    def _generate_test_queries(self, n: int) -> List[str]:
        templates = ["What is {}?", "Who invented {}?", "How does {} work?", "Why is {} important?"]
        topics = ["AI", "quantum computing", "blockchain", "neural networks", "deep learning", "NLP", "transformers", "RAG"]
        return [templates[i % len(templates)].format(topics[i % len(topics)]) for i in range(n)]
    
    def setup_test_collection(self, collection_name: str, num_vectors: int, hnsw_m: int = 16, hnsw_ef_construct: int = 128) -> float:
        print(f"  Setting up '{collection_name}' with {num_vectors:,} vectors...")
        try:
            self.client.delete_collection(collection_name)
        except:
            pass
        
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(m=hnsw_m, ef_construct=hnsw_ef_construct),
        )
        
        batch_size = 1000
        start_time = time.time()
        
        for batch_start in range(0, num_vectors, batch_size):
            batch_end = min(batch_start + batch_size, num_vectors)
            batch_vectors = self._generate_test_vectors(batch_end - batch_start)
            points = [PointStruct(id=batch_start + i, vector=v.tolist(), payload={"text": f"Doc {batch_start + i}"}) for i, v in enumerate(batch_vectors)]
            self.client.upsert(collection_name=collection_name, points=points)
            if (batch_start + batch_size) % 10000 == 0:
                print(f"    Inserted {batch_end:,}/{num_vectors:,}...")
        
        insert_time = time.time() - start_time
        print(f"  Done: {num_vectors:,} vectors in {insert_time:.1f}s ({num_vectors/insert_time:.0f} vec/s)")
        return insert_time
    
    def benchmark_retrieval(self, collection_name: str, num_queries: int = 100, top_k: int = 10, ef_search: int = 50) -> BenchmarkResult:
        queries = self._generate_test_queries(num_queries)
        query_vectors = self.embed_model.encode(queries, show_progress_bar=False)
        
        # Warm up
        for qv in query_vectors[:5]:
            self.client.query_points(collection_name=collection_name, query=qv.tolist(), limit=top_k, search_params=models.SearchParams(hnsw_ef=ef_search))
        
        latencies = []
        for qv in query_vectors:
            start = time.perf_counter()
            self.client.query_points(collection_name=collection_name, query=qv.tolist(), limit=top_k, search_params=models.SearchParams(hnsw_ef=ef_search))
            latencies.append((time.perf_counter() - start) * 1000)
        
        p50, p95, p99, mean = get_percentiles(latencies)
        qps = num_queries / (sum(latencies) / 1000)
        
        return BenchmarkResult(experiment="retrieval", config={"collection": collection_name, "top_k": top_k, "ef_search": ef_search},
                               num_queries=num_queries, latencies_ms=latencies, p50_ms=p50, p95_ms=p95, p99_ms=p99, mean_ms=mean, qps=qps, timestamp=datetime.now().isoformat())
    
    def run_corpus_scaling(self, corpus_sizes: List[int] = [10000, 50000, 100000], num_queries: int = 100) -> List[BenchmarkResult]:
        print("\n" + "="*60)
        print("EXPERIMENT 1: Corpus Scaling")
        print("="*60)
        
        if self.use_real_collections:
            # Use existing real collections
            collections_to_test = ["oltp_chunks", "olap_chunks"]
            results = []
            for collection_name in collections_to_test:
                if not self.client.collection_exists(collection_name):
                    print(f"  Skipping {collection_name} (does not exist)")
                    continue
                
                info = self.client.get_collection(collection_name)
                actual_size = info.points_count
                print(f"\n--- Collection: {collection_name} ({actual_size:,} vectors) ---")
                print(f"  Running {num_queries} queries...")
                result = self.benchmark_retrieval(collection_name=collection_name, num_queries=num_queries, ef_search=50)
                result.config["corpus_size"] = actual_size
                results.append(result)
                print(f"  P50={result.p50_ms:.1f}ms, P95={result.p95_ms:.1f}ms, QPS={result.qps:.1f}")
            self.results.extend(results)
            return results
        else:
            # Original synthetic data approach
            results = []
            for size in corpus_sizes:
                print(f"\n--- Corpus Size: {size:,} ---")
                collection_name = f"bench_corpus_{size}"
                self.setup_test_collection(collection_name, size)
                time.sleep(2)
                print(f"  Running {num_queries} queries...")
                result = self.benchmark_retrieval(collection_name=collection_name, num_queries=num_queries, ef_search=50)
                result.config["corpus_size"] = size
                results.append(result)
                print(f"  P50={result.p50_ms:.1f}ms, P95={result.p95_ms:.1f}ms, QPS={result.qps:.1f}")
                self.client.delete_collection(collection_name)
            self.results.extend(results)
            return results
    
    def run_hnsw_tuning(self, corpus_size: int = 100000, ef_values: List[int] = [50, 100, 200, 400], num_queries: int = 100) -> List[BenchmarkResult]:
        print("\n" + "="*60)
        print("EXPERIMENT 2: HNSW ef_search Tuning")
        print("="*60)
        
        if self.use_real_collections:
            # Test on both real collections
            collections_to_test = ["oltp_chunks", "olap_chunks"]
            results = []
            for collection_name in collections_to_test:
                if not self.client.collection_exists(collection_name):
                    print(f"  Skipping {collection_name} (does not exist)")
                    continue
                
                info = self.client.get_collection(collection_name)
                actual_size = info.points_count
                print(f"\n--- Collection: {collection_name} ({actual_size:,} vectors) ---")
                for ef in ef_values:
                    print(f"  ef_search = {ef}")
                    result = self.benchmark_retrieval(collection_name=collection_name, num_queries=num_queries, ef_search=ef)
                    result.config["corpus_size"] = actual_size
                    result.config["collection"] = collection_name
                    results.append(result)
                    print(f"    P50={result.p50_ms:.1f}ms, P95={result.p95_ms:.1f}ms, QPS={result.qps:.1f}")
            self.results.extend(results)
            return results
        else:
            # Original synthetic data approach
            collection_name = "bench_hnsw_tuning"
            self.setup_test_collection(collection_name, corpus_size)
            time.sleep(2)
            results = []
            for ef in ef_values:
                print(f"\n--- ef_search = {ef} ---")
                result = self.benchmark_retrieval(collection_name=collection_name, num_queries=num_queries, ef_search=ef)
                result.config["corpus_size"] = corpus_size
                results.append(result)
                print(f"  P50={result.p50_ms:.1f}ms, P95={result.p95_ms:.1f}ms, QPS={result.qps:.1f}")
            self.client.delete_collection(collection_name)
            self.results.extend(results)
            return results
    
    def run_qps_stress_test(self, corpus_size: int = 100000, duration_seconds: int = 30) -> BenchmarkResult:
        print("\n" + "="*60)
        print("EXPERIMENT 3: QPS Stress Test")
        print("="*60)
        
        if self.use_real_collections:
            # Use oltp_chunks for stress test (typically larger)
            collection_name = "oltp_chunks"
            if not self.client.collection_exists(collection_name):
                print(f"  Collection {collection_name} does not exist, falling back to synthetic")
                collection_name = "bench_qps_stress"
                self.setup_test_collection(collection_name, corpus_size)
                time.sleep(2)
            else:
                info = self.client.get_collection(collection_name)
                print(f"  Using {collection_name} ({info.points_count:,} vectors)")
        else:
            collection_name = "bench_qps_stress"
            self.setup_test_collection(collection_name, corpus_size)
            time.sleep(2)
        
        queries = self._generate_test_queries(1000)
        query_vectors = self.embed_model.encode(queries, show_progress_bar=False)
        print(f"  Running for {duration_seconds}s...")
        
        latencies = []
        start_time = time.time()
        query_idx = 0
        while time.time() - start_time < duration_seconds:
            qv = query_vectors[query_idx % len(query_vectors)]
            query_idx += 1
            start = time.perf_counter()
            self.client.query_points(collection_name=collection_name, query=qv.tolist(), limit=10, search_params=models.SearchParams(hnsw_ef=50))
            latencies.append((time.perf_counter() - start) * 1000)
        
        p50, p95, p99, mean = get_percentiles(latencies)
        qps = len(latencies) / (time.time() - start_time)
        print(f"  {len(latencies)} queries, QPS={qps:.1f}, P50={p50:.1f}ms, P95={p95:.1f}ms")
        if not self.use_real_collections:
            self.client.delete_collection(collection_name)
        
        result = BenchmarkResult(experiment="qps_stress", config={"corpus_size": corpus_size, "duration_seconds": duration_seconds, "total_queries": len(latencies)},
                                 num_queries=len(latencies), latencies_ms=latencies[:100], p50_ms=p50, p95_ms=p95, p99_ms=p99, mean_ms=mean, qps=qps, timestamp=datetime.now().isoformat())
        self.results.append(result)
        return result
    
    def save_results(self, output_dir: str = "results/benchmarks"):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_file = f"{output_dir}/scaling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump([{k: v if k != 'latencies_ms' else v[:50] for k, v in asdict(r).items()} for r in self.results], f, indent=2)
        print(f"\nSaved to: {output_file}")
    
    def print_summary(self):
        print("\n" + "="*80)
        print(f"{'Experiment':<15} {'Config':<25} {'P50 (ms)':<12} {'P95 (ms)':<12} {'QPS':<10}")
        print("-"*80)
        for r in self.results:
            cfg = f"size={r.config.get('corpus_size', '')} ef={r.config.get('ef_search', '')}"
            print(f"{r.experiment:<15} {cfg:<25} {r.p50_ms:<12.1f} {r.p95_ms:<12.1f} {r.qps:<10.1f}")


def main():
    parser = argparse.ArgumentParser(description="Scaling Benchmark")
    parser.add_argument("--qdrant-host", default="localhost")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--corpus-scaling", action="store_true")
    parser.add_argument("--hnsw-tuning", action="store_true")
    parser.add_argument("--qps-test", action="store_true")
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--corpus-sizes", default="10000,50000,100000")
    parser.add_argument("--use-real-collections", action="store_true", 
                       help="Use existing oltp_chunks/olap_chunks instead of synthetic data")
    args = parser.parse_args()
    
    corpus_sizes = [int(x) for x in args.corpus_sizes.split(",")]
    print(f"Qdrant: {args.qdrant_host}:{args.qdrant_port}")
    if args.use_real_collections:
        print("Mode: Using real collections (oltp_chunks, olap_chunks)")
    else:
        print("Mode: Using synthetic data")
    
    bench = ScalingBenchmark(args.qdrant_host, args.qdrant_port, args.use_real_collections)
    run_all = args.all or not (args.corpus_scaling or args.hnsw_tuning or args.qps_test)
    
    if run_all or args.corpus_scaling:
        bench.run_corpus_scaling(corpus_sizes, args.num_queries)
    if run_all or args.hnsw_tuning:
        bench.run_hnsw_tuning(num_queries=args.num_queries)
    if run_all or args.qps_test:
        bench.run_qps_stress_test()
    
    bench.print_summary()
    bench.save_results()


if __name__ == "__main__":
    main()
