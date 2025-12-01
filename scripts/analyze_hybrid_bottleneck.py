#!/usr/bin/env python3
"""
Analyze bottleneck in hybrid retrieval to understand why QPS is constant.

Measures time spent in each phase:
- BM25 lookup
- Dense Qdrant search
- Individual Qdrant retrieves
- RRF fusion
"""

import argparse
import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.rag_pipeline import QueryAwareRAGPipeline


def analyze_single_query(pipeline, query, num_runs=10):
    """Analyze a single query to measure time breakdown."""
    print(f"\nAnalyzing query: '{query}'")
    print(f"Running {num_runs} times to get average...")
    
    # Ensure BM25 index is built first (lazy loading)
    print("  Building BM25 index if needed...")
    collection = "oltp_chunks"
    if collection not in pipeline.bm25_indexes:
        pipeline._build_bm25_index(collection)
    
    if collection not in pipeline.bm25_indexes:
        print(f"  ⚠️  Warning: BM25 index not available for {collection}")
        return {
            'bm25': [],
            'dense': [],
            'retrieve': [],
            'fusion': [],
            'total': []
        }
    
    times = {
        'bm25': [],
        'dense': [],
        'retrieve': [],
        'fusion': [],
        'total': []
    }
    
    for i in range(num_runs):
        # Monkey-patch to measure times
        original_bm25_retrieve = pipeline.bm25_indexes[collection].retrieve
        original_qdrant_query = pipeline.qdrant_client.query_points
        original_qdrant_retrieve = pipeline.qdrant_client.retrieve
        original_rrf = pipeline._reciprocal_rank_fusion
        
        bm25_time = 0
        dense_time = 0
        retrieve_time = 0
        fusion_time = 0
        
        def timed_bm25_retrieve(*args, **kwargs):
            nonlocal bm25_time
            start = time.time()
            result = original_bm25_retrieve(*args, **kwargs)
            bm25_time += (time.time() - start) * 1000
            return result
        
        def timed_qdrant_query(*args, **kwargs):
            nonlocal dense_time
            start = time.time()
            result = original_qdrant_query(*args, **kwargs)
            dense_time += (time.time() - start) * 1000
            return result
        
        def timed_qdrant_retrieve(*args, **kwargs):
            nonlocal retrieve_time
            start = time.time()
            result = original_qdrant_retrieve(*args, **kwargs)
            retrieve_time += (time.time() - start) * 1000
            return result
        
        def timed_rrf(*args, **kwargs):
            nonlocal fusion_time
            start = time.time()
            result = original_rrf(*args, **kwargs)
            fusion_time += (time.time() - start) * 1000
            return result
        
        # Patch methods
        pipeline.bm25_indexes[collection].retrieve = timed_bm25_retrieve
        pipeline.qdrant_client.query_points = timed_qdrant_query
        pipeline.qdrant_client.retrieve = timed_qdrant_retrieve
        pipeline._reciprocal_rank_fusion = timed_rrf
        
        # Run query
        start_total = time.time()
        try:
            result = pipeline.query(query, top_k=10, use_reranker=False)
            total_time = (time.time() - start_total) * 1000
            
            times['bm25'].append(bm25_time)
            times['dense'].append(dense_time)
            times['retrieve'].append(retrieve_time)
            times['fusion'].append(fusion_time)
            times['total'].append(total_time)
        except Exception as e:
            print(f"  Error on run {i+1}: {e}")
        
        # Restore original methods
        pipeline.bm25_indexes[collection].retrieve = original_bm25_retrieve
        pipeline.qdrant_client.query_points = original_qdrant_query
        pipeline.qdrant_client.retrieve = original_qdrant_retrieve
        pipeline._reciprocal_rank_fusion = original_rrf
    
    # Calculate averages
    print(f"\n{'Phase':<20} {'Avg (ms)':<15} {'Min (ms)':<15} {'Max (ms)':<15} {'% of Total':<15}")
    print("-" * 80)
    
    total_avg = sum(times['total']) / len(times['total'])
    
    for phase in ['bm25', 'dense', 'retrieve', 'fusion', 'total']:
        if times[phase]:
            avg = sum(times[phase]) / len(times[phase])
            min_val = min(times[phase])
            max_val = max(times[phase])
            pct = (avg / total_avg * 100) if total_avg > 0 else 0
            print(f"{phase.capitalize():<20} {avg:<15.2f} {min_val:<15.2f} {max_val:<15.2f} {pct:<15.1f}%")
    
    return times


def main():
    parser = argparse.ArgumentParser(description="Analyze hybrid retrieval bottleneck")
    parser.add_argument("--qdrant-host", type=str, default="localhost", help="Qdrant server host")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant server port")
    parser.add_argument("--num-runs", type=int, default=10, help="Number of runs per query")
    parser.add_argument("--queries", type=str, nargs="+", 
                       default=["What is machine learning?", "Who founded Microsoft?"],
                       help="Queries to analyze")
    
    args = parser.parse_args()
    
    print("="*80)
    print("HYBRID RETRIEVAL BOTTLENECK ANALYSIS")
    print("="*80)
    
    # Initialize pipeline with hybrid enabled (disable query expansion for performance testing)
    print("\nInitializing pipeline with hybrid retrieval...")
    pipeline = QueryAwareRAGPipeline(
        classifier_type="feature",
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        use_hybrid=True,
        use_query_expansion=False,  # Disable for performance testing
    )
    
    print("✓ Pipeline initialized")
    
    # Analyze each query
    all_times = {
        'bm25': [],
        'dense': [],
        'retrieve': [],
        'fusion': [],
        'total': []
    }
    
    for query in args.queries:
        times = analyze_single_query(pipeline, query, args.num_runs)
        for phase in all_times:
            all_times[phase].extend(times[phase])
    
    # Overall summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY (across all queries)")
    print("="*80)
    print(f"{'Phase':<20} {'Avg (ms)':<15} {'% of Total':<15}")
    print("-" * 50)
    
    total_avg = sum(all_times['total']) / len(all_times['total']) if all_times['total'] else 0
    
    for phase in ['bm25', 'dense', 'retrieve', 'fusion', 'total']:
        if all_times[phase]:
            avg = sum(all_times[phase]) / len(all_times[phase])
            pct = (avg / total_avg * 100) if total_avg > 0 else 0
            print(f"{phase.capitalize():<20} {avg:<15.2f} {pct:<15.1f}%")
    
    # Identify bottleneck
    print("\n" + "="*80)
    print("BOTTLENECK IDENTIFICATION")
    print("="*80)
    
    phase_avgs = {}
    for phase in ['bm25', 'dense', 'retrieve', 'fusion']:
        if all_times[phase]:
            phase_avgs[phase] = sum(all_times[phase]) / len(all_times[phase])
    
    if phase_avgs:
        bottleneck = max(phase_avgs.items(), key=lambda x: x[1])
        print(f"Primary bottleneck: {bottleneck[0].upper()} ({bottleneck[1]:.2f}ms, {bottleneck[1]/total_avg*100:.1f}% of total)")
        
        # Recommendations
        print("\nRecommendations:")
        if bottleneck[0] == 'retrieve':
            print("  ⚠️  Individual Qdrant retrieves are the bottleneck!")
            print("  → Batch retrieves: Use qdrant_client.retrieve() with multiple IDs")
            print("  → Or: Pre-fetch metadata during dense search")
        elif bottleneck[0] == 'dense':
            print("  ⚠️  Dense Qdrant search is the bottleneck!")
            print("  → Reduce ef_search parameter")
            print("  → Use connection pooling")
        elif bottleneck[0] == 'bm25':
            print("  ⚠️  BM25 lookup is the bottleneck!")
            print("  → Optimize BM25 index (already in-memory, should be fast)")
            print("  → Check if there's thread contention")
        elif bottleneck[0] == 'fusion':
            print("  ⚠️  RRF fusion is the bottleneck!")
            print("  → Optimize fusion algorithm")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

