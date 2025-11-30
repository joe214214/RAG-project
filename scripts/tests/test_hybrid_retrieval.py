#!/usr/bin/env python3
"""
Test Hybrid Retrieval on ecetesla with Qdrant.

Usage:
    # On ecetesla0 (where Qdrant runs)
    python scripts/test_hybrid_retrieval.py --qdrant-host localhost
    
    # From another node
    python scripts/test_hybrid_retrieval.py --qdrant-host ecetesla0
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_bm25_only():
    """Test BM25 retriever standalone."""
    print("\n" + "=" * 60)
    print("TEST 1: BM25 Sparse Retrieval")
    print("=" * 60)
    
    from retrievers import BM25Retriever
    
    # Sample documents
    docs = {
        "doc1": "Apple Inc is a technology company founded by Steve Jobs",
        "doc2": "Microsoft was founded by Bill Gates and Paul Allen",
        "doc3": "Tesla electric vehicles are manufactured by Tesla Inc",
        "doc4": "Machine learning is a subset of artificial intelligence",
        "doc5": "Steve Jobs and Bill Gates were both technology pioneers",
    }
    
    bm25 = BM25Retriever.from_id_to_text(docs)
    print(f"Created BM25 index with {len(bm25)} documents")
    
    # Test queries
    queries = [
        "Steve Jobs Apple",
        "Bill Gates Microsoft",
        "electric vehicles Tesla",
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = bm25.retrieve(query, top_k=3)
        for doc_id, score in results:
            print(f"  [{doc_id}] score={score:.3f}: {docs[doc_id][:50]}...")
    
    print("\n✅ BM25 test PASSED")
    return bm25, docs


def test_hybrid_with_qdrant(qdrant_host: str, qdrant_port: int, collection: str, hnsw_ef: int = 50):
    """Test full hybrid retrieval with Qdrant."""
    print("\n" + "=" * 60)
    print(f"TEST 2: Hybrid Retrieval with Qdrant (HNSW ef={hnsw_ef})")
    print("=" * 60)
    
    from retrievers import BM25Retriever
    from qdrant_client import QdrantClient
    from qdrant_client import models
    from sentence_transformers import SentenceTransformer
    import numpy as np
    
    # Connect to Qdrant
    print(f"Connecting to Qdrant at {qdrant_host}:{qdrant_port}...")
    
    try:
        client = QdrantClient(host=qdrant_host, port=qdrant_port, check_compatibility=False)
        coll_info = client.get_collection(collection)
        print(f"✅ Connected to Qdrant collection: {collection} ({coll_info.points_count} points)")
    except Exception as e:
        print(f"❌ Failed to connect to Qdrant: {e}")
        return
    
    # Load embedder
    print("Loading embedding model...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # Get some documents from Qdrant to build BM25 index
    print("Building BM25 index from Qdrant data...")
    
    # HNSW search parameters (ef controls search quality/speed tradeoff)
    # OLTP: ef=50 (faster), OLAP: ef=200 (higher recall)
    search_params = models.SearchParams(hnsw_ef=hnsw_ef, exact=False)
    
    # Sample query to get documents
    sample_vec = embedder.encode("sample query")
    sample_results = client.query_points(
        collection_name=collection,
        query=sample_vec.tolist(),
        limit=100,
        search_params=search_params,
    )
    
    if not sample_results.points:
        print("❌ No documents found in Qdrant. Ingest data first.")
        return
    
    # Build BM25 from retrieved documents
    id_to_text = {}
    for hit in sample_results.points:
        doc_id = str(hit.id)
        text = hit.payload.get("text", "") if hit.payload else ""
        if text:
            id_to_text[doc_id] = text
    
    print(f"Built BM25 index from {len(id_to_text)} documents")
    
    if len(id_to_text) < 5:
        print("⚠️ Warning: Very few documents found. Results may be limited.")
    
    bm25 = BM25Retriever.from_id_to_text(id_to_text)
    
    # Test queries
    test_queries = [
        "Who founded Microsoft?",
        "What is machine learning?",
        "capital of France",
    ]
    
    print("\n--- Hybrid Search Results ---")
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        query_vec = embedder.encode(query)
        
        # Dense only search (with HNSW ef=50 for OLTP-style fast search)
        print("\n  [Dense Only]:")
        dense_results = client.query_points(
            collection_name=collection,
            query=query_vec.tolist(),
            limit=3,
            search_params=search_params,
        )
        for hit in dense_results.points:
            text = hit.payload.get("text", "")[:60] if hit.payload else ""
            print(f"    [{hit.id}] score={hit.score:.3f}: {text}...")
        
        # BM25 only search
        print("\n  [BM25 Only]:")
        bm25_results = bm25.retrieve(query, top_k=3)
        for doc_id, score in bm25_results:
            text = id_to_text.get(doc_id, "")[:60]
            print(f"    [{doc_id}] score={score:.3f}: {text}...")
        
        # Hybrid search (manual fusion)
        print("\n  [Hybrid (α=0.3)]:")
        alpha = 0.3
        
        # Get BM25 scores
        bm25_all = bm25.retrieve(query, top_k=50)
        bm25_scores = {doc_id: score for doc_id, score in bm25_all}
        
        # Get dense scores (with HNSW parameters)
        dense_all = client.query_points(
            collection_name=collection,
            query=query_vec.tolist(),
            limit=50,
            search_params=search_params,
        )
        dense_scores = {str(hit.id): hit.score for hit in dense_all.points}
        
        # Normalize and combine
        def normalize(scores):
            if not scores:
                return scores
            min_v, max_v = min(scores.values()), max(scores.values())
            if max_v == min_v:
                return {k: 1.0 for k in scores}
            return {k: (v - min_v) / (max_v - min_v) for k, v in scores.items()}
        
        bm25_norm = normalize(bm25_scores)
        dense_norm = normalize(dense_scores)
        
        all_ids = set(bm25_norm.keys()) | set(dense_norm.keys())
        hybrid_scores = {}
        for doc_id in all_ids:
            b = bm25_norm.get(doc_id, 0.0)
            d = dense_norm.get(doc_id, 0.0)
            hybrid_scores[doc_id] = alpha * b + (1 - alpha) * d
        
        # Sort and display top 3
        sorted_hybrid = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        for doc_id, score in sorted_hybrid:
            text = id_to_text.get(doc_id, "")[:60]
            b = bm25_norm.get(doc_id, 0.0)
            d = dense_norm.get(doc_id, 0.0)
            print(f"    [{doc_id}] hybrid={score:.3f} (bm25={b:.3f}, dense={d:.3f})")
            print(f"           {text}...")
    
    print("\n✅ Hybrid retrieval test PASSED")


def main():
    parser = argparse.ArgumentParser(description="Test hybrid retrieval")
    parser.add_argument("--qdrant-host", default="localhost", help="Qdrant host")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant port")
    parser.add_argument("--collection", default="oltp_chunks", help="Qdrant collection")
    parser.add_argument("--bm25-only", action="store_true", help="Only test BM25")
    parser.add_argument("--hnsw-ef", type=int, default=50, 
                        help="HNSW ef parameter (50=fast/OLTP, 200=recall/OLAP)")
    
    args = parser.parse_args()
    
    print(f"Using HNSW ef={args.hnsw_ef}")
    
    # Always test BM25
    test_bm25_only()
    
    # Test hybrid with Qdrant unless --bm25-only
    if not args.bm25_only:
        test_hybrid_with_qdrant(args.qdrant_host, args.qdrant_port, args.collection, args.hnsw_ef)
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
