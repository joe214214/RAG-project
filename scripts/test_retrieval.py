#!/usr/bin/env python3
"""
Test Retrieval from Qdrant

Verifies that the ingested data can be searched correctly.

Usage:
    python scripts/test_retrieval.py --query "What is machine learning?"
    python scripts/test_retrieval.py --interactive
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def search_qdrant(
    query: str,
    collection: str,
    top_k: int = 5,
    host: str = "localhost",
    port: int = 6333,
):
    """Search Qdrant collection."""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU mode
    
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer
    
    # Load model on CPU (Tesla P4 not compatible with current PyTorch)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    query_vector = model.encode(query, normalize_embeddings=True)
    
    # Search
    client = QdrantClient(host=host, port=port, check_compatibility=False)
    
    results = client.query_points(
        collection_name=collection,
        query=query_vector.tolist(),
        limit=top_k,
    )
    
    return results.points


def print_results(results, collection: str):
    """Pretty print search results."""
    print(f"\n📊 Results from {collection}:")
    print("-" * 50)
    
    for i, hit in enumerate(results, 1):
        text = hit.payload.get("text", "")[:200]
        score = hit.score
        source = hit.payload.get("source_file", "unknown")
        
        print(f"\n{i}. Score: {score:.4f}")
        print(f"   Source: {source}")
        print(f"   Text: {text}...")


def run_interactive(host: str, port: int):
    """Interactive query mode."""
    from qdrant_client import QdrantClient
    
    client = QdrantClient(host=host, port=port, check_compatibility=False)
    collections = client.get_collections().collections
    
    print("\n📚 Available collections:")
    for c in collections:
        info = client.get_collection(c.name)
        print(f"   - {c.name}: {info.points_count} points")
    
    print("\nEnter queries (Ctrl+C to exit):")
    
    while True:
        try:
            query = input("\n🔍 Query: ").strip()
            if not query:
                continue
            
            # Search both collections
            for coll in ["oltp_chunks", "olap_chunks"]:
                try:
                    results = search_qdrant(query, coll, top_k=3, host=host, port=port)
                    print_results(results, coll)
                except Exception as e:
                    print(f"   {coll}: Not available ({e})")
                    
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break


def main():
    parser = argparse.ArgumentParser(description="Test Retrieval")
    parser.add_argument("--query", type=str, help="Search query")
    parser.add_argument("--collection", default="oltp_chunks")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6333)
    parser.add_argument("--interactive", action="store_true")
    
    args = parser.parse_args()
    
    if args.interactive:
        run_interactive(args.host, args.port)
        return
    
    if not args.query:
        parser.error("--query required (or use --interactive)")
    
    print(f"🔍 Query: {args.query}")
    results = search_qdrant(args.query, args.collection, args.top_k, args.host, args.port)
    print_results(results, args.collection)


if __name__ == "__main__":
    main()

