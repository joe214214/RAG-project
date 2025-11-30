#!/usr/bin/env python3
"""
Quick verification script to check Qdrant collections and embeddings.

Usage:
    python scripts/verify_qdrant.py --qdrant-host ecetesla0
"""

import argparse
from qdrant_client import QdrantClient
from qdrant_client.models import Distance


def verify_collections(host: str = "localhost", port: int = 6333):
    """Verify Qdrant collections and show statistics."""
    print("=" * 70)
    print("Qdrant Collection Verification")
    print("=" * 70)
    
    # Connect to Qdrant
    print(f"\n🔌 Connecting to Qdrant at {host}:{port}...")
    try:
        client = QdrantClient(host=host, port=port, check_compatibility=False)
        print("✅ Connected successfully!")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return
    
    # List all collections
    print("\n📚 Available Collections:")
    print("-" * 70)
    try:
        collections = client.get_collections().collections
        if not collections:
            print("   No collections found.")
            return
        
        total_points = 0
        for coll in collections:
            try:
                info = client.get_collection(coll.name)
                points_count = info.points_count
                total_points += points_count
                
                # Get collection config
                config = info.config
                vector_size = config.params.vectors.size if hasattr(config.params.vectors, 'size') else "N/A"
                distance = config.params.vectors.distance if hasattr(config.params.vectors, 'distance') else "N/A"
                
                # HNSW config
                hnsw_config = config.hnsw_config if hasattr(config, 'hnsw_config') else None
                hnsw_m = hnsw_config.m if hnsw_config else "N/A"
                hnsw_ef = hnsw_config.ef_construct if hnsw_config else "N/A"
                
                print(f"\n   Collection: {coll.name}")
                print(f"   ├─ Points: {points_count:,}")
                print(f"   ├─ Vector Size: {vector_size}")
                print(f"   ├─ Distance: {distance}")
                print(f"   └─ HNSW: M={hnsw_m}, ef_construct={hnsw_ef}")
                
            except Exception as e:
                print(f"   ❌ Error reading collection '{coll.name}': {e}")
        
        print("\n" + "-" * 70)
        print(f"   Total Points Across All Collections: {total_points:,}")
        
    except Exception as e:
        print(f"❌ Error listing collections: {e}")
        return
    
    # Verify specific collections
    print("\n🔍 Verifying Target Collections:")
    print("-" * 70)
    
    target_collections = ["oltp_chunks", "olap_chunks"]
    for coll_name in target_collections:
        try:
            if client.collection_exists(coll_name):
                info = client.get_collection(coll_name)
                print(f"\n   ✅ {coll_name}: {info.points_count:,} points")
                
                # Try to retrieve a sample point
                if info.points_count > 0:
                    try:
                        # Get first point
                        sample = client.retrieve(
                            collection_name=coll_name,
                            ids=[0],
                            with_payload=True,
                            with_vectors=False,
                        )
                        if sample:
                            point = sample[0]
                            payload = point.payload if hasattr(point, 'payload') else {}
                            print(f"      Sample payload keys: {list(payload.keys())}")
                            if 'text' in payload:
                                text_preview = payload['text'][:100] + "..." if len(payload['text']) > 100 else payload['text']
                                print(f"      Sample text: {text_preview}")
                    except Exception as e:
                        print(f"      ⚠ Could not retrieve sample: {e}")
            else:
                print(f"   ❌ {coll_name}: Collection does not exist")
        except Exception as e:
            print(f"   ❌ Error checking {coll_name}: {e}")
    
    # Test search (if collections exist)
    print("\n🧪 Testing Search Functionality:")
    print("-" * 70)
    
    for coll_name in target_collections:
        try:
            if client.collection_exists(coll_name):
                info = client.get_collection(coll_name)
                if info.points_count > 0:
                    # Get vector size from config
                    config = info.config
                    vector_size = config.params.vectors.size if hasattr(config.params.vectors, 'size') else 384
                    
                    # Create a dummy query vector
                    import numpy as np
                    query_vec = np.random.rand(vector_size).astype(np.float32).tolist()
                    
                    # Try a search
                    results = client.query_points(
                        collection_name=coll_name,
                        query=query_vec,
                        limit=1,
                    )
                    
                    if results.points:
                        print(f"   ✅ {coll_name}: Search works (returned {len(results.points)} result)")
                    else:
                        print(f"   ⚠ {coll_name}: Search returned no results")
                else:
                    print(f"   ⚠ {coll_name}: Empty collection, skipping search test")
        except Exception as e:
            print(f"   ❌ {coll_name}: Search test failed: {e}")
    
    print("\n" + "=" * 70)
    print("✅ Verification Complete!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Verify Qdrant collections and embeddings")
    parser.add_argument("--qdrant-host", type=str, default="localhost", help="Qdrant host")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant port")
    
    args = parser.parse_args()
    verify_collections(host=args.qdrant_host, port=args.qdrant_port)


if __name__ == "__main__":
    main()

