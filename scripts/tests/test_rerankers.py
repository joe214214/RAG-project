#!/usr/bin/env python3
"""
Test Rerankers with Qdrant Retrieval

Tests different reranking strategies on real data:
- OLTP: No reranker vs TinyBERT
- OLAP: MiniLM vs BGE vs Zerank-2

Usage:
    # CPU mode
    python scripts/test_rerankers.py --qdrant-host localhost
    
    # GPU mode (on ecetesla1/2/4 with RTX 3070)
    python scripts/test_rerankers.py --qdrant-host ecetesla0.uwaterloo.ca
    
    # Test specific OLAP model
    python scripts/test_rerankers.py --olap-model zerank-2
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_gpu():
    """Check GPU availability and return device."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✓ GPU: {gpu_name} ({vram:.1f} GB VRAM)")
            return "cuda"
        else:
            print("⚠ No CUDA GPU available, using CPU")
            return "cpu"
    except ImportError:
        print("⚠ PyTorch not found, using CPU")
        return "cpu"


def search_qdrant(query: str, collection: str, top_k: int, host: str, port: int):
    """Search Qdrant and return documents."""
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query_vector = model.encode(query, normalize_embeddings=True)
    
    client = QdrantClient(host=host, port=port, check_compatibility=False)
    results = client.query_points(
        collection_name=collection,
        query=query_vector.tolist(),
        limit=top_k,
    )
    
    documents = []
    for hit in results.points:
        documents.append({
            "text": hit.payload.get("text", ""),
            "score": hit.score,
            "metadata": hit.payload,
        })
    
    return documents


def test_oltp_rerankers(documents, query, device):
    """Test OLTP reranking strategies."""
    from rerankers import OLTPReranker
    
    print("\n" + "=" * 60)
    print("OLTP Reranker Comparison")
    print("=" * 60)
    
    # Option 1: No reranker
    print("\n1. No Reranker (vector search only):")
    start = time.time()
    no_rerank = OLTPReranker(use_reranker=False)
    results = no_rerank.rerank(query, documents, top_k=5)
    elapsed = time.time() - start
    print(f"   Time: {elapsed*1000:.1f}ms")
    for i, r in enumerate(results, 1):
        print(f"   {i}. [{r.score:.4f}] {r.text[:60]}...")
    
    # Option 2: TinyBERT
    print("\n2. TinyBERT Reranker (ms-marco-TinyBERT-L-2-v2):")
    start = time.time()
    tiny_rerank = OLTPReranker(use_reranker=True, device=device)
    results = tiny_rerank.rerank(query, documents, top_k=5)
    elapsed = time.time() - start
    print(f"   Time: {elapsed*1000:.1f}ms (includes model load)")
    for i, r in enumerate(results, 1):
        print(f"   {i}. [{r.score:.4f}] (was #{r.original_rank+1}) {r.text[:50]}...")


def test_olap_rerankers(documents, query, device, models_to_test):
    """Test OLAP reranking strategies."""
    from rerankers import OLAPReranker
    
    print("\n" + "=" * 60)
    print("OLAP Reranker Comparison")
    print("=" * 60)
    
    for model_name in models_to_test:
        print(f"\n• {model_name}:")
        try:
            start = time.time()
            reranker = OLAPReranker(model_name=model_name, device=device)
            results = reranker.rerank(query, documents, top_k=5)
            elapsed = time.time() - start
            print(f"   Time: {elapsed*1000:.1f}ms (includes model load)")
            for i, r in enumerate(results, 1):
                print(f"   {i}. [{r.score:.4f}] (was #{r.original_rank+1}) {r.text[:50]}...")
        except Exception as e:
            print(f"   ❌ Error: {e}")


def test_factory(documents, query, device, olap_model):
    """Test RerankerFactory routing."""
    from rerankers import RerankerFactory
    
    print("\n" + "=" * 60)
    print("RerankerFactory (Auto-routing)")
    print("=" * 60)
    
    factory = RerankerFactory(
        device=device,
        oltp_use_reranker=False,  # No reranker for OLTP
        olap_model=olap_model,
    )
    
    # OLTP query
    print(f"\n1. Simulated OLTP query (no reranker):")
    start = time.time()
    results = factory.rerank(query, documents, query_type="oltp", top_k=3)
    elapsed = time.time() - start
    print(f"   Time: {elapsed*1000:.1f}ms")
    for i, r in enumerate(results, 1):
        print(f"   {i}. [{r.score:.4f}] {r.text[:50]}...")
    
    # OLAP query
    print(f"\n2. Simulated OLAP query ({olap_model}):")
    start = time.time()
    results = factory.rerank(query, documents, query_type="olap", top_k=5)
    elapsed = time.time() - start
    print(f"   Time: {elapsed*1000:.1f}ms")
    for i, r in enumerate(results, 1):
        print(f"   {i}. [{r.score:.4f}] {r.text[:50]}...")


def main():
    parser = argparse.ArgumentParser(description="Test Rerankers")
    parser.add_argument("--qdrant-host", default="localhost")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument("--collection", default="oltp_chunks")
    parser.add_argument("--query", default="What causes disease?")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--olap-model", default="minilm",
                        choices=["minilm", "bge-base", "bge-large", "zerank-1", "zerank-2"],
                        help="OLAP reranker model to test")
    parser.add_argument("--test-all-olap", action="store_true",
                        help="Test all OLAP models (requires significant VRAM for larger models)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Reranker Comparison Test")
    print("=" * 60)
    
    # Check GPU
    device = check_gpu()
    
    print(f"\nQdrant: {args.qdrant_host}:{args.qdrant_port}")
    print(f"Collection: {args.collection}")
    print(f"Query: {args.query}")
    
    # Step 1: Retrieve from Qdrant
    print(f"\n📥 Retrieving top-{args.top_k} from Qdrant...")
    try:
        documents = search_qdrant(
            args.query, args.collection, args.top_k,
            args.qdrant_host, args.qdrant_port
        )
        print(f"   Retrieved {len(documents)} documents")
    except Exception as e:
        print(f"❌ Qdrant error: {e}")
        print("\nUsing sample documents for testing...")
        documents = [
            {"text": "Parkinson's disease is caused by loss of dopamine neurons in the brain.", "score": 0.8, "metadata": {}},
            {"text": "Heart disease is the leading cause of death worldwide, often caused by poor diet.", "score": 0.75, "metadata": {}},
            {"text": "Machine learning algorithms can predict disease outcomes from patient data.", "score": 0.7, "metadata": {}},
            {"text": "Infectious diseases are caused by pathogenic microorganisms like bacteria and viruses.", "score": 0.65, "metadata": {}},
            {"text": "The weather forecast predicts sunny skies tomorrow.", "score": 0.5, "metadata": {}},
            {"text": "Autoimmune diseases occur when the immune system attacks the body's own tissues.", "score": 0.6, "metadata": {}},
            {"text": "Cancer is a disease characterized by uncontrolled cell growth and division.", "score": 0.72, "metadata": {}},
        ]
    
    # Step 2: Test OLTP rerankers
    test_oltp_rerankers(documents, args.query, device)
    
    # Step 3: Test OLAP rerankers
    if args.test_all_olap:
        models = ["minilm", "bge-base"]  # Add larger models if you have VRAM
        if device == "cuda":
            print("\n⚠ Testing all OLAP models. Zerank-2 requires ~8GB+ VRAM.")
            models.extend(["zerank-2"])
    else:
        models = [args.olap_model]
    
    test_olap_rerankers(documents, args.query, device, models)
    
    # Step 4: Test Factory
    test_factory(documents, args.query, device, args.olap_model)
    
    print("\n" + "=" * 60)
    print("✅ Reranker tests complete!")
    print("=" * 60)
    print("\nAvailable OLAP models:")
    print("   minilm     - cross-encoder/ms-marco-MiniLM-L-6-v2 (22M params)")
    print("   bge-base   - BAAI/bge-reranker-base (109M params)")
    print("   bge-large  - BAAI/bge-reranker-large (335M params)")
    print("   zerank-1   - zeroentropy/zerank-1 (~1B params)")
    print("   zerank-2   - zeroentropy/zerank-2 (~4B params, SOTA)")


if __name__ == "__main__":
    main()
