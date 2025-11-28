#!/usr/bin/env python3
"""
End-to-End Ingestion Pipeline

Connects all components: Dataset → Chunk → Embed → Store in Qdrant

Usage:
    # Full pipeline with MS MARCO
    python scripts/ingest_pipeline.py --dataset msmarco --max-docs 1000

    # Full pipeline with HotpotQA  
    python scripts/ingest_pipeline.py --dataset hotpotqa --max-docs 500

    # Test mode (small sample, no Qdrant)
    python scripts/ingest_pipeline.py --test
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_dataset(dataset_name: str, max_docs: int = 1000) -> List[Dict]:
    """Load passages/contexts from specified dataset."""
    
    if dataset_name == "msmarco":
        from data.loaders.msmarco_loader import MSMARCOLoader
        loader = MSMARCOLoader()
        passages = loader.load_passages(max_passages=max_docs)
        return [{"id": p.id, "text": p.text, "title": p.title or ""} for p in passages]
    
    elif dataset_name == "hotpotqa":
        from data.loaders.hotpotqa_loader import HotpotQALoader
        loader = HotpotQALoader()
        contexts = loader.load_contexts(max_contexts=max_docs)
        return [{"id": c.id, "text": c.text, "title": c.title} for c in contexts]
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def chunk_documents(docs: List[Dict]) -> Dict:
    """Chunk documents using multi-granular chunker."""
    from preprocess.chunking import create_chunker
    
    chunker = create_chunker()
    result = chunker.chunk_passages(docs, id_field="id", text_field="text")
    return result


def generate_embeddings(
    chunks: List,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    use_gpu: bool = False,
) -> np.ndarray:
    """Generate embeddings for chunks."""
    from sentence_transformers import SentenceTransformer
    
    device = "cuda:0" if use_gpu else "cpu"
    print(f"Loading embedding model on {device}...")
    model = SentenceTransformer(model_name, device=device)
    
    texts = [c.text for c in chunks]
    
    print(f"Generating embeddings for {len(texts)} chunks...")
    start_time = time.time()
    
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    
    elapsed = time.time() - start_time
    print(f"Generated {len(embeddings)} embeddings in {elapsed:.2f}s "
          f"({len(embeddings)/elapsed:.1f} chunks/sec)")
    
    return embeddings


def store_in_qdrant(
    chunks: List,
    embeddings: np.ndarray,
    collection_name: str,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    hnsw_m: int = 16,
    hnsw_ef_construct: int = 128,
):
    """Store chunks and embeddings in Qdrant."""
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import (
        Distance, VectorParams, HnswConfigDiff, PointStruct
    )
    
    print(f"Connecting to Qdrant at {qdrant_host}:{qdrant_port}...")
    client = QdrantClient(host=qdrant_host, port=qdrant_port, check_compatibility=False)
    
    vector_size = embeddings.shape[1]
    
    # Create or recreate collection
    if client.collection_exists(collection_name):
        print(f"Deleting existing collection: {collection_name}")
        client.delete_collection(collection_name)
    
    print(f"Creating collection: {collection_name} (HNSW M={hnsw_m}, ef={hnsw_ef_construct})")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        hnsw_config=HnswConfigDiff(m=hnsw_m, ef_construct=hnsw_ef_construct),
    )
    
    # Prepare and insert points
    points = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        payload = {
            "text": chunk.text,
            "chunk_type": chunk.chunk_type,
            "source_file": chunk.source_file,
            "section": chunk.section,
            "level": chunk.level,
        }
        if chunk.parent_id:
            payload["parent_id"] = chunk.parent_id
        
        points.append(PointStruct(id=i, vector=emb.tolist(), payload=payload))
    
    # Batch upsert
    print(f"Inserting {len(points)} vectors...")
    batch_size = 100
    for i in range(0, len(points), batch_size):
        client.upsert(collection_name=collection_name, points=points[i:i+batch_size])
    
    print(f"✓ Stored {len(points)} vectors in {collection_name}")
    return {"collection": collection_name, "count": len(points)}


def run_test_mode():
    """Quick test without Qdrant."""
    print("=" * 60)
    print("TEST MODE (no Qdrant required)")
    print("=" * 60)
    
    sample_docs = [
        {"id": "test_1", "text": "Machine learning is a subset of artificial intelligence.", "title": "ML"},
        {"id": "test_2", "text": "Deep learning uses neural networks with many layers.", "title": "DL"},
    ]
    
    print("\n1. Testing Chunking...")
    result = chunk_documents(sample_docs)
    print(f"   OLTP: {len(result['oltp'])}, OLAP parents: {len(result['olap_parents'])}")
    
    print("\n2. Testing Embedding...")
    if result['oltp']:
        embeddings = generate_embeddings(result['oltp'][:2], use_gpu=False)
        print(f"   Shape: {embeddings.shape}")
    
    print("\n✓ Test completed! Use --dataset msmarco for full pipeline.")


def main():
    parser = argparse.ArgumentParser(description="Ingestion Pipeline")
    parser.add_argument("--dataset", choices=["msmarco", "hotpotqa"])
    parser.add_argument("--max-docs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--qdrant-host", default="localhost")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument("--skip-qdrant", action="store_true")
    parser.add_argument("--test", action="store_true")
    
    args = parser.parse_args()
    
    if args.test:
        run_test_mode()
        return
    
    if not args.dataset:
        parser.error("--dataset required (or use --test)")
    
    print("=" * 60)
    print(f"Ingestion Pipeline - {args.dataset.upper()}")
    print("=" * 60)
    
    # Step 1: Load
    print(f"\n📥 Loading {args.dataset}...")
    docs = load_dataset(args.dataset, args.max_docs)
    print(f"   Loaded {len(docs)} documents")
    
    # Step 2: Chunk
    print(f"\n✂️  Chunking...")
    chunks = chunk_documents(docs)
    print(f"   OLTP: {len(chunks['oltp'])}, OLAP: {len(chunks['olap_parents'])}")
    
    # Step 3: Embed
    print(f"\n🧮 Embedding...")
    oltp_emb = generate_embeddings(chunks['oltp'], batch_size=args.batch_size, use_gpu=args.use_gpu)
    olap_all = chunks['olap_parents'] + chunks['olap_children']
    olap_emb = generate_embeddings(olap_all, batch_size=args.batch_size, use_gpu=args.use_gpu) if olap_all else None
    
    # Step 4: Store
    if not args.skip_qdrant:
        print(f"\n💾 Storing in Qdrant...")
        store_in_qdrant(chunks['oltp'], oltp_emb, "oltp_chunks", 
                       args.qdrant_host, args.qdrant_port, hnsw_m=16, hnsw_ef_construct=128)
        if olap_emb is not None:
            store_in_qdrant(olap_all, olap_emb, "olap_chunks",
                          args.qdrant_host, args.qdrant_port, hnsw_m=32, hnsw_ef_construct=200)
    
    print("\n✅ Pipeline complete!")


if __name__ == "__main__":
    main()
