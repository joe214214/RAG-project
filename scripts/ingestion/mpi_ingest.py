#!/usr/bin/env python3
"""
MPI-Distributed Ingestion Pipeline

Distributes dataset loading, chunking, and embedding across multiple MPI workers.
All workers write directly to Qdrant to avoid memory bottlenecks.

Usage:
    # Single process (no MPI)
    python scripts/mpi_ingest.py --dataset msmarco --max-docs 10000
    
    # MPI with 2 workers
    mpirun -np 2 python scripts/mpi_ingest.py --dataset msmarco --max-docs 50000 --use-gpu
    
    # MPI across multiple nodes
    mpirun -np 4 -host ecetesla1,ecetesla2,ecetesla4,ecetesla4 \
        python scripts/mpi_ingest.py --dataset msmarco --max-docs 100000 --use-gpu
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# MPI support
try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False
    print("Warning: mpi4py not installed. Running in single-process mode.")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_dataset_portion(
    dataset_name: str,
    max_docs: int,
    rank: int,
    size: int,
) -> List[Dict]:
    """Load a portion of the dataset for this MPI worker."""
    
    if dataset_name == "msmarco":
        from data.loaders.msmarco_loader import MSMARCOLoader
        loader = MSMARCOLoader()
        # Each worker loads its portion
        docs_per_worker = max_docs // size
        start_idx = rank * docs_per_worker
        end_idx = max_docs if rank == size - 1 else start_idx + docs_per_worker
        
        # Load full dataset (cached, so fast), but only keep our portion
        # This is necessary because the loader caches the full dataset
        passages = loader.load_passages(max_passages=max_docs)
        my_passages = passages[start_idx:end_idx]
        
        # Clear reference to full dataset to free memory
        del passages
        
        return [{"id": p.id, "text": p.text, "title": p.title or ""} for p in my_passages]
    
    elif dataset_name == "hotpotqa":
        from data.loaders.hotpotqa_loader import HotpotQALoader
        loader = HotpotQALoader()
        contexts = loader.load_contexts(max_contexts=max_docs)
        
        # Distribute across workers
        docs_per_worker = len(contexts) // size
        start_idx = rank * docs_per_worker
        end_idx = len(contexts) if rank == size - 1 else start_idx + docs_per_worker
        
        my_contexts = contexts[start_idx:end_idx]
        
        # Clear reference to full dataset to free memory
        del contexts
        
        # For HotpotQA, preserve title for exact matching (research standard)
        return [{"id": c.id, "text": c.text, "title": c.title} for c in my_contexts]
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def chunk_documents(docs: List[Dict]) -> Dict:
    """Chunk documents using multi-granular chunker."""
    from preprocess.chunking import create_chunker
    
    chunker = create_chunker()
    result = chunker.chunk_passages(docs, id_field="id", text_field="text")
    return result


def generate_embeddings_mpi(
    chunks: List,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    use_gpu: bool = False,
    rank: int = 0,
) -> np.ndarray:
    """Generate embeddings for chunks (MPI-aware GPU assignment)."""
    from sentence_transformers import SentenceTransformer
    
    # Assign GPU based on rank, but check availability
    if use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                # Use GPU 0 for all workers (each node has its own GPU 0)
                # This avoids conflicts when multiple workers are on same node
                device = "cuda:0"
            else:
                device = "cpu"
                if rank == 0:
                    print("⚠ GPU requested but CUDA not available, using CPU")
        except ImportError:
            device = "cpu"
            if rank == 0:
                print("⚠ PyTorch not available, using CPU")
    else:
        device = "cpu"
    
    if rank == 0:
        print(f"Loading embedding model on {device}...")
    
    model = SentenceTransformer(model_name, device=device)
    
    texts = [c.text for c in chunks]
    
    if rank == 0:
        print(f"Generating embeddings for {len(texts)} chunks (worker {rank})...")
    
    start_time = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=(rank == 0),
        normalize_embeddings=True,
    )
    elapsed = time.time() - start_time
    
    if rank == 0:
        print(f"Worker {rank}: {len(embeddings)} embeddings in {elapsed:.2f}s ({len(embeddings)/elapsed:.1f} chunks/s)")
    
    return embeddings


def get_collection_info(collection_name: str, qdrant_host: str, qdrant_port: int, append: bool) -> int:
    """Get existing collection count if appending."""
    if not append:
        return 0
    
    from qdrant_client import QdrantClient
    client = QdrantClient(host=qdrant_host, port=qdrant_port, check_compatibility=False)
    
    if client.collection_exists(collection_name):
        info = client.get_collection(collection_name)
        return info.points_count
    return 0


def store_in_qdrant_mpi(
    chunks: List,
    embeddings: np.ndarray,
    collection_name: str,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    hnsw_m: int = 16,
    hnsw_ef_construct: int = 128,
    rank: int = 0,
    start_id_offset: int = 0,
    append: bool = False,
    existing_count: int = 0,
):
    """Store chunks and embeddings in Qdrant (each worker writes directly)."""
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import (
        Distance, VectorParams, HnswConfigDiff, PointStruct
    )
    
    client = QdrantClient(host=qdrant_host, port=qdrant_port, check_compatibility=False)
    
    vector_size = embeddings.shape[1]
    
    # Adjust start_id_offset if appending
    if append:
        start_id_offset += existing_count
    
    # Only rank 0 creates/recreates collection
    if rank == 0:
        if client.collection_exists(collection_name):
            if append:
                print(f"Appending to existing collection: {collection_name} ({existing_count} existing points)")
            else:
                print(f"Deleting existing collection: {collection_name}")
                client.delete_collection(collection_name)
                print(f"Creating collection: {collection_name} (HNSW M={hnsw_m}, ef={hnsw_ef_construct})")
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                    hnsw_config=HnswConfigDiff(m=hnsw_m, ef_construct=hnsw_ef_construct),
                )
        else:
            print(f"Creating collection: {collection_name} (HNSW M={hnsw_m}, ef={hnsw_ef_construct})")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                hnsw_config=HnswConfigDiff(m=hnsw_m, ef_construct=hnsw_ef_construct),
            )
    
    # Wait for collection creation
    if HAS_MPI:
        MPI.COMM_WORLD.Barrier()
    
    # Prepare points with unique IDs
    points = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        point_id = start_id_offset + i
        payload = {
            "text": chunk.text,
            "chunk_type": chunk.chunk_type,
            "source_file": chunk.source_file,
            "section": chunk.section,
            "level": chunk.level,
            "worker_rank": rank,  # Track which worker created this
        }
        if chunk.parent_id:
            payload["parent_id"] = chunk.parent_id
        
        # Store original passage/doc IDs for exact matching (research standard)
        if chunk.original_passage_id:
            payload["original_passage_id"] = chunk.original_passage_id
        if chunk.original_doc_id:
            payload["original_doc_id"] = chunk.original_doc_id
        
        # Store HotpotQA-specific fields (title-based identifiers)
        if chunk.original_title:
            payload["original_title"] = chunk.original_title
        if chunk.original_sent_id is not None:
            payload["original_sent_id"] = chunk.original_sent_id
        
        points.append(PointStruct(id=point_id, vector=emb.tolist(), payload=payload))
    
    # Batch upsert
    batch_size = 100
    for i in range(0, len(points), batch_size):
        client.upsert(collection_name=collection_name, points=points[i:i+batch_size])
    
    if rank == 0:
        print(f"Worker {rank}: Stored {len(points)} vectors in {collection_name}")
    
    return len(points)


def run_single_process(
    dataset_name: str,
    max_docs: int,
    batch_size: int,
    use_gpu: bool,
    qdrant_host: str,
    qdrant_port: int,
    append: bool = False,
    output_stats: str = None,
):
    """Run ingestion in single-process mode (no MPI)."""
    print("=" * 60)
    print(f"Single-Process Ingestion - {dataset_name.upper()}")
    print("=" * 60)
    
    # Performance tracking
    timings = {
        "total_start": time.time(),
        "load_time": 0.0,
        "chunk_time": 0.0,
        "embed_time": 0.0,
        "store_time": 0.0,
    }
    
    # Load
    print(f"\n📥 Loading {dataset_name}...")
    load_start = time.time()
    docs = load_dataset_portion(dataset_name, max_docs, rank=0, size=1)
    timings["load_time"] = time.time() - load_start
    print(f"   Loaded {len(docs)} documents")
    
    # Chunk
    print(f"\n✂️  Chunking...")
    chunk_start = time.time()
    chunks = chunk_documents(docs)
    timings["chunk_time"] = time.time() - chunk_start
    print(f"   OLTP: {len(chunks['oltp'])}, OLAP parents: {len(chunks['olap_parents'])}, OLAP children: {len(chunks['olap_children'])}")
    
    # Embed
    print(f"\n🧮 Embedding...")
    embed_start = time.time()
    oltp_emb = generate_embeddings_mpi(chunks['oltp'], batch_size=batch_size, use_gpu=use_gpu, rank=0)
    olap_all = chunks['olap_parents'] + chunks['olap_children']
    olap_emb = generate_embeddings_mpi(olap_all, batch_size=batch_size, use_gpu=use_gpu, rank=0) if olap_all else None
    timings["embed_time"] = time.time() - embed_start
    
    # Store
    print(f"\n💾 Storing in Qdrant...")
    store_start = time.time()
    oltp_existing = get_collection_info("oltp_chunks", qdrant_host, qdrant_port, append)
    oltp_count = store_in_qdrant_mpi(
        chunks['oltp'], oltp_emb, "oltp_chunks",
        qdrant_host, qdrant_port, hnsw_m=16, hnsw_ef_construct=128,
        rank=0, start_id_offset=0, append=append, existing_count=oltp_existing
    )
    
    olap_count = 0
    if olap_emb is not None:
        olap_existing = get_collection_info("olap_chunks", qdrant_host, qdrant_port, append)
        olap_count = store_in_qdrant_mpi(
            olap_all, olap_emb, "olap_chunks",
            qdrant_host, qdrant_port, hnsw_m=32, hnsw_ef_construct=200,
            rank=0, start_id_offset=0, append=append, existing_count=olap_existing
        )
    
    timings["store_time"] = time.time() - store_start
    timings["total_time"] = time.time() - timings["total_start"]
    
    total_chunks = oltp_count + olap_count
    
    print(f"\n✅ Complete!")
    print(f"   Stored {oltp_count} OLTP chunks")
    if olap_count > 0:
        print(f"   Stored {olap_count} OLAP chunks")
    
    print(f"\n📊 Performance Metrics:")
    print(f"   Total time: {timings['total_time']:.2f}s")
    print(f"   Load time: {timings['load_time']:.2f}s ({timings['load_time']/timings['total_time']*100:.1f}%)")
    print(f"   Chunk time: {timings['chunk_time']:.2f}s ({timings['chunk_time']/timings['total_time']*100:.1f}%)")
    print(f"   Embed time: {timings['embed_time']:.2f}s ({timings['embed_time']/timings['total_time']*100:.1f}%)")
    print(f"   Store time: {timings['store_time']:.2f}s ({timings['store_time']/timings['total_time']*100:.1f}%)")
    print(f"\n   Throughput: {total_chunks/timings['total_time']:.1f} chunks/sec")
    print(f"   Docs/sec: {max_docs/timings['total_time']:.1f} docs/sec")
    print(f"   Chunks/doc: {total_chunks/max_docs:.1f}")
    
    # Save stats if requested
    if output_stats:
        # Create directory if it doesn't exist
        output_path = Path(output_stats)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        stats = {
            "dataset": dataset_name,
            "workers": 1,
            "max_docs": max_docs,
            "use_gpu": use_gpu,
            "batch_size": batch_size,
            "total_chunks": total_chunks,
            "oltp_chunks": oltp_count,
            "olap_chunks": olap_count,
            "timings": {
                "total": timings["total_time"],
                "load": timings["load_time"],
                "chunk": timings["chunk_time"],
                "embed": timings["embed_time"],
                "store": timings["store_time"],
            },
            "throughput": {
                "chunks_per_sec": total_chunks / timings["total_time"],
                "docs_per_sec": max_docs / timings["total_time"],
                "chunks_per_doc": total_chunks / max_docs,
            },
            "per_worker_timings": [timings],
        }
        with open(output_stats, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\n   Stats saved to: {output_stats}")


def run_mpi(
    dataset_name: str,
    max_docs: int,
    batch_size: int,
    use_gpu: bool,
    qdrant_host: str,
    qdrant_port: int,
    append: bool = False,
    output_stats: str = None,
):
    """Run ingestion with MPI distribution."""
    if not HAS_MPI:
        return run_single_process(dataset_name, max_docs, batch_size, use_gpu, qdrant_host, qdrant_port)
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Performance tracking
    timings = {
        "total_start": time.time(),
        "load_time": 0.0,
        "chunk_time": 0.0,
        "embed_time": 0.0,
        "store_time": 0.0,
    }
    
    if rank == 0:
        print("=" * 60)
        print(f"MPI-Distributed Ingestion - {dataset_name.upper()}")
        print("=" * 60)
        print(f"Workers: {size}")
        print(f"Total documents: {max_docs}")
        print(f"Docs per worker: ~{max_docs // size}")
        print(f"GPU: {'Yes' if use_gpu else 'No'}")
    
    comm.Barrier()
    
    # Step 1: Load dataset portion
    if rank == 0:
        print(f"\n📥 Loading {dataset_name} (distributed across {size} workers)...")
    
    load_start = time.time()
    docs = load_dataset_portion(dataset_name, max_docs, rank, size)
    timings["load_time"] = time.time() - load_start
    
    # Gather only counts (integers), not the docs themselves
    doc_count = len(docs)
    if rank == 0:
        all_counts = comm.gather(doc_count, root=0)
        print(f"   Loaded: {sum(all_counts)} total documents")
        for i, count in enumerate(all_counts):
            print(f"     Worker {i}: {count} docs")
    else:
        comm.gather(doc_count, root=0)
    
    comm.Barrier()
    
    # Step 2: Chunk
    if rank == 0:
        print(f"\n✂️  Chunking (parallel across workers)...")
    
    chunk_start = time.time()
    chunks = chunk_documents(docs)
    timings["chunk_time"] = time.time() - chunk_start
    
    if rank == 0:
        all_oltp = comm.gather(len(chunks['oltp']), root=0)
        all_olap_p = comm.gather(len(chunks['olap_parents']), root=0)
        all_olap_c = comm.gather(len(chunks['olap_children']), root=0)
        print(f"   Total chunks:")
        print(f"     OLTP: {sum(all_oltp)}")
        print(f"     OLAP parents: {sum(all_olap_p)}")
        print(f"     OLAP children: {sum(all_olap_c)}")
    else:
        comm.gather(len(chunks['oltp']), root=0)
        comm.gather(len(chunks['olap_parents']), root=0)
        comm.gather(len(chunks['olap_children']), root=0)
    
    comm.Barrier()
    
    # Step 3: Embed (parallel, each worker on its GPU)
    if rank == 0:
        print(f"\n🧮 Embedding (distributed across {size} GPUs)...")
    
    embed_start = time.time()
    oltp_emb = generate_embeddings_mpi(chunks['oltp'], batch_size=batch_size, use_gpu=use_gpu, rank=rank)
    olap_all = chunks['olap_parents'] + chunks['olap_children']
    olap_emb = generate_embeddings_mpi(olap_all, batch_size=batch_size, use_gpu=use_gpu, rank=rank) if olap_all else None
    timings["embed_time"] = time.time() - embed_start
    
    comm.Barrier()
    
    # Step 4: Store (all workers write directly to Qdrant)
    if rank == 0:
        print(f"\n💾 Storing in Qdrant (all workers writing directly)...")
    
    store_start = time.time()
    
    # Get existing counts if appending (only rank 0 queries)
    oltp_existing = get_collection_info("oltp_chunks", qdrant_host, qdrant_port, append) if rank == 0 else 0
    olap_existing = get_collection_info("olap_chunks", qdrant_host, qdrant_port, append) if rank == 0 else 0
    
    # Broadcast to all workers
    if HAS_MPI:
        oltp_existing = comm.bcast(oltp_existing, root=0)
        olap_existing = comm.bcast(olap_existing, root=0)
    
    # Calculate ID offsets to avoid collisions
    all_oltp_counts = comm.allgather(len(chunks['oltp']))
    oltp_offset = sum(all_oltp_counts[:rank])
    
    oltp_count = store_in_qdrant_mpi(
        chunks['oltp'], oltp_emb, "oltp_chunks",
        qdrant_host, qdrant_port, hnsw_m=16, hnsw_ef_construct=128,
        rank=rank, start_id_offset=oltp_offset, append=append, existing_count=oltp_existing
    )
    
    olap_count = 0
    if olap_emb is not None:
        all_olap_counts = comm.allgather(len(olap_all))
        olap_offset = sum(all_olap_counts[:rank])
        
        olap_count = store_in_qdrant_mpi(
            olap_all, olap_emb, "olap_chunks",
            qdrant_host, qdrant_port, hnsw_m=32, hnsw_ef_construct=200,
            rank=rank, start_id_offset=olap_offset, append=append, existing_count=olap_existing
        )
    
    timings["store_time"] = time.time() - store_start
    timings["total_time"] = time.time() - timings["total_start"]
    
    comm.Barrier()
    
    if rank == 0:
        # Gather all timings and counts
        all_timings = comm.gather(timings, root=0)
        total_oltp = comm.reduce(oltp_count, op=MPI.SUM, root=0)
        total_olap = comm.reduce(olap_count, op=MPI.SUM, root=0)
        
        # Calculate aggregate stats
        total_chunks = total_oltp + total_olap
        max_total_time = max(t["total_time"] for t in all_timings)
        max_load_time = max(t["load_time"] for t in all_timings)
        max_chunk_time = max(t["chunk_time"] for t in all_timings)
        max_embed_time = max(t["embed_time"] for t in all_timings)
        max_store_time = max(t["store_time"] for t in all_timings)
        
        print(f"\n✅ Complete!")
        print(f"   Stored {total_oltp} OLTP chunks")
        print(f"   Stored {total_olap} OLAP chunks")
        print(f"\n📊 Performance Metrics:")
        print(f"   Total time: {max_total_time:.2f}s")
        print(f"   Load time: {max_load_time:.2f}s ({max_load_time/max_total_time*100:.1f}%)")
        print(f"   Chunk time: {max_chunk_time:.2f}s ({max_chunk_time/max_total_time*100:.1f}%)")
        print(f"   Embed time: {max_embed_time:.2f}s ({max_embed_time/max_total_time*100:.1f}%)")
        print(f"   Store time: {max_store_time:.2f}s ({max_store_time/max_total_time*100:.1f}%)")
        print(f"\n   Throughput: {total_chunks/max_total_time:.1f} chunks/sec")
        print(f"   Docs/sec: {max_docs/max_total_time:.1f} docs/sec")
        print(f"   Chunks/doc: {total_chunks/max_docs:.1f}")
        
        # Save stats to JSON if requested
        if output_stats:
            # Create directory if it doesn't exist
            output_path = Path(output_stats)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            stats = {
                "dataset": dataset_name,
                "workers": size,
                "max_docs": max_docs,
                "use_gpu": use_gpu,
                "batch_size": batch_size,
                "total_chunks": total_chunks,
                "oltp_chunks": total_oltp,
                "olap_chunks": total_olap,
                "timings": {
                    "total": max_total_time,
                    "load": max_load_time,
                    "chunk": max_chunk_time,
                    "embed": max_embed_time,
                    "store": max_store_time,
                },
                "throughput": {
                    "chunks_per_sec": total_chunks / max_total_time,
                    "docs_per_sec": max_docs / max_total_time,
                    "chunks_per_doc": total_chunks / max_docs,
                },
                "per_worker_timings": all_timings,
            }
            with open(output_stats, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"\n   Stats saved to: {output_stats}")
    else:
        comm.gather(timings, root=0)
        comm.reduce(oltp_count, op=MPI.SUM, root=0)
        comm.reduce(olap_count, op=MPI.SUM, root=0)


def main():
    parser = argparse.ArgumentParser(description="MPI-Distributed Ingestion Pipeline")
    parser.add_argument("--dataset", choices=["msmarco", "hotpotqa"], required=True)
    parser.add_argument("--max-docs", type=int, default=10000, help="Total documents to ingest")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU for embedding")
    parser.add_argument("--qdrant-host", default="localhost", help="Qdrant server host")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant server port")
    parser.add_argument("--append", action="store_true", help="Append to existing collections instead of recreating")
    parser.add_argument("--output-stats", type=str, default=None, help="Save performance stats to JSON file")
    
    args = parser.parse_args()
    
    # Check if running with MPI
    if HAS_MPI:
        size = MPI.COMM_WORLD.Get_size()
        if size > 1:
            run_mpi(
                args.dataset, args.max_docs, args.batch_size,
                args.use_gpu, args.qdrant_host, args.qdrant_port, args.append, args.output_stats
            )
        else:
            run_single_process(
                args.dataset, args.max_docs, args.batch_size,
                args.use_gpu, args.qdrant_host, args.qdrant_port, args.append, args.output_stats
            )
    else:
        run_single_process(
            args.dataset, args.max_docs, args.batch_size,
            args.use_gpu, args.qdrant_host, args.qdrant_port, args.append, args.output_stats
        )


if __name__ == "__main__":
    main()

