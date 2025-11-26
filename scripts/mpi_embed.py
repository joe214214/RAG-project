#!/usr/bin/env python3
"""
MPI-Distributed Embedding Generation

Distributes embedding computation across multiple GPU nodes using MPI.
This is the distributed component that demonstrates scalability.

Usage:
    mpirun -np 4 python scripts/mpi_embed.py --input data/chunks.json --output data/embeddings

Requirements:
    - mpi4py
    - sentence-transformers
    - numpy
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

# MPI imports - gracefully handle if not available
try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False
    print("Warning: mpi4py not available. Running in single-process mode.")

from sentence_transformers import SentenceTransformer


def load_chunks(input_path: str) -> List[dict]:
    """Load chunks from JSON file."""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_embeddings(
    output_dir: str,
    ids: List[str],
    embeddings: np.ndarray,
    chunk_type: str = "all",
):
    """Save embeddings to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save embeddings as numpy array
    np.save(output_path / f"embeddings_{chunk_type}.npy", embeddings)
    
    # Save IDs
    with open(output_path / f"ids_{chunk_type}.json", "w") as f:
        json.dump(ids, f)
    
    print(f"Saved {len(ids)} embeddings to {output_path}")


def embed_texts(
    texts: List[str],
    model: SentenceTransformer,
    batch_size: int = 32,
    show_progress: bool = True,
) -> np.ndarray:
    """Embed texts using sentence transformer."""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
        normalize_embeddings=True,  # For cosine similarity
    )
    return embeddings


def distribute_chunks_mpi(chunks: List[dict], comm) -> List[dict]:
    """
    Distribute chunks across MPI ranks.
    
    Each rank gets approximately len(chunks) / size chunks.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Only rank 0 has all chunks initially
    if rank == 0:
        # Split chunks into roughly equal parts
        chunk_splits = np.array_split(chunks, size)
        print(f"Rank 0: Distributing {len(chunks)} chunks across {size} ranks")
    else:
        chunk_splits = None
    
    # Scatter chunks to all ranks
    local_chunks = comm.scatter(chunk_splits, root=0)
    
    print(f"Rank {rank}: Received {len(local_chunks)} chunks")
    return list(local_chunks)


def gather_embeddings_mpi(
    local_ids: List[str],
    local_embeddings: np.ndarray,
    comm,
) -> Tuple[List[str], np.ndarray]:
    """
    Gather embeddings from all MPI ranks to rank 0.
    """
    rank = comm.Get_rank()
    
    # Gather all IDs and embeddings to rank 0
    all_ids = comm.gather(local_ids, root=0)
    all_embeddings = comm.gather(local_embeddings, root=0)
    
    if rank == 0:
        # Flatten lists
        combined_ids = [id for sublist in all_ids for id in sublist]
        combined_embeddings = np.vstack(all_embeddings)
        print(f"Rank 0: Gathered {len(combined_ids)} embeddings")
        return combined_ids, combined_embeddings
    else:
        return [], np.array([])


def run_distributed(args):
    """Run embedding generation with MPI distribution."""
    
    if not HAS_MPI:
        print("MPI not available. Use --no-mpi flag for single-process mode.")
        return
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    print(f"Rank {rank}/{size}: Starting distributed embedding generation")
    
    # Load model on each rank
    device = f"cuda:{rank % args.gpus_per_node}" if args.use_gpu else "cpu"
    print(f"Rank {rank}: Loading model on {device}")
    
    model = SentenceTransformer(args.model_name, device=device)
    
    # Rank 0 loads and distributes chunks
    if rank == 0:
        print(f"Rank 0: Loading chunks from {args.input}")
        chunks = load_chunks(args.input)
        total_chunks = len(chunks)
    else:
        chunks = None
        total_chunks = None
    
    # Broadcast total count for timing estimates
    total_chunks = comm.bcast(total_chunks, root=0)
    
    # Distribute chunks across ranks
    local_chunks = distribute_chunks_mpi(chunks if rank == 0 else [], comm)
    
    # Extract texts and IDs
    local_ids = [c["id"] for c in local_chunks]
    local_texts = [c["text"] for c in local_chunks]
    
    # Generate embeddings
    start_time = time.time()
    print(f"Rank {rank}: Embedding {len(local_texts)} texts...")
    
    local_embeddings = embed_texts(
        local_texts,
        model,
        batch_size=args.batch_size,
        show_progress=(rank == 0),  # Only show progress on rank 0
    )
    
    local_time = time.time() - start_time
    print(f"Rank {rank}: Embedded {len(local_texts)} texts in {local_time:.2f}s "
          f"({len(local_texts)/local_time:.1f} texts/sec)")
    
    # Synchronize before gathering
    comm.Barrier()
    
    # Gather all embeddings to rank 0
    all_ids, all_embeddings = gather_embeddings_mpi(local_ids, local_embeddings, comm)
    
    # Rank 0 saves results
    if rank == 0:
        total_time = time.time() - start_time
        print(f"\n=== Summary ===")
        print(f"Total chunks: {total_chunks}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Throughput: {total_chunks/total_time:.1f} chunks/sec")
        print(f"Speedup vs 1 rank: ~{size}x (ideal)")
        
        save_embeddings(args.output, all_ids, all_embeddings)


def run_single_process(args):
    """Run embedding generation without MPI (for testing/debugging)."""
    
    print("Running in single-process mode")
    
    device = "cuda:0" if args.use_gpu else "cpu"
    print(f"Loading model on {device}")
    
    model = SentenceTransformer(args.model_name, device=device)
    
    print(f"Loading chunks from {args.input}")
    chunks = load_chunks(args.input)
    
    ids = [c["id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    
    start_time = time.time()
    print(f"Embedding {len(texts)} texts...")
    
    embeddings = embed_texts(texts, model, batch_size=args.batch_size)
    
    total_time = time.time() - start_time
    print(f"\n=== Summary ===")
    print(f"Total chunks: {len(chunks)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {len(chunks)/total_time:.1f} chunks/sec")
    
    save_embeddings(args.output, ids, embeddings)


def create_sample_chunks(output_path: str, n_chunks: int = 1000):
    """Create sample chunks for testing."""
    chunks = [
        {
            "id": f"chunk_{i}",
            "text": f"This is sample chunk number {i}. It contains some text for testing the embedding pipeline. "
                    f"The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.",
            "source": f"doc_{i // 100}",
        }
        for i in range(n_chunks)
    ]
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(chunks, f)
    
    print(f"Created {n_chunks} sample chunks at {output_path}")


def main():
    parser = argparse.ArgumentParser(description="MPI-Distributed Embedding Generation")
    
    parser.add_argument("--input", type=str, default="data/chunks/all_chunks.json",
                        help="Input JSON file with chunks")
    parser.add_argument("--output", type=str, default="data/embeddings/mpi",
                        help="Output directory for embeddings")
    parser.add_argument("--model-name", type=str, 
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Sentence transformer model name")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for embedding")
    parser.add_argument("--use-gpu", action="store_true",
                        help="Use GPU for embedding")
    parser.add_argument("--gpus-per-node", type=int, default=1,
                        help="Number of GPUs per node")
    parser.add_argument("--no-mpi", action="store_true",
                        help="Run without MPI (single process)")
    parser.add_argument("--create-sample", action="store_true",
                        help="Create sample chunks for testing")
    parser.add_argument("--sample-size", type=int, default=1000,
                        help="Number of sample chunks to create")
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_chunks(args.input, args.sample_size)
        return
    
    if args.no_mpi or not HAS_MPI:
        run_single_process(args)
    else:
        run_distributed(args)


if __name__ == "__main__":
    main()

