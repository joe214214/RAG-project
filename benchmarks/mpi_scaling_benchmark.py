#!/usr/bin/env python3
"""
MPI Scaling Benchmark for Distributed Embedding Generation

Usage:
    # Single node
    python benchmarks/mpi_scaling_benchmark.py --num-docs 10000
    
    # MPI with multiple workers
    mpirun -np 4 python benchmarks/mpi_scaling_benchmark.py --num-docs 100000 --use-gpu
"""

import argparse
import json
import time
import os
from pathlib import Path
from datetime import datetime
from typing import List
import numpy as np

try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False

from sentence_transformers import SentenceTransformer


def generate_documents(n: int) -> List[str]:
    templates = ["This discusses {t} in {f}.", "{t} is key in {f}.", "Understanding {t} advances {f}.", "{t} transforms {f}."]
    topics = ["ML", "DL", "NLP", "CV", "RL", "transformers", "RAG", "embeddings"]
    fields = ["AI", "data science", "robotics", "healthcare", "finance"]
    return [templates[i % len(templates)].format(t=topics[i % len(topics)], f=fields[i % len(fields)]) + f" Doc {i}." for i in range(n)]


def run_single(num_docs: int, batch_size: int = 32, use_gpu: bool = False):
    print("\n" + "="*60)
    print("Single-Process Embedding Benchmark")
    print("="*60)
    
    device = "cuda" if use_gpu else "cpu"
    print(f"Device: {device}, Docs: {num_docs}")
    
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    documents = generate_documents(num_docs)
    
    start = time.time()
    embeddings = model.encode(documents, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    total_time = time.time() - start
    
    print(f"\nDocs: {num_docs}, Time: {total_time:.2f}s, Throughput: {num_docs/total_time:.1f} docs/s")
    return {"mode": "single", "num_docs": num_docs, "time_s": total_time, "throughput": num_docs/total_time, "device": device}


def run_mpi(num_docs: int, batch_size: int = 32, use_gpu: bool = False):
    if not HAS_MPI:
        return run_single(num_docs, batch_size, use_gpu)
    
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    
    if rank == 0:
        print("\n" + "="*60)
        print(f"MPI Embedding Benchmark ({size} workers)")
        print("="*60)
    
    # Distribute documents
    docs_per_worker = num_docs // size
    start_idx = rank * docs_per_worker
    end_idx = num_docs if rank == size - 1 else start_idx + docs_per_worker
    my_docs = generate_documents(end_idx - start_idx)
    
    device = "cuda" if use_gpu else "cpu"
    if use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % 4)
    
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    
    comm.Barrier()
    start = MPI.Wtime()
    
    my_embeddings = model.encode(my_docs, batch_size=batch_size, show_progress_bar=(rank == 0), convert_to_numpy=True)
    
    comm.Barrier()
    total_time = MPI.Wtime() - start
    
    all_times = comm.gather(total_time, root=0)
    all_counts = comm.gather(len(my_docs), root=0)
    
    if rank == 0:
        max_time = max(all_times)
        total_docs = sum(all_counts)
        throughput = total_docs / max_time
        
        print(f"\nWorkers: {size}, Docs: {total_docs}, Time: {max_time:.2f}s, Throughput: {throughput:.1f} docs/s")
        for i, (t, c) in enumerate(zip(all_times, all_counts)):
            print(f"  Worker {i}: {c} docs, {t:.2f}s, {c/t:.1f} docs/s")
        
        result = {"mode": "mpi", "workers": size, "num_docs": total_docs, "time_s": max_time, "throughput": throughput, "device": device}
        Path("results/benchmarks").mkdir(parents=True, exist_ok=True)
        with open(f"results/benchmarks/mpi_{size}w_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(result, f, indent=2)
        return result
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-docs", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--use-gpu", action="store_true")
    args = parser.parse_args()
    
    if HAS_MPI and MPI.COMM_WORLD.Get_size() > 1:
        run_mpi(args.num_docs, args.batch_size, args.use_gpu)
    else:
        run_single(args.num_docs, args.batch_size, args.use_gpu)


if __name__ == "__main__":
    main()
