# MPI-Distributed Ingestion Guide

## Overview

`scripts/mpi_ingest.py` distributes dataset loading, chunking, and embedding across multiple MPI workers for scalable data ingestion.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              MPI-Distributed Ingestion                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Worker 0 (ecetesla1)  │  Worker 1 (ecetesla2)  │  ...     │
│  ┌──────────────────┐   │  ┌──────────────────┐  │         │
│  │ Load docs 0-N    │   │  │ Load docs N-2N    │  │         │
│  │ Chunk            │   │  │ Chunk             │  │         │
│  │ Embed (GPU 0)    │   │  │ Embed (GPU 1)     │  │         │
│  └────────┬─────────┘   │  └────────┬─────────┘  │         │
│           │              │           │             │         │
│           └──────────────┴───────────┘             │         │
│                      │                              │         │
│                      ▼                              │         │
│              ┌───────────────┐                      │         │
│              │ Qdrant Server │                      │         │
│              │ (ecetesla0)   │                      │         │
│              │ oltp_chunks   │                      │         │
│              │ olap_chunks   │                      │         │
│              └───────────────┘                      │         │
└─────────────────────────────────────────────────────────────┘
```

## Usage

### Single-Process Mode (Testing)

```bash
# Test on small sample
python scripts/mpi_ingest.py --dataset msmarco --max-docs 1000

# With GPU
python scripts/mpi_ingest.py --dataset msmarco --max-docs 10000 --use-gpu
```

### MPI Mode (Distributed)

```bash
# 2 workers on same node
mpirun -np 2 python scripts/mpi_ingest.py \
    --dataset msmarco \
    --max-docs 50000 \
    --use-gpu \
    --qdrant-host ecetesla0

# 4 workers across 3 nodes (ecetesla1, ecetesla2, ecetesla4)
mpirun -np 4 -host ecetesla1,ecetesla2,ecetesla4,ecetesla4 \
    python scripts/mpi_ingest.py \
    --dataset msmarco \
    --max-docs 100000 \
    --use-gpu \
    --qdrant-host ecetesla0
```

### Incremental Ingestion

```bash
# First batch
python scripts/mpi_ingest.py --dataset msmarco --max-docs 10000

# Append more data
python scripts/mpi_ingest.py --dataset msmarco --max-docs 20000 --append
```

## Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Dataset: `msmarco` or `hotpotqa` | Required |
| `--max-docs` | Total documents to ingest | 10000 |
| `--batch-size` | Embedding batch size | 32 |
| `--use-gpu` | Use GPU for embedding | False |
| `--qdrant-host` | Qdrant server hostname | localhost |
| `--qdrant-port` | Qdrant server port | 6333 |
| `--append` | Append to existing collections | False |

## How It Works

### 1. Dataset Distribution

Each MPI worker loads a portion of the dataset:
- Worker 0: docs 0 to N/size
- Worker 1: docs N/size to 2N/size
- Worker 2: docs 2N/size to 3N/size
- ...

### 2. Parallel Chunking

Each worker chunks its documents independently:
- OLTP chunks (fine-grained, ~450 tokens)
- OLAP parents (coarse-grained, ~1500 tokens)
- OLAP children (medium, ~500 tokens)

### 3. Distributed Embedding

Each worker embeds its chunks on its assigned GPU:
- Worker 0 → GPU 0
- Worker 1 → GPU 1
- Worker 2 → GPU 2
- Worker 3 → GPU 0 (cycles)

### 4. Direct Qdrant Writes

All workers write directly to Qdrant to avoid memory bottlenecks:
- Unique IDs calculated per worker (no collisions)
- Batch upserts (100 points per batch)
- Rank 0 creates collections, others wait

## Expected Performance

| Workers | Documents | Expected Time | Throughput |
|---------|-----------|---------------|------------|
| 1 (CPU) | 10K | ~5 min | ~33 docs/s |
| 1 (GPU) | 10K | ~1 min | ~167 docs/s |
| 4 (GPU) | 100K | ~10 min | ~167 docs/s per worker |

## Troubleshooting

### "Connection refused" to Qdrant
- Check Qdrant is running: `ps aux | grep qdrant`
- Verify hostname: Use `ecetesla0` not `localhost` from other nodes

### GPU out of memory
- Reduce `--batch-size` (try 16 or 8)
- Use fewer workers per node

### ID collisions
- Shouldn't happen (offsets calculated per worker)
- If it does, check MPI communication is working

## Next Steps

After ingestion:
1. Verify data: `python scripts/test_retrieval.py --qdrant-host ecetesla0`
2. Run scaling benchmarks: `python benchmarks/scaling_benchmark.py --qdrant-host ecetesla0`
3. Evaluate quality: `python scripts/evaluate_rerankers.py --qdrant-host ecetesla0`



