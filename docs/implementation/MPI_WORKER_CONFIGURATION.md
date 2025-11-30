# MPI Worker Configuration Guide

## Available Nodes

Based on your cluster setup, you have access to:

| Node | GPU | Purpose |
|------|-----|---------|
| **ecetesla0** | Tesla P4 (CUDA 6.1) | Qdrant server (no workers) |
| **ecetesla1** | RTX 3070 (CUDA 8.6) | Worker node |
| **ecetesla2** | RTX 3070 (CUDA 8.6) | Worker node |
| **ecetesla4** | RTX 3070 (CUDA 8.6) | Worker node |

**Note:** ecetesla0 runs Qdrant, so we don't use it as a worker to avoid resource contention.

---

## How Workers Are Distributed

### MPI Command Format

```bash
mpirun -np <NUM_WORKERS> -host <NODE_LIST> python script.py
```

### Worker Distribution Examples

#### 1 Worker (Single Process)
```bash
python scripts/mpi_ingest.py --dataset msmarco --max-docs 10000
```
- **Worker 0:** Runs on current node (ecetesla1)
- **Work:** Processes all 10K documents
- **GPU:** Uses GPU 0 on ecetesla1

#### 2 Workers (Same Node)
```bash
mpirun -np 2 python scripts/mpi_ingest.py --dataset msmarco --max-docs 10000
```
- **Worker 0:** ecetesla1, GPU 0, docs 0-5K
- **Worker 1:** ecetesla1, GPU 1, docs 5K-10K
- **Work:** Each processes 5K documents

#### 2 Workers (Different Nodes)
```bash
mpirun -np 2 -host ecetesla1,ecetesla2 \
  python scripts/mpi_ingest.py --dataset msmarco --max-docs 10000
```
- **Worker 0:** ecetesla1, GPU 0, docs 0-5K
- **Worker 1:** ecetesla2, GPU 0, docs 5K-10K
- **Work:** Each processes 5K documents

#### 4 Workers (Across 3 Nodes)
```bash
mpirun -np 4 -host ecetesla1,ecetesla2,ecetesla4,ecetesla4 \
  python scripts/mpi_ingest.py --dataset msmarco --max-docs 10000
```
- **Worker 0:** ecetesla1, GPU 0, docs 0-2.5K
- **Worker 1:** ecetesla2, GPU 0, docs 2.5K-5K
- **Worker 2:** ecetesla4, GPU 0, docs 5K-7.5K
- **Worker 3:** ecetesla4, GPU 1, docs 7.5K-10K
- **Work:** Each processes 2.5K documents

---

## How Work Is Divided

### Document Distribution

The script divides work using **rank-based slicing**:

```python
# In load_dataset_portion()
docs_per_worker = max_docs // size
start_idx = rank * docs_per_worker
end_idx = max_docs if rank == size - 1 else start_idx + docs_per_worker
my_docs = all_docs[start_idx:end_idx]
```

**Example with 10K docs, 4 workers:**
- Worker 0 (rank=0): docs[0:2500] → 2,500 docs
- Worker 1 (rank=1): docs[2500:5000] → 2,500 docs
- Worker 2 (rank=2): docs[5000:7500] → 2,500 docs
- Worker 3 (rank=3): docs[7500:10000] → 2,500 docs (last worker gets remainder)

### GPU Assignment

Each worker gets a GPU based on its rank:

```python
# In generate_embeddings_mpi()
gpu_id = rank % 4  # Cycles through GPUs 0-3
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
```

**Example with 4 workers:**
- Worker 0 (rank=0): GPU 0
- Worker 1 (rank=1): GPU 1
- Worker 2 (rank=2): GPU 2
- Worker 3 (rank=3): GPU 3 (or cycles back to 0 if only 1 GPU per node)

---

## Recommended Configurations

### For Scaling Tests (10K docs)

| Workers | Command | Node Distribution | Expected Speedup |
|---------|---------|-------------------|------------------|
| **1** | `python scripts/mpi_ingest.py ...` | ecetesla1 only | Baseline (1x) |
| **2** | `mpirun -np 2 -host ecetesla1,ecetesla2 ...` | 1 worker per node | ~1.8x |
| **4** | `mpirun -np 4 -host ecetesla1,ecetesla2,ecetesla4,ecetesla4 ...` | 2 nodes with 1 worker, 1 node with 2 workers | ~3.2x |

### For Full Ingestion (100K docs)

```bash
# 4 workers across 3 nodes (best balance)
mpirun -np 4 -host ecetesla1,ecetesla2,ecetesla4,ecetesla4 \
  python scripts/mpi_ingest.py \
  --dataset msmarco \
  --max-docs 100000 \
  --use-gpu \
  --qdrant-host ecetesla0 \
  --append \
  --output-stats results/ingest_msmarco_100k_4w.json
```

**Why this configuration?**
- ✅ Uses all available GPU nodes
- ✅ Distributes load evenly (25K docs per worker)
- ✅ Avoids overloading ecetesla0 (Qdrant server)
- ✅ Good balance of parallelism vs. overhead

---

## Alternative Configurations

### More Workers on Same Node

If you want more parallelism on a single node:

```bash
# 4 workers, all on ecetesla1 (if it has 4 GPUs)
mpirun -np 4 -host ecetesla1,ecetesla1,ecetesla1,ecetesla1 \
  python scripts/mpi_ingest.py ...
```

**Pros:** Lower network overhead  
**Cons:** GPU memory contention, node overload

### Fewer Workers

If you want to test with fewer resources:

```bash
# 2 workers, same node
mpirun -np 2 python scripts/mpi_ingest.py ...
```

**Pros:** Simpler setup, less resource usage  
**Cons:** Slower ingestion

---

## How to Check Worker Distribution

### Verify MPI is Working

```bash
# Test MPI communication
mpirun -np 4 -host ecetesla1,ecetesla2,ecetesla4,ecetesla4 \
  python -c "from mpi4py import MPI; comm = MPI.COMM_WORLD; print(f'Rank {comm.Get_rank()}/{comm.Get_size()} on {MPI.Get_processor_name()}')"
```

**Expected output:**
```
Rank 0/4 on ecetesla1
Rank 1/4 on ecetesla2
Rank 2/4 on ecetesla4
Rank 3/4 on ecetesla4
```

### Check GPU Assignment

Each worker prints its GPU assignment:
```
Worker 0: Loading embedding model on cuda...
Worker 1: Loading embedding model on cuda...
Worker 2: Loading embedding model on cuda...
Worker 3: Loading embedding model on cuda...
```

### Monitor Resource Usage

```bash
# On each worker node, check GPU usage
ssh ecetesla1 "nvidia-smi"
ssh ecetesla2 "nvidia-smi"
ssh ecetesla4 "nvidia-smi"
```

---

## Workflow Summary

### Step-by-Step Process

1. **MPI Launches Workers**
   - `mpirun` starts N processes across specified nodes
   - Each process gets a unique rank (0, 1, 2, ...)

2. **Each Worker Loads Its Portion**
   - Worker calculates: `start_idx = rank * (total_docs / num_workers)`
   - Loads only its slice of the dataset

3. **Each Worker Chunks Independently**
   - Processes its documents into OLTP/OLAP chunks
   - No communication needed (embarrassingly parallel)

4. **Each Worker Embeds on Its GPU**
   - GPU assigned: `rank % num_gpus`
   - Generates embeddings in parallel

5. **All Workers Write to Qdrant**
   - Each worker calculates unique IDs: `id = start_id_offset + local_idx`
   - Writes directly to Qdrant (no coordination needed)

6. **Rank 0 Aggregates Results**
   - Collects timing stats from all workers
   - Prints summary and saves JSON stats

---

## Troubleshooting

### Issue: Workers Not Starting

**Check:**
```bash
# Verify MPI is installed
python -c "import mpi4py; print('OK')"

# Check SSH access to nodes
ssh ecetesla1 "hostname"
ssh ecetesla2 "hostname"
ssh ecetesla4 "hostname"
```

### Issue: GPU Not Found

**Check:**
```bash
# Verify GPU on each node
ssh ecetesla1 "nvidia-smi"
ssh ecetesla2 "nvidia-smi"
ssh ecetesla4 "nvidia-smi"
```

### Issue: Load Imbalance

**Symptom:** Some workers finish much faster than others

**Cause:** Uneven document sizes or GPU performance differences

**Solution:** This is normal - the slowest worker determines total time (which is what we measure)

---

## Summary

**Key Points:**
- Workers are distributed by MPI based on `-host` argument
- Work is divided by rank (round-robin document slicing)
- Each worker gets its own GPU (rank % num_gpus)
- All workers write directly to Qdrant (no bottleneck)
- Rank 0 aggregates and reports results

**Recommended Setup:**
- **Scaling tests:** 1, 2, 4 workers
- **Full ingestion:** 4 workers across ecetesla1, ecetesla2, ecetesla4
- **Qdrant server:** ecetesla0 (separate from workers)

