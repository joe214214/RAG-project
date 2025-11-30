# Full Ingestion Workflow with Performance Measurement

## Overview

This document explains the complete workflow for ingesting MS MARCO and HotpotQA datasets into Qdrant using MPI-distributed processing, with comprehensive performance measurement.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MPI Workers (ecetesla*)                   │
│                                                              │
│  Worker 0 (ecetesla1)    Worker 1 (ecetesla2)              │
│  ├─ Load docs 0-25K      ├─ Load docs 25K-50K             │
│  ├─ Chunk                 ├─ Chunk                         │
│  ├─ Embed (GPU)           ├─ Embed (GPU)                   │
│  └─ Write to Qdrant       └─ Write to Qdrant                │
│                                                              │
│  Worker 2 (ecetesla4)    Worker 3 (ecetesla4)              │
│  ├─ Load docs 50K-75K    ├─ Load docs 75K-100K            │
│  ├─ Chunk                 ├─ Chunk                          │
│  ├─ Embed (GPU)           ├─ Embed (GPU)                    │
│  └─ Write to Qdrant       └─ Write to Qdrant                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  Qdrant DB    │
                    │  (ecetesla0)  │
                    │               │
                    │  oltp_chunks  │
                    │  olap_chunks  │
                    └───────────────┘
```

---

## Step-by-Step Workflow

### Phase 1: Preparation

#### 1.1 Check Cluster Status

```bash
# SSH to cluster
ssh oankit@eceterm1.uwaterloo.ca
ssh ecetesla0  # Qdrant server
ssh ecetesla1  # Worker node
ssh ecetesla2  # Worker node
ssh ecetesla4  # Worker node

# Verify Qdrant is running
curl http://ecetesla0:6333/collections

# Check disk space (on ecetesla0 where Qdrant stores data)
df -h ~/qdrant-data
```

#### 1.2 Create Results Directory

```bash
# On your local machine or cluster
mkdir -p results/
```

---

### Phase 2: Baseline Measurement (1 Worker)

**Purpose:** Establish baseline performance for comparison.

#### 2.1 Run Single-Worker Ingestion

```bash
# SSH to ecetesla1
ssh ecetesla1
cd ~/RAG-project
source ~/rag-env/bin/activate

# Run baseline (no MPI, single process)
python scripts/mpi_ingest.py \
  --dataset msmarco \
  --max-docs 10000 \
  --use-gpu \
  --qdrant-host ecetesla0 \
  --output-stats results/ingest_1w.json
```

**What Happens:**
1. **Load:** Downloads/loads 10K MS MARCO passages
2. **Chunk:** Creates ~50K OLTP chunks (5 chunks/doc average)
3. **Embed:** Generates embeddings using GPU
4. **Store:** Writes to Qdrant `oltp_chunks` collection
5. **Measure:** Records timing for each phase
6. **Save:** Writes stats to `results/ingest_1w.json`

**Expected Output:**
```
MPI-Distributed Ingestion - MSMARCO
============================================================
Workers: 1
Total documents: 10000
Docs per worker: ~10000
GPU: Yes

📥 Loading msmarco (distributed across 1 workers)...
   Loaded: 10000 total documents
     Worker 0: 10000 docs

✂️  Chunking (parallel across workers)...
   Total chunks:
     OLTP: 50023
     OLAP parents: 10000
     OLAP children: 25000

🧮 Embedding (distributed across 1 GPUs)...
Worker 0: 50023 embeddings in 245.32s (203.9 chunks/s)

💾 Storing in Qdrant (all workers writing directly)...
✅ Complete!
   Stored 50023 OLTP chunks
   Stored 35000 OLAP chunks

📊 Performance Metrics:
   Total time: 312.45s
   Load time: 15.23s (4.9%)
   Chunk time: 12.34s (3.9%)
   Embed time: 245.32s (78.5%)
   Store time: 39.56s (12.7%)

   Throughput: 272.3 chunks/sec
   Docs/sec: 32.0 docs/sec
   Chunks/doc: 8.5

   Stats saved to: results/ingest_1w.json
```

---

### Phase 3: Distributed Ingestion (2 Workers)

**Purpose:** Measure 2x scaling efficiency.

#### 3.1 Run 2-Worker Ingestion

```bash
# On ecetesla1 (or any node with MPI)
mpirun -np 2 \
  python scripts/mpi_ingest.py \
  --dataset msmarco \
  --max-docs 10000 \
  --use-gpu \
  --qdrant-host ecetesla0 \
  --append \
  --output-stats results/ingest_2w.json
```

**What Happens:**
- **Worker 0:** Processes docs 0-5K
- **Worker 1:** Processes docs 5K-10K
- Both workers embed in parallel on their GPUs
- Both write directly to Qdrant (no coordination needed for appends)
- Stats aggregated across workers

**Expected Output:**
```
MPI-Distributed Ingestion - MSMARCO
============================================================
Workers: 2
Total documents: 10000
Docs per worker: ~5000
GPU: Yes

📥 Loading msmarco (distributed across 2 workers)...
   Loaded: 10000 total documents
     Worker 0: 5000 docs
     Worker 1: 5000 docs

✂️  Chunking (parallel across workers)...
   Total chunks:
     OLTP: 50023
     OLAP parents: 10000
     OLAP children: 25000

🧮 Embedding (distributed across 2 GPUs)...
Worker 0: 25011 embeddings in 128.45s (194.7 chunks/s)
Worker 1: 25012 embeddings in 129.12s (193.7 chunks/s)

💾 Storing in Qdrant (all workers writing directly)...
✅ Complete!
   Stored 50023 OLTP chunks
   Stored 35000 OLAP chunks

📊 Performance Metrics:
   Total time: 178.23s
   Load time: 8.12s (4.6%)
   Chunk time: 6.78s (3.8%)
   Embed time: 129.12s (72.5%)  ← Max across workers
   Store time: 34.21s (19.2%)

   Throughput: 477.2 chunks/sec  ← ~1.75x speedup
   Docs/sec: 56.1 docs/sec
   Chunks/doc: 8.5

   Stats saved to: results/ingest_2w.json
```

**Key Metrics:**
- **Speedup:** 312.45s / 178.23s = **1.75x**
- **Efficiency:** 1.75 / 2 = **87.5%** (ideal would be 2x)

---

### Phase 4: Distributed Ingestion (4 Workers)

**Purpose:** Measure 4x scaling efficiency across multiple nodes.

#### 4.1 Run 4-Worker Ingestion

```bash
# Across 3 GPU nodes (ecetesla1, ecetesla2, ecetesla4)
mpirun -np 4 -host ecetesla1,ecetesla2,ecetesla4,ecetesla4 \
  python scripts/mpi_ingest.py \
  --dataset msmarco \
  --max-docs 10000 \
  --use-gpu \
  --qdrant-host ecetesla0 \
  --append \
  --output-stats results/ingest_4w.json
```

**What Happens:**
- **Worker 0 (ecetesla1):** Docs 0-2.5K
- **Worker 1 (ecetesla2):** Docs 2.5K-5K
- **Worker 2 (ecetesla4):** Docs 5K-7.5K
- **Worker 3 (ecetesla4):** Docs 7.5K-10K
- All workers embed in parallel
- All write directly to Qdrant

**Expected Output:**
```
MPI-Distributed Ingestion - MSMARCO
============================================================
Workers: 4
Total documents: 10000
Docs per worker: ~2500
GPU: Yes

📥 Loading msmarco (distributed across 4 workers)...
   Loaded: 10000 total documents
     Worker 0: 2500 docs
     Worker 1: 2500 docs
     Worker 2: 2500 docs
     Worker 3: 2500 docs

✂️  Chunking (parallel across workers)...
   Total chunks:
     OLTP: 50023
     OLAP parents: 10000
     OLAP children: 25000

🧮 Embedding (distributed across 4 GPUs)...
Worker 0: 12506 embeddings in 68.23s (183.2 chunks/s)
Worker 1: 12505 embeddings in 67.89s (184.1 chunks/s)
Worker 2: 12506 embeddings in 69.12s (180.9 chunks/s)
Worker 3: 12506 embeddings in 68.45s (182.7 chunks/s)

💾 Storing in Qdrant (all workers writing directly)...
✅ Complete!
   Stored 50023 OLTP chunks
   Stored 35000 OLAP chunks

📊 Performance Metrics:
   Total time: 95.67s
   Load time: 4.23s (4.4%)
   Chunk time: 3.45s (3.6%)
   Embed time: 69.12s (72.2%)  ← Max across workers
   Store time: 18.87s (19.7%)

   Throughput: 888.9 chunks/sec  ← ~3.26x speedup
   Docs/sec: 104.5 docs/sec
   Chunks/doc: 8.5

   Stats saved to: results/ingest_4w.json
```

**Key Metrics:**
- **Speedup:** 312.45s / 95.67s = **3.26x**
- **Efficiency:** 3.26 / 4 = **81.5%**

---

### Phase 5: Full Dataset Ingestion

**Purpose:** Ingest the full datasets for evaluation.

#### 5.1 Ingest MS MARCO (100K documents)

```bash
# 4 workers, 100K documents
mpirun -np 4 -host ecetesla1,ecetesla2,ecetesla4,ecetesla4 \
  python scripts/mpi_ingest.py \
  --dataset msmarco \
  --max-docs 100000 \
  --use-gpu \
  --qdrant-host ecetesla0 \
  --append \
  --output-stats results/ingest_msmarco_100k_4w.json
```

**Expected:**
- ~500K OLTP chunks
- ~100K OLAP parent chunks
- ~250K OLAP child chunks
- Time: ~15-20 minutes (depending on network/GPU)

#### 5.2 Ingest HotpotQA (Full contexts)

```bash
# 4 workers, 50K contexts (full HotpotQA)
mpirun -np 4 -host ecetesla1,ecetesla2,ecetesla4,ecetesla4 \
  python scripts/mpi_ingest.py \
  --dataset hotpotqa \
  --max-docs 50000 \
  --use-gpu \
  --qdrant-host ecetesla0 \
  --append \
  --output-stats results/ingest_hotpotqa_50k_4w.json
```

**Expected:**
- ~50K-100K OLAP chunks
- Time: ~5-10 minutes

---

### Phase 6: Analyze Scaling Performance

#### 6.1 Create Analysis Script

```bash
# On local machine or cluster
python benchmarks/ingestion_scaling_benchmark.py \
  --results-dir results/ \
  --plot results/scaling_plot.png \
  --output results/scaling_analysis.json
```

**Output:**
```
======================================================================
Ingestion Scaling Analysis
======================================================================

Baseline (1 worker):
  Time: 312.45s
  Throughput: 272.3 chunks/sec

Scaling Results:
Workers    Time (s)     Speedup      Efficiency   Throughput      
----------------------------------------------------------------------
2          178.23      1.75x        87.5%        477.2            
4          95.67       3.26x        81.5%        888.9            

Average Scaling Efficiency: 84.5%
(100% = perfect linear scaling)

Plot saved to: results/scaling_plot.png
```

#### 6.2 Interpret Results

**Good Scaling Indicators:**
- ✅ Efficiency > 80%: Good parallelization
- ✅ Embedding time dominates: Expected (GPU-bound)
- ✅ Throughput increases with workers: System scales

**Bottleneck Analysis:**
- **Embedding:** Should be ~70-80% of total time (GPU-bound)
- **Storage:** Should be <20% (network/disk I/O)
- **Load/Chunk:** Should be <10% (CPU-bound, fast)

**Efficiency Loss Sources:**
- Network overhead (Qdrant writes)
- Load imbalance (some workers finish faster)
- GPU memory contention (if multiple workers on same node)

---

## Complete Example: Full Workflow

```bash
# ============================================================
# Step 1: Baseline (1 worker, 10K docs)
# ============================================================
python scripts/mpi_ingest.py \
  --dataset msmarco \
  --max-docs 10000 \
  --use-gpu \
  --qdrant-host ecetesla0 \
  --output-stats results/ingest_1w.json

# ============================================================
# Step 2: 2 workers (10K docs, append)
# ============================================================
mpirun -np 2 python scripts/mpi_ingest.py \
  --dataset msmarco \
  --max-docs 10000 \
  --use-gpu \
  --qdrant-host ecetesla0 \
  --append \
  --output-stats results/ingest_2w.json

# ============================================================
# Step 3: 4 workers (10K docs, append)
# ============================================================
mpirun -np 4 -host ecetesla1,ecetesla2,ecetesla4,ecetesla4 \
  python scripts/mpi_ingest.py \
  --dataset msmarco \
  --max-docs 10000 \
  --use-gpu \
  --qdrant-host ecetesla0 \
  --append \
  --output-stats results/ingest_4w.json

# ============================================================
# Step 4: Analyze scaling
# ============================================================
python benchmarks/ingestion_scaling_benchmark.py \
  --results-dir results/ \
  --plot results/scaling_plot.png \
  --output results/scaling_analysis.json

# ============================================================
# Step 5: Full MS MARCO ingestion (100K docs)
# ============================================================
mpirun -np 4 -host ecetesla1,ecetesla2,ecetesla4,ecetesla4 \
  python scripts/mpi_ingest.py \
  --dataset msmarco \
  --max-docs 100000 \
  --use-gpu \
  --qdrant-host ecetesla0 \
  --append \
  --output-stats results/ingest_msmarco_100k.json

# ============================================================
# Step 6: Full HotpotQA ingestion (50K contexts)
# ============================================================
mpirun -np 4 -host ecetesla1,ecetesla2,ecetesla4,ecetesla4 \
  python scripts/mpi_ingest.py \
  --dataset hotpotqa \
  --max-docs 50000 \
  --use-gpu \
  --qdrant-host ecetesla0 \
  --append \
  --output-stats results/ingest_hotpotqa_50k.json

# ============================================================
# Step 7: Verify collections
# ============================================================
curl http://ecetesla0:6333/collections/oltp_chunks
curl http://ecetesla0:6333/collections/olap_chunks
```

---

## Performance Metrics Explained

### Timing Breakdown

| Phase | What It Measures | Expected % |
|-------|------------------|------------|
| **Load** | Dataset download/loading time | 3-5% |
| **Chunk** | Document chunking (CPU) | 3-5% |
| **Embed** | Embedding generation (GPU) | 70-80% |
| **Store** | Qdrant write operations | 10-20% |

### Throughput Metrics

- **chunks/sec:** Overall ingestion rate (higher is better)
- **docs/sec:** Document processing rate
- **chunks/doc:** Average chunks per document (dataset-dependent)

### Scaling Metrics

- **Speedup:** `baseline_time / parallel_time`
- **Efficiency:** `speedup / num_workers * 100%`
  - 100% = perfect linear scaling
  - 80%+ = good scaling
  - <60% = poor scaling (bottleneck)

---

## Troubleshooting

### Issue: Low Efficiency (<60%)

**Possible Causes:**
- Network bottleneck (Qdrant writes)
- Load imbalance (uneven work distribution)
- GPU memory contention

**Solutions:**
- Check network latency: `ping ecetesla0`
- Monitor GPU usage: `nvidia-smi -l 1`
- Increase batch size: `--batch-size 64`

### Issue: Storage Time Too High (>30%)

**Possible Causes:**
- Qdrant server overloaded
- Network congestion
- Disk I/O bottleneck

**Solutions:**
- Check Qdrant logs: `tail -f ~/qdrant-data/logs/qdrant.log`
- Monitor disk I/O: `iostat -x 1`
- Reduce concurrent writes (fewer workers)

### Issue: Embedding Time Not Dominating

**Possible Causes:**
- GPU not being used
- Batch size too small
- Model loading overhead

**Solutions:**
- Verify GPU: `nvidia-smi`
- Increase batch size: `--batch-size 128`
- Pre-warm model (first run is slower)

---

## Next Steps After Ingestion

1. **Verify Collections:**
   ```bash
   curl http://ecetesla0:6333/collections/oltp_chunks
   curl http://ecetesla0:6333/collections/olap_chunks
   ```

2. **Test Retrieval:**
   ```bash
   python scripts/rag_pipeline.py \
     --query "What is AI?" \
     --qdrant-host ecetesla0 \
     --classifier-type feature
   ```

3. **Run Evaluation:**
   ```bash
   python scripts/evaluate_rerankers.py \
     --qdrant-host ecetesla0 \
     --device cuda
   ```

4. **Run Ablation Studies:**
   ```bash
   python benchmarks/reranker_ablation.py \
     --qdrant-host ecetesla0 \
     --device cuda
   ```

---

## Summary

**Workflow:**
1. Baseline (1 worker) → Establish performance baseline
2. Scaling tests (2, 4 workers) → Measure parallel efficiency
3. Full ingestion (100K MS MARCO, 50K HotpotQA) → Populate collections
4. Analysis → Generate scaling plots and reports
5. Evaluation → Run retrieval/reranking benchmarks

**Key Metrics:**
- **Time breakdown:** Identify bottlenecks
- **Throughput:** Measure ingestion rate
- **Scaling efficiency:** Validate distributed performance
- **Collection size:** Verify data ingestion

This workflow demonstrates **distributed system scalability** for your course project! 🚀

