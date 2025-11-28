# Query-Aware RAG System - Implementation Guide

This document explains what has been implemented and how to reproduce the setup on the UWaterloo ECE cluster.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Completed Components](#completed-components)
4. [Environment Setup](#environment-setup)
5. [Component Details](#component-details)
6. [How to Reproduce](#how-to-reproduce)
7. [File Structure](#file-structure)
8. [Next Steps](#next-steps)

---

## Project Overview

This project implements a **Query-Aware RAG (Retrieval-Augmented Generation) System** that routes queries to specialized pipelines:

- **OLTP Pipeline**: For factoid/lookup queries (e.g., "What is the capital of France?")
- **OLAP Pipeline**: For analytical/multi-hop queries (e.g., "Compare the economic policies of X and Y")

### Key Features

- Multi-granular chunking (fine-grained for OLTP, coarse-grained for OLAP)
- Qdrant vector database with HNSW indexing
- Hierarchical parent-child chunk linking
- MPI-distributed embedding generation (for scalability)
- Dataset loaders for MS MARCO and HotpotQA

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Query Input                              │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Query Classifier                              │
│                 (DistilBERT - TODO)                             │
│              Classifies: OLTP vs OLAP                           │
└─────────────────────────┬───────────────────────────────────────┘
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
┌───────────────────────┐   ┌───────────────────────┐
│    OLTP Pipeline      │   │    OLAP Pipeline      │
├───────────────────────┤   ├───────────────────────┤
│ • Fine-grained chunks │   │ • Coarse chunks       │
│ • HNSW M=16, ef=50    │   │ • HNSW M=32, ef=200   │
│ • Fast retrieval      │   │ • Hierarchical search │
│ • Optional reranker   │   │ • Cross-encoder       │
└───────────────────────┘   └───────────────────────┘
            │                           │
            └─────────────┬─────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Qdrant Vector Store                          │
│                  (Running on ecetesla0)                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Completed Components

| Phase | Component | Status | Description |
|-------|-----------|--------|-------------|
| 1 | Qdrant Setup | ✅ Done | Vector DB with HNSW on ecetesla0 |
| 2 | Data Loaders | ✅ Done | MS MARCO, HotpotQA, Classifier data |
| 3 | Chunking | ✅ Done | Multi-granular OLTP/OLAP chunking |
| 4 | MPI Embedding | ✅ Created | Script ready for testing |
| 4 | Qdrant Client | ✅ Done | Python wrapper with HNSW config |
| 4 | Ingestion Pipeline | ✅ Done | End-to-end: chunk → embed → store |
| 5 | Rerankers | ✅ Done | TinyBERT (OLTP), BGE-large (OLAP) |
| 5 | Quality Evaluation | ✅ Done | MRR, Recall@k, NDCG@k metrics |
| 5 | Classifier | ⏳ Pending | DistilBERT OLTP/OLAP router |
| 6 | LLM Integration | ⏳ Pending | OpenAI/Azure API |
| 7 | Scaling Tests | ⏳ Pending | Corpus and MPI experiments |

---

## Environment Setup

### Prerequisites

- SSH access to UWaterloo ECE cluster
- Account on eceterm1 and ecetesla0

### Step 1: SSH to Cluster

```bash
# From local machine
ssh oankit@eceterm1.uwaterloo.ca

# From eceterm1, SSH to GPU node
ssh oankit@ecetesla0.uwaterloo.ca
```

### Step 2: Create Python Virtual Environment

```bash
# On ecetesla0 (uses tcsh shell)
cd ~
python3 -m venv rag-env
source ~/rag-env/bin/activate.csh

# Verify
which python
# Should show: /home/oankit/rag-env/bin/python
```

### Step 3: Install Python Dependencies

```bash
pip install --upgrade pip
pip install numpy qdrant-client sentence-transformers datasets
```

### Step 4: Install and Start Qdrant

```bash
# Download Qdrant binary
mkdir -p ~/qdrant
cd ~/qdrant
wget https://github.com/qdrant/qdrant/releases/download/v1.12.4/qdrant-x86_64-unknown-linux-gnu.tar.gz
tar -xzf qdrant-x86_64-unknown-linux-gnu.tar.gz

# Create data directory
mkdir -p ~/qdrant-data

# Start Qdrant (runs on port 6333)
./qdrant &

# Verify it's running
curl http://localhost:6333/collections
# Should return: {"result":{"collections":[]},"status":"ok",...}
```

### Step 5: Create Project Directory Structure

```bash
mkdir -p ~/RAG-project/data/loaders
mkdir -p ~/RAG-project/data/datasets
mkdir -p ~/RAG-project/preprocess
mkdir -p ~/RAG-project/scripts
mkdir -p ~/RAG-project/vector_stores
mkdir -p ~/RAG-project/configs
```

---

## Component Details

### 1. Data Loaders

**Location:** `data/loaders/`

| File | Purpose |
|------|---------|
| `msmarco_loader.py` | Loads MS MARCO passages and queries (OLTP) |
| `hotpotqa_loader.py` | Loads HotpotQA multi-hop queries (OLAP) |
| `classifier_data.py` | Creates balanced OLTP/OLAP training data |

**Usage:**
```python
from data.loaders import MSMARCOLoader, HotpotQALoader, ClassifierDataLoader

# Load MS MARCO (OLTP)
loader = MSMARCOLoader()
passages = loader.load_passages(max_passages=1000)
queries = loader.load_queries(max_queries=500)

# Load HotpotQA (OLAP)
loader = HotpotQALoader()
queries = loader.load_queries(max_queries=500)
contexts = loader.load_contexts(max_contexts=1000)

# Create classifier training data
loader = ClassifierDataLoader()
examples = loader.create_training_data(n_oltp=500, n_olap=500)
train, val = loader.split_data(examples)
```

### 2. Multi-Granular Chunking

**Location:** `preprocess/chunking.py`

**Strategy (from `docs/Approach/chunking2.md`):**

| Pipeline | Chunk Type | Target Size | Method |
|----------|------------|-------------|--------|
| OLTP | Fine-grained | 384-512 tokens | Sentence groups |
| OLAP | Coarse-grained | 1500 tokens (parent) | Section-level |
| OLAP | Child chunks | 500 tokens | Paragraph-level |

**Features:**
- Semantic sentence splitting (not fixed-size)
- Heading-aware chunking (markdown headers)
- Parent-child hierarchical linking
- Metadata preservation (heading_path, section, source)
- MD5-based deduplication

**Usage:**
```python
from preprocess.chunking import create_chunker

chunker = create_chunker(
    oltp_tokens=450,
    olap_parent_tokens=1500,
    olap_child_tokens=500,
)

result = chunker.chunk_document(text, source_file="doc.md")
# Returns:
# {
#     "oltp": [Chunk, ...],           # Fine-grained chunks
#     "olap_parents": [Chunk, ...],   # Section-level chunks
#     "olap_children": [Chunk, ...],  # Paragraph-level chunks
# }
```

**Chunk Metadata:**
```python
@dataclass
class Chunk:
    id: str
    text: str
    chunk_type: str          # "oltp" or "olap"
    source_file: str
    heading_path: List[str]  # ["Chapter 1", "Section 1.2"]
    section: str
    chunk_index: int
    parent_id: Optional[str] # For OLAP children
    child_ids: List[str]     # For OLAP parents
    level: str               # "parent", "child", or "leaf"
    text_hash: str           # For deduplication
```

### 3. Qdrant Vector Store

**Location:** `vector_stores/qdrant_store.py`

**HNSW Configuration:**
```yaml
# OLTP Collection (fast, precise)
oltp_collection:
  name: "oltp_chunks"
  hnsw_m: 16
  hnsw_ef_construct: 128
  search_ef: 50

# OLAP Collection (thorough search)
olap_collection:
  name: "olap_chunks"
  hnsw_m: 32
  hnsw_ef_construct: 200
  search_ef: 200
```

**Usage:**
```python
from vector_stores.qdrant_store import QdrantVectorStore, QdrantConfig, QdrantCollectionConfig

config = QdrantConfig(host="localhost", port=6333)
collection_config = QdrantCollectionConfig(
    name="oltp_chunks",
    vector_size=384,
    hnsw_m=16,
    hnsw_ef_construct=128,
    search_ef=50,
)

store = QdrantVectorStore(config, collection_config)
store.upsert(ids, vectors, payloads)
results = store.search(query_vector, top_k=10)
```

### 4. MPI Embedding Script

**Location:** `scripts/mpi_embed.py`

**Purpose:** Distribute embedding generation across multiple GPU nodes for scalability.

**Usage:**
```bash
# Single process test
python scripts/mpi_embed.py --single-process --sample-chunks

# MPI distributed (4 processes)
mpirun -np 4 python scripts/mpi_embed.py --input chunks.json --output embeddings.npy
```

### 5. Rerankers

**Location:** `rerankers/cross_encoder_reranker.py`

**Available Models:**

| Pipeline | Model | HuggingFace ID | Parameters | Latency (cached) |
|----------|-------|----------------|------------|------------------|
| OLTP | None | - | - | 0ms |
| OLTP | TinyBERT | cross-encoder/ms-marco-TinyBERT-L-2-v2 | 4.4M | 500ms |
| OLAP | MiniLM | cross-encoder/ms-marco-MiniLM-L-6-v2 | 22M | 550ms |
| OLAP | BGE-base | BAAI/bge-reranker-base | 109M | 1.1s |
| OLAP | **BGE-large** | BAAI/bge-reranker-large | 335M | **1.4s** |

**Recommended Configuration:**
- **OLTP:** No reranker (vector search is fast and sufficient)
- **OLAP:** BGE-large (best quality/latency tradeoff)

**Usage:**
```python
from rerankers import OLTPReranker, OLAPReranker, RerankerFactory

# OLTP: No reranker (default) or TinyBERT
oltp_reranker = OLTPReranker(use_reranker=False)
# or
oltp_reranker = OLTPReranker(use_reranker=True)  # TinyBERT

# OLAP: BGE-large (recommended)
olap_reranker = OLAPReranker(model="bge-large")

# Auto-routing based on query type
factory = RerankerFactory(olap_model="bge-large")
results = factory.rerank(query, documents, query_type="olap", top_k=10)
```

**Running on GPU (ecetesla1):**
```bash
# ecetesla0 has Tesla P4 (CUDA 6.1, incompatible with PyTorch)
# ecetesla1 has RTX 3070 (CUDA 8.6, compatible)

ssh ecetesla1
source ~/rag-env/bin/activate.csh
cd ~/RAG-project
python scripts/test_rerankers.py --qdrant-host ecetesla0 --olap-model bge-large
```

### 6. Quality Evaluation

**Location:** `scripts/evaluate_rerankers.py`

**Metrics:**

| Metric | Formula | Description |
|--------|---------|-------------|
| **MRR** | 1/rank of first relevant | How quickly you find a relevant doc |
| **Recall@k** | relevant_in_top_k / total_relevant | Coverage of relevant docs |
| **NDCG@k** | DCG / ideal_DCG | Ranking quality with position weighting |
| **Precision@k** | relevant_in_top_k / k | Proportion of top-k that are relevant |

**Ground Truth Sources:**
- **MS MARCO:** qrels (query relevance judgments)
- **HotpotQA:** supporting_facts (passages that answer the question)

**Usage:**
```bash
# Evaluate baseline (no reranker)
python scripts/evaluate_rerankers.py --qdrant-host ecetesla0 --olap-model none

# Evaluate with BGE-large
python scripts/evaluate_rerankers.py --qdrant-host ecetesla0 --olap-model bge-large
```

---

## How to Reproduce

### Step 1: Copy Files to Cluster

From Windows PowerShell:
```powershell
# Create directories on eceterm1
ssh oankit@eceterm1.uwaterloo.ca "mkdir -p ~/RAG-project/data/loaders ~/RAG-project/preprocess ~/RAG-project/scripts ~/RAG-project/vector_stores ~/RAG-project/configs"

# Copy files to eceterm1
scp -r data/loaders oankit@eceterm1.uwaterloo.ca:~/RAG-project/data/
scp -r preprocess oankit@eceterm1.uwaterloo.ca:~/RAG-project/
scp -r scripts oankit@eceterm1.uwaterloo.ca:~/RAG-project/
scp -r vector_stores oankit@eceterm1.uwaterloo.ca:~/RAG-project/
scp -r configs oankit@eceterm1.uwaterloo.ca:~/RAG-project/
```

From ecetesla0:
```bash
# Copy from eceterm1 to ecetesla0
mkdir -p ~/RAG-project
scp -r eceterm1:~/RAG-project/* ~/RAG-project/
```

### Step 2: Run Tests

```bash
# Activate environment
source ~/rag-env/bin/activate.csh

# Test data loaders
cd ~/RAG-project
python scripts/test_loaders.py

# Test chunking
python scripts/test_chunking.py

# Test Qdrant HNSW
python scripts/test_hnsw.py
```

### Expected Output

**Data Loaders Test:**
```
Testing MS MARCO Loader (OLTP)
✓ Loaded 100 passages
✓ Loaded 50 queries

Testing HotpotQA Loader (OLAP)
✓ Loaded 50 queries
✓ Loaded 100 contexts

Testing Classifier Data Loader
✓ Created 50 examples

🎉 All tests passed!
```

**Chunking Test:**
```
Test 1: Basic Chunking
✓ OLTP chunks created: 1
✓ OLAP parent chunks: 5

Test 2: Chunking MS MARCO Passages
✓ OLTP chunks: 22
✓ OLAP parents: 50

🎉 All chunking tests passed!
```

**Qdrant HNSW Test:**
```
Created OLTP collection
Inserted 100 vectors
Search returned 5 results
  id=70, score=0.7916
  id=1, score=0.7809

Qdrant HNSW test PASSED!
```

---

## File Structure

```
RAG-project/
├── configs/
│   └── qdrant_config.yaml      # Qdrant connection settings
├── data/
│   ├── loaders/
│   │   ├── __init__.py
│   │   ├── msmarco_loader.py   # MS MARCO dataset loader
│   │   ├── hotpotqa_loader.py  # HotpotQA dataset loader
│   │   └── classifier_data.py  # Classifier training data
│   └── datasets/               # Cached downloaded data
├── preprocess/
│   └── chunking.py             # Multi-granular chunking
├── rerankers/
│   ├── __init__.py             # Exports OLTPReranker, OLAPReranker
│   └── cross_encoder_reranker.py  # CrossEncoder implementations
├── scripts/
│   ├── test_loaders.py         # Test data loaders
│   ├── test_chunking.py        # Test chunking
│   ├── test_hnsw.py            # Test Qdrant HNSW
│   ├── test_retrieval.py       # Test retrieval from Qdrant
│   ├── test_rerankers.py       # Test reranker models
│   ├── evaluate_rerankers.py   # Quality evaluation (MRR, Recall, NDCG)
│   ├── ingest_pipeline.py      # End-to-end ingestion pipeline
│   ├── mpi_embed.py            # MPI embedding generation
│   └── setup_qdrant.sh         # Qdrant setup script
├── vector_stores/
│   └── qdrant_store.py         # Qdrant client wrapper
└── docs/
    ├── IMPLEMENTATION_GUIDE.md # This document
    └── Approach/               # Design documents
```

---

## Next Steps

| Phase | Task | Priority | Status |
|-------|------|----------|--------|
| 5 | Train DistilBERT classifier | High | ⏳ Pending |
| 6 | Integrate OpenAI/Azure LLM | Medium | ⏳ Pending |
| 7 | Run corpus scaling experiments (10K→1M) | High | ⏳ Pending |
| 7 | Run MPI node scaling experiments (1→4 nodes) | High | ⏳ Pending |
| 7 | Run quality evaluation with labeled data | High | ⏳ Pending |
| 8 | Write IEEE report and documentation | High | ⏳ Pending |

### Completed
- ✅ Qdrant setup with HNSW on ecetesla0
- ✅ Data loaders (MS MARCO, HotpotQA)
- ✅ Multi-granular chunking (OLTP/OLAP)
- ✅ Ingestion pipeline (chunk → embed → store)
- ✅ Rerankers (TinyBERT for OLTP, BGE-large for OLAP)
- ✅ Quality evaluation script (MRR, Recall@k, NDCG@k)

---

## Troubleshooting

### tcsh Shell Issues

ecetesla0 uses `tcsh`, not `bash`. Common issues:

```bash
# Wrong (bash syntax)
source rag-env/bin/activate

# Correct (tcsh syntax)
source ~/rag-env/bin/activate.csh
```

### Qdrant Not Running

```bash
# Check if running
curl http://localhost:6333/collections

# If not, start it
cd ~/qdrant
./qdrant &
```

### Python Import Errors

```bash
# Ensure you're in the project directory
cd ~/RAG-project

# Or add to PYTHONPATH
setenv PYTHONPATH ~/RAG-project:$PYTHONPATH
```

---

## References

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [HNSW Algorithm](https://arxiv.org/abs/1603.09320)
- [MS MARCO Dataset](https://microsoft.github.io/msmarco/)
- [HotpotQA Dataset](https://hotpotqa.github.io/)
- Project design docs: `docs/Approach/`

