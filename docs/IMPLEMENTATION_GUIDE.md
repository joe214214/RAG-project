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
| 5 | Reranker | ⏳ Pending | CrossEncoder implementation |
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
├── scripts/
│   ├── test_loaders.py         # Test data loaders
│   ├── test_chunking.py        # Test chunking
│   ├── test_hnsw.py            # Test Qdrant HNSW
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

| Phase | Task | Priority |
|-------|------|----------|
| 4 | Test MPI embedding on ecetesla0 | High |
| 4 | Ingest chunks into Qdrant | High |
| 5 | Implement CrossEncoder reranker | Medium |
| 5 | Train DistilBERT classifier | Medium |
| 6 | Integrate OpenAI/Azure LLM | Medium |
| 7 | Run corpus scaling experiments | High |
| 7 | Run MPI node scaling experiments | High |
| 8 | Write IEEE report and documentation | High |

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

