# Scripts Directory

This directory contains all executable scripts for the RAG system, organized by functionality.

## Directory Structure

### `core/` - Core Pipeline Components
Main RAG pipeline and supporting modules:
- **`rag_pipeline.py`** - Main query-aware RAG pipeline (OLTP/OLAP routing)
- **`llm_answer_generator.py`** - LLM answer generation with cost tracking
- **`feature_router.py`** - Feature-based query classifier

### `ingestion/` - Data Ingestion Scripts
Scripts for ingesting data into Qdrant:
- **`ingest_pipeline.py`** - Single-node ingestion pipeline
- **`mpi_ingest.py`** - MPI-distributed ingestion (multi-node)
- **`mpi_embed.py`** - MPI-distributed embedding generation

### `evaluation/` - Evaluation Scripts
Comprehensive evaluation and benchmarking:
- **`comprehensive_evaluation.py`** - Full system evaluation (all configs)
- **`evaluate_classifier.py`** - Classifier performance evaluation
- **`evaluate_cost.py`** - Cost analysis (baseline vs best config)
- **`evaluate_rerankers.py`** - Reranker ablation study

### `tests/` - Testing & Verification Scripts
Unit tests and verification tools:
- **`test_chunking.py`** - Test chunking strategies
- **`test_hnsw.py`** - Test HNSW index parameters
- **`test_hybrid_retrieval.py`** - Test hybrid retrieval
- **`test_ir_datasets.py`** - Test ir_datasets integration
- **`test_loaders.py`** - Test data loaders
- **`test_query_expansion.py`** - Test query expansion
- **`test_rerankers.py`** - Test reranker models
- **`test_retrieval.py`** - Test retrieval components
- **`check_ground_truth_coverage.py`** - Verify ground truth coverage in Qdrant
- **`validate_classifier.py`** - Validate classifier predictions
- **`verify_hotpotqa_fields.py`** - Verify HotpotQA data fields
- **`verify_id_alignment.py`** - Verify ID alignment between datasets and Qdrant
- **`verify_qdrant.py`** - Verify Qdrant connection and collections

### `utils/` - Utility Scripts
Helper scripts for tuning and optimization:
- **`tune_hybrid_alpha.py`** - Tune hybrid retrieval alpha parameter

### `setup/` - Setup Scripts
Infrastructure setup:
- **`setup_qdrant.sh`** - Qdrant setup script

---

## Usage Examples

### Core Pipeline
```bash
# Run a query
python scripts/core/rag_pipeline.py --query "What is machine learning?"
```

### Ingestion
```bash
# Single node
python scripts/ingestion/ingest_pipeline.py --dataset msmarco --max-docs 10000

# Distributed (MPI)
mpirun -np 2 python scripts/ingestion/mpi_ingest.py --dataset msmarco --max-docs 100000 --use-gpu
```

### Evaluation
```bash
# Comprehensive evaluation
python scripts/evaluation/comprehensive_evaluation.py \
    --qdrant-host ecetesla0 \
    --use-ground-truth \
    --output results/evaluation/my_eval.json

# Cost evaluation
python scripts/evaluation/evaluate_cost.py \
    --qdrant-host ecetesla0 \
    --output results/evaluation/cost_eval.json
```

### Testing
```bash
# Verify ground truth coverage
python scripts/tests/check_ground_truth_coverage.py \
    --qdrant-host ecetesla0 \
    --num-queries 50

# Test retrieval
python scripts/tests/test_retrieval.py
```

---

## Note on Import Paths

Scripts use relative imports from the project root. When running scripts from subdirectories, ensure you're in the project root or adjust paths accordingly:

```bash
# From project root (recommended)
python scripts/core/rag_pipeline.py ...

# Or add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python scripts/core/rag_pipeline.py ...
```

