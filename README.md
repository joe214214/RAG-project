# Query-Aware RAG System with OLTP/OLAP Routing

**ECE 750 - Advanced Topics in Computer Systems**  
**University of Waterloo - Winter 2025**

---

## Overview

A query-aware Retrieval-Augmented Generation (RAG) system that intelligently routes queries to optimized retrieval pipelines:
- **OLTP (Factoid) queries**: Fast, precise retrieval with optional lightweight reranking
- **OLAP (Analytical) queries**: Comprehensive retrieval with cross-encoder reranking for multi-hop reasoning

## Key Features

- **Query Classification**: Transformer-based (MiniLM) and feature-based classifiers for OLTP/OLAP routing
- **Hybrid Retrieval**: Combines BM25 (sparse) and dense vector search with Reciprocal Rank Fusion (RRF)
- **Multi-granular Chunking**: Fine-grained chunks for OLTP, hierarchical chunks for OLAP
- **Distributed Ingestion**: MPI-based parallel embedding generation across GPU nodes
- **Reranker Ablation**: Support for TinyBERT, MiniLM, and BGE cross-encoders
- **LLM Integration**: Answer generation with OpenAI gpt-4o-mini and cost tracking

---

## Project Structure

```
RAG-project/
├── benchmarks/           # Performance benchmarks
│   ├── parallel_load_test.py      # Concurrent query load testing
│   ├── mpi_load_test.py           # Distributed load testing
│   ├── reranker_ablation.py       # Reranker comparison
│   ├── scaling_benchmark.py       # Corpus/HNSW scaling
│   └── ingestion_scaling_benchmark.py  # MPI ingestion analysis
│
├── classifier/           # Trained classifier models
│   ├── bert-base-uncased/         # BERT classifier
│   ├── distilbert-base-uncased/   # DistilBERT classifier
│   ├── microsoft_MiniLM-L12.../   # MiniLM classifier (best)
│   ├── ml/                        # Training code
│   └── CLASSIFIER_TRAINING.md     # Training documentation
│
├── data/                 # Data loading utilities
│   └── loaders/                   # Dataset loaders (MS MARCO, HotpotQA)
│
├── docs/                 # Documentation
│   ├── approach/                  # Design decisions & recommendations
│   │   ├── classifier_recommendations.md
│   │   ├── embedding_strategy_recommendations.md
│   │   ├── reranker_recommendations.md
│   │   ├── vector_db_recommendations.md
│   │   └── novelty_assessment_report.md
│   ├── implementation/            # How-to guides
│   │   ├── IMPLEMENTATION_GUIDE.md
│   │   ├── MPI_INGESTION_GUIDE.md
│   │   └── INGESTION_WORKFLOW.md
│   └── results/                   # Evaluation reports
│       ├── EVALUATION_REPORT.md
│       ├── COMPREHENSIVE_EVALUATION_REPORT.md
│       ├── 1M_CHUNK_INGESTION_REPORT.md
│       ├── COST_EVALUATION_REPORT.md
│       ├── PARALLEL_LOAD_TEST_REPORT.md
│       └── archive/               # Iteration reports
│
├── models/               # Trained models
│   └── feature_router.pkl         # Feature-based classifier
│
├── preprocess/           # Data preprocessing
│   └── chunking.py                # Multi-granular chunking
│
├── rerankers/            # Reranking modules
│   └── cross_encoder_reranker.py  # Cross-encoder implementation
│
├── results/              # Experiment results (JSON)
│   ├── evaluation/                # Final evaluation results
│   ├── scaling/                   # Scaling experiment data
│   ├── plots/                     # Generated visualizations
│   └── archive/                   # Iteration results
│
├── retrievers/           # Retrieval modules
│   ├── bm25_retriever.py          # BM25 sparse retrieval
│   └── hybrid_retriever.py        # Hybrid (BM25 + dense) retrieval
│
├── scripts/              # Main scripts
│   ├── rag_pipeline.py            # Main RAG pipeline
│   ├── comprehensive_evaluation.py # Full system evaluation
│   ├── mpi_ingest.py              # Distributed ingestion
│   ├── evaluate_cost.py           # Cost analysis
│   └── ...                        # Various utilities
│
├── vector_stores/        # Vector database
│   └── qdrant_store.py            # Qdrant wrapper
│
└── requirements.txt      # Python dependencies
```

---

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv rag-env
source rag-env/bin/activate  # Linux/Mac
# or: rag-env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Qdrant

```bash
# Using Docker
docker run -p 6333:6333 qdrant/qdrant:latest
```

### 3. Ingest Data

```bash
# Single node
python scripts/ingest_pipeline.py --dataset msmarco --max-docs 10000

# Distributed (MPI)
mpirun -np 2 -host node1,node2 python scripts/mpi_ingest.py \
    --dataset msmarco --max-docs 100000 --use-gpu
```

### 4. Run Queries

```bash
python scripts/rag_pipeline.py --query "What is machine learning?"
```

### 5. Evaluate

```bash
python scripts/comprehensive_evaluation.py \
    --qdrant-host localhost \
    --use-ground-truth \
    --output results/evaluation/my_eval.json
```

---

## Key Results

### System Performance

| Configuration | Latency (ms) | MRR | Recall@10 |
|--------------|-------------|-----|-----------|
| Feature + Dense | 46 | 0.33 | 0.55 |
| Transformer + Dense | 56 | 0.34 | 0.58 |
| Transformer + Hybrid + Rerank | 520 | 0.45 | 0.63 |

### Scalability

- **1M chunks ingested** in ~13 minutes using 2 GPU nodes
- **Throughput**: ~1,000 chunks/second
- **Load testing**: Up to 50 concurrent clients, 100% success rate

### Cost Efficiency

- **Best configuration**: 12.7% cheaper than baseline
- **Answer generation**: ~$0.0002 per query (gpt-4o-mini)

---

## Cluster Deployment (UWaterloo eceTesla)

```bash
# SSH to cluster
ssh user@eceterm1.uwaterloo.ca
ssh user@ecetesla1.uwaterloo.ca

# Start Qdrant on ecetesla0
cd ~/qdrant && ./qdrant &

# Run distributed ingestion
mpirun -np 2 -host ecetesla1,ecetesla2 python scripts/mpi_ingest.py \
    --qdrant-host ecetesla0 --dataset msmarco --max-docs 700000 --use-gpu

# Run evaluation
python scripts/comprehensive_evaluation.py --qdrant-host ecetesla0 --use-ground-truth
```

---

## References

- **MS MARCO**: Microsoft Machine Reading Comprehension dataset
- **HotpotQA**: Multi-hop question answering dataset
- **Qdrant**: Vector similarity search engine
- **Sentence Transformers**: all-MiniLM-L6-v2 embeddings

---

## License

This project was developed for ECE 750 at the University of Waterloo.

