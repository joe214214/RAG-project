<!-- c6429d9b-a127-4415-84e4-ce8183bf4eaa 05fb0e00-9b6b-4831-93db-843d029e000d -->
# RAG System Upgrade Plan

## Architecture Overview

```
ecetesla (Single GPU Node - RTX 3070)
├── Milvus Standalone (Docker) - HNSW index
├── Query Classifier (DistilBERT)
├── Embedder (MiniLM / e5-small)
├── Reranker (CrossEncoder for OLAP only)
└── Pipeline Orchestration

ecehadoop (Optional - Batch Processing)
└── PySpark jobs for bulk embedding generation

External
└── OpenAI/Azure API for LLM generation
```

## Phase 1: Milvus + HNSW Setup

Replace FAISS flat index with Milvus HNSW on ecetesla.

**Files to modify:**

- `vector_stores/superdb.py` - Replace FaissVectorStore with MilvusVectorStore
- `vector_stores/finedb.py` - Update to use Milvus
- `configs/base_config.yaml` - Add Milvus connection settings

**HNSW Parameters:**

- `M=16` (graph connectivity)
- `efConstruction=256` (build quality)
- `ef_search=64` for OLTP, `ef_search=256` for OLAP

**New dependency:** `pymilvus>=2.3.0`

---

## Phase 2: Cross-Encoder Reranker (OLAP only)

Add reranking for complex OLAP queries; skip for OLTP (use hybrid scores).

**Files to create/modify:**

- NEW: `retrievers/reranker.py` - CrossEncoderReranker class
- `pipelines/olap_pipeline.py` - Integrate reranker after retrieval
- `pipelines/oltp_pipeline.py` - No reranking (fast path)

**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`

**Strategy:**

- OLTP: Hybrid retrieval scores only (no reranking)
- OLAP: CrossEncoder reranks top-k candidates

---

## Phase 3: Embedding Upgrade

Improve embedding quality with better models.

**Files to modify:**

- `ml/embeddings.py` - Add model selection, consider e5-small-v2
- `preprocess/chunking.py` - Add semantic chunking with overlap

**Models:**

- Default: `all-MiniLM-L6-v2` (current, fast)
- Quality: `intfloat/e5-small-v2` (better recall)

---

## Phase 4: LLM API Integration

Replace stub with real LLM calls.

**Files to modify:**

- `pipelines/llm_client.py` - Add OpenAI/Azure client
- `configs/base_config.yaml` - LLM provider settings

**Features:**

- Environment variable config (`OPENAI_API_KEY`)
- Retry logic with exponential backoff
- Token counting and cost tracking

**New dependency:** `openai>=1.0.0`

---

## Phase 5: ecetesla Deployment

Deploy all components on single GPU node.

**Steps:**

1. SSH: `ssh userID@ecetesla2.uwaterloo.ca`
2. Check GPU: `nvidia-smi`
3. Install Milvus via Docker
4. Clone repo and install dependencies
5. Run pipeline with GPU acceleration

---

## Phase 6 (Optional): Spark Batch Embedding

For large corpus (>100K documents), use ecehadoop for bulk embedding.

**Files to create:**

- NEW: `distributed/spark_embed.py` - PySpark embedding job

**Constraints:**

- HDFS usage < 1GB in `/user/userID/`
- Use `/tmp/` for shared large inputs
- Jobs < 30 minutes

---

## Dependencies to Add

```
pymilvus>=2.3.0
openai>=1.0.0
```

---

## Distribution Summary

| Component | Where | Distributed? |

|-----------|-------|--------------|

| Milvus | ecetesla | No (standalone) |

| Embedder | ecetesla | No |

| Reranker | ecetesla | No |

| Classifier | ecetesla | No |

| LLM | API calls | N/A |

| Batch embed | ecehadoop | Yes (optional) |

### To-dos

- [ ] Replace FAISS with Milvus + HNSW in vector_stores/
- [ ] Add CrossEncoder reranker for OLAP pipeline
- [ ] Upgrade embedding models and add semantic chunking
- [ ] Integrate OpenAI/Azure LLM API in llm_client.py
- [ ] Deploy on ecetesla GPU node
- [ ] (Optional) Create PySpark batch embedding job for ecehadoop