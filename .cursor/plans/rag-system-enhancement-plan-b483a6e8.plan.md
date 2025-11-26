<!-- b483a6e8-11d5-4583-a5b4-b444dba14246 1eebcf32-9d7a-4a85-9b39-64f718c77c99 -->
# RAG System Enhancement Plan

## Overview

This plan addresses five major components to enhance the query-aware RAG system: (1) distributed deployment on ECE Hadoop cluster, (2) Milvus migration with HNSW indexing, (3) embedding strategy revision, (4) reranker implementation, and (5) proposal documentation updates.

## Task Order and Dependencies

### Phase 1: Distributed Environment Foundation (Week 1-2)

**Goal**: Set up cluster infrastructure and identify distributed components

**Tasks**:

1. **Cluster Access & Environment Setup**

   - Connect to ECE Hadoop cluster (`ecehadoop.private.uwaterloo.ca`)
   - Set up Python environment with MPI support (`mpi4py`)
   - Install base dependencies on cluster nodes
   - Test SSH keyless access across nodes (ecehadoop0-15)

2. **Component Distribution Analysis**

   - Document which components need distribution:
     - **Embedding generation**: Batch processing across nodes (parallelize document chunks)
     - **Vector search**: Milvus distributed deployment (OLAP pipeline)
     - **Graph construction**: Leiden clustering can parallelize community detection
     - **Reranking**: Can distribute candidate reranking across nodes
   - Keep query classifier and routing logic centralized (low latency requirement)

3. **Create Distribution Architecture**

   - Design MPI-based parallel embedding generation
   - Plan Milvus cluster deployment (standalone mode initially, scale to distributed)
   - Create hostfile for MPI jobs (`mpi_ecehadoop_hosts`)

**Files to modify**:

- Create `docs/ECE hadoop/distribution_architecture.md`
- Create `scripts/cluster_setup.sh`
- Modify `ml/embeddings.py` to support MPI parallelization

---

### Phase 2: Milvus Migration with HNSW (Week 2-3)

**Goal**: Replace FAISS with Milvus, enable HNSW indexing

**Tasks**:

1. **Milvus Installation & Configuration**

   - Install Milvus standalone on cluster (start with single node, scale later)
   - Configure Milvus with HNSW index type
   - Set HNSW parameters: `M=32` (M=32), `ef_construction=200`, `ef_search=50` (OLTP) / `ef_search=200` (OLAP)
   - Create separate collections for super-chunks and fine-chunks

2. **Migrate Vector Store Interface**

   - Create `vector_stores/milvus_store.py` implementing same interface as `FaissVectorStore`
   - Update `vector_stores/superdb.py` and `vector_stores/finedb.py` to use Milvus backend
   - Implement collection management (create, load, search)
   - Add HNSW index configuration per collection

3. **Data Migration Script**

   - Create `scripts/migrate_faiss_to_milvus.py`
   - Load existing FAISS embeddings into Milvus collections
   - Verify search results match FAISS (recall@k comparison)

4. **Update Configuration**

   - Add Milvus connection settings to `configs/base_config.yaml`
   - Add HNSW tuning parameters (M, ef_construction, ef_search) per pipeline

**Files to modify**:

- Create `vector_stores/milvus_store.py`
- Modify `vector_stores/superdb.py`, `vector_stores/finedb.py`
- Modify `configs/base_config.yaml`
- Create `scripts/migrate_faiss_to_milvus.py`
- Update `main.py` to initialize Milvus connections

**Dependencies**: Phase 1 complete

---

### Phase 3: Embedding Strategy Revision (Week 3-4)

**Goal**: Implement multi-scale semantic chunking and improved embedding models

**Tasks**:

1. **Upgrade Embedding Models**

   - **OLTP**: Migrate from `all-MiniLM-L6-v2` to `sentence-transformers/all-mpnet-base-v2` or `intfloat/e5-large-v2` (better quality)
   - **OLAP**: Add support for hierarchical embeddings (consider `QZhou-Embedding` or `BGE-large` for complex queries)
   - Update `ml/embeddings.py` to support model selection per pipeline

2. **Implement Semantic Chunking**

   - Enhance `preprocess/chunking.py` with semantic segmentation:
     - Use sentence-transformers for semantic similarity clustering
     - Implement HDBSCAN or Agglomerative clustering for document-level chunks
     - Add overlap handling to preserve entity continuity
   - Create multi-scale chunks: fine (sentence-level), medium (paragraph), coarse (document-level)

3. **Small-to-Big Retrieval**

   - Modify `retrievers/hierarchical_retriever.py` to support progressive retrieval
   - Start with fine chunks, expand to super chunks if evidence insufficient
   - Add evidence sufficiency scoring

4. **Update Embedding Generation Pipeline**

   - Modify `main.py` `embed_chunks()` to support batch processing with MPI
   - Add distributed embedding generation script for cluster deployment

**Files to modify**:

- Modify `ml/embeddings.py` (model selection, batch processing)
- Enhance `preprocess/chunking.py` (semantic clustering)
- Modify `retrievers/hierarchical_retriever.py` (progressive retrieval)
- Modify `main.py` (embedding pipeline)
- Create `scripts/distributed_embedding.py` (MPI-based)

**Dependencies**: Phase 2 complete (needs Milvus for storage)

---

### Phase 4: Reranker Implementation (Week 4-5)

**Goal**: Add query-aware reranking to both OLTP and OLAP pipelines

**Tasks**:

1. **OLTP Reranker (Fast Path)**

   - Implement ColBERT-style reranker or lightweight cross-encoder
   - Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (fast, sub-second latency)
   - Create `retrievers/rerankers/oltp_reranker.py`
   - Integrate into `pipelines/oltp_pipeline.py` after hybrid retrieval

2. **OLAP Reranker (Quality Path)**

   - Implement cross-encoder reranker: `BAAI/bge-reranker-large` or `Zerank-1`
   - For multi-document: Consider Dynamic Passage Selector (DPS) or setwise reranking
   - Create `retrievers/rerankers/olap_reranker.py`
   - Integrate into `pipelines/olap_pipeline.py` after hierarchical retrieval

3. **Policy Integration**

   - Update `PolicyDecision` to actually use `rerank_depth` parameter
   - Implement adaptive reranking: adjust rerank set size based on query complexity
   - Add reranker latency tracking to metrics

4. **Reranker Abstraction**

   - Create base `Reranker` interface in `retrievers/rerankers/__init__.py`
   - Implement factory pattern for query-type-based reranker selection

**Files to modify**:

- Create `retrievers/rerankers/__init__.py`
- Create `retrievers/rerankers/oltp_reranker.py`
- Create `retrievers/rerankers/olap_reranker.py`
- Modify `pipelines/oltp_pipeline.py` (add reranking step)
- Modify `pipelines/olap_pipeline.py` (add reranking step)
- Modify `pipelines/policy.py` (use rerank_depth)
- Update `requirements.txt` (add reranker dependencies)

**Dependencies**: Phase 3 complete (needs improved embeddings)

---

### Phase 5: Proposal Updates & Documentation (Week 5-6)

**Goal**: Update proposal to reflect implemented improvements

**Tasks**:

1. **Address Feedback Points**

   - Clarify "dynamically adjusts" = heuristic policy layer (not ML-based yet)
   - Define "pipelines" explicitly (OLTP vs OLAP retrieval strategies)
   - Update work plan to reflect Milvus deployment (not just "set up")
   - Clarify classifier training vs pretrained model usage
   - Define success criteria: latency (P95 <500ms OLTP, <2s OLAP), accuracy (F1 >0.8), cost reduction (20-30% vs baseline)

2. **Update Technical Details**

   - Document HNSW implementation and tuning
   - Add reranker architecture to solution design
   - Update scalability section with distributed deployment details
   - Add Milvus cluster architecture diagram

3. **Revise Literature Review**

   - Cite QA-Dragon and other Aug 2025 papers (from novelty assessment)
   - Differentiate from existing work more clearly
   - Add references to reranking papers (REBEL, DPS, etc.)

4. **Create Distribution Documentation**

   - Document cluster deployment process
   - Add performance benchmarks (local vs distributed)
   - Create `docs/ECE hadoop/deployment_guide.md`

**Files to modify**:

- `docs/Proposal_and_feedback.MD` (address all feedback points)
- Create `docs/ECE hadoop/deployment_guide.md`
- Update `README_EN.md` with distributed deployment instructions

**Dependencies**: All previous phases (document what was built)

---

## Critical Path Dependencies

```
Phase 1 (Distributed Setup)
    ↓
Phase 2 (Milvus Migration) ──→ Phase 3 (Embeddings) ──→ Phase 4 (Reranker)
    ↓                                                          ↓
Phase 5 (Documentation) ←─────────────────────────────────────┘
```

## Success Metrics

- **Distributed**: Embedding generation scales linearly with nodes (up to 16 nodes)
- **Milvus**: HNSW index achieves >95% recall@10 with <50ms latency (OLTP)
- **Embeddings**: Semantic chunking improves retrieval F1 by 10-15%
- **Reranker**: OLTP reranker adds <100ms latency, OLAP reranker improves F1 by 5-10%
- **Proposal**: All feedback points addressed, success criteria defined

## Risk Mitigation

- **Milvus complexity**: Start with standalone mode, validate before distributed
- **Cluster access**: Test early, have fallback to local development
- **Embedding model size**: Monitor memory usage, use quantization if needed
- **Reranker latency**: Profile early, fallback to no reranking if too slow

### To-dos

- [ ] Set up ECE Hadoop cluster access, MPI environment, and test connectivity across nodes
- [ ] Analyze which components need distribution (embeddings, vector search, graph construction) and create architecture doc
- [ ] Install Milvus standalone on cluster, configure HNSW indexes with appropriate parameters
- [ ] Create Milvus vector store interface, migrate FAISS code to use Milvus backend
- [ ] Migrate existing FAISS embeddings to Milvus collections and verify search accuracy
- [ ] Upgrade embedding models (E5-large for OLTP, BGE/QZhou for OLAP) and update embedding pipeline
- [ ] Implement semantic chunking with HDBSCAN clustering and multi-scale chunk generation
- [ ] Create MPI-based distributed embedding generation script for cluster deployment
- [ ] Implement fast OLTP reranker (ColBERT or ms-marco cross-encoder) and integrate into OLTP pipeline
- [ ] Implement quality-focused OLAP reranker (BGE-large or DPS) and integrate into OLAP pipeline
- [ ] Update PolicyDecision to use rerank_depth parameter and implement adaptive reranking
- [ ] Address all proposal feedback points: clarify heuristics, define pipelines, add success criteria
- [ ] Update proposal with HNSW details, reranker architecture, distributed deployment info
- [ ] Create cluster deployment guide and update README with distributed setup instructions