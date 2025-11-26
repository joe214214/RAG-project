<!-- 6d26a0c1-5467-4b86-a9e3-c389dcedf710 51a624c4-a828-4bd7-b28f-c52f8e47f81d -->
# Distributed RAG System Migration Plan

This plan outlines the steps to migrate the current local RAG implementation to a distributed environment on the ECE cluster, integrating Milvus as the vector database, implementing a local Cross-Encoder reranker, and refining the embedding strategy.

## 1. Vector Database Setup (Milvus)

- **Objective:** Replace local FAISS with a distributed Milvus instance for scalability.
- **Tasks:**
    - [ ] **Infrastructure Setup:** Deploy Milvus on the ECE cluster (likely Docker/Kubernetes or standalone binary depending on cluster constraints). *Note: Need to verify if Docker is allowed on `ecetesla`/`ecehadoop` nodes or if a standalone build is required.*
    - [ ] **Connector Implementation:** Create `vector_stores/milvus_store.py` implementing the `VectorStore` interface.
        - Implement connection logic (host/port from config).
        - Implement `build_index` using HNSW parameters (M, efConstruction).
        - Implement `search` with `ef` tuning.
    - [ ] **Migration:** Update `main.py` and `initialize_system` to use `MilvusStore` instead of `FaissVectorStore` based on config.
    - [ ] **Configuration:** Add Milvus connection params (host, port, collection names) to `configs/base_config.yaml`.

## 2. Reranker Implementation (Local Cross-Encoder)

- **Objective:** Improve retrieval precision using a local Cross-Encoder model, replacing the placeholder/stub.
- **Tasks:**
    - [ ] **Model Integration:** Add `cross-encoder/ms-marco-MiniLM-L-6-v2` (or similar) to `ml/reranker.py` using `sentence-transformers`.
    - [ ] **Pipeline Update:** Modify `OltpPipeline` and `OlapPipeline` to include a reranking step.
        - **OLTP:** Retrieve Top-K (e.g., 50) -> Rerank -> Return Top-N (e.g., 5).
        - **OLAP:** Rerank candidates before graph construction/summarization.
    - [ ] **Policy Integration:** Wire `rerank_depth` from `PolicyDecision` to the reranker logic.

## 3. Embedding Strategy Refinement

- **Objective:** optimize embeddings for dual-pipeline needs.
- **Tasks:**
    - [ ] **Model Selection:**
        - **OLTP:** Keep `sentence-transformers/all-MiniLM-L6-v2` (fast, effective).
        - **OLAP:** Evaluate/Switch to a larger model like `Gemma-2-9B` or `Qwen2.5-7B` (if GPU RAM permits on `ecetesla`) for better semantic capture of complex queries.
    - [ ] **Chunking Strategy:**
        - Implement **Semantic Chunking** in `preprocess/chunking.py` (overlapping windows based on sentence boundaries/semantic similarity) instead of fixed token windows.
        - Ensure "Small-to-Big" mapping is preserved for hierarchical retrieval.

## 4. Distributed Execution on ECE Cluster

- **Objective:** Run the system across multiple nodes (MPI/Spark).
- **Tasks:**
    - [ ] **Environment Setup:** Create a Conda/venv setup script for ECE nodes (handling restricted internet access if necessary).
    - [ ] **Parallel Ingestion (Spark/MPI):**
        - Use **Spark** (on `ecehadoop`) for heavy preprocessing: PDF extraction -> Chunking -> Embedding generation.
        - Write a PySpark job to process documents in parallel and push vectors to Milvus.
    - [ ] **Inference Deployment:**
        - Deploy the API/System on `ecetesla` nodes (GPU enabled) for Inference (Embedder + Reranker + LLM).
        - Use MPI or a simple HTTP load balancer if running multiple inference workers.

## 5. Testing & Validation

- **Tasks:**
    - [ ] **Unit Tests:** Verify Milvus connector and Reranker logic locally.
    - [ ] **Integration Test:** Run a small scale end-to-end test on the cluster.
    - [ ] **Benchmarks:** Re-run `benchmarks/ablation.py` and `scale_qps.py` on the cluster to measure performance gains.

## Questions/Clarifications Needed:

1.  **Docker Availability:** Can we run Docker containers (for Milvus) on `ecehadoop` or `ecetesla` nodes? Or do we need to build from source/use a provided instance?
2.  **GPU Access:** Do you have specific `ecetesla` nodes assigned, or is it a shared queue? (Affects model size selection for OLAP).
3.  **Spark/HDFS:** Do you have a persistent HDFS directory setup on `ecehadoop`?

This plan prioritizes the Milvus migration and Reranker implementation as they are core architectural changes required before scaling out.

### To-dos

- [ ] Create Milvus vector store implementation
- [ ] Implement Cross-Encoder reranker
- [ ] Update config for Milvus and Reranker
- [ ] Refine chunking strategy (Semantic Chunking)
- [ ] Develop Spark job for parallel ingestion
- [ ] Create cluster deployment scripts