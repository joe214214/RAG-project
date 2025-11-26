<!-- ed90473d-4317-4cd6-b904-cedddf58ed68 26b0e471-075e-41e6-b72a-e9e70c2fa039 -->
# Distributed RAG Upgrade Plan

## 1. Cluster & Distributed Runbook (docs/ECE hadoop/…)

- Consolidate requirements from `CLUSTER_QUICKSTART.md`, `CLUSTER_SETUP.md`, `ECE_HADOOP_CLUSTER_GUIDE.md`, `ece_servers_how-to.pdf`, `ece_server_how_to.md`, `what needs to be distributed.md`, and `ecetesla.md` into a single runbook (SSH hops via eceterm hosts, GPU discovery on ecetesla with `nvidia-smi`, Spark/MPI etiquette, HDFS quotas, port range 10 000–11 000).
- Decide which services run on UW clusters vs. locally (Faiss/Milvus pods, FastAPI service, monitoring), capture job submission templates, and outline HDFS/rsync data-sync procedures (prefer `/tmp` per cluster guide).

## 2. Embedding Pipeline Upgrade (docs/embedding_strategy_recommendations.md)

- Implement semantic multi-scale chunking with overlap controls in `preprocess/chunking.py` and expose dual embedding backbones (SBERT/E5 for OLTP, QZhou/Gemma/hyperbolic for OLAP) via `configs/base_config.yaml` + `ml/embeddings.py`.
- Add optional adapter/LoRA hooks plus monitoring (recall@k, MRR, latency) to validate new embeddings and document BM25/dense fusion tuning prior to cluster deployment.

## 3. Vector Store & HNSW Integration (docs/vector_db_recommendations.md, /facebookresearch/faiss)

- Extend `vector_stores/{superdb,finedb}.py` to support FAISS HNSW/IndexHNSWCagra knobs (`M`, `ef_construction`, `ef_search`, `base_level_only`) following Faiss guidance on GPU→CPU conversion and stats.
- If OLAP requires distributed storage, prototype Milvus/TigerVector scripts aligned with cluster docs and add ANN sweep utilities (e.g., in `benchmarks/ablation.py`) to profile recall/latency tradeoffs.

## 4. Reranker Implementation (docs/reranker_recommendations.md, /huggingface/sentence-transformers)

- Create a `rerankers/` package with (a) ColBERT or optimized pairwise LLM reranker for OLTP and (b) CrossEncoder/DPS-style reranker for OLAP; wire them into `pipelines/oltp_pipeline.py`, `pipelines/olap_pipeline.py`, and the policy engine for adaptive `rerank_depth`.
- Extend `benchmarks/ablation.py` with reranking experiments (MAP, MRR@10) to quantify latency vs. quality improvements per pipeline.

## 5. Proposal & Documentation Refresh (docs/Proposal_and_feedback.MD)

- Update the proposal, README, and the new runbook to reflect the distributed deployment, embedding overhaul, HNSW/vector DB setup, and reranker design, explicitly addressing reviewer questions about policy learning, baselines, and success criteria.

### To-dos

- [ ] Draft UW cluster runbook + packaging plan
- [ ] Implement multi-scale embeddings + config hooks
- [ ] Add FAISS HNSW support + tuning
- [ ] Implement OLTP/OLAP rerankers
- [ ] Revise proposal + docs