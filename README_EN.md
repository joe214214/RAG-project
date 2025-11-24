# Query-Aware Routing for Scalable RAG

Retrieval-Augmented Generation (RAG) system with query-aware routing, hierarchical super-chunk retrieval, and dual OLTP/OLAP pipelines. Modular design so preprocessing, retrieval, policy, pipelines, and benchmarking can be swapped or extended.

## Features
- **Query-aware routing** — DistilBERT classifier outputs `oltp` vs `olap`.
- **Dual pipelines** — Fast OLTP hybrid retriever for factoids; OLAP graph pipeline with entity graphs + Leiden communities.
- **Hierarchical retrieval** — PDF → text → super-chunks (~2–3 pages) → fine chunks (~512 tokens). Super/Fine FAISS indexes plus BM25 + hybrid scoring.
- **Policy layer** — Dynamic knobs for `top_k`, hybrid weight, rerank depth, hierarchical toggle, and graceful degradation under load.
- **Benchmark suite** — Corpus scaling, QPS scaling, and ablation studies output charts to `results/`.
- **Config-driven** — `configs/base_config.yaml` centralizes paths and model settings.

## Repository Layout
```
project_root/
├── configs/                 # YAML configs
├── data/                    # Input + intermediate artifacts
│   ├── pdf/                 # Raw PDFs
│   ├── text/                # Extracted text
│   ├── chunks/{super,fine}/ # Hierarchical chunks + metadata
│   └── embeddings/{...}/    # Saved numpy embeddings
├── vector_stores/           # FAISS wrappers (super/fine)
├── retrievers/              # BM25, hybrid, hierarchical retrievers
├── pipelines/               # OLTP, OLAP, policy, LLM shim
├── preprocess/              # PDF extractor + chunking
├── ml/                      # Embeddings + classifier (train/infer)
├── graph/                   # Entity graph + Leiden + summaries
├── benchmarks/              # Scale/QPS/ablation scripts
├── results/                 # Benchmark plots
└── main.py                  # `rag_answer` entrypoint
```

## Quickstart (Demo Mode)
```bash
pip install -r requirements.txt
python main.py --demo --query "Summarize AI investments by Apple and Microsoft."
```
`--demo` populates synthetic documents, builds chunks and FAISS indexes, and runs the full stack locally (no external APIs).

## Using Your Own PDFs
1. Place PDFs under `data/pdf/`.
2. Convert PDFs to text and create hierarchical chunks:
```python
from pathlib import Path
from preprocess.pdf_reader import extract_pdfs_to_text
from preprocess.chunking import create_hierarchical_chunks

extract_pdfs_to_text(Path("data/pdf"), Path("data/text"))
create_hierarchical_chunks(
    text_dir=Path("data/text"),
    super_chunk_dir=Path("data/chunks/super"),
    fine_chunk_dir=Path("data/chunks/fine"),
)
```
3. Rebuild embeddings/indexes and run inference:
```bash
python main.py --query "Your question here"
```

## Training the Query Classifier
```bash
python ml/classifier/train_classifier.py \
  --model-name distilbert-base-uncased \
  --output-dir artifacts/classifier
```
Then point `classifier.model_name_or_path` in `configs/base_config.yaml` (or env var) to the exported directory.

## Benchmarks
All benchmark scripts write JSON + charts into `results/`.
```bash
python benchmarks/scale_corpus.py
python benchmarks/scale_qps.py
python benchmarks/ablation.py
```
Expected figures:
- `results/latency_vs_kb.png`
- `results/qps_tail_latency.png`
- `results/routing_ablation.png`
- `results/hierarchical_vs_flat.png`

## Extending the System
- Swap embedding models in `configs/base_config.yaml`.
- Plug a real LLM by extending `pipelines/llm_client.LLMClient`.
- Add retrieval pipelines (e.g., GraphRAG variants) and register them in routing/policy logic.
- Replace vector backends by implementing the same interface in `vector_stores` (e.g., Milvus).

See `HOWTO_EN.md` for more step-by-step usage.***
