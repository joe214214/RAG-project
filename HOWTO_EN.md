# How to Use the Query-Aware RAG Project

Step-by-step: install → prepare data → build vectors/indexes → run QA → benchmarks.

## 1) Install Dependencies
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # Windows; use source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
python -m spacy download en_core_web_sm  # for the graph pipeline
```

## 2) Prepare Data
### Option A — Demo
```bash
python main.py --demo --query "Summarize AI investments by Apple and Microsoft."
```
Generates synthetic docs, chunks, indexes, and runs the full pipeline.

### Option B — Your PDFs
1. Put PDFs in `data/pdf/`.
2. Extract text:
```python
from pathlib import Path
from preprocess.pdf_reader import extract_pdfs_to_text

extract_pdfs_to_text(Path("data/pdf"), Path("data/text"))
```
3. Create hierarchical chunks:
```python
from pathlib import Path
from preprocess.chunking import create_hierarchical_chunks

create_hierarchical_chunks(
    text_dir=Path("data/text"),
    super_chunk_dir=Path("data/chunks/super"),
    fine_chunk_dir=Path("data/chunks/fine"),
)
```
4. Run inference (rebuilds embeddings + indexes on the fly):
```bash
python main.py --query "Summarize the latest AI investments."
```

## 3) Configure (Optional)
Edit `configs/base_config.yaml` to change:
- Embedding model and batch size
- Paths (PDF/chunks/embeddings)
- Default retrieval depths and hybrid weight

## 4) Query Classifier
- Train: `python ml/classifier/train_classifier.py --output-dir artifacts/classifier`
- Inference API:
```python
from ml.classifier.classifier_inference import classify_query
label = classify_query("Summarize quarterly performance of Apple and Microsoft.")
```
Point `classifier.model_name_or_path` to your trained model.

## 5) Pipelines & Policy
- Policy: `pipelines/policy.py`
- OLTP pipeline: `pipelines/oltp_pipeline.py`
- OLAP pipeline: `pipelines/olap_pipeline.py`
- Hierarchical retrieval: `retrievers/hierarchical_retriever.py`

To customize, instantiate your pipeline and register via `configure_oltp_pipeline()` / `configure_olap_pipeline()`.

## 6) Benchmarks (Charts)
```bash
python benchmarks/scale_corpus.py   # KB size vs latency/throughput
python benchmarks/scale_qps.py      # QPS vs P50/P95/P99
python benchmarks/ablation.py       # routing/hierarchical/graph ablations
```
Charts land in `results/`:
- `latency_vs_kb.png`
- `qps_tail_latency.png`
- `routing_ablation.png`
- `hierarchical_vs_flat.png`

## 7) Frequently Used Paths
- Raw PDFs: `data/pdf/`
- Extracted text: `data/text/`
- Chunks: `data/chunks/{super,fine}/`
- Embeddings: `data/embeddings/{super,fine}/`
- Benchmark outputs: `results/`

With these steps you can run the baseline, extend with new retrieval/graph strategies, and collect scalability measurements.***
