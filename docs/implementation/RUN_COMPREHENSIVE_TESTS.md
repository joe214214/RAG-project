# Running Comprehensive Evaluation Tests

This guide shows how to run the same comprehensive evaluation tests documented in:
- `docs/results/initialFindings/ID_VS_TEXT_SIMILARITY_COMPARISON.md`
- `docs/results/initialFindings/COMPREHENSIVE_EVAL_IMPROVED_ANALYSIS.md`

## Quick Start

### Option 1: Use the Script (Linux/Mac)

```bash
chmod +x scripts/run_comprehensive_tests.sh
./scripts/run_comprehensive_tests.sh
```

### Option 2: Use the Script (Windows)

```cmd
scripts\run_comprehensive_tests.bat
```

### Option 3: Run Manually

Run each test individually:

## Test 1: ID-Based Evaluation (Exact Matching)

This matches the `COMPREHENSIVE_EVAL_IMPROVED_ANALYSIS.md` test.

```bash
python scripts/evaluation/comprehensive_evaluation.py \
    --qdrant-host ecetesla0 \
    --qdrant-port 6333 \
    --device cpu \
    --use-ground-truth \
    --oltp-limit 50 \
    --olap-limit 50 \
    --output results/comprehensive_eval_id_based.json
```

**What it tests:**
- Exact ID matching (MS MARCO passage IDs, HotpotQA title+sentence IDs)
- 7 configurations (baseline, feature, transformer, hybrid, reranked)
- Metrics: MRR, Recall@10, NDCG@10

**Expected output:** `results/comprehensive_eval_id_based.json`

---

## Test 2: Text Similarity - Jaccard (Word-Level)

```bash
python scripts/evaluation/comprehensive_evaluation.py \
    --qdrant-host ecetesla0 \
    --qdrant-port 6333 \
    --device cpu \
    --use-ground-truth \
    --use-text-first \
    --similarity-method jaccard \
    --similarity-threshold 0.15 \
    --oltp-limit 50 \
    --olap-limit 50 \
    --output results/comprehensive_eval_jaccard.json
```

**What it tests:**
- Word-level Jaccard similarity (text overlap on word sets)
- Same 7 configurations
- More lenient relevance matching than ID-based

**Expected output:** `results/comprehensive_eval_jaccard.json`

---

## Test 3: Text Similarity - N-gram (Character 3-grams)

```bash
python scripts/evaluation/comprehensive_evaluation.py \
    --qdrant-host ecetesla0 \
    --qdrant-port 6333 \
    --device cpu \
    --use-ground-truth \
    --use-text-first \
    --similarity-method ngram \
    --similarity-threshold 0.15 \
    --oltp-limit 50 \
    --olap-limit 50 \
    --output results/comprehensive_eval_ngram.json
```

**What it tests:**
- Character 3-gram overlap similarity
- Same 7 configurations
- Best performing method according to previous tests (99% Recall@10)

**Expected output:** `results/comprehensive_eval_ngram.json`

---

## Test 4: Text Similarity - Hybrid (Combined)

```bash
python scripts/evaluation/comprehensive_evaluation.py \
    --qdrant-host ecetesla0 \
    --qdrant-port 6333 \
    --device cpu \
    --use-ground-truth \
    --use-text-first \
    --similarity-method hybrid \
    --similarity-threshold 0.15 \
    --oltp-limit 50 \
    --olap-limit 50 \
    --output results/comprehensive_eval_hybrid_similarity.json
```

**What it tests:**
- Weighted combination: 50% word Jaccard + 30% character 3-gram + 20% length ratio
- Same 7 configurations
- Balanced approach between word and character-level matching

**Expected output:** `results/comprehensive_eval_hybrid_similarity.json`

---

## Expected Results

Based on previous tests:

| Method | Best MRR | Best Recall@10 | Best NDCG@10 |
|--------|----------|----------------|--------------|
| **ID-Based** | 0.454 | 0.273 | 0.456 |
| **Jaccard** | 0.525 | 0.403 | 0.535 |
| **Hybrid** | 0.948 | 0.978 | 0.940 |
| **N-gram** | **0.973** | **0.990** | **0.968** |

## Configurations Tested

Each test evaluates 7 configurations:

1. `baseline_feature_dense` - Baseline (feature classifier, dense-only)
2. `feature_hybrid` - Feature classifier + hybrid retrieval
3. `feature_hybrid_rerank_oltp` - Feature + hybrid + reranking
4. `transformer_dense` - Transformer classifier + dense-only
5. `transformer_hybrid` - Transformer + hybrid retrieval
6. `transformer_hybrid_rerank` - Transformer + hybrid + reranking (best)
7. `feature_dense_rerank` - Feature + dense + reranking

## Running on Cluster

If running on the cluster (ecetesla1, etc.):

```bash
# SSH to cluster node
ssh oankit@ecetesla1.uwaterloo.ca

# Navigate to project
cd ~/RAG-project

# Activate environment
source ~/rag-env/bin/activate

# Run tests
chmod +x scripts/run_comprehensive_tests.sh
./scripts/run_comprehensive_tests.sh
```

## Analyzing Results

After running tests, compare results:

```python
import json

# Load results
with open('results/comprehensive_eval_id_based.json') as f:
    id_based = json.load(f)

with open('results/comprehensive_eval_ngram.json') as f:
    ngram = json.load(f)

# Compare best configs
print("ID-Based Best:", id_based['best_config'])
print("N-gram Best:", ngram['best_config'])
```

## Notes

- **Test duration**: Each test takes ~10-15 minutes (100 queries × 7 configs)
- **Total time**: ~40-60 minutes for all 4 tests
- **Qdrant**: Ensure Qdrant is running on `ecetesla0:6333`
- **Ground truth**: Requires `ir_datasets` package and dataset access

## Troubleshooting

### Error: "No module named 'ir_datasets'"
```bash
pip install ir-datasets
```

### Error: "Connection refused" to Qdrant
- Check Qdrant is running: `ssh ecetesla0 "systemctl status qdrant"`
- Verify host/port: `--qdrant-host ecetesla0 --qdrant-port 6333`

### Error: "Dataset not found"
- Ensure datasets are downloaded: `python -c "import ir_datasets; print(ir_datasets.load('msmarco-passage/train'))"`


