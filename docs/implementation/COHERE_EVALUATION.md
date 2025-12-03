# Running Comprehensive Evaluation with Cohere API

This guide shows how to run the comprehensive evaluation tests with Cohere API enabled for embeddings and reranking.

## Quick Start

### Option 1: Use the Script

```bash
# Set API key (if not already set)
export COHERE_API_KEY="your-api-key-here"

# Make script executable
chmod +x scripts/run_cohere_evaluation.sh

# Run evaluation
./scripts/run_cohere_evaluation.sh
```

### Option 2: Run Manually

```bash
python scripts/evaluation/comprehensive_evaluation.py \
    --qdrant-host ecetesla0 \
    --qdrant-port 6333 \
    --device cpu \
    --use-ground-truth \
    --oltp-limit 50 \
    --olap-limit 50 \
    --use-cohere \
    --cohere-api-key "your-api-key-here" \
    --cohere-embed-model "embed-english-light-v3.0" \
    --cohere-rerank-model "rerank-english-v3.0" \
    --output results/comprehensive_eval_cohere_id_based.json
```

## What This Tests

This runs the **ID-Based (Exact Matching)** evaluation with Cohere API:

- **Embeddings**: Uses Cohere `embed-english-light-v3.0` (384 dimensions) instead of local `sentence-transformers`
- **Reranking**: Uses Cohere `rerank-english-v3.0` instead of local cross-encoder rerankers
- **Evaluation Method**: Exact ID matching (MS MARCO passage IDs, HotpotQA title+sentence IDs)
- **Configurations**: Tests all 7 configurations with Cohere API

## Configurations Tested

1. `baseline_feature_dense` - Baseline with Cohere embeddings
2. `feature_hybrid` - Feature classifier + hybrid (BM25 + Cohere embeddings)
3. `feature_hybrid_rerank_oltp` - Feature + hybrid + Cohere reranking
4. `transformer_dense` - Transformer classifier + Cohere embeddings
5. `transformer_hybrid` - Transformer + hybrid (BM25 + Cohere embeddings)
6. `transformer_hybrid_rerank` - Transformer + hybrid + Cohere reranking
7. `feature_dense_rerank` - Feature + dense + Cohere reranking

## Expected Output

Results will be saved to `results/comprehensive_eval_cohere_id_based.json` with:
- MRR (Mean Reciprocal Rank)
- Recall@10, Recall@5, Recall@20
- NDCG@10
- Latency metrics
- Per-configuration breakdowns

## Comparison with Local Models

After running, you can compare:

| Metric | Local Models | Cohere API |
|--------|--------------|------------|
| **Best MRR** | 0.454 | TBD |
| **Best Recall@10** | 0.273 | TBD |
| **Best NDCG@10** | 0.456 | TBD |
| **Latency** | ~305ms | TBD |

## Notes

- **Vector Dimensions**: Uses `embed-english-light-v3.0` (384 dims) to match your existing Qdrant collection
- **Cost**: Each evaluation run costs ~$5-10 in Cohere API calls (100 queries × 7 configs × embeddings + reranking)
- **Duration**: Takes ~15-20 minutes (longer than local due to API latency)
- **Fallback**: If Cohere API fails, automatically falls back to local embeddings/rerankers

## Troubleshooting

### Error: "Cohere API key required"
```bash
export COHERE_API_KEY="your-api-key-here"
# or pass --cohere-api-key "your-key"
```

### Error: "Vector dimension error: expected dim: 384, got 1024"
- This means you're using `embed-english-v3.0` (1024 dims) instead of `embed-english-light-v3.0` (384 dims)
- Fix: Use `--cohere-embed-model "embed-english-light-v3.0"`

### Slow Performance
- Expected: Cohere API adds ~200-500ms latency per query
- Total time: ~15-20 minutes for 100 queries × 7 configs

## Running on Cluster

```bash
# SSH to cluster
ssh oankit@ecetesla1.uwaterloo.ca

# Navigate to project
cd ~/RAG-project

# Activate environment
source ~/rag-env/bin/activate

# Set API key (tcsh/csh shell)
setenv COHERE_API_KEY "your-api-key-here"

# Run evaluation
chmod +x scripts/run_cohere_evaluation.sh
./scripts/run_cohere_evaluation.sh
```

## Next Steps

After running, compare results:
1. **Cohere vs Local**: Compare `comprehensive_eval_cohere_id_based.json` vs `comprehensive_eval_id_based.json`
2. **Quality Improvement**: Check if Cohere improves MRR/Recall@10
3. **Cost Analysis**: Calculate cost per query improvement
4. **Latency Trade-off**: Evaluate if quality improvement justifies latency increase


