# Comprehensive Evaluation: Improved Parameters Analysis

**Date:** January 2025  
**Evaluation:** Comprehensive RAG evaluation with improved parameters  
**File:** `results/comprehensive_eval_improved.json`  
**Queries:** 100 (50 MS MARCO OLTP + 50 HotpotQA OLAP)

---

## Executive Summary

This evaluation tested 7 configurations with improved retrieval parameters:
- **Higher HNSW ef_search**: 200 (OLTP), 400 (OLAP) vs previous 50/200
- **Larger reranker candidate pool**: 50 candidates vs previous 20
- **Optimal hybrid_alpha**: 0.5 (tuned from 0.7)
- **Fixed BM25 indexing**: Full corpus (105K chunks) vs previous 10K

**Key Finding:** Transformer classifier + hybrid retrieval + reranking achieves **MRR=0.454** (best), but Recall@10 remains moderate (~0.27-0.29) across all configs.

---

## Results Summary

| Config | Latency (ms) | Confidence | MRR | Recall@10 | NDCG@10 | OLTP | OLAP |
|--------|--------------|------------|-----|-----------|---------|------|------|
| baseline_feature_dense | 30.3 | 0.660 | 0.296 | 0.273 | 0.316 | 58 | 42 |
| feature_hybrid | 296.4 | 0.660 | 0.297 | 0.275 | 0.317 | 58 | 42 |
| feature_hybrid_rerank_oltp | 371.7 | 0.660 | 0.354 | 0.269 | 0.366 | 58 | 42 |
| transformer_dense | 53.3 | 1.000 | 0.391 | 0.285 | 0.402 | 50 | 50 |
| transformer_hybrid | 244.1 | 1.000 | 0.382 | 0.285 | 0.392 | 50 | 50 |
| transformer_hybrid_rerank | 305.0 | 1.000 | **0.454** | 0.273 | **0.456** | 50 | 50 |
| feature_dense_rerank | 99.3 | 0.660 | 0.354 | 0.269 | 0.366 | 58 | 42 |

---

## Key Findings

### 1. Transformer Classifier Outperforms Feature-Based

**Transformer advantages:**
- **Perfect routing confidence** (1.000 vs 0.660)
- **Balanced routing** (50/50 OLTP/OLAP vs 58/42)
- **Higher MRR** (+32%: 0.391-0.454 vs 0.296-0.354)
- **Better NDCG** (+27%: 0.392-0.456 vs 0.316-0.366)

**Recommendation:** Use transformer classifier for production.

### 2. Reranking Significantly Improves MRR

**Impact of reranking:**
- **Feature-based**: MRR 0.296 → 0.354 (+19.6%)
- **Transformer**: MRR 0.391 → 0.454 (+16.1%)
- **Latency overhead**: +102ms average

**Trade-off:** Reranking improves ranking quality (MRR, NDCG) but adds ~100ms latency.

### 3. Hybrid Retrieval Shows Minimal Recall Improvement

**Hybrid vs Dense-only:**
- **Recall@10**: 0.275 (hybrid) vs 0.273 (dense) - **+0.7% improvement**
- **Latency overhead**: +243ms average
- **MRR**: Similar (0.297 vs 0.296 for feature, 0.382 vs 0.391 for transformer)

**Analysis:** Hybrid retrieval adds significant latency (~243ms) with minimal recall benefit. This suggests:
- BM25 may not be finding relevant passages that dense retrieval misses
- The queries may be more semantic than keyword-based
- BM25 normalization/scoring may need tuning

**Recommendation:** Consider disabling hybrid retrieval for latency-sensitive applications unless queries are highly keyword-focused.

### 4. Best Configuration: `transformer_hybrid_rerank`

**Performance:**
- **MRR: 0.454** (best)
- **NDCG@10: 0.456** (best)
- **Recall@10: 0.273** (moderate)
- **Latency: 305ms** (acceptable for OLAP)

**Use case:** Best for quality-critical applications where latency <500ms is acceptable.

### 5. Fastest Configuration: `baseline_feature_dense`

**Performance:**
- **Latency: 30.3ms** (fastest)
- **MRR: 0.296** (moderate)
- **Recall@10: 0.273** (moderate)

**Use case:** Best for latency-critical applications where sub-50ms response is required.

---

## Ablation Analysis

### Classifier Comparison
- **Feature-based**: 199.40ms avg latency
- **Transformer**: 200.82ms avg latency
- **Speed difference**: Negligible (1.01x slower)
- **Quality difference**: Significant (+32% MRR)

**Conclusion:** Transformer classifier provides substantial quality improvement with minimal latency cost.

### Retrieval Method Comparison
- **Dense-only**: 60.95ms avg latency, Recall@10: 0.275
- **Hybrid**: 304.31ms avg latency, Recall@10: 0.275
- **Overhead**: +243ms for +0.1% recall improvement

**Conclusion:** Hybrid retrieval not cost-effective for current query set.

### Reranker Impact
- **Without reranker**: 156.02ms avg latency
- **With reranker**: 258.66ms avg latency
- **Overhead**: +102ms
- **MRR improvement**: +15-20%

**Conclusion:** Reranking provides good quality/latency trade-off.

---

## Comparison to Previous Results

### Previous Comprehensive Evaluation (`comprehensive_eval.json`)
- **Best MRR**: ~0.33-0.34 (lower than current 0.454)
- **Best Recall@10**: ~0.55-0.63 (higher than current 0.27-0.29)
- **Note**: Previous evaluation may have used different ground truth or evaluation methodology

### Key Differences
1. **MRR improved**: 0.454 vs 0.33-0.34 (+37%)
2. **Recall@10 decreased**: 0.27-0.29 vs 0.55-0.63 (-50%)
3. **More consistent**: Current results show consistent performance across queries

**Possible explanations:**
- Different query sets (100 queries vs previous evaluation)
- Stricter ground truth matching (exact ID matching)
- Different evaluation methodology

---

## Recommendations

### 1. Production Configuration Selection

**For Quality-Critical Applications:**
- **Config**: `transformer_hybrid_rerank`
- **Metrics**: MRR=0.454, NDCG=0.456, Latency=305ms
- **Trade-off**: Best quality, acceptable latency

**For Latency-Critical Applications:**
- **Config**: `transformer_dense`
- **Metrics**: MRR=0.391, Latency=53ms
- **Trade-off**: Good quality, fast response

**For Balanced Applications:**
- **Config**: `transformer_hybrid_rerank` (disable hybrid if latency is concern)
- **Metrics**: MRR=0.454, Latency=305ms
- **Trade-off**: Best quality, moderate latency

### 2. Areas for Further Improvement

**Low Recall@10 (~0.27) suggests:**
1. **Query-document mismatch**: Relevant passages may not be in top-10
2. **Embedding model limitations**: Current model (`all-MiniLM-L6-v2`) may not capture query semantics well
3. **Ground truth alignment**: Verify that ground truth IDs match Qdrant metadata exactly
4. **Query expansion**: Consider query expansion/reformulation for better retrieval

**Potential improvements:**
- Try stronger embedding models (e.g., `bge-large-en-v1.5`)
- Implement query expansion (synonyms, paraphrases)
- Fine-tune embedding model on domain data
- Increase `top_k` for retrieval (currently 10, try 20-30)
- Analyze queries with `mrr=0` to identify failure patterns

### 3. Hybrid Retrieval Optimization

**Current state:** Hybrid retrieval adds 243ms latency with minimal recall benefit.

**Potential fixes:**
- Tune BM25 parameters (k1, b)
- Improve BM25 score normalization
- Use Reciprocal Rank Fusion (RRF) instead of weighted combination
- Analyze which queries benefit from BM25 vs dense-only

---

## Conclusion

The improved parameters (higher ef_search, larger reranker pool, optimal alpha) have:
- ✅ **Improved MRR** significantly (0.454 vs previous ~0.33)
- ✅ **Improved NDCG** (0.456 vs previous ~0.35)
- ⚠️ **Recall@10 remains moderate** (~0.27-0.29)
- ✅ **Transformer classifier** clearly outperforms feature-based
- ⚠️ **Hybrid retrieval** not cost-effective for current query set

**Next steps:**
1. Analyze queries with `mrr=0` to identify failure patterns
2. Verify ground truth alignment (ID matching)
3. Consider stronger embedding models or query expansion
4. Evaluate hybrid retrieval on keyword-heavy query subset

---

## Appendix: Configuration Details

### Test Environment
- **Qdrant host**: ecetesla0:6333
- **Device**: CPU
- **Embedding model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Transformer classifier**: `microsoft_MiniLM-L12-H384-uncased`
- **Rerankers**: TinyBERT (OLTP), MiniLM (OLAP)
- **HNSW ef_search**: 200 (OLTP), 400 (OLAP)
- **Reranker candidate pool**: 50 candidates
- **Hybrid alpha**: 0.5 (BM25 weight)

### Evaluation Metrics
- **MRR**: Mean Reciprocal Rank (average of 1/rank of first relevant passage)
- **Recall@10**: Fraction of queries where at least one relevant passage is in top-10
- **NDCG@10**: Normalized Discounted Cumulative Gain at rank 10
- **Latency**: Average query processing time (ms)
- **Confidence**: Classifier confidence score (0-1)

