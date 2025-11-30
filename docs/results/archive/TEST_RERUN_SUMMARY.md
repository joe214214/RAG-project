# Test Rerun Summary - BM25 Index Fix

**Date:** January 2025  
**Issue:** BM25 index was only building from first 10K chunks (~1% of corpus)  
**Fix:** Updated `_build_bm25_index()` to scroll through ALL chunks with thread-safe implementation  
**Impact:** Hybrid retrieval tests need rerun for accurate results

---

## Tests That Need Rerun

### 1. Comprehensive Evaluation (`scripts/comprehensive_evaluation.py`)

**Affected Configurations (4 out of 7):**
- ❌ `feature_hybrid` - Feature classifier + Hybrid retrieval
- ❌ `feature_hybrid_rerank_oltp` - Feature + Hybrid + Reranker
- ❌ `transformer_hybrid` - Transformer + Hybrid retrieval
- ❌ `transformer_hybrid_rerank` - Transformer + Hybrid + Reranker

**Command:**
```bash
python scripts/comprehensive_evaluation.py \
  --qdrant-host ecetesla0 \
  --output results/comprehensive_eval_fixed.json
```

**Expected Changes:**
- Better BM25 recall (searching full 813K corpus vs 10K)
- More accurate hybrid scores (BM25 + dense fusion)
- Possibly different latency (initial indexing slower, but retrieval more accurate)

---

### 2. Parallel Load Test - Hybrid (`benchmarks/parallel_load_test.py`)

**Affected Test:**
- ❌ Hybrid retrieval load test (already run, but with incomplete BM25)

**Command:**
```bash
python benchmarks/parallel_load_test.py \
  --qdrant-host ecetesla0 \
  --num-queries 1000 \
  --concurrency-levels 1,10,20,50,100 \
  --use-hybrid \
  --output results/load_test_hybrid_fixed.json
```

**Expected Changes:**
- More accurate QPS/latency metrics
- Better hybrid retrieval performance
- More realistic load test results

---

## Tests That DON'T Need Rerun

### ✅ Still Valid (No Hybrid Retrieval)

1. **Baseline Tests:**
   - `baseline_feature_dense` - Dense-only, no BM25
   - `transformer_dense` - Dense-only, no BM25
   - `feature_dense_rerank` - Dense-only + reranker, no BM25

2. **Reranker Ablation (`benchmarks/reranker_ablation.py`):**
   - All tests use dense-only retrieval
   - No BM25/hybrid involved

3. **Load Tests (Non-Hybrid):**
   - Baseline load test (no hybrid)
   - Reranker load test (no hybrid)

---

## What Changed in the Code

### Before (Broken):
```python
scroll_result = self.qdrant_client.scroll(
    collection_name=collection,
    limit=10000,  # Only first 10K chunks!
    ...
)
```

### After (Fixed):
```python
# Scroll through ALL chunks using pagination
id_to_text = {}
offset = None
batch_size = 10000

while True:
    scroll_result = self.qdrant_client.scroll(
        collection_name=collection,
        limit=batch_size,
        offset=offset,
        ...
    )
    # Process batch and continue until all chunks fetched
    ...
```

**Additional Fixes:**
- Added `threading.Lock()` for thread-safe index building
- Progress indicators for large collections
- Proper pagination handling

---

## Impact Analysis

### Before Fix:
- BM25 indexed: **10,000 chunks** (~1% of 813K corpus)
- BM25 searched: **~1% of data**
- Hybrid retrieval: Mostly dense retrieval (BM25 contribution minimal)

### After Fix:
- BM25 indexed: **~813,000 chunks** (100% of corpus)
- BM25 searches: **Full corpus**
- Hybrid retrieval: True BM25 + dense fusion

### Expected Improvements:
1. **Better Recall:** BM25 can now find keyword matches across entire corpus
2. **More Accurate Hybrid Scores:** BM25 scores reflect full corpus statistics
3. **Better Quality Metrics:** MRR, Recall@10 should improve for hybrid configs

---

## Priority Order for Rerun

1. **High Priority:**
   - Comprehensive evaluation (affects main results report)
   - Hybrid load test (affects scalability analysis)

2. **Medium Priority:**
   - Any custom hybrid retrieval tests
   - Hybrid-specific ablation studies

3. **Low Priority:**
   - Test scripts (`test_hybrid_retrieval.py`) - for validation only

---

## Files to Update After Rerun

1. **`docs/Results/EVALUATION_REPORT.md`**
   - Update hybrid retrieval metrics (Section 4.2)
   - Update comparison tables (Section B.1)
   - Revise "Hybrid may improve recall" section

2. **`results/comprehensive_eval.json`**
   - Replace with new results from fixed BM25

3. **`results/load_test_hybrid.json`**
   - Replace with `load_test_hybrid_fixed.json` results

---

## Verification Steps

After rerunning tests, verify:

1. **BM25 Index Size:**
   ```python
   # Check that BM25 index contains all chunks
   # Should see: "Building BM25 index from 813,178 chunks..."
   ```

2. **Hybrid Retrieval Quality:**
   - Compare Recall@10 before/after fix
   - Should see improvement in hybrid configs

3. **Latency Impact:**
   - Initial indexing will be slower (indexing 813K vs 10K)
   - But retrieval should be more accurate

---

**Status:** Ready for rerun  
**Next Step:** Run comprehensive evaluation and hybrid load test with fixed BM25

