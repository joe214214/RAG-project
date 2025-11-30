# RAG Retrieval Improvements - Implementation Status

## ✅ Completed (Phase 1 - Quick Wins)

### 1. ID Alignment Verification Script
**File**: `scripts/verify_id_alignment.py`

**Purpose**: Verify that ground truth passage IDs match what's stored in Qdrant.

**Usage**:
```bash
python scripts/verify_id_alignment.py --qdrant-host ecetesla0 --num-samples 10
```

**What it does**:
- Samples queries with `mrr=0` and `num_relevant_total=1`
- Fetches ground truth IDs from MS MARCO/HotpotQA datasets
- Checks if those IDs exist in Qdrant
- Reports alignment issues

**Expected Impact**: Identifies data alignment problems that could explain 20-30% of retrieval failures.

---

### 2. HNSW Parameter Tuning
**File**: `scripts/rag_pipeline.py` (updated)

**Changes**:
- **OLTP queries**: `ef_search` increased from 50 → **200**
- **OLAP queries**: `ef_search` increased from 200 → **400**
- **Hybrid retrieval**: `ef_search` increased from 50 → **200**

**Research Basis**: Marqo V2 uses `ef_search=2000` by default. Qdrant research shows significant recall improvements with higher `ef_search` values.

**Expected Impact**: +5-10% Recall@10 improvement with minimal code changes.

**Note**: These are evaluation settings. For production, you may want lower values (50-100) for latency.

---

### 3. Increased Reranker Candidate Pool
**File**: `scripts/rag_pipeline.py` (updated)

**Changes**:
- **Dense-only with reranker**: Increased from `top_k * 2` (20) → **`top_k * 5` (50)**
- **Hybrid with reranker**: Already retrieving 50-100, kept as is

**Research Basis**: Cross-encoder rerankers perform significantly better with 50-200 candidates instead of 20.

**Expected Impact**: +3-5% MRR improvement from better reranking.

---

### 4. Hybrid Alpha Tuning Script
**File**: `scripts/tune_hybrid_alpha.py`

**Purpose**: Test different α values (BM25 vs dense weight) to find optimal balance.

**Usage**:
```bash
python scripts/tune_hybrid_alpha.py --qdrant-host ecetesla0 --num-queries 20
```

**What it does**:
- Tests α values: [0.3, 0.5, 0.7, 0.9]
- Measures MRR, Recall@10, NDCG@10 for each
- Recommends optimal α value

**Expected Impact**: +2-5% improvement if current α=0.7 is suboptimal.

---

## 📋 Next Steps

### Immediate Actions:
1. **Run ID verification**:
   ```bash
   python scripts/verify_id_alignment.py --qdrant-host ecetesla0
   ```
   This will tell us if data alignment is the issue.

2. **Re-run comprehensive evaluation** with new parameters:
   ```bash
   python scripts/comprehensive_evaluation.py --qdrant-host ecetesla0 --use-ground-truth --oltp-limit 50 --olap-limit 50
   ```
   Compare results to baseline to measure improvement.

3. **Tune hybrid alpha**:
   ```bash
   python scripts/tune_hybrid_alpha.py --qdrant-host ecetesla0
   ```
   Then update `rag_pipeline.py` with optimal α value.

---

## 🔄 Phase 2 (Medium Effort - Next)

### 5. Reciprocal Rank Fusion (RRF)
**Status**: Not yet implemented

**What**: Replace weighted score combination with RRF for hybrid retrieval.

**Expected Impact**: +2-4% improvement in hybrid retrieval quality.

### 6. Query Expansion
**Status**: Not yet implemented

**What**: Generate query variants using LLM, retrieve with all variants.

**Expected Impact**: +5-10% Recall@10, especially for keyword-heavy queries.

### 7. Semantic Chunking Improvements
**Status**: Not yet implemented

**What**: Implement paragraph/section-aware chunking instead of fixed-size.

**Expected Impact**: +3-5% improvement in retrieval precision.

---

## 📊 Expected Cumulative Impact

**Current Performance** (from comprehensive_eval.json):
- Best MRR: 0.443 (transformer_hybrid_rerank)
- Best Recall@10: 0.336 (transformer_dense)

**After Phase 1 Improvements**:
- Expected MRR: 0.50-0.55 (+15-25%)
- Expected Recall@10: 0.40-0.45 (+20-35%)

**After Phase 2 Improvements**:
- Expected MRR: 0.55-0.60 (+25-35%)
- Expected Recall@10: 0.45-0.50 (+35-50%)

---

## 📝 Files Modified

1. `scripts/rag_pipeline.py`:
   - Increased HNSW `ef_search` parameters
   - Increased reranker candidate pool

2. `scripts/verify_id_alignment.py` (NEW):
   - ID alignment verification tool

3. `scripts/tune_hybrid_alpha.py` (NEW):
   - Hybrid alpha parameter tuning tool

4. `docs/Results/RAG_RETRIEVAL_IMPROVEMENT_PLAN.md` (NEW):
   - Comprehensive improvement plan with research citations

---

## ⚠️ Important Notes

1. **HNSW ef_search**: The increased values (200/400) are for **evaluation**. For production with latency constraints, you may want to use lower values (50-100).

2. **Reranker candidate pool**: Larger pools (50-100) improve quality but increase latency. Monitor latency impact.

3. **ID Verification**: Run this first! If IDs are misaligned, other improvements won't help.

4. **Alpha Tuning**: Results depend on your specific data. Run the tuning script to find optimal α for your use case.

---

## 🚀 Quick Start

To test the improvements:

```bash
# 1. Verify data alignment
python scripts/verify_id_alignment.py --qdrant-host ecetesla0

# 2. Re-run evaluation with improved parameters
python scripts/comprehensive_evaluation.py --qdrant-host ecetesla0 --use-ground-truth --oltp-limit 50 --olap-limit 50 --output results/comprehensive_eval_improved.json

# 3. Compare results
python -c "import json; old=json.load(open('results/comprehensive_eval.json')); new=json.load(open('results/comprehensive_eval_improved.json')); print('Compare MRR and Recall@10 between files')"
```

