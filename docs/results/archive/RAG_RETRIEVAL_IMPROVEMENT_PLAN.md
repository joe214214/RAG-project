# RAG Retrieval Improvement Plan

Based on current evaluation results and research into advanced RAG techniques, this document outlines actionable improvements to boost retrieval quality (MRR, Recall@k, NDCG@k).

## Current Performance Summary

- **Best Configuration**: `transformer_hybrid_rerank` (MRR: 0.443, Recall@10: 0.334)
- **Key Issue**: Only 41.6% of queries find relevant chunks, with 29.2% coverage of relevant chunks
- **Transformer classifier** significantly outperforms feature-based (+41% MRR)
- **Hybrid retrieval** shows minimal improvement over dense-only

---

## Priority 1: Data Alignment & Verification (Critical)

### 1.1 Verify ID Matching
**Problem**: Low recall (28-34%) suggests potential ID misalignment between ground truth and Qdrant.

**Actions**:
```python
# Create verification script
# 1. Sample 10 queries with mrr=0 and num_relevant_total=1
# 2. For each, fetch the relevant passage ID from ground truth
# 3. Query Qdrant for that exact ID
# 4. Compare:
#    - Does the passage exist in Qdrant?
#    - Does the ID format match (msmarco_passage_123 vs 123)?
#    - Is the text content the same?
```

**Expected Impact**: Could explain 20-30% of failures if IDs are misaligned.

### 1.2 Check Chunking Alignment
**Problem**: Documents may be chunked differently than expected by ground truth.

**Actions**:
- Verify that MS MARCO passages are stored as single chunks (not split)
- Verify HotpotQA contexts preserve title boundaries
- Check if `original_passage_id`/`original_title` are correctly stored in Qdrant payloads

---

## Priority 2: HNSW Parameter Tuning (Quick Win)

### 2.1 Increase `ef_search` for Evaluation
**Current**: Using default `ef=50` for OLTP, `ef=200` for OLAP

**Research Finding**: Higher `ef_search` significantly improves recall. Marqo V2 uses `ef_search=2000` by default. Qdrant research shows `ef_search` in range 50-200 is typical, but can go higher for better recall.

**Actions**:
```python
# In rag_pipeline.py, update search_params:
# For evaluation (comprehensive_evaluation.py):
search_params = models.SearchParams(hnsw_ef=200, exact=False)  # OLTP: 50→200
search_params = models.SearchParams(hnsw_ef=400, exact=False)  # OLAP: 200→400

# For production, keep lower values for latency
```

**Expected Impact**: +5-10% Recall@10 improvement with minimal code changes.

### 2.2 Tune `ef_construct` During Indexing
**Current**: Using Qdrant defaults

**Research Finding**: Higher `ef_construct` builds better graph structure, enabling better recall even with lower `ef_search` at query time.

**Actions**:
- Re-index collections with `ef_construct=200-400` (current default is ~100)
- Trade-off: Slower indexing, but better long-term recall

---

## Priority 3: Retrieval Hyperparameter Tuning

### 3.1 Increase Retrieval `k` for Reranking
**Current**: Retrieving `top_k * 2` (20) before reranking to 10

**Research Finding**: Rerankers perform better with larger candidate pools. Cross-encoders benefit from 50-200 candidates.

**Actions**:
```python
# In rag_pipeline.py _retrieve_dense_only and _retrieve_hybrid:
# When use_reranker=True:
limit = top_k * 5  # Instead of top_k * 2 (50 candidates for reranking)

# After reranking, return top_k
```

**Expected Impact**: +3-5% MRR improvement from better reranking.

### 3.2 Tune Hybrid Weight α
**Current**: Fixed `α=0.7` (70% dense, 30% BM25)

**Research Finding**: Optimal α varies by dataset. Should be tuned empirically.

**Actions**:
```python
# Run ablation: α ∈ {0.3, 0.5, 0.7, 0.9}
# Measure MRR, Recall@10 for each
# Select best α per query type (OLTP vs OLAP)
```

**Expected Impact**: +2-5% improvement if current α is suboptimal.

### 3.3 Evaluate at Higher k
**Current**: Evaluating at k=10

**Actions**:
- Run evaluation with k=20, k=30 to see if recall improves
- Helps identify if issue is ranking (low MRR) vs retrieval (low recall)

---

## Priority 4: Advanced Reranking Techniques

### 4.1 Reciprocal Rank Fusion (RRF)
**Research Finding**: RRF combines rankings from multiple retrieval methods without score normalization issues.

**Current**: Using weighted score combination for hybrid

**Actions**:
```python
# Implement RRF for hybrid retrieval:
# 1. Get BM25 results (ranked list)
# 2. Get dense results (ranked list)
# 3. Apply RRF: score = 1/(rank + 60) for each method
# 4. Sum scores, re-rank
```

**Expected Impact**: +2-4% improvement in hybrid retrieval quality.

### 4.2 Larger Reranker Candidate Pool
**Current**: Reranking top 20

**Actions**:
- Increase to 50-100 candidates before reranking
- Use faster reranker (TinyBERT) for initial filtering, then MiniLM for final ranking

**Expected Impact**: +3-5% MRR improvement.

---

## Priority 5: Query Optimization

### 5.1 Query Expansion
**Research Finding**: Expanding queries with synonyms/related terms improves recall, especially for BM25.

**Actions**:
```python
# Use LLM to generate 2-3 query variants:
# Original: "machine learning frameworks"
# Expanded: ["ML libraries", "AI development tools", "deep learning platforms"]
# Retrieve with all variants, merge results
```

**Expected Impact**: +5-10% Recall@10, especially for keyword-heavy queries.

### 5.2 Query Reformulation
**Research Finding**: RQ-RAG framework shows query rewriting improves retrieval for ambiguous queries.

**Actions**:
- Use LLM to rewrite queries: clarify terms, decompose complex queries
- Particularly effective for multi-hop questions

**Expected Impact**: +3-7% improvement on complex queries.

---

## Priority 6: Chunking Optimization

### 6.1 Semantic Chunking
**Current**: Using fixed-size chunking (384-512 tokens)

**Research Finding**: Semantic chunking preserves coherence and improves embedding quality.

**Actions**:
- Implement paragraph/section-aware chunking
- Use sentence transformers to identify semantic boundaries
- Preserve context at chunk boundaries

**Expected Impact**: +3-5% improvement in retrieval precision.

### 6.2 Contextual Retrieval
**Research Finding**: Anthropic's contextual retrieval prepends explanatory context to chunks before embedding, reducing failure rate by 35-49%.

**Actions**:
```python
# For each chunk, generate context:
# "In [document title], section [section name], discussing [topic]:"
# Prepend to chunk before embedding and BM25 indexing
```

**Expected Impact**: Significant improvement in retrieval accuracy (+10-15% recall).

---

## Priority 7: Embedding Model Improvements

### 7.1 Domain Fine-Tuning
**Research Finding**: Fine-tuning embedding models on domain data improves Recall@10 by 10-15 percentage points.

**Actions**:
- Collect query-document pairs from MS MARCO/HotpotQA
- Fine-tune `all-MiniLM-L6-v2` using contrastive learning
- Use synthetic data generation if needed

**Expected Impact**: +10-15% Recall@10 improvement (largest potential gain).

### 7.2 Model Upgrade
**Current**: `all-MiniLM-L6-v2` (384 dim)

**Alternatives**:
- `BGE-base-en-v1.5` (768 dim, better performance)
- `sentence-transformers/all-mpnet-base-v2` (768 dim)
- Domain-specific models if available

**Expected Impact**: +5-10% improvement, but requires re-indexing.

---

## Priority 8: Multi-Stage Retrieval

### 8.1 Hierarchical Retrieval (RAPTOR)
**Research Finding**: RAPTOR creates multi-level document summaries, improving retrieval for complex queries by 20%.

**Actions**:
- Implement recursive clustering and summarization
- Retrieve at multiple abstraction levels
- Combine results

**Expected Impact**: Significant improvement for multi-hop queries (+15-20%).

### 8.2 Parent Document Retrieval
**Research Finding**: Retrieve fine-grained chunks but return parent documents for context.

**Actions**:
- Chunk at sentence level for retrieval
- Return paragraph/section-level chunks to LLM
- Preserves context while maintaining precision

**Expected Impact**: +5-8% improvement in answer quality.

---

## Priority 9: Analysis by Query Type

### 9.1 Separate OLTP/OLAP Analysis
**Current**: Aggregating metrics across query types

**Actions**:
```python
# In comprehensive_evaluation.py, add:
# - Average MRR per query_type per config
# - Identify which query type is underperforming
# - Tune retrieval separately for OLTP vs OLAP
```

**Expected Impact**: Better understanding of failure modes.

### 9.2 Error Analysis
**Actions**:
- Identify queries with `mrr=0` but high `top_scores`
- Manually inspect: near-misses vs completely off-topic
- Adjust strategy based on error type

---

## Priority 10: Advanced Techniques (If Needed)

### 10.1 Late Chunking
**Research Finding**: Embed full documents first, then chunk, preserving document-level context in embeddings.

**Actions**:
- Use long-context embedding models (8K+ tokens)
- Embed full documents, derive chunk embeddings from full-doc embeddings

**Expected Impact**: +5-10% improvement, especially for long documents.

### 10.2 Adaptive Retrieval
**Research Finding**: TARG (Training-Free Adaptive Retrieval Gating) skips retrieval when model is confident, reducing latency by 70-90%.

**Actions**:
- Analyze model confidence before retrieval
- Skip retrieval for simple factual queries
- Use multiple retrieval rounds for complex queries

**Expected Impact**: Efficiency gains, but may not improve correctness metrics.

---

## Implementation Roadmap

### Phase 1 (Quick Wins - 1-2 days):
1. ✅ Verify ID alignment (critical)
2. ✅ Increase `ef_search` for evaluation
3. ✅ Increase reranker candidate pool (20→50)
4. ✅ Tune hybrid α parameter

### Phase 2 (Medium Effort - 3-5 days):
5. ✅ Implement RRF for hybrid retrieval
6. ✅ Query expansion for BM25
7. ✅ Semantic chunking improvements
8. ✅ Separate OLTP/OLAP analysis

### Phase 3 (Higher Effort - 1-2 weeks):
9. ✅ Contextual retrieval (prepend context to chunks)
10. ✅ Embedding model fine-tuning
11. ✅ Hierarchical retrieval (RAPTOR)

### Phase 4 (Research - Future):
12. ✅ Late chunking
13. ✅ Adaptive retrieval gating
14. ✅ Model architecture upgrades

---

## Expected Cumulative Impact

If all Phase 1-2 improvements are implemented:
- **Current**: MRR=0.443, Recall@10=0.334
- **Target**: MRR=0.55-0.60, Recall@10=0.45-0.50
- **Improvement**: +25-35% in both metrics

If Phase 3 improvements are added:
- **Target**: MRR=0.65-0.70, Recall@10=0.55-0.60
- **Improvement**: +50-60% over baseline

---

## Key Research Insights

1. **HNSW Tuning**: `ef_search` is the most impactful parameter for recall. Can increase from 50 to 200-400 for evaluation.

2. **Reranking**: Larger candidate pools (50-100) significantly improve reranker effectiveness.

3. **Contextual Retrieval**: Prepending context to chunks is a simple but highly effective technique (+35-49% failure reduction).

4. **Query Expansion**: Particularly effective for keyword-based retrieval, can improve recall by 5-10%.

5. **Fine-Tuning**: Domain-specific fine-tuning provides the largest gains (+10-15% Recall@10).

6. **Multi-Stage Retrieval**: Hierarchical approaches (RAPTOR) excel at complex, multi-hop queries.

---

## References

- Perplexity Research: Advanced RAG retrieval techniques
- Qdrant Documentation: HNSW parameter tuning
- Anthropic: Contextual Retrieval
- Marqo: HNSW recall optimization
- RAPTOR: Hierarchical document retrieval
- RQ-RAG: Query reformulation framework

