# Cohere API Integration - Usage Guide

This guide explains how to use Cohere API integration in your RAG pipeline.

## Overview

The RAG pipeline now supports Cohere API for:
- **Embeddings**: Using Cohere Embed API instead of local sentence-transformers
- **Reranking**: Using Cohere Rerank API instead of local cross-encoder rerankers

## Setup

### 1. Install Dependencies

```bash
pip install cohere>=4.0.0
```

### 2. Set API Key

**Option 1: Environment Variable (Recommended)**
```bash
export COHERE_API_KEY="your-api-key-here"
```

**Option 2: Command-Line Argument**
```bash
python scripts/rag_pipeline.py --use-cohere --cohere-api-key "your-api-key-here" ...
```

## Usage Examples

### Basic Usage with Cohere Embeddings

```bash
# Use Cohere for embeddings (still uses local rerankers)
python scripts/rag_pipeline.py \
    --query "What is retrieval augmented generation?" \
    --use-cohere \
    --qdrant-host ecetesla0 \
    --top-k 10
```

### Cohere Embeddings + Cohere Reranking

```bash
# Use Cohere for both embeddings and reranking
python scripts/rag_pipeline.py \
    --query "How does vector database improve search?" \
    --use-cohere \
    --use-reranker \
    --qdrant-host ecetesla0 \
    --top-k 5
```

### Custom Cohere Models

```bash
# Use specific Cohere models
python scripts/rag_pipeline.py \
    --query "What is machine learning?" \
    --use-cohere \
    --cohere-embed-model "embed-english-v3.0" \
    --cohere-rerank-model "rerank-english-v3.0" \
    --qdrant-host ecetesla0
```

### Interactive Mode with Cohere

```bash
python scripts/rag_pipeline.py \
    --interactive \
    --use-cohere \
    --qdrant-host ecetesla0
```

## How It Works

### When `--use-cohere` is enabled:

1. **Query Embedding**: Uses Cohere Embed API (`embed-english-v3.0`) instead of local `sentence-transformers`
2. **Qdrant Search**: Searches Qdrant using Cohere embeddings (your existing Qdrant collections work as-is)
3. **Reranking** (if `--use-reranker` is set): Uses Cohere Rerank API (`rerank-english-v3.0`) instead of local cross-encoders

### Fallback Behavior

- If Cohere API fails, the system automatically falls back to:
  - Local embeddings (sentence-transformers)
  - Local rerankers (cross-encoders)

## Comparison: Cohere API vs Local Models

| Feature | Local (Current) | Cohere API |
|---------|----------------|------------|
| **Embedding Model** | sentence-transformers/all-MiniLM-L6-v2 | embed-english-v3.0 |
| **Reranker** | Cross-encoder (TinyBERT, MiniLM) | rerank-english-v3.0 |
| **Latency** | ~943ms (retrieval) | ~1,100-1,400ms (estimated) |
| **Cost** | $0.0002/query | ~$0.001-0.01/query |
| **Control** | Full control | API-dependent |
| **Scalability** | 1.87× speedup with MPI | Limited by API rate limits |

## Integration with Existing Features

### Works with:
- ✅ Query classification (OLTP/OLAP routing)
- ✅ Hybrid retrieval (BM25 + dense) - Cohere replaces dense embeddings only
- ✅ Query expansion (LLM-based)
- ✅ LLM answer generation
- ✅ All evaluation scripts

### Note:
- When `--use-cohere` is enabled, hybrid retrieval still uses BM25 for keyword search, but uses Cohere embeddings for dense search
- Cohere reranking replaces local rerankers when `--use-reranker` is set

## Testing Cohere API

### Quick Test

```bash
# Test Cohere embeddings
python scripts/rag_pipeline.py \
    --query "What is RAG?" \
    --use-cohere \
    --qdrant-host ecetesla0 \
    --top-k 5 \
    --output-json > test_cohere.json
```

### Compare Performance

```bash
# Test 1: Local embeddings
python scripts/rag_pipeline.py \
    --query "What is RAG?" \
    --qdrant-host ecetesla0 \
    --top-k 5 \
    --output-json > test_local.json

# Test 2: Cohere API
python scripts/rag_pipeline.py \
    --query "What is RAG?" \
    --use-cohere \
    --qdrant-host ecetesla0 \
    --top-k 5 \
    --output-json > test_cohere.json

# Compare results
diff test_local.json test_cohere.json
```

## Cost Estimation

### Cohere API Pricing (Approximate)

- **Embed API**: ~$0.001 per 1K tokens
- **Rerank API**: ~$0.001-0.01 per 1K tokens

### Example Cost Calculation

For 1,000 queries/day:
- Average query: 20 tokens
- Average documents retrieved: 10 documents × 500 tokens = 5,000 tokens

**Daily Cost:**
- Embedding: 1,000 queries × 20 tokens / 1,000 × $0.001 = $0.02
- Reranking: 1,000 queries × 5,000 tokens / 1,000 × $0.001 = $5.00
- **Total: ~$5.02/day (~$150/month)**

**Your Current System:**
- ~$0.0002/query × 1,000 = **$0.20/day (~$6/month)**

## Recommendations

### Use Cohere API when:
- ✅ Quality is more important than latency
- ✅ Small-scale application (<10K queries/day)
- ✅ You want managed service (no infrastructure)
- ✅ Development/prototyping phase
- ✅ Multilingual support needed

### Stick with Local Models when:
- ✅ Latency is critical (<100ms requirement)
- ✅ Cost optimization is important
- ✅ Large-scale application (>100K queries/day)
- ✅ You need full control over indexing
- ✅ You want to leverage distributed processing (MPI)

## Troubleshooting

### Error: "Cohere API key required"

**Solution:** Set `COHERE_API_KEY` environment variable or pass `--cohere-api-key`

```bash
export COHERE_API_KEY="your-key"
# or
python scripts/rag_pipeline.py --use-cohere --cohere-api-key "your-key" ...
```

### Error: "Cohere embedding failed"

**Solution:** The system automatically falls back to local embeddings. Check:
- API key is valid
- Internet connection
- Cohere API status

### Slow Performance

**Expected:** Cohere API adds ~200-500ms latency compared to local models due to:
- Network latency
- API processing time
- Rate limiting

## Code Integration

### Programmatic Usage

```python
from scripts.rag_pipeline import QueryAwareRAGPipeline

# Initialize with Cohere API
pipeline = QueryAwareRAGPipeline(
    qdrant_host="ecetesla0",
    use_cohere=True,
    cohere_api_key="your-api-key",  # or set COHERE_API_KEY env var
    cohere_embed_model="embed-english-v3.0",
    cohere_rerank_model="rerank-english-v3.0",
)

# Query with Cohere
result = pipeline.query(
    query="What is RAG?",
    top_k=10,
    use_reranker=True  # Uses Cohere Rerank API
)

print(f"Retrieved {len(result.retrieved_chunks)} chunks")
print(f"Using Cohere: {result.metadata.get('cohere_retrieval', False)}")
```

## Next Steps

1. **Test Cohere API** on a subset of queries (100 queries)
2. **Measure**:
   - Latency vs your current system
   - Quality improvement (Recall@10, MRR)
   - Cost per query
3. **Decide** if quality improvement justifies cost/latency trade-off
4. **Consider hybrid approach**: Use Cohere for complex queries, local for simple queries

## References

- [Cohere Python SDK](https://github.com/cohere-ai/cohere-python)
- [Cohere Embed API Docs](https://docs.cohere.com/reference/embed)
- [Cohere Rerank API Docs](https://docs.cohere.com/reference/rerank)


