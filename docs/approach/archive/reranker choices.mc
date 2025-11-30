Based on testing so far, here's the recommendation:

## Final Reranker Configuration

| Pipeline | Reranker | Rationale |
|----------|----------|-----------|
| **OLTP** | **None** | Fast (11-17ms), TinyBERT adds 500ms for minimal gain |
| **OLAP** | **BGE-base** | Good quality, fits in quota (1.1GB vs 2.2GB for bge-large) |

## Why These Choices?

### OLTP: No Reranker
- Baseline latency: **17ms** ✅
- TinyBERT adds: **+500ms** for ~2% improvement
- Not worth it for simple factoid lookups

### OLAP: BGE-base (not BGE-large)
- **BGE-large** (2.2GB) - download kept timing out, disk quota issues
- **BGE-base** (1.1GB) - downloaded successfully, good quality
- Latency: ~1.1s (acceptable for OLAP <2s target)

## For Your Paper

Report:
> *"For OLTP queries, we use vector search without reranking (17ms latency). For OLAP queries, we apply BGE-base cross-encoder reranking (~1.1s latency) to improve recall on complex multi-hop queries."*

## Code Configuration

```python
# In your pipeline
if query_type == "oltp":
    reranker = None  # No reranking
else:  # olap
    reranker = OLAPReranker(model_name="bge-base")
```

This gives you a **differentiated approach** that you can justify with latency/quality tradeoffs in your ablation study.