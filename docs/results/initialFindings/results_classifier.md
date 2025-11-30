Excellent results! Here's the **comparison summary**:

## Classifier Evaluation Results

| Model | Accuracy | Macro F1 | Latency | Edge Cases | Hard Cases |
|-------|----------|----------|---------|------------|------------|
| **DistilBERT** | **94.7%** | **0.947** | **3.6ms** ⚡ | 53.8% | 33.3% |
| BERT-base | 94.7% | 0.947 | 5.9ms | 53.8% | 33.3% |
| MiniLM-L12 | 92.9% | 0.929 | 6.5ms | 38.5% | 0.0% |

## Key Insights

1. **Standard queries (MS MARCO + HotpotQA): 100% accuracy** across all models
2. **Edge cases expose weaknesses** - all models struggle with ambiguous queries
3. **DistilBERT wins** - same accuracy as BERT-base but **40% faster** (3.6ms vs 5.9ms)

## Commonly Misclassified (OLAP→OLTP)
- "Compare Tesla and Ford profits..." — comparison looks factoid
- "Summarize Apple's strategy..." — synthesis without multi-hop keywords
- "Is Paris bigger than London?" — comparison without explicit analysis words
- "What is the average salary at Google?" — aggregation looks like lookup

## Recommendation

**Use DistilBERT** for production:
- Best speed/accuracy tradeoff
- 3.6ms latency is excellent for real-time routing
- Same F1 as BERT-base at nearly half the latency

## Next Steps

| Task | Status |
|------|--------|
| ✅ Classifier trained | Complete |
| ✅ Classifier evaluated | Complete |
| ⏳ LLM integration | Pending |
| ⏳ Scaling experiments | Pending |
| ⏳ Final report | Pending |

Should I update the todos and proceed with **LLM integration** (OpenAI/Azure API)?