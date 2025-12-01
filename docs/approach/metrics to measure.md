Based on the research and your work on **query-aware routing for scalable RAG systems**, here are the **key plots and visualizations** you should include in your presentation:

## **Core Retrieval & Ranking Performance Plots**

### 1. **MRR (Mean Reciprocal Rank) Comparison**
   - **Why**: This is the standard metric you've been tracking with your baseline vs. reranker models
   - **Format**: Bar chart comparing different retrieval strategies (e.g., "Baseline BM25", "Vector-only", "Reranker-enhanced")
   - **Datasets**: Side-by-side bars for MS MARCO (OLTP) vs. other datasets to show how routing impacts different corpus types

### 2. **Precision & Recall Curves**
   - **Why**: Shows trade-off between retrieval quality and latency
   - **Format**: Line plot with Precision@K (K=1,5,10,20) showing how retrieval quality drops at different cutoffs
   - **Insight**: Critical for demonstrating why your query-aware router matters—different query types need different retrieval depths

### 3. **Reranker Impact Analysis**
   - **Why**: Shows MRR gain/loss before/after reranking (you noted OLTP MRR dropped with the reranker)
   - **Format**: Heatmap or grouped bar chart showing:
     - MRR before reranker
     - MRR after reranker
     - Query routing decisions that led to that outcome

***

## **Scalability & Efficiency Plots**

### 4. **Throughput & Latency vs. Vector DB Size**
   - **Why**: Central to "scalable" narrative
   - **Format**: Dual-axis line chart showing:
     - X-axis: Vector DB size (corpus scale)
     - Y-axis left: Docs/second throughput
     - Y-axis right: Query latency (ms)
   - **Data**: From your ingestion profiling (load/embed/store times you tracked)

### 5. **Ingestion Performance Breakdown**
   - **Why**: Shows where time is spent during MPI-distributed ingestion
   - **Format**: Stacked bar chart with segments:
     - Embed time (embedding model inference)
     - Load time (data parsing)
     - Store time (vector DB insertion)
   - **Variation**: Show scaling across different MPI process counts (1, 4, 8 processes) to demonstrate parallelization efficiency

### 6. **Chunk Utilization Analysis**
   - **Why**: Shows which chunks/documents actually get used in responses
   - **Format**: Histogram of chunk utilization scores or heatmap showing:
     - X-axis: Queries
     - Y-axis: Chunks ranked by relevance
     - Color intensity: Whether chunk was used in final response

***

## **Query-Aware Routing Plots** (Your Novel Contribution)

### 7. **Query Routing Decision Distribution**
   - **Why**: Visualizes your core innovation
   - **Format**: Pie or stacked bar chart showing:
     - What % of queries were routed to which retrieval strategy (BM25, dense vector, hybrid)
     - Breakdown by query type/complexity
   - **Insight**: Demonstrates that routing optimization is learned, not arbitrary

### 8. **Query Complexity vs. Retrieval Strategy Performance**
   - **Why**: Shows why routing matters
   - **Format**: Scatter plot with trend lines:
     - X-axis: Query complexity metric (length, number of entities, etc.)
     - Y-axis: MRR
     - Color/marker: Retrieval strategy used
   - **Insight**: Easy queries get routed to fast strategies, complex queries to expensive rerankers

### 9. **Cost-Benefit of Routing Decisions**
   - **Why**: Balances accuracy vs. computational cost
   - **Format**: Bubble chart:
     - X-axis: Latency (ms)
     - Y-axis: MRR
     - Bubble size: Query frequency
     - Color: Routing strategy
   - **Insight**: Shows Pareto frontier—routing makes decisions that optimize for both speed and accuracy

***

## **Statistical & Comparative Plots**

### 10. **Statistical Significance Testing**
   - **Why**: Validates that improvements aren't noise
   - **Format**: Confidence interval plot:
     - Y-axis: Different routing strategies
     - X-axis: MRR with error bars (95% CI)
     - Points where CIs don't overlap = significant difference
   - **Data**: From multiple evaluation runs with different random seeds

### 11. **Per-Dataset Performance Summary**
   - **Why**: Shows generalization across corpus types
   - **Format**: Grid of small multiples:
     - Each cell = one dataset (MS MARCO OLTP, TREC, etc.)
     - Bar chart within: routing strategy performance
   - **Insight**: Demonstrates your method works across diverse retrieval tasks

### 12. **Retrieval vs. Generation Quality**
   - **Why**: Shows end-to-end RAG performance, not just retrieval
   - **Format**: Scatter plot:
     - X-axis: Retrieval MRR
     - Y-axis: Generation quality (ROUGE, BLEU, semantic similarity)
     - Shows that better retrieval = better generation (validates the pipeline)

***

## **Recommended Slide Structure**

1. **Motivation**: Plot showing scalability problem (throughput vs. corpus size with knee in curve)
2. **Baseline Performance**: MRR comparison across strategies
3. **Query-Aware Routing**: Routing decision distribution
4. **Impact of Routing**: Query complexity vs. performance (scatterplot showing your optimization)
5. **Scalability Results**: Throughput + latency breakdown across system sizes
6. **Statistical Validation**: Confidence intervals proving significance
7. **End-to-End Results**: Per-dataset performance grid

***

## **Implementation Tips for Your Presentation**

- **Use matplotlib + seaborn** or **Plotly** for interactive charts
- **Color code consistently**: One color per routing strategy across all charts
- **Add trend lines** to scatter plots for clarity
- **Include error bars/confidence intervals** to show statistical rigor
- **Normalize metrics** so they're comparable across datasets (e.g., MRR as % improvement over baseline)

Given your ingestion profiling and reranker evaluation work, you likely have the data for plots 1-6 and 10-12. The query-aware routing plots (7-9) are your key differentiators for the presentation—make sure they clearly show *why* routing decisions matter.

Would you like me to create a presentation template with placeholder charts for these, or help you generate any specific plot from your evaluation data?

[1](https://developers.llamaindex.ai/python/framework/optimizing/production_rag/)
[2](https://www.reddit.com/r/LangChain/comments/1e8oct1/rag_in_production_best_practices_for_robust_and/)
[3](https://falkordb.com/blog/advanced-rag/)
[4](https://www.domo.com/blog/a-complete-guide-to-retrieval-augmented-generation)
[5](https://thesequence.substack.com/p/guest-post-advanced-rag-techniques)
[6](https://promptengineering.org/optimizing-small-scale-rag-systems-techniques-for-efficient-data-retrieval-and-enhanced-performance/)
[7](https://galileo.ai/blog/top-metrics-to-monitor-and-improve-rag-performance)
[8](https://arxiv.org/abs/2502.01113)
[9](https://www.youtube.com/watch?v=dI_TmTW9S4c)
[10](https://www.rohan-paul.com/p/handling-graphs-and-charts-in-rag)
[11](https://www.evidentlyai.com/llm-guide/rag-evaluation)
[12](https://intuitionlabs.ai/articles/rag-performance-pharmaceutical-documents)
[13](https://www.cliffsnotes.com/study-notes/27781006)
[14](https://zilliz.com/blog/advanced-rag-techniques-bridging-text-and-visuals-for-accurate-responses)
[15](https://www.braintrust.dev/articles/best-rag-evaluation-tools)
[16](https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/)
[17](https://arxiv.org/html/2510.12323)
[18](https://neo4j.com/blog/developer/graphrag-field-guide-rag-patterns/)
[19](https://neo4j.com/blog/developer/rag-tutorial/)
[20](https://www.legionintel.com/blog/chat-metrics-for-enterprise-scale-rag)

You already cover a lot; what is missing is mostly finer-grained, routing-specific and ops-style metrics that will make the story sharper and more defensible.[1][2]

## Clarify success criteria

Make success criteria explicit per dimension, not just “improve cost, latency, stability”:

- Quality: e.g., “≥X% F1 / LLM-judge win-rate vs best baseline on OLTP and OLAP subsets.”[3][2]
- Cost: “≥Y% reduction in cost-per-correct-answer on OLTP at same quality; ≤Z% cost increase on OLAP for ≥Δ quality gain.”[2][4]
- Latency: “P95 latency for OLTP stays under A ms at 100 QPS; OLAP under B ms at 50 QPS.”[5][1]
- Stability: “RSS/RSC not worse than baseline for given reader; routing should not create new instability regimes.”[6][7]

These can be summarized in one slide as your “go/no-go” metrics.[4]

## Routing- and classifier-specific metrics

Right now you mention a classifier and oracle routing but not how you’ll measure routing quality itself. Add:

- Classifier performance: accuracy, precision/recall/F1 on OLTP vs OLAP labels; confusion matrix to show where it fails.[8][2]
- Impact of misrouting: quality and cost when:  
  - Forced OLTP-only, OLAP-only, oracle routing, learned routing.  
  - You can report “regret” = oracle quality – actual quality, and “cost gap” = actual cost – oracle cost.[9][2]
- Routing load share: % of queries sent to each pipeline (BM25+vector vs GraphRAG) and how that shifts under different traffic mixes.[3][9]

These answer the feedback about “standard baselines”, “oracle”, and whether query-aware routing is actually doing something non-trivial.[2][9]

## Per-stage RAG metrics (retrieval vs generation)

Most RAG guides stress evaluating both retrieval and generation, not just end-to-end.[1][2]

For retrieval:

- Recall@k, Precision@k, MRR/NDCG for each pipeline and query type.[10][2]
- k-sensitivity curves for OLTP vs OLAP (which you already use for RSS/RSC); those also justify your routing and pipeline design.[6][2]

For generation:

- Faithfulness/groundedness metrics (e.g., via LLM-as-judge or ragas-style scores) to ensure GraphRAG summaries don’t hallucinate.[11][8]
- Citation precision/recall (how often cited chunks actually support the answer), especially important for OLAP/graph summaries.[11][2]

End-to-end:

- Overall answer correctness/ROUGE/F1 or LLM win-rate vs a strong baseline per query type.[4][3]

## Cost and resource metrics that fit your “scalable” angle

You already track cost/query and latency percentiles; for a “scalable systems” course you can lean more into systems-style metrics.[5][1]

Add:

- Token-level cost decomposition:  
  - Indexing cost: tokens and dollars to build BM25+vector vs GraphRAG/TREX-style graphs.[12][9]
  - Query-time cost: prompt+completion tokens per query for OLTP vs OLAP pipelines.[13][12]
- Cost-per-correct-answer (you already mention this—make it central): plot against corpus size and QPS for each pipeline and router.[3][4]
- Performance-to-cost ratio (e.g., F1 per dollar or per 1k tokens) for each configuration.[14][4]
- Resource utilization under load: CPU/GPU utilization and saturation points at different QPS levels; helps argue about headroom and scalability.[14][2]

## Stability and robustness measurements

You already plan RSS and RSC; make sure they are broken down by query type and pipeline.[7][6]

Add:

- RSS/RSC per reader model and per query type (OLTP vs OLAP) so you can say “routing avoids unstable regimes for this reader.”[7][6]
- Sensitivity to traffic mix: vary OLTP:OLAP ratios (e.g., 80:20, 50:50, 20:80) and measure:  
  - Tail latency (P95/P99) per type.  
  - Cost-per-correct-answer per type and overall.[1][2]
- Degradation under overload: run beyond your target QPS and show where each configuration breaks (queueing, timeouts, severe RSS drops).[5][1]

## User- and ops-facing metrics

Finally, a few “production-y” measurements that many RAG eval guides recommend and that map nicely to your “stability” story.[15][5]

- Aborted / timed-out queries and error rate per pipeline and under high QPS.[2][1]
- Caching effectiveness: cache hit rate on query or retrieval-level caching and resulting cost/latency savings.[15][5]
- Simple human eval slice: for a small subset, ask humans or an LLM-as-judge which answers they prefer and whether they look stable as k or load changes.[16][17]

If you want, the next step could be to turn this into a concrete “metrics table” for your report and a minimal subset that you actually implement given course-time constraints.

[1](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/tutorials/ai-cookbook/evaluate-assess-performance)
[2](https://neptune.ai/blog/evaluating-rag-pipelines)
[3](https://labelyourdata.com/articles/llm-fine-tuning/rag-evaluation)
[4](https://futureagi.com/blogs/rag-evaluation-metrics-2025)
[5](https://www.getmaxim.ai/articles/rag-evaluation-a-complete-guide-for-2025/)
[6](https://arxiv.org/html/2403.09040v3)
[7](https://chatpaper.com/paper/165564)
[8](https://blog.takanabe.tokyo/en/2023/09/ee194398-aff2-4065-bf2a-1acbe7b081d5/)
[9](https://arxiv.org/html/2503.02922v1)
[10](https://weaviate.io/blog/rag-evaluation)
[11](https://docs.ragas.io/en/stable/concepts/metrics/overview/)
[12](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/graphrag-costs-explained-what-you-need-to-know/4207978)
[13](https://www.falkordb.com/blog/reduce-graphrag-indexing-costs/)
[14](https://galileo.ai/blog/top-metrics-to-monitor-and-improve-rag-performance)
[15](https://dev.to/debmckinney/rag-evaluation-metrics-a-practical-guide-for-measuring-retrieval-augmented-generation-with-maxim-ai-2dkj)
[16](https://www.statsig.com/perspectives/rag-evaluation-metrics-methods-benchmarks)
[17](https://www.microsoft.com/en-us/research/blog/benchmarkqed-automated-benchmarking-of-rag-systems/)
[18](https://www.investopedia.com/terms/r/residual-sum-of-squares.asp)
[19](https://arxiv.org/html/2511.02754)
[20](https://rss.org.uk/RSS/media/File-library/Conference/abstracts_booklet_rss_2018_online.pdf)