Query-Aware Routing for Scalable RAG Systems
 Omar Ankit Yifan Zhou Ruodie Yu Runpeng Gu Cheng Fang
 ECE 750- Scalable Computer System Design
 University of Waterloo
 1 Problem and Research Question
 Can query-aware routing that classifies queries as OLTP-style (factoid) or OLAP-style (synthetic)
 improve RAG system cost, latency, and stability at scale versus query-agnostic baselines?
 Current RAG systems treat all queries the same. In practice, OLTP-style (factoid) queries are simple lookups
 (“What year was Apple founded?”), while OLAP-style (synthetic) queries require combining information from
 many documents (“Summarize recent AI investments by major tech companies”). Microsoft shows OLAP queries
 cost 10× more [3]. Treating both the same wastes resources and hurts complex queries.
 2 Solution Design
 We propose a query-aware RAG system:
 (1) Query Classifier: A lightweight AI (DistilBERT) predicts if each question is simple or complex based on syntax
 and meaning.
 (2) Dual Retrieval Pipelines:- For OLTP: Uses hybrid BM25+embeddings—combining keyword (BM25) and
 meaning-based (embeddings) search—with a fast HNSW index to quickly fetch the top k = 5–10 matches.- For
 OLAP:Builds an entity graph of concepts found in documents, groups them using Leiden community detection,
 and provides layered (hierarchical) summaries so the system can answer complex questions by aggregating related
 information.
 (3) Policy Layer: Dynamically adjusts search settings (e.g., k, rerank depth, hybrid weights) based on query type
 and model robustness [1].
 3 Methodology and Scalability
 Datasets: HotpotQA (5.4K questions), MS MARCO (6.9K queries), Earnings transcripts (1.7M tokens). Scale from
 106 to 108 document chunks. Metrics: Quality (F1, LLM-as-judge), Efficiency (P50/P95/P99 latency, cost/query),
 Stability (RSS, RSC from RAGGED framework). Baselines: Vector RAG (fixed k = 10), GraphRAG (all queries),
 TREX[3], oracle routing. Scalability Requirements: Target 50–1000 QPS, measure tail latency under load, track
 cost-per-correct-answer across corpus sizes.
 4 Literature Review
 RAGGED identifies reader robustness as the key RAG determinant—some models (FLAN-T5, RSS=0.99) benefit
 from large k, others (LLaMA-2, RSS=0.7) degrade [1]. GraphRAG achieves 72–83% win rate on OLAP queries but
 costs $389/corpus vs. $36 for TREX [2]. REBEL demonstrates multi-criteria reranking (relevance+diversity+coverage)
 enables 15–25% quality gains via adaptive inference-time compute [4]. Wang et al. show hybrid search (α = 0.3) and
 monoT5 reranking improve performance but evaluate at small scale (500 examples) [5]. Gap: No work combines
 query-aware routing with specialized pipelines optimized per query type.
 5 Preliminary Evidence
 Microsoft data: OLAP queries cost $0.20–$0.50 vs. $0.02–$0.05 for OLTP [3]. Reader RSS varies 4×: FLAN-T5
 (0.99), GPT-3.5 (0.99), LLaMA-2 (0.95), Claude (0.70) [1]. Manual analysis of 100 Natural Questions: 43% OLTP,
 57% OLAP (Cohen’s κ = 0.82). GraphRAG requires 26–33% fewer tokens than source text summarization at
 scale [2].
 6 Work Plan
 • Deploy vector database (Milvus), implement BM25 and Cohere hybrid search, and ingest datasets.
 • Build the GraphRAG OLAP pipeline including entity extraction, community detection with Leiden clustering,
 and document summarization.
 • Train a DistilBERT query classifier using labeled data (derived from quality vs. k retrieval depth curves).
 • Integrate routing logic to direct queries to appropriate pipelines, and run A/B tests vs. standard baselines.
 1
• Perform evaluation across document scales and query rates; compute trade-off curves for quality, cost, and
 latency.
 7 Limitations and Risks
 Classifier errors (10–15% misclassification) mitigated by routing ambiguous queries to OLAP (quality bias) and
 reporting oracle upper bound. OLAP complexity addressed via dynamic throttling under load and cost-per-answer
 tracking. Router latency (<50ms) minimized using distilled model and query caching. Dataset bias mitigated
 by testing diverse domains (finance, general knowledge).
 8 Expected Deliverables
 • Empirical evaluation of query-aware routing for RAG, measuring cost reduction and accuracy trade-offs when
 classifying and routing queries as OLTP (factoid) or OLAP (synthetic).
 • Open-source implementation of a modular OLTP/OLAP RAG architecture for further research and practical
 use.
 • Benchmarkdataset and code for evaluating query classifiers and routing effectiveness, supporting reproducibility
 and comparison.
 • Analysis of cost, latency, and accuracy trade-offs across query types, models, and system settings to inform
 future RAG design and deployment.
 • Addresses a practical gap: providing evidence and reusable tools for matching retrieval complexity to query
 requirements in large-scale systems.
 References
 [1] J. Hsia, A. Shaikh, Z. Wang, and G. Neubig, “RAGGED: Towards informed design of retrieval augmented
 generation systems,” arXiv preprint arXiv:2403.09040, March 2024.
 [2] D. Edge, H. Trinh, N. Cheng, J. Bradley, A. Chao, A. Mody, S. Truitt, and J. Larson, “From local to global: A
 graph RAG approach to query-focused summarization,” arXiv preprint arXiv:2404.16130, April 2024.
 [3] J. Cahoon, P. Singh, H. Trinh, F. Psallidas, and C. Curino, “Optimizing open-domain question answering with
 graph-based retrieval augmented generation,” arXiv preprint arXiv:2503.02922, March 2025.
 [4] W. LeVine and B. Varjavand, “Relevance isn’t all you need: Scaling RAG systems with inference-time compute
 via multi-criteria reranking,” arXiv preprint arXiv:2504.07104, Published at ICLR 2025, March 2025.
 [5] X. Wang, Z. Wang, X. Gao, F. Zhang, Y. Wu, Z. Xu, T. Shi, Z. Wang, S. Li, Q. Qian, R. Yin, C. Lv, X. Zheng,
 and X. Huang, “Searching for best practices in retrieval-augmented generation,” arXiv preprint arXiv:2407.01219,
 July 2024.
 [6] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. K¨uttler, M. Lewis, W. Yih, T. Rockt¨aschel,
 S. Riedel, and D. Kiela, “Retrieval-augmented generation for knowledge-intensive NLP tasks,” Advances in Neural
 Information Processing Systems, vol. 33, pp. 9459–9474, 2020.
 [7] W. Cohen, H. Sun, R. A. Hofer, and M. Siegler, “Scalable neural methods for reasoning with a symbolic knowledge
 base,” arXiv preprint arXiv:2002.06115, Published at ICLR 2020, February 2020.
 [8] U. Oche and F. Adepoju, “A systematic review of key retrieval-augmented generation challenges and prospects,”
 arXiv preprint arXiv:2507.18910, July 2025.
 2

 feedback:

 Overall Feedback
Project proposal:

1. Interesting problem.

2. "Dynamically adjusts search settings" - this is learning based or heuristics only?

3. "Gap: No work combines
query-aware routing with specialized pipelines optimized per query type." - is this an open problem in the RAG community? What do you mean by "pipelines"?

4. " implement BM25 and Cohere hybrid search" - or set up? After all you are using an existing vector database.

5. "Train a DistilBERT query classifier using labeled data " - how long this will take? Can't you use a pretrained model?

6. "standard baselines" - make it clear.

7. Note that you have limited time. The research objectives needs to be feasible and not overly ambitious. I would try to limit the scope of expected contributions.

8. What are the success criteria (e.g., accuracy/error rate, latency/throughput etc.) 