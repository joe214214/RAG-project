SLIDE 1: Title Slide
Query-Aware Routing for Scalable RAG Systems
[Your Name(s)]
Department of Electrical and Computer Engineering
University of Waterloo
Waterloo, Ontario, Canada
{your.email}@uwaterloo.ca
ECE 750 - Scalable Computer System Design
January 2025
SLIDE 2: Background - RAG Systems
Retrieval-Augmented Generation (RAG) combines retrieval with language models
Problem: Current RAG systems treat all queries uniformly
Reality: Queries have vastly different complexity and information needs
Two Query Types:
OLTP (Factoid): "What year was Apple founded?" → Simple lookup
OLAP (Analytical): "Summarize AI investments" → Complex synthesis
Microsoft Research: OLAP queries cost 10× more than OLTP queries
SLIDE 3: Background - Query Types
OLTP-style queries (Factoid):
Require precise, factual answers
Single document lookup sufficient
Example: "What year was Apple founded?"
Characteristics: Short, keyword-focused, single answer
OLAP-style queries (Analytical):
Require information synthesis across multiple documents
Need broader context and reasoning
Example: "Summarize recent AI investments by major tech companies"
Characteristics: Complex, multi-faceted, requires aggregation
Challenge: Treating both identically wastes resources and hurts performance
SLIDE 4: Background - Vector Similarity Search
Vector embeddings: Numerical representations in d-dimensional space
Similarity functions:
Cosine similarity: $\cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\|\|\mathbf{B}\|}$
Euclidean distance: $\|\mathbf{A}-\mathbf{B}\| = \sqrt{\sum_{i=1}^{n}(A_i-B_i)^2}$
Problem: Top-k search in large vector datasets
Approximate Nearest Neighbor (ANN) search often suffices
Popular tools: FAISS, Qdrant, Milvus, Pinecone
SLIDE 5: Overview - Problem Statement
Research Question:
Can query-aware routing that classifies queries as OLTP-style or OLAP-style improve RAG system cost, latency, and stability at scale versus query-agnostic baselines?
Requirements for Production:
✅ High query processing throughput
✅ Low query processing latency (milliseconds)
✅ Robustness and scalability
✅ Cost efficiency
Challenge: Large-scale datasets require distributed processing and efficient indexing

SLIDE 6: Overview - System Architecture
Our query-aware RAG system consists of:
Query Classifier: Routes queries to appropriate pipelines (OLTP vs OLAP)
Multi-granular Chunking: Fine-grained (OLTP) and coarse-grained (OLAP) chunks
Hybrid Retrieval: BM25 sparse + dense vector search with RRF
Reranking: Cross-encoder models improve result quality
Distributed Ingestion: MPI-based parallel processing
Vector Database: Qdrant with HNSW indexing
Visual: System architecture diagram showing query flow

SLIDE 7: Contribution Summary
Pyramid: A distributed solution based on HNSW for similarity search
Our System: A scalable RAG solution with query-aware routing
Key Contributions:
✅ Query-aware routing: Classifies queries as OLTP/OLAP for optimized retrieval
✅ Distributed ingestion: MPI-based parallel processing achieves 1.87× speedup with 4 workers
✅ Multi-granular chunking: Separate fine/coarse chunks for different query types
✅ Hybrid retrieval: BM25 + dense vectors with Reciprocal Rank Fusion (RRF)
✅ Comprehensive evaluation: 7 configurations, multiple metrics, cost analysis
Results: 1M chunks ingested in ~7 minutes, 99% Recall@10, 12.7% cost reduction
S
LIDE 8: Background - HNSW (Hierarchical Navigable Small World)
HNSW proximity graph has multiple layers
Bottom layer (layer 0): Contains all items in dataset
Upper layers: Sampled uniformly from previous layer
Index construction: Each layer approximates k-nearest neighbor graph
Query processing: Graph walk starts at top layer, moves to best neighbor
Complexity: O(log n) search complexity
Visual: HNSW graph illustration (similar to Pyramid Fig. 1)

SLIDE 9: Methodology - Query-Aware Routing
Two Classifier Approaches:
Feature-Based Classifier:
Random forest using query features (length, keyword density)
Performance: 0.759 confidence, balanced routing (55% OLTP)
Latency: <1ms overhead
Transformer Classifier:
MiniLM-L12-H384-uncased (sentence transformers)
Performance: 0.991 confidence, balanced routing (50% OLTP)
Latency: ~10ms overhead
Key Finding: Transformer achieves near-perfect routing confidence
SLIDE 10: Methodology - Multi-Granular Chunking
OLTP Chunks (Fine-grained):
Size: 384-512 tokens
Granularity: Sentence-level
Purpose: Precision-focused retrieval
OLAP Chunks (Coarse-grained):
Size: Section-level, variable length
Granularity: Paragraph/section boundaries
Purpose: Context-rich retrieval
Hierarchical Structure: Parent-child relationships preserved
Advantage: Separate collections enable query-type-specific optimization
SLIDE 11: Methodology - Hybrid Retrieval
Components:
BM25 Sparse Retrieval: Full corpus indexing (105K+ chunks)
Dense Vector Search: sentence-transformers/all-MiniLM-L6-v2 (384-dim)
Reciprocal Rank Fusion (RRF): Combines rankings without score normalization
Performance Trade-offs:
Method	Latency	QPS	Quality (MRR)	Use Case
Dense-Only	55ms	82.3	0.963	High throughput
Hybrid	943ms	7	0.973	Better recall
Key Finding: Hybrid provides better recall but 8.5× latency overhead
SLIDE 12: Methodology - Distributed Ingestion
MPI-Based Parallel Processing:
Framework: Message Passing Interface (MPI)
Nodes: ecetesla1, ecetesla2, ecetesla4 (GPU-enabled)
Work Distribution: Round-robin document assignment
Processing Pipeline:
Load: Distributed dataset loading
Chunk: Multi-granular chunking (parallel)
Embed: GPU-accelerated embedding generation (batch size 32)
Store: Concurrent writes to Qdrant
Performance: ~1,900 chunks/sec throughput with 4 workers
SLIDE 13: Evaluation - Datasets
Name	# Documents	# Chunks	Purpose
MS MARCO	700,000	813,178	OLTP queries (factoid)
HotpotQA	50,000	~50,000	OLAP queries (analytical)
Test Queries:
MS MARCO: 26 queries with ground truth passage IDs
HotpotQA: 50 queries with Wikipedia title + sentence ID ground truth
Evaluation Method: N-gram similarity (character 3-grams) for robust matching
SLIDE 14: Evaluation - Scaling Results (1M Chunks)
Visual: results/plots/scaling_1M_plot.png (4-panel plot)
Workers	Time	Speedup	Efficiency	Throughput
1	786.5s (13.1 min)	1.00x	100%	1,034 chunks/s
2	795.6s (13.3 min)	0.99x	49.5%	1,022 chunks/s
4	421.4s (7.0 min)	1.87x	46.7%	1,930 chunks/s
Key Findings:
2 Workers: Minimal improvement (storage I/O bottleneck)
4 Workers: 1.87× speedup (significant scaling benefit)
Efficiency: 46.7% (load imbalance reduces parallel efficiency)
SLIDE 15: Evaluation - Time Breakdown
1 Worker:
Embed: 387.8s (49.3%)
Store: 386.1s (49.1%)
Load/Chunk: 12.7s (1.6%)
4 Workers (Max):
Embed: 271.2s (64.4%) ← Worker 1 bottleneck
Store: 143.4s (34.0%)
Load/Chunk: 6.8s (1.6%)
Load Imbalance:
Worker 0: 98.6s embedding (fastest)
Worker 1: 271.2s embedding (2.75× slower)
Root Causes: Uneven work distribution, GPU performance differences, storage contention
SLIDE 16: Evaluation - Load Testing Results
Visual: results/plots/best/report_comprehensive_load_testing.png
Baseline (Dense-Only):
Peak QPS: 82.3 at concurrency=10
P95 Latency: 19ms → 878ms (46× increase)
Bottleneck: Qdrant I/O saturates beyond optimal concurrency
Hybrid Retrieval:
Peak QPS: ~7 (11.7× slower)
P95 Latency: 300ms → 30,671ms
Trade-off: Better recall vs higher latency
Key Insight: Storage bottleneck limits concurrency beyond 10 queries
SLIDE 17: Evaluation - Quality Metrics
Best Configuration: transformer_hybrid_rerank
Metric	Value
MRR	0.973
Recall@10	0.990 (99%)
NDCG@10	0.968
Latency	302.7ms
Confidence	0.9999
Fastest High-Quality: transformer_dense
Latency: 54.1ms
MRR: 0.963, Recall@10: 0.985
Visual: results/plots/best/report_evaluation_metrics.png
SLIDE 18: Evaluation - Cost Analysis
Visual: results/plots/best/cost_evaluation.png
Configuration	Cost (50 queries)	Cost/Query	Reduction
Baseline	$0.0111	$0.0002	-
Best	$0.0096	$0.0002	12.7%
Comparison with Microsoft Benchmarks:
Microsoft: $0.02-$0.50 per query
Our System: ~$0.0002 per query (100× lower)
Reason: Using gpt-4o-mini (cheaper model) + optimized routing
SLIDE 19: Evaluation - Comparison Summary
Scaling Achievements:
✅ 1M chunks ingested in ~7 minutes (1.87× speedup)
✅ ~1,900 chunks/sec throughput with 4 workers
✅ 100% success rate (no data loss)
Quality Achievements:
✅ 99% Recall@10 with best configuration
✅ MRR 0.973 (excellent ranking quality)
✅ 12.7% cost reduction vs baseline
Limitations:
⚠️ 46.7% efficiency (load imbalance)
⚠️ Storage bottleneck (single-node Qdrant)
SLIDE 20: Bottleneck Analysis
Primary Bottlenecks:
Storage I/O (Qdrant):
Single-node deployment limits scaling
Concurrent writes saturate network/disk bandwidth
Solution: Distributed Qdrant cluster
Load Imbalance:
Worker 1 consistently 2.4-2.75× slower
Uneven work distribution
Solution: Dynamic load balancing (work stealing)
Network Contention:
Sequential Qdrant operations
No connection pooling
Solution: Batch operations, connection pooling
SLIDE 21: Conclusions
Key Achievements:
✅ Scalability: 1.87× speedup with 4 workers (production-scale)
✅ Quality: 99% Recall@10 with transformer classifier
✅ Cost: 12.7% reduction vs baseline
✅ Performance: 82 QPS baseline, 54ms best latency
Key Contributions:
Query-aware routing improves cost and performance
Distributed ingestion demonstrates scalability
Comprehensive evaluation with bottleneck identification
Future Work:
Distributed Qdrant setup
Dynamic load balancing
Better ID alignment for evaluation
SLIDE 22: Questions?
Thank you!
Contact: {your.email}@uwaterloo.ca
Repository: Available upon request
Presentation Flow Summary
Total Slides: 22 slides
Estimated Time: 15-20 minutes + 3-minute Q&A
Flow Structure (Following Pyramid Paper Format):
Title (Slide 1)
Background (Slides 2-4): RAG systems, query types, vector search
Overview (Slides 5-6): Problem statement, system architecture
Contribution (Slide 7): Key contributions summary
Technical Background (Slide 8): HNSW explanation
Methodology (Slides 9-12): Routing, chunking, retrieval, ingestion
Evaluation (Slides 13-19): Datasets, scaling, load testing, quality, cost
Bottleneck Analysis (Slide 20): System limitations
Conclusions (Slide 21): Achievements and future work
Questions (Slide 22)
Key Visuals to Include:

**Architecture Diagrams (Mermaid code available in SLIDES_VISUALS.md):**
1. System Architecture Diagram (Slide 6)
   - Complete query flow: Input → Classifier → OLTP/OLAP pipelines → Qdrant → Output
   - Shows embedding options (Local/Cohere API) and reranking options
   - Mermaid code: Section 1 in SLIDES_VISUALS.md

2. Distributed Ingestion Architecture (Slide 12)
   - MPI worker nodes (ecetesla1, ecetesla2, ecetesla4)
   - Parallel processing flow: Load → Chunk → Embed → Store
   - Shows load imbalance and bottleneck
   - Mermaid code: Section 2 in SLIDES_VISUALS.md

3. HNSW Graph Illustration (Slide 8)
   - Multi-layer graph structure with query walk path
   - Mermaid code: Section 3 in SLIDES_VISUALS.md

4. Query Routing Flow Diagram (Slide 9)
   - Classifier decision tree showing Feature-based vs Transformer
   - Routing percentages and confidence scores
   - Mermaid code: Section 4 in SLIDES_VISUALS.md

5. Hybrid Retrieval Architecture (Slide 11)
   - BM25 + Dense paths with RRF combination
   - Local vs Cohere API options
   - Mermaid code: Section 5 in SLIDES_VISUALS.md

6. Cohere API vs Local Models Comparison (Slide 17B or update Slide 17)
   - Side-by-side metrics comparison
   - Mermaid code: Section 6 in SLIDES_VISUALS.md

7. Scaling Results Visualization (Slide 14)
   - Worker comparison showing 1, 2, and 4 worker configurations
   - Mermaid code: Section 7 in SLIDES_VISUALS.md

**Evaluation Plots:**
8. results/plots/scaling_1M_plot.png (Slide 14)
9. results/plots/best/report_comprehensive_load_testing.png (Slide 16)
10. results/plots/best/report_evaluation_metrics.png (Slide 17)
11. results/plots/best/cost_evaluation.png (Slide 18)
12. results/plots/best/cohere_vs_local_comparison.png (Slide 17B or update Slide 17)

**How to Use Mermaid Diagrams:**
- Option 1: Use online editor (https://mermaid.live/) to render and export as PNG/SVG
- Option 2: Use Mermaid plugins in Markdown viewers (GitHub, GitLab, etc.)
- Option 3: Use Mermaid CLI: `mmdc -i diagram.mmd -o diagram.png`
- Option 4: Include directly in HTML presentations with Mermaid.js library

**Note:** All Mermaid diagram code is available in `docs/SLIDES_VISUALS.md` for easy copy-paste and customization.

This structure follows the Pyramid paper format while highlighting your RAG project's contributions and results.
