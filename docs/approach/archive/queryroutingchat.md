Perfect! Now I have comprehensive information. Let me create a detailed summary of the papers mentioned, the query classifier approach, and analysis of the gap claim.

## Analysis of the Query-Aware Routing for Scalable RAG Systems Paper Review

### **Core Research Proposal Summary**

The paper proposes a query-aware routing architecture for RAG systems that classifies queries as **OLTP-style (factoid/transactional)** or **OLAP-style (analytical/aggregation)** and routes them to specialized retrieval pipelines. The system uses **DistilBERT** as a lightweight classifier combined with a dynamic policy layer that adjusts retrieval parameters (k-matches, rerank depth, hybrid weighting) based on query complexity and model confidence.[1]

***

### **Key Papers Cited and Related Work**

#### **1. QA-Dragon (Aug 2025) - The Closest Competitor**
**Most Similar Approach to Your Proposal**

QA-Dragon presents a query-aware dynamic RAG system for multimodal visual question answering. It introduces two core components: a domain router (identifying subject domains like medical, sports) and a search router (dynamically selecting optimal retrieval strategies).[2]

Key overlap with your work:
- ✅ All three core elements: query classification + dynamic routing + specialized retrieval strategies
- ✅ Addresses query heterogeneity with tailored approaches

Key differences:
- Focuses on **multimodal VQA** rather than text-only RAG
- Uses domain classification (categorical) vs. your **complexity-based OLTP/OLAP distinction**
- No explicit mention of cost optimization or the documented 10x efficiency gap
- No entity graph construction or hierarchical summarization for complex queries

**Assessment**: QA-Dragon is the most relevant existing work but operates in a different domain and lacks the efficiency-focused, database-inspired taxonomy that defines your contribution.

#### **2. Adaptive-RAG - Complexity-Based Routing Precedent**
Adaptive-RAG already demonstrates complexity-based routing across heterogeneous pipelines, routing queries to: LLM-only, single-step retrieval, or multi-step iterative retrieval based on complexity. This represents an important precedent that the review flags as needing explicit differentiation in your work. The core distinction is that Adaptive-RAG uses iterative retrieval for complex queries, while your approach uses graph-based entity aggregation.[1]

#### **3. Mixture of Retrievers (MoR) - Per-Query Optimization**
MoR computes per-query, per-retriever weights using pre-retrieval and post-retrieval signals, enabling adaptive fusion of heterogeneous retrievers without manual selection. While MoR optimizes retriever combinations within a pipeline, your work selects between fundamentally different algorithm architectures (BM25+embeddings vs. entity graphs).[3]

#### **4. Self-RAG - Adaptive Retrieval-Generation**
Self-RAG uses special "reflection tokens" to allow language models to adaptively retrieve passages on-demand and critique both retrievals and generations. It represents an alternative approach to managing query heterogeneity through learned control tokens rather than explicit classification.[4]

#### **5. TREX (Microsoft Research) - Cost-Performance Balance**
TREX combines graph-based and vector-based retrieval to handle both OLTP-style (fact-based) and OLAP-style (thematic) queries with balanced cost-performance tradeoffs. This work validates the OLTP/OLAP taxonomy for RAG but doesn't explicitly use a classifier for routing—instead, it evaluates fixed configurations of GraphRAG and vector retrieval separately.[5]

#### **6. HySemRAG (Aug 2025) - Parallel vs. Selective Routing**
HySemRAG combines semantic search, keyword filtering, and knowledge graph traversal in a parallel hybrid retrieval pipeline with self-correction. Unlike your selective routing (pick one pipeline), HySemRAG uses all methods simultaneously, avoiding routing latency but increasing computational cost.[1]

#### **7. EfficientRAG - Iterative Multi-Hop Retrieval**
EfficientRAG iteratively generates new queries for multi-hop question answering without LLM calls at each iteration, offering a computationally efficient alternative to your OLAP graph construction approach for handling complex reasoning.[6]

#### **8. QCG-RAG (Query-Centric Graph RAG) - Granularity Control**
QCG-RAG addresses the granularity dilemma in graph-based RAG by enabling query-centric graph construction with controllable node/edge granularity (avoiding both overly fine-grained entity-level and overly coarse document-level graphs). This could serve as a more scalable alternative to your full entity graph approach for OLAP queries.[7]

#### **9. HyPA-RAG - Hybrid Parameter Adaptation**
Uses DistilBERT for query complexity classification (2 or 3 classes) and adapts multiple parameters: top-k, query rewrites, knowledge graph depth, and keyword filtering. Highly relevant as it demonstrates practical DistilBERT classification working at scale with parameter tuning.[8]

#### **10. Adaptive-k - Router-Free Top-k Selection**
Analyzes similarity score distributions to adaptively select k without explicit query classification, achieving significant cost savings by selecting only relevant documents without routing overhead.[1]

#### **11. RAGGED Framework - Reader Robustness Metrics**
Introduces metrics for measuring RAG system stability:
- **RSS (Retrieval Sensitivity Score)**: Measures performance degradation when retrieval depth increases
- **RSC (Retrieval Scaling Coefficient)**: Measures scalability with increased context[9]

The framework demonstrates that **reader robustness to noise** is the key determinant of RAG stability, validating your policy layer's dependence on reader confidence signals.

***

### **The Query Classifier: DistilBERT Approach**

#### **Technical Architecture**

Your proposal uses **DistilBERT** (97% of BERT's performance at 40% smaller, 60% faster) as a lightweight binary classifier to predict OLTP vs. OLAP query types.[10]

**Training Label Generation Challenge** (flagged by reviewers as critical):
The review identifies a significant weakness: labels derived from "quality vs. k curves" require:
- Multiple retrieval runs per query (expensive)
- Risk of circularity (labeling based on current retriever configuration)
- Domain overfitting (labels may not transfer across domains/LLMs)

**Reviewer Recommendations for Label Generation**:
- Combine silver labels (model-based) with heuristic features (query decomposition depth, entity count, verb types)
- Use dataset priors (e.g., Natural Questions for OLTP, ASQA for OLAP)
- Follow Adaptive-RAG's hybrid labeling strategy rather than pure reward-based approaches

#### **Classification Performance Concerns**

The single manual analysis on 100 Natural Questions samples is statistically insufficient to validate:
- Classifier accuracy across diverse query distributions
- Misclassification impact on system performance
- Generalization across different document collections and LLMs

#### **Calibration and Uncertainty Handling**

The review questions whether the system handles low-confidence classifications:
- When should the system fall back or escalate?
- Can pipelines hybridize (e.g., start with OLTP, escalate to OLAP if needed)?
- How is router uncertainty propagated to the policy layer?

***

### **Is This an Open Problem in the RAG Community?**

#### **The Gap Claim Evaluation: 8.5/10 Validity Score**

**Your Original Claim**: "No work combines query-aware routing with specialized pipelines optimized per query type."

**Review Assessment**: ✅ **SUBSTANTIALLY VALID** with important nuances.

**Why It's Valid**:
1. **No exact combination exists**: While QA-Dragon has similar architecture, no work applies the database-inspired OLTP/OLAP taxonomy to RAG with classifier-based routing between algorithmically distinct pipelines optimized for cost-efficiency.

2. **OLTP/OLAP taxonomy is novel for RAG**: Directly adapting database concepts to query routing is a genuinely new angle, validated by TREX's work demonstrating 10x cost differences between query types.

3. **Efficiency focus differentiates you**: Most other works optimize for accuracy/quality; your cost-efficiency angle is underexplored in the routing literature.

4. **Unique technical contributions**:
   - Dual-pipeline architecture (hybrid BM25+embeddings vs. entity graphs with Leiden clustering)
   - Dynamic policy layer adjusting multiple parameters simultaneously
   - Integration of reader robustness (RSS/RSC from RAGGED)

#### **Why the Gap Is Partially Nuanced**:
1. **QA-Dragon (Aug 2025)**: Published very recently, has domain + search routing for multimodal VQA. Represents active parallel research but in different problem space.

2. **Multiple works have 2 of 3 elements**: Adaptive-RAG (classification + routing but to iterative depth), HySemRAG (multiple pipelines but parallel not selective), MoR (per-query optimization but same pipeline).

3. **Rapid field evolution**: 7 of 10 closest papers published Jul-Aug 2025, indicating this is a hot research area. The field is converging toward similar solutions.

#### **Practical Validity: Industry Perspective**

The review notes that major RAG providers (OpenAI, Anthropic, Google) likely have internal routing systems, but your work would be the **first open, documented, rigorously benchmarked approach** combining these elements—which has significant value regardless of perfect novelty.

***

### **Cost Efficiency Claims: The 10x Gap**

The review validates the cost optimization angle:

**Evidence from Industry Data**:
- 60-75% of RAG queries are straightforward[11]
- Alternative models can respond 27x faster for simple queries[11]
- Cost savings of 98% possible for lightweight models on simple queries[11]
- Real customers have reduced costs by 82% through routing[11]

This empirical validation strengthens your motivation, though the proposal doesn't yet provide concrete cost measurements from your proposed system.

***

### **Critical Weaknesses Identified by Reviewers**

1. **Premature submission**: No end-to-end experimental results yet; mostly preliminary evidence
2. **Label generation risks**: Potential for domain overfitting, circular dependencies
3. **OLAP cost underspecified**: No details on incremental update costs for dynamic corpora at 10^8 chunks
4. **Missing empirical comparisons**: No direct benchmarks against router-free (TREX, Adaptive-k) or agent-based (agentic RAG) alternatives
5. **System-level gaps**: Sharding, backpressure policies, caching, multi-tenant isolation not specified for 1000 QPS+ scale

***

### **Recommended Positioning & Next Steps**

**Revised Gap Statement** (per review):

> "While recent work explores query routing (QA-Dragon for multimodal VQA) and hybrid retrieval strategies (HySemRAG), no existing approach applies database query complexity concepts (OLTP/OLAP) to text RAG systems with classifier-based routing between algorithmically distinct pipelines optimized for cost-efficiency at scale. Prior work either focuses on routing to different data sources rather than algorithms, uses agent-based selection with variable latency overhead, or applies all retrieval strategies in parallel without selective routing."

**Critical Comparisons to Include**:
- Against QA-Dragon (differentiate domain, taxonomy, efficiency focus)
- Against Adaptive-RAG (show how OLTP/OLAP differs from complexity-based iterative routing)
- Against HySemRAG (selective vs. parallel benefits/tradeoffs)
- Against TREX (show classifier advantage over fixed routing)
- Against Adaptive-k (demonstrate routing benefits vs. router-free k selection)

[1](https://arxiv.org/html/2507.18910v1)
[2](https://arxiv.org/html/2505.23052v1)
[3](https://dev.to/exploredataaiml/building-an-intelligent-rag-system-with-query-routing-validation-and-self-correction-2e4k)
[4](https://aclanthology.org/2025.iwsds-1.14.pdf)
[5](https://www.facebook.com/groups/cto.platform/posts/2016154118830620/)
[6](https://arxiv.org/abs/2508.05197)
[7](https://www.deepchecks.com/build-high-performance-rag-pipelines-scale/)
[8](https://dataloop.ai/library/model/haider0941_distilbert-base-educationl/)
[9](https://4561480.fs1.hubspotusercontent-na1.net/hubfs/4561480/Ebooks%20whitepapers%20and%20reports/O%E2%80%99Reilly%20Guide%20-%20RAG%20in%20Production%20with%20Haystack/OReilly%20Guide%20-%20RAG_in_production_with_Haystack-FINAL.pdf)
[10](https://chatpaper.com/paper/174269)
[11](https://www.chitika.com/adaptive-rag-systems-langchain-langgraph/)
[12](https://arxiv.org/html/2506.15862v1)
[13](https://arxiv.org/html/2503.02922v1)
[14](https://www.promptingguide.ai/research/rag)
[15](https://arxiv.org/html/2409.09046v2)
[16](https://arxiv.org/abs/2408.04259)
[17](https://openreview.net/pdf/b44801e94a5e3c70b2ce5462a1962363bc9031f7.pdf)
[18](https://datasciocean.com/en/paper-intro/adaptive-k/)
[19](https://arxiv.org/abs/2310.11511)
[20](https://clickhouse.com/resources/engineering/oltp-vs-olap)
[21](https://arxiv.org/abs/2503.02922)
[22](https://zilliz.com/learn/distilbert-distilled-version-of-bert)
[23](https://arxiv.org/pdf/2204.06522.pdf)
[24](https://arxiv.org/pdf/2505.23052.pdf)
[25](https://github.com/dblilienthal/Multiclass-Text-Classification-with-DistilBERT-on-COVID-19-Tweets)
[26](https://aclanthology.org/2025.emnlp-main.1674.pdf)
[27](https://openreview.net/forum?id=4ufjBV6S4I)
[28](https://airouter.io/news/2024-12-11-optimizing-rag-systems-prioritizing-efficiency-and-cost-effectiveness)
[29](https://towardsdatascience.com/how-to-build-an-overengineered-retrieval-system/)

Your current DistilBERT setup is fundamentally learning “which dataset is this from?” instead of “how complex is this query?”, so trying to tweak it around the edges won’t fix the core issue.[1][2]

## Why the current DistilBERT setup is broken

The proxy-label design (MS MARCO = OLTP, Hotpot = OLAP) makes dataset identity a much easier signal than true complexity, so the model converges to that shortcut.  Symptoms you’re seeing (heavy OLTP bias, worse than keywords, overconfident when wrong) are exactly what you’d expect from a classifier trained on confounded labels.  Given that:[2][1]

- It is not just underfitting or lack of data; the target function itself is wrong.
- Further fine-tuning on the same labeling scheme will only overfit more to dataset style.

So the options are: change the labeling/target, change the model family/architecture, or both.

## Option A: Heuristic / rule-based router (short term, reliable)

For your current project, a simple heuristic classifier like the one you wrote is actually a very defensible baseline:

- Keyword/structure-based rules (“why”, “how”, “compare”, “impact of”, “pros and cons”, multiple clauses, conjunctions) often correlate strongly with complex/OLAP-style questions.[3][1]
- Papers on “simple vs complex question classification” and “Optimizing RAG: Classifying Queries for Dynamic Processing” show that very simple features can separate factoid vs complex questions reasonably well.[1][2][3]

Concretely:

- Use your heuristic as the primary router.
- Add a small set of extra patterns:
  - OLAP signals: “why”, “how does”, “what are the effects/impacts of”, “advantages and disadvantages”, “compare”, “difference between”, “relationship between”, “pros and cons”, “explain”, “summarize”, “overview of”.
  - OLTP signals: “when”, “where”, “who”, “how many”, “how much”, explicit entities + single relation (“capital of X”, “population of X”, “birthday of X”).[4][1]
- Optionally, use length and punctuation:
  - Very short, single-clause questions → OLTP.
  - Long questions with multiple clauses, commas, “and/or” → more likely OLAP.[1]

This will likely outperform your current DistilBERT model and is easy to debug and explain.

## Option B: Fix the labeling (LLM or human labels)

If you want a learned classifier, you must change how labels are created so they reflect complexity, not dataset:

1) Build a mixed dataset first

- Sample queries from multiple sources: MS MARCO, HotpotQA, TriviaQA, NQ, maybe your real production logs.[5][4]
- Do NOT tie label to source; every query gets its own OLTP/OLAP tag.

2) Use LLM-assisted labeling

- Prompt a strong LLM to classify each query as “simple factoid/local lookup” vs “requires synthesis/multi-hop/aggregation”.[3][5]
- Add explicit criteria in the prompt (number of entities/relations, need to aggregate across multiple facts, explain vs lookup).
- Manually spot-check and correct a subset to estimate noise; if needed, relabel borderline examples.

3) Train a new classifier (can still be DistilBERT)

- Now fine-tune DistilBERT on this mixed, genuinely-labeled dataset.
- Add simple features if you want (length, presence of “why/how/compare”) either:
  - As extra tokens in the input (“[LEN_LONG] why is the sky blue?”), or
  - Via a tiny MLP on top of the CLS embedding concatenated with these numeric features.[6]

This turns DistilBERT back into a legitimate complexity classifier instead of a dataset detector.

## Option C: Use a small “query complexity” head on top of heuristics

A strong pattern from related work is: combine cheap rules with a small learned component rather than relying purely on the model.[7][2]

A practical hybrid:

- Stage 1: Apply heuristics.
  - If clearly OLAP (strong OLAP phrases) → route to OLAP.
  - If clearly OLTP (short, single-fact factoid) → route to OLTP.
- Stage 2: For ambiguous cases only (no strong patterns), run a small DistilBERT classifier trained on the improved labels above.

This gives you:

- Low cost (most queries handled by rules).
- Model capacity spent only on genuinely ambiguous cases.
- Much better calibration than your current “always OLTP” behavior.

## Option D: Replace DistilBERT with a routing approach closer to RAG literature

Several recent RAG routing works suggest alternatives to a pure query-text classifier:

1) RAGRouter-style routing

- RAGRouter learns to route queries across LLMs using contrastive learning over query + document embeddings and model “capability” embeddings.[8][9]
- Adapting the idea: instead of predicting OLTP/OLAP from query text alone, encode:
  - The query embedding.
  - Cheap retrieval signals (e.g., entropy of BM25 scores, spread of top-k similarities).
- Use those to decide whether the query likely needs a deep/graph-style pipeline vs shallow/local retrieval.

2) Query complexity from structure/graph features

- QA work on “simple or complex questions” uses features like number of entities, relations, dependency tree depth, etc., rather than dataset identity.[10][3][1]
- You could engineer features such as:
  - Number of named entities.
  - Number of verbs/clauses.
  - Presence of temporal/causal connectors (“because”, “after”, “due to”).
- Train a small classifier (even logistic regression or a shallow MLP) on these features + some LLM-labeled data.

This may be easier to get right and more interpretable than a pure DistilBERT text model.

## What to do right now for your project

Given where you are and the review feedback you already have:

- For the current experimental paper:
  - Use the heuristic classifier (Option A) as the main router.
  - Explicitly report that naive DistilBERT fine-tuning on dataset-proxy labels failed (with the confusion matrix, calibration behavior, and heuristic vs model comparison).
  - Frame this as an important negative result supporting your claim that careful labeling and evaluation are crucial for query-aware routing.[2][7]

- For “future work” / revised implementation:
  - Plan Option B or C: LLM-assisted labeling on a mixed dataset + retrained DistilBERT, optionally with rules for easy cases.
  - Mention that more advanced routing approaches (RAGRouter-style, complexity features, adaptive-k without classification) are viable alternatives that you plan to compare against.[9][11][7][8]

In short: don’t fight the current DistilBERT model; either (1) rely on your heuristic router for this iteration and treat DistilBERT as a cautionary tale, or (2) rebuild the dataset with true complexity labels and retrain, ideally in a hybrid rules + model setup.[7][2][3][1]

[1](https://aclanthology.org/W12-6001.pdf)
[2](https://aclanthology.org/2025.iwsds-1.14/)
[3](https://dl.acm.org/doi/10.1145/3744663)
[4](https://uwspace.uwaterloo.ca/bitstreams/c1d314d6-6843-41df-912f-e22bdb61748a/download)
[5](https://www.sciencedirect.com/science/article/pii/S2666651022000249)
[6](https://arxiv.org/pdf/2204.06522.pdf)
[7](https://www.meilisearch.com/blog/adaptive-rag)
[8](https://arxiv.org/html/2505.23052v2)
[9](https://arxiv.org/abs/2505.23052)
[10](https://www.ijcai.org/proceedings/2020/0519.pdf)
[11](https://datasciocean.com/en/paper-intro/adaptive-k/)
[12](https://arxiv.org/html/2507.11954v1)
[13](https://www.tandfonline.com/doi/full/10.1080/1206212X.2024.2426512?scroll=top&needAccess=true)
[14](https://aclanthology.org/2025.iwsds-1.14.pdf)
[15](https://openreview.net/pdf?id=4VKVUmE1I8)
[16](https://pure.ulster.ac.uk/en/publications/optimizing-rag-classifying-queries-for-dynamic-processing/)
[17](https://learn.microsoft.com/en-gb/answers/questions/2239952/optimizing-rag-dynamic-query-routing-for-multi-sou)
[18](https://www.semanticscholar.org/paper/RAGRouter:-Learning-to-Route-Queries-to-Multiple-Zhang-Liu/bbc2dcac1d4e52e607dc17104414b6b0cee5fb44)
[19](https://github.com/NirDiamant/RAG_Techniques)
[20](https://github.com/OwwO99/RAGRouter)
[21](https://www.edenai.co/post/the-2025-guide-to-retrieval-augmented-generation-rag)
[22](https://www.themoonlight.io/en/review/learning-to-route-queries-across-knowledge-bases-for-step-wise-retrieval-augmented-reasoning)

## Practical Application to Your Work
Based on the approaches outlined above and the related work you've reviewed, here's how to move forward:

### **Why the Alternatives Work Better Than Retraining DistilBERT on the Same Data**
The core issue is that your training signal was fundamentally corrupted. Simply retraining the same model on bad data won't fix it—you need to break the dataset-label confound first. Approach 4 (fixed DistilBERT with proper labels) works because it uses **genuinely mixed data** where OLTP and OLAP queries exist across all sources, not segregated by dataset.

### **For Your Paper Right Now**
Present your classifier challenge as a strength, not weakness:

1. **Show the diagnostic work you've done** (confusion matrix, calibration curves, worse-than-baseline comparison to heuristics). This is valuable negative results that many papers skip.

2. **Position Approach 1 (heuristics) as your immediate experimental baseline.** Frame it as: "Given labeling challenges with purely neural approaches, we adopt a linguistically-motivated heuristic baseline inspired by query complexity literature (cf. Bisk et al. 2012, Shaikh et al. 2013)."

3. **Use Approach 2 (dependency parsing features) as your main empirical contribution** for routing decisions. This aligns with the academic rigor your reviewers want—it's grounded in prior work on question complexity and avoids the neural model's pitfalls.

4. **Compare against TREX as a router-free alternative.** Your paper review specifically flagged that Adaptive-k exists, so benchmark your routing against "no routing" to show the value you add.

### **Which Approach Most Aligns With Your Paper's Contributions**
Looking at your proposal review feedback, the reviewers want:

- ✅ **Explicit query routing** (rules or learned, but *transparent*) → Approaches 1, 2, or 5 deliver this
- ✅ **Cost/efficiency focus** (vs. pure accuracy) → All approaches show this implicitly, but Approach 3 (retrieval signals) is most direct
- ✅ **Dual-pipeline architecture** (OLTP vs. OLAP distinction) → Independent of classifier choice
- ❌ **Overly-engineered neural models without clear validation** → Your DistilBERT failure shows this risk; switch to Approach 2 or 5

**Recommendation:** Use **Approach 2 (dependency parsing)** as your primary routing mechanism, with **Approach 1 (heuristics)** as a simpler fallback baseline. This demonstrates both academic rigor and practical robustness—exactly what reviewers want to see.

### **Specific Implementation Path**
1. **This week**: Replace DistilBERT with heuristic router (Approach 1). Get your dual-pipeline system working end-to-end with this baseline.

2. **Next week**: Add dependency parsing features (Approach 2). Show empirical comparison: heuristics vs. features vs. retrieval signals.

3. **Week 3**: If time permits, implement hybrid router (Approach 5) as your final system. Benchmark against TREX and Adaptive-k as mentioned in your review.

4. **Paper positioning**: "We investigated neural classifiers for query routing but found that linguistic and retrieval-based features, combined with simple decision rules, outperform naive fine-tuning while remaining interpretable and robust to domain shift. This aligns with recent findings in query complexity detection..."

This positioning turns your DistilBERT failure into a methodological strength rather than a weakness.

[1](https://arxiv.org/abs/2310.11511)
[2](https://www.kore.ai/blog/self-reflective-retrieval-augmented-generation-self-rag)
[3](https://www.projectpro.io/article/self-rag/1176)
[4](https://selfrag.github.io)
[5](https://www.thoughtworks.com/en-ca/insights/blog/generative-ai/four-retrieval-techniques-improve-rag)
[6](https://arxiv.org/html/2505.23052v1)
[7](https://www.shaped.ai/blog/unlock-text-data-nlp-feature-engineering-for-search-recs)
[8](https://github.com/AkariAsai/self-rag)
[9](https://arxiv.org/abs/2505.23052)
[10](https://www.infoq.com/presentations/nlp-ml-dl/)
[11](https://ceur-ws.org/Vol-2414/paper3.pdf)
[12](https://odr.chalmers.se/bitstreams/7a0b9845-9b04-4fff-8f1f-ea84297d6a3b/download)
[13](https://www.sciencedirect.com/science/article/abs/pii/S095741742302612X)
[14](https://arxiv.org/pdf/2501.07727.pdf)
[15](https://spaces.facsci.ualberta.ca/casalberta/wp-content/uploads/sites/12/2019/06/castle_2019_Presentation-1.pdf)
[16](https://arxiv.org/html/2503.02435v2)
[17](https://arxiv.org/html/2510.02388v1)
[18](https://rbcborealis.com/research-blogs/generating-labels-model-training-using-weak-supervision/)
[19](https://liu-nlp.ai/dl4nlp/modules/module3/intro-to-dependency-parsing.pdf)
[20](https://www.ijraset.com/research-paper/hybrid-algorithm-combining-bellman-ford-dijkstra-and-machine-learning-for-dynamic-network-routing)

You’re right to feel that a pure keyword heuristic router is not very impressive as a “solution” for a researchy OLTP/OLAP router. It’s a good debugging tool and baseline, but not something you want as the main intellectual contribution.[1][2]

## How to position heuristics vs “real” solutions

- Use heuristics as:
  - A sanity-check baseline.
  - A way to show how badly the naive DistilBERT setup fails (20% worse than even dumb rules).
- Do NOT present heuristics as your final answer. Instead, make them:
  - The simplest baseline in your experiments.
  - The starting point for more principled methods (feature-based or learned).

That way, you’re not selling heuristics as the contribution; you’re using them to demonstrate why better routing is non-trivial.

## More “serious” approaches you can use

Here are options that feel less naive and more publishable, without going full Self-RAG:

1) Feature-based complexity classifier (structural, not just keywords)
- Use linguistic features (number of clauses, entities, dependency depth, coordination, etc.) from a parser plus some lightweight rules or a small model.[3][4]
- This lets you say: “We operationalize query complexity using structural features inspired by question complexity literature, and show it improves routing over simple keyword heuristics and naive neural classifiers.”
- It’s still interpretable, but much less naive than raw keyword checks.

2) Retrieval-signal-based router (behavioral complexity)
- Use signals from retrieval itself: entropy/variance of scores, spread between top-1 and top-k, disagreement between sparse and dense rankings.[5]
- Intuition: “easy/local” queries produce confident, peaky distributions; complex/OLAP queries look messy and uncertain.
- That sounds much more sophisticated than “if query contains ‘why’ ⇒ OLAP” and is grounded in the RAG behavior itself.

3) Fixed DistilBERT with proper labels
- Build a small mixed dataset and label complexity with an LLM + some manual checks, then retrain DistilBERT on that.[2][1]
- You can explicitly say: “We tried the naive dataset-proxy labeling and show it fails; with proper complexity labels, a small DistilBERT model outperforms heuristics.”

4) Hybrid router (rules + model)
- Rules for obvious cases, model for ambiguous ones.
- This aligns with what many “Adaptive-RAG” / RAGRouter-style systems do: simple logic first, learned component when needed.[6][2]

## How to make this look non-naive in the paper

You can structure the story like this:

- “Naive neural classifier”: DistilBERT with proxy labels fails (your confusion matrix, overconfidence, etc.).
- “Naive heuristics”: surprisingly strong but limited; used as a baseline.
- “Our approach”: a feature- and/or retrieval-signal-based classifier, optionally with a small DistilBERT head trained on real complexity labels, that:
  - Beats heuristics.
  - Is interpretable.
  - Aligns with prior work on question complexity and RAG routing.[1][3][6]

So you don’t have to stick with the heuristic router as “the solution”—use it as the dumb but honest baseline, and then build something more principled on top that you can be proud to defend.

[1](https://aclanthology.org/2025.iwsds-1.14/)
[2](https://www.meilisearch.com/blog/adaptive-rag)
[3](https://aclanthology.org/W12-6001.pdf)
[4](https://dl.acm.org/doi/10.1145/3744663)
[5](https://www.thoughtworks.com/en-ca/insights/blog/generative-ai/four-retrieval-techniques-improve-rag)
[6](https://arxiv.org/html/2505.23052v2)

That approach is actually very strong and, for your use case, arguably cleaner and more “systems-y” than a pure DistilBERT text classifier.[1][2]

## Why this RAGRouter-style + structural features approach is good

- It uses signals that are directly tied to “is this query hard for retrieval?” rather than “does it look like dataset X?”.[3][1]
- Structural features (entities, clauses, connectors) come from question-complexity literature and map naturally to OLTP vs OLAP.[2][4]
- A small classifier on top (logistic regression / shallow MLP) is:
  - Easy to train and debug.
  - Less likely to pick up dataset-style shortcuts than a big LM.
  - Easy to ablate and explain in the paper.

## How you could concretely structure it

You can define three feature groups:

1) Text features (structure/semantics):
- Number of tokens, clauses (approx by verb count), entities.[2]
- Presence of causal/comparative phrases: “because”, “due to”, “impact of”, “compare”, “difference between”.[4][2]
- Presence of “why/how/compare/pros and cons” vs “who/when/where/how many”.[2]

2) Retrieval behavior features:
- Entropy of BM25 scores over top-k docs.[3]
- Entropy or variance of embedding similarities over top-k.
- Score spread between top-1 and top-k.
- Correlation between sparse and dense rankings.[3]

3) Query embedding:
- A single dense vector from a sentence embedding model (e.g., MiniLM/E5), reduced via PCA if needed.

Then:

- Concatenate: [hand-crafted features] + [reduced embedding dims] → small classifier.
- Train on a few thousand LLM-labeled OLTP/OLAP examples (your Option B labeling).[5][6]

## Advantages vs. fine-tuned DistilBERT

- More interpretable: you can show which features matter (e.g., “multiple entities + high entropy retrieval → OLAP”).[4][2]
- Better aligned with routing goal: it “sees” retrieval difficulty, not just question wording.[3]
- Simpler model: logistic regression / small MLP is trivial to train and cheap to deploy.
- Great story in the paper: “Inspired by RAGRouter and question complexity work, we design a feature-based router that combines structural and retrieval signals, avoiding the failure modes of naive text-only neural classifiers.”[1][5][2]

## How this fits with your OLTP/OLAP framing

- OLTP path: triggered when
  - Structural features suggest single-entity, single-relation factoid.
  - Retrieval shows low entropy, peaky scores, high sparse–dense agreement.
- OLAP path: triggered when
  - Multiple entities/clauses, causal/comparative language.
  - Retrieval shows high entropy, flat scores, sparse–dense disagreement.

That is very consistent with your “OLTP vs OLAP” story and clearly a step beyond naive heuristics without requiring heavy LM finetuning.

If you want, next step can be: design a minimal feature set (10–15 features), a small labeling plan, and the exact classifier/metric setup to run in the next week.

[1](https://arxiv.org/html/2505.23052v2)
[2](https://aclanthology.org/W12-6001.pdf)
[3](https://www.thoughtworks.com/en-ca/insights/blog/generative-ai/four-retrieval-techniques-improve-rag)
[4](https://dl.acm.org/doi/10.1145/3744663)
[5](https://aclanthology.org/2025.iwsds-1.14/)
[6](https://www.meilisearch.com/blog/adaptive-rag)