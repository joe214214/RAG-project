#!/usr/bin/env python3
"""
RAG Quality Evaluation Script

Based on best practices from Answer_Quality_Evaluation.md:

RETRIEVAL METRICS:
- MRR (Mean Reciprocal Rank)
- Recall@k, Precision@k
- NDCG@k
- Context Precision (fraction of retrieved that are relevant)
- Context Recall (fraction of relevant that are retrieved)

ANSWER QUALITY METRICS (when LLM is integrated):
- Faithfulness (claims traceable to sources)
- Answer Relevance (semantic match to query)
- Hallucination Rate (claims without source support)

DIFFERENT PROTOCOLS:
- OLTP: Exact match, fast binary checks
- OLAP: Claim-level analysis, multi-step completeness

Uses real ground truth from MS MARCO qrels and HotpotQA supporting_facts.
"""

import argparse
import time
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field, asdict
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from qdrant_client import models
    from sentence_transformers import SentenceTransformer
    from datasets import load_dataset
    from vector_stores.qdrant_store import (
        QdrantVectorStore, 
        QdrantConfig, 
        QdrantCollectionConfig,
        create_oltp_store,
        create_olap_store,
    )
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install qdrant-client sentence-transformers datasets")
    sys.exit(1)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EvalQuery:
    """A query with ground truth relevance labels."""
    qid: str
    query: str
    relevant_texts: List[str]       # Ground truth relevant passages
    query_type: str                 # 'oltp' or 'olap'
    answer: Optional[str] = None    # Ground truth answer (if available)


@dataclass
class RetrievalResult:
    """Results from retrieval (before/after reranking)."""
    doc_id: str
    score: float
    text: str
    is_relevant: bool = False


@dataclass 
class QueryEvalResult:
    """Comprehensive evaluation results for a single query."""
    qid: str
    query_type: str
    
    # Retrieval metrics
    mrr: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    recall_at_20: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    ndcg_at_10: float = 0.0
    
    # Context quality (from Answer_Quality_Evaluation.md)
    context_precision: float = 0.0  # Fraction of retrieved that are relevant
    context_recall: float = 0.0     # Fraction of relevant that are retrieved
    
    # Timing
    retrieval_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    
    # Counts
    num_retrieved: int = 0
    num_relevant_found: int = 0
    num_relevant_total: int = 0


# =============================================================================
# Metric Calculations
# =============================================================================

def dcg_at_k(relevances: List[int], k: int) -> float:
    """Compute Discounted Cumulative Gain at k."""
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(relevances: List[int], k: int) -> float:
    """Compute Normalized DCG at k."""
    dcg = dcg_at_k(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)
    return dcg / idcg if idcg > 0 else 0.0


def reciprocal_rank(relevances: List[int]) -> float:
    """Compute reciprocal rank (1/position of first relevant doc)."""
    for i, rel in enumerate(relevances):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0


def text_similarity(text1: str, text2: str) -> float:
    """
    Jaccard similarity for text matching.
    Used when we can't match by exact IDs.
    """
    t1 = set(text1.lower().split())
    t2 = set(text2.lower().split())
    
    if not t1 or not t2:
        return 0.0
    
    intersection = len(t1 & t2)
    union = len(t1 | t2)
    
    return intersection / union if union > 0 else 0.0


def is_text_relevant(retrieved_text: str, relevant_texts: List[str], threshold: float = 0.25) -> bool:
    """Check if retrieved text matches any relevant text."""
    for rel_text in relevant_texts:
        if text_similarity(retrieved_text, rel_text) >= threshold:
            return True
    return False


# =============================================================================
# Evaluator Class
# =============================================================================

class RAGEvaluator:
    """
    Comprehensive RAG evaluation following Answer_Quality_Evaluation.md guidelines.
    
    Supports different evaluation protocols for OLTP vs OLAP:
    - OLTP: Exact match, fast binary checks, precision-focused
    - OLAP: Recall-focused, multi-document coverage
    
    Uses QdrantVectorStore wrapper for proper HNSW parameter control:
    - OLTP: search_ef=50 (faster, lower recall)
    - OLAP: search_ef=200 (slower, higher recall)
    """
    
    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection: str = "oltp_chunks",
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda",
        query_type: str = "oltp",  # 'oltp' or 'olap' - affects HNSW search_ef
    ):
        print(f"Loading embedding model on {device}...")
        self.embedder = SentenceTransformer(embed_model, device=device)
        self.collection = collection
        self.reranker = None
        self.device = device
        self.query_type = query_type
        
        # Create vector store with appropriate HNSW parameters
        config = QdrantConfig(host=qdrant_host, port=qdrant_port)
        
        # Different search_ef for OLTP vs OLAP (per project proposal)
        if query_type == "oltp":
            search_ef = 50   # Fast, good for factoid queries
        else:
            search_ef = 200  # Higher recall, better for multi-hop
        
        self.vector_store = QdrantVectorStore(
            config=config,
            collection_config=QdrantCollectionConfig(
                name=collection,
                vector_size=384,  # MiniLM-L6-v2 dimension
                hnsw_m=16,
                hnsw_ef_construct=128,
                search_ef=search_ef,
            )
        )
        print(f"Connected to Qdrant (HNSW search_ef={search_ef} for {query_type})")
    
    def set_reranker(self, reranker):
        """Set the reranker to use."""
        self.reranker = reranker
    
    def set_query_type(self, query_type: str):
        """
        Update HNSW search_ef based on query type.
        - OLTP: ef=50 (faster)
        - OLAP: ef=200 (higher recall)
        """
        if query_type == "oltp":
            self.vector_store.collection_config.search_ef = 50
        else:
            self.vector_store.collection_config.search_ef = 200
        self.query_type = query_type
    
    def retrieve(self, query: str, top_k: int = 100) -> Tuple[List[RetrievalResult], float]:
        """
        Retrieve documents from Qdrant using wrapper with proper HNSW params.
        Returns results and latency.
        """
        start = time.time()
        query_vec = self.embedder.encode(query)
        
        # Use wrapper's search method (applies search_ef via SearchParams)
        results = self.vector_store.search(query_vec, top_k=top_k)
        latency = (time.time() - start) * 1000
        
        return [
            RetrievalResult(
                doc_id=doc_id,
                score=score,
                text=payload.get("text", ""),
            )
            for doc_id, score, payload in results
        ], latency
    
    def rerank(self, query: str, results: List[RetrievalResult], top_k: int = 20) -> Tuple[List[RetrievalResult], float]:
        """Apply reranker if set. Returns reranked results and latency."""
        if self.reranker is None:
            return results[:top_k], 0.0
        
        start = time.time()
        # Reranker expects list of dicts with 'text', 'score', 'metadata' keys
        documents = [
            {"text": r.text, "score": r.score, "metadata": {"doc_id": r.doc_id}}
            for r in results[:top_k]
        ]
        reranked = self.reranker.rerank(query, documents, top_k=top_k)
        latency = (time.time() - start) * 1000
        
        # Convert RerankResult objects back to RetrievalResult
        return [
            RetrievalResult(
                doc_id=r.metadata.get("doc_id", ""),
                score=r.score,
                text=r.text,
            )
            for r in reranked
        ], latency
    
    def evaluate_query(
        self,
        eval_query: EvalQuery,
        retrieve_k: int = 100,
        rerank_k: int = 20,
        similarity_threshold: float = 0.25,
    ) -> QueryEvalResult:
        """
        Evaluate a single query with full metrics.
        
        Protocol differs by query type (per Answer_Quality_Evaluation.md):
        - OLTP: Focus on precision, exact match, fast checks (search_ef=50)
        - OLAP: Focus on recall, coverage, multi-document synthesis (search_ef=200)
        """
        # Adjust HNSW search_ef based on query type
        self.set_query_type(eval_query.query_type)
        
        # Step 1: Retrieve (now uses appropriate search_ef)
        candidates, retrieval_latency = self.retrieve(eval_query.query, top_k=retrieve_k)
        
        # Step 2: Rerank
        results, rerank_latency = self.rerank(eval_query.query, candidates, top_k=rerank_k)
        
        # Step 3: Mark relevance for each result
        for r in results:
            r.is_relevant = is_text_relevant(r.text, eval_query.relevant_texts, similarity_threshold)
        
        # Step 4: Compute metrics
        relevances = [1 if r.is_relevant else 0 for r in results]
        num_relevant_total = len(eval_query.relevant_texts)
        
        # Count relevant at different k
        rel_at_5 = sum(relevances[:5])
        rel_at_10 = sum(relevances[:10])
        rel_at_20 = sum(relevances[:20])
        
        # Context precision: fraction of retrieved that are relevant
        context_precision = sum(relevances) / len(relevances) if relevances else 0.0
        
        # Context recall: fraction of relevant that are retrieved
        # (approximated since we use text matching, not exact IDs)
        context_recall = min(sum(relevances), num_relevant_total) / num_relevant_total if num_relevant_total > 0 else 0.0
        
        return QueryEvalResult(
            qid=eval_query.qid,
            query_type=eval_query.query_type,
            
            # Retrieval metrics
            mrr=reciprocal_rank(relevances),
            recall_at_5=rel_at_5 / num_relevant_total if num_relevant_total > 0 else 0.0,
            recall_at_10=rel_at_10 / num_relevant_total if num_relevant_total > 0 else 0.0,
            recall_at_20=rel_at_20 / num_relevant_total if num_relevant_total > 0 else 0.0,
            precision_at_5=rel_at_5 / 5,
            precision_at_10=rel_at_10 / 10,
            ndcg_at_10=ndcg_at_k(relevances, 10),
            
            # Context quality
            context_precision=context_precision,
            context_recall=context_recall,
            
            # Timing
            retrieval_latency_ms=retrieval_latency,
            rerank_latency_ms=rerank_latency,
            total_latency_ms=retrieval_latency + rerank_latency,
            
            # Counts
            num_retrieved=len(results),
            num_relevant_found=sum(relevances),
            num_relevant_total=num_relevant_total,
        )
    
    def evaluate_batch(
        self,
        queries: List[EvalQuery],
        retrieve_k: int = 100,
        rerank_k: int = 20,
        verbose: bool = True,
        similarity_threshold: float = 0.15,
    ) -> Dict[str, any]:
        """
        Evaluate a batch of queries and return aggregate metrics.
        
        Returns separate metrics for OLTP and OLAP queries.
        """
        results = []
        oltp_results = []
        olap_results = []
        
        for i, q in enumerate(queries):
            result = self.evaluate_query(q, retrieve_k, rerank_k, similarity_threshold)
            results.append(result)
            
            if q.query_type == "oltp":
                oltp_results.append(result)
            else:
                olap_results.append(result)
            
            if verbose and (i + 1) % 10 == 0:
                print(f"  Evaluated {i + 1}/{len(queries)} queries...")
        
        def aggregate(res_list: List[QueryEvalResult]) -> Dict[str, float]:
            if not res_list:
                return {}
            return {
                "num_queries": len(res_list),
                "MRR": np.mean([r.mrr for r in res_list]),
                "Recall@5": np.mean([r.recall_at_5 for r in res_list]),
                "Recall@10": np.mean([r.recall_at_10 for r in res_list]),
                "Recall@20": np.mean([r.recall_at_20 for r in res_list]),
                "Precision@5": np.mean([r.precision_at_5 for r in res_list]),
                "Precision@10": np.mean([r.precision_at_10 for r in res_list]),
                "NDCG@10": np.mean([r.ndcg_at_10 for r in res_list]),
                "Context_Precision": np.mean([r.context_precision for r in res_list]),
                "Context_Recall": np.mean([r.context_recall for r in res_list]),
                "Avg_Latency_ms": np.mean([r.total_latency_ms for r in res_list]),
                "P95_Latency_ms": np.percentile([r.total_latency_ms for r in res_list], 95),
                "Avg_Retrieval_ms": np.mean([r.retrieval_latency_ms for r in res_list]),
                "Avg_Rerank_ms": np.mean([r.rerank_latency_ms for r in res_list]),
            }
        
        return {
            "all": aggregate(results),
            "oltp": aggregate(oltp_results),
            "olap": aggregate(olap_results),
        }


# =============================================================================
# Data Loaders
# =============================================================================

def load_msmarco_queries(limit: int = 50) -> List[EvalQuery]:
    """Load MS MARCO queries with relevance judgments (OLTP)."""
    print("Loading MS MARCO queries...")
    
    try:
        dataset = load_dataset(
            "microsoft/ms_marco",
            "v1.1",
            split="validation",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"  Error: {e}")
        return []
    
    queries = []
    for item in dataset:
        if len(queries) >= limit:
            break
        
        query = item.get("query", "")
        passages = item.get("passages", {})
        
        relevant_texts = []
        if isinstance(passages, dict):
            texts = passages.get("passage_text", [])
            selected = passages.get("is_selected", [])
            for text, sel in zip(texts, selected):
                if sel == 1:
                    relevant_texts.append(text)
        
        if query and relevant_texts:
            queries.append(EvalQuery(
                qid=f"msmarco_{len(queries)}",
                query=query,
                relevant_texts=relevant_texts,
                query_type="oltp",
            ))
    
    print(f"  Loaded {len(queries)} MS MARCO queries")
    return queries


def load_hotpotqa_queries(limit: int = 50) -> List[EvalQuery]:
    """Load HotpotQA queries with supporting facts (OLAP)."""
    print("Loading HotpotQA queries...")
    
    try:
        dataset = load_dataset(
            "hotpot_qa",
            "fullwiki",
            split="validation",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"  Error: {e}")
        return []
    
    queries = []
    for item in dataset:
        if len(queries) >= limit:
            break
        
        question = item.get("question", "")
        context = item.get("context", {})
        supporting_facts = item.get("supporting_facts", {})
        answer = item.get("answer", "")
        
        relevant_texts = []
        if isinstance(context, dict) and isinstance(supporting_facts, dict):
            titles = context.get("title", [])
            sentences_list = context.get("sentences", [])
            sf_titles = supporting_facts.get("title", [])
            sf_sent_ids = supporting_facts.get("sent_id", [])
            
            for sf_title, sf_sent_id in zip(sf_titles, sf_sent_ids):
                if sf_title in titles:
                    idx = titles.index(sf_title)
                    if idx < len(sentences_list):
                        sents = sentences_list[idx]
                        if sf_sent_id < len(sents):
                            relevant_texts.append(sents[sf_sent_id])
        
        if question and relevant_texts:
            queries.append(EvalQuery(
                qid=f"hotpot_{len(queries)}",
                query=question,
                relevant_texts=relevant_texts,
                query_type="olap",
                answer=answer,
            ))
    
    print(f"  Loaded {len(queries)} HotpotQA queries")
    return queries


# =============================================================================
# Output Formatting
# =============================================================================

def print_metrics(name: str, metrics: Dict[str, float], baseline: Dict[str, float] = None):
    """Print metrics with optional comparison to baseline."""
    if not metrics:
        print(f"\n{name}: No data")
        return
    
    print(f"\n{'─' * 60}")
    print(f"{name}")
    print(f"{'─' * 60}")
    
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            if baseline and key in baseline and baseline[key] > 0:
                delta = ((value - baseline[key]) / baseline[key]) * 100
                print(f"  {key:<20} {value:>10.4f}  ({delta:+.1f}%)")
            else:
                if "Latency" in key or "_ms" in key:
                    print(f"  {key:<20} {value:>10.1f} ms")
                else:
                    print(f"  {key:<20} {value:>10.4f}")
        else:
            print(f"  {key:<20} {value}")


def print_summary_table(baseline: Dict, reranked: Dict, reranker_name: str):
    """Print a side-by-side comparison table."""
    print(f"\n{'═' * 70}")
    print("SUMMARY COMPARISON")
    print(f"{'═' * 70}")
    print(f"{'Metric':<25} {'Baseline':<15} {reranker_name:<15} {'Δ':<15}")
    print(f"{'─' * 70}")
    
    key_metrics = [
        ("MRR", "MRR"),
        ("Recall@10", "Recall@10"),
        ("Precision@10", "Precision@10"),
        ("NDCG@10", "NDCG@10"),
        ("Context_Precision", "Ctx Precision"),
        ("Context_Recall", "Ctx Recall"),
        ("Avg_Latency_ms", "Latency (ms)"),
    ]
    
    for metric_key, display_name in key_metrics:
        base_val = baseline.get("all", {}).get(metric_key, 0)
        rerank_val = reranked.get("all", {}).get(metric_key, 0) if reranked else 0
        
        if "Latency" in metric_key:
            delta = f"{rerank_val - base_val:+.1f} ms"
            print(f"{display_name:<25} {base_val:<15.1f} {rerank_val:<15.1f} {delta:<15}")
        else:
            delta = f"{(rerank_val - base_val):+.4f}"
            print(f"{display_name:<25} {base_val:<15.4f} {rerank_val:<15.4f} {delta:<15}")


def save_results(results: Dict, filename: str):
    """Save results to JSON file."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {filename}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="RAG Quality Evaluation (per Answer_Quality_Evaluation.md)"
    )
    parser.add_argument("--qdrant-host", default="localhost")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument("--collection", default="oltp_chunks")
    parser.add_argument("--num-queries", type=int, default=30)
    parser.add_argument("--olap-model", default="bge-large",
                        choices=["none", "minilm", "bge-base", "bge-large"])
    parser.add_argument("--dataset", default="both",
                        choices=["msmarco", "hotpotqa", "both"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    parser.add_argument("--debug", action="store_true", help="Show debug info for first few queries")
    parser.add_argument("--similarity-threshold", type=float, default=0.15, 
                        help="Text similarity threshold for relevance matching (default: 0.15)")
    args = parser.parse_args()
    
    print("═" * 70)
    print("RAG QUALITY EVALUATION")
    print("Based on Answer_Quality_Evaluation.md best practices")
    print("═" * 70)
    print(f"\nQdrant: {args.qdrant_host}:{args.qdrant_port}")
    print(f"Collection: {args.collection}")
    print(f"Device: {args.device}")
    
    # Load queries
    queries = []
    if args.dataset in ["msmarco", "both"]:
        queries.extend(load_msmarco_queries(args.num_queries))
    if args.dataset in ["hotpotqa", "both"]:
        queries.extend(load_hotpotqa_queries(args.num_queries))
    
    if not queries:
        print("\n❌ No queries loaded!")
        return
    
    oltp_count = sum(1 for q in queries if q.query_type == "oltp")
    olap_count = sum(1 for q in queries if q.query_type == "olap")
    print(f"\nLoaded {len(queries)} queries: {oltp_count} OLTP, {olap_count} OLAP")
    
    # Initialize evaluator
    evaluator = RAGEvaluator(
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        collection=args.collection,
        device=args.device,
    )
    
    # Debug: show first query details
    if args.debug and queries:
        print("\n" + "─" * 70)
        print("DEBUG: First Query Analysis")
        print("─" * 70)
        q = queries[0]
        print(f"Query: {q.query[:100]}...")
        print(f"Type: {q.query_type}")
        print(f"Num relevant texts: {len(q.relevant_texts)}")
        if q.relevant_texts:
            print(f"Sample relevant text: {q.relevant_texts[0][:150]}...")
        
        # Retrieve and show what we got
        results, _ = evaluator.retrieve(q.query, top_k=5)
        print(f"\nTop 5 retrieved:")
        for i, r in enumerate(results[:5]):
            print(f"  {i+1}. [{r.score:.3f}] {r.text[:100]}...")
            # Check similarity with each relevant text
            for j, rel_text in enumerate(q.relevant_texts[:2]):
                sim = text_similarity(r.text, rel_text)
                print(f"      Similarity with rel[{j}]: {sim:.3f}")
        print("─" * 70)
    
    # Baseline evaluation (no reranker)
    print("\n" + "═" * 70)
    print("BASELINE (Vector Search Only)")
    print("═" * 70)
    evaluator.set_reranker(None)
    baseline_results = evaluator.evaluate_batch(queries, similarity_threshold=args.similarity_threshold)
    
    print_metrics("All Queries", baseline_results["all"])
    if baseline_results["oltp"]:
        print_metrics("OLTP Queries (MS MARCO)", baseline_results["oltp"])
    if baseline_results["olap"]:
        print_metrics("OLAP Queries (HotpotQA)", baseline_results["olap"])
    
    # Reranker evaluation
    reranker_results = None
    if args.olap_model != "none":
        print("\n" + "═" * 70)
        print(f"WITH RERANKER ({args.olap_model})")
        print("═" * 70)
        
        try:
            from rerankers import OLAPReranker
            reranker = OLAPReranker(model_name=args.olap_model)
            evaluator.set_reranker(reranker)
            reranker_results = evaluator.evaluate_batch(queries, similarity_threshold=args.similarity_threshold)
            
            print_metrics("All Queries", reranker_results["all"], baseline_results["all"])
            if reranker_results["oltp"]:
                print_metrics("OLTP Queries", reranker_results["oltp"], baseline_results["oltp"])
            if reranker_results["olap"]:
                print_metrics("OLAP Queries", reranker_results["olap"], baseline_results["olap"])
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary table
    print_summary_table(baseline_results, reranker_results, args.olap_model)
    
    # Save results
    if args.output:
        save_results({
            "baseline": baseline_results,
            "reranker": reranker_results,
            "config": vars(args),
        }, args.output)
    
    print("\n✅ Evaluation complete!")
    print("\nNOTE: For full evaluation (faithfulness, hallucination detection),")
    print("integrate an LLM and implement claim-level grounding checks per")
    print("docs/Approach/Answer_Quality_Evaluation.md")


if __name__ == "__main__":
    main()
