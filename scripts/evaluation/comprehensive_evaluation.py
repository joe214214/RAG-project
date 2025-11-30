#!/usr/bin/env python3
"""
Comprehensive RAG System Evaluation

Tests multiple configurations:
1. Classifier comparison (feature vs transformer)
2. Reranker ablation (with/without, different models)
3. Retrieval method comparison (hybrid vs dense-only)
4. End-to-end performance metrics
5. Retrieval correctness (MRR, Recall@k, NDCG@k)

Usage:
    # Run all evaluations with ground truth (from datasets, 50 queries per type by default)
    # Results saved to: results/comprehensive_eval_optimized.json (default)
    python scripts/comprehensive_evaluation.py --qdrant-host ecetesla0 --use-ground-truth
    
    # With custom limits
    python scripts/comprehensive_evaluation.py --qdrant-host ecetesla0 --use-ground-truth --oltp-limit 100 --olap-limit 100
    
    # Run with custom queries (no correctness metrics)
    python scripts/comprehensive_evaluation.py --qdrant-host ecetesla0 --output results/eval_results.json
    
    # Custom output filename
    python scripts/comprehensive_evaluation.py --qdrant-host ecetesla0 --use-ground-truth --output results/my_eval.json
    
    # Test specific components
    python scripts/comprehensive_evaluation.py --qdrant-host ecetesla0 --test-classifiers
    python scripts/comprehensive_evaluation.py --qdrant-host ecetesla0 --test-rerankers
    python scripts/comprehensive_evaluation.py --qdrant-host ecetesla0 --test-retrieval
"""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict, field
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.rag_pipeline import QueryAwareRAGPipeline, RAGResult
from qdrant_client import QdrantClient, models

# Import evaluation utilities from evaluate_rerankers.py
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("⚠ datasets library not available. Ground truth evaluation disabled.")

# Import ir_datasets for research-standard MS MARCO evaluation
try:
    import ir_datasets
    HAS_IR_DATASETS = True
except ImportError:
    HAS_IR_DATASETS = False
    print("⚠ ir_datasets library not available. Install with: pip install ir-datasets")


@dataclass
class EvalConfig:
    """Configuration for a single evaluation run."""
    name: str
    classifier_type: str  # "feature" or "transformer"
    classifier_model: Optional[str] = None
    router_model: Optional[str] = None
    use_hybrid: bool = False
    use_reranker: bool = False
    oltp_reranker: Optional[str] = None  # "none", "tinybert", "minilm"
    olap_reranker: Optional[str] = None   # "minilm", "bge-base", "bge-large"
    description: str = ""


@dataclass
class QueryResult:
    """Results for a single query."""
    query: str
    query_type: str  # Predicted by classifier
    confidence: float
    num_retrieved: int
    top_scores: List[float]
    latency_ms: float
    config_name: str
    # Correctness metrics (if ground truth available)
    mrr: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    recall_at_20: float = 0.0
    ndcg_at_10: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    num_relevant_found: int = 0
    num_relevant_total: int = 0


@dataclass
class EvalResults:
    """Aggregated evaluation results."""
    config: EvalConfig
    queries: List[QueryResult]
    avg_latency_ms: float
    avg_confidence: float
    routing_accuracy: Dict[str, float]  # Distribution of query types
    num_queries: int
    # Correctness metrics (if ground truth available)
    mrr: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    recall_at_20: float = 0.0
    ndcg_at_10: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    has_ground_truth: bool = False


# Test queries for evaluation (fallback when ground truth not available)
TEST_QUERIES = {
    "oltp": [
        "What is machine learning?",
        "Who founded Microsoft?",
        "What is the capital of France?",
        "When was Python created?",
        "What is the speed of light?",
        "Who wrote Romeo and Juliet?",
        "What is the largest planet?",
        "What is artificial intelligence?",
        "Who invented the telephone?",
        "What is the chemical formula for water?",
    ],
    "olap": [
        "Compare machine learning and deep learning",
        "What are the differences between AI and ML?",
        "Explain the relationship between neural networks and deep learning",
        "How do transformers differ from RNNs?",
        "What are the advantages and disadvantages of supervised learning?",
        "Compare supervised and unsupervised learning approaches",
        "How does reinforcement learning relate to other ML paradigms?",
        "What are the key differences between CNNs and RNNs?",
        "Explain the evolution of NLP from rule-based to transformer models",
        "Compare different optimization algorithms used in deep learning",
    ],
}


# =============================================================================
# Ground Truth Data Loaders (from evaluate_rerankers.py)
# =============================================================================

@dataclass
class EvalQueryWithGT:
    """Query with ground truth relevance labels."""
    qid: str
    query: str
    query_type: str
    relevant_texts: List[str] = field(default_factory=list)
    relevant_passage_ids: List[str] = field(default_factory=list)
    relevant_titles: List[str] = field(default_factory=list)  # HotpotQA
    relevant_sent_ids: List[int] = field(default_factory=list)  # HotpotQA
    answer: Optional[str] = None


def load_msmarco_queries(limit: int = 50) -> List[EvalQueryWithGT]:
    """
    Load MS MARCO queries with relevance judgments (OLTP) using ir_datasets.
    
    Uses ir_datasets for research-standard evaluation with proper qrels (passage IDs).
    Falls back to HuggingFace datasets if ir_datasets is not available.
    """
    # Try ir_datasets first (research-standard approach)
    if HAS_IR_DATASETS:
        print("Loading MS MARCO queries with ground truth using ir_datasets (research-standard)...")
        try:
            dataset = ir_datasets.load("msmarco-passage/dev")
            
            # Build query_id -> query_text mapping
            queries_dict = {}
            for query in dataset.queries_iter():
                queries_dict[str(query.query_id)] = query.text
                if len(queries_dict) >= limit * 2:  # Load extra for matching
                    break
            
            # Build query_id -> relevant_passage_ids mapping from qrels
            qrels_dict = {}
            for qrel in dataset.qrels_iter():
                query_id = str(qrel.query_id)
                passage_id = str(qrel.doc_id)
                if query_id not in qrels_dict:
                    qrels_dict[query_id] = []
                qrels_dict[query_id].append(passage_id)
            
            # Combine queries and qrels
            queries = []
            for query_id, query_text in list(queries_dict.items())[:limit]:
                relevant_ids = qrels_dict.get(query_id, [])
                if relevant_ids:  # Only include queries with relevant passages
                    queries.append(EvalQueryWithGT(
                        qid=query_id,
                        query=query_text,
                        query_type="oltp",
                        relevant_passage_ids=relevant_ids,
                        relevant_texts=[],  # Can be loaded from corpus if needed
                    ))
            
            print(f"  Loaded {len(queries)} MS MARCO queries with {sum(len(q.relevant_passage_ids) for q in queries)} relevant passage IDs")
            return queries
            
        except Exception as e:
            print(f"  Error loading MS MARCO with ir_datasets: {e}")
            print("  Falling back to HuggingFace datasets (may not have passage_id)...")
    
    # Fallback to HuggingFace datasets (may not have passage_id)
    if not HAS_DATASETS:
        print("  ⚠ Neither ir_datasets nor datasets available. Cannot load MS MARCO queries.")
        return []
    
    print("Loading MS MARCO queries with ground truth using HuggingFace datasets...")
    print("  ⚠ Warning: microsoft/ms_marco v1.1 validation split does not include passage_id")
    print("  ⚠ Consider installing ir_datasets for research-standard evaluation: pip install ir-datasets")
    
    try:
        dataset = load_dataset(
            "microsoft/ms_marco",
            "v1.1",
            split="validation",
        )
    except Exception as e:
        print(f"  Error loading MS MARCO: {e}")
        return []
    
    queries = []
    for item in dataset:
        if len(queries) >= limit:
            break
        
        query = item.get("query", "")
        passages = item.get("passages", {})
        
        queryid = item.get("queryid") or item.get("query_id") or f"msmarco_{len(queries)}"
        queryid = str(queryid)
        
        relevant_texts = []
        relevant_passage_ids = []
        
        if isinstance(passages, dict):
            texts = passages.get("passage_text", [])
            selected = passages.get("is_selected", [])
            passage_ids = passages.get("passage_id", [])
            
            for idx, (text, sel) in enumerate(zip(texts, selected)):
                if sel == 1:
                    relevant_texts.append(text)
                    if passage_ids and idx < len(passage_ids):
                        relevant_passage_ids.append(str(passage_ids[idx]))
        
        if query and (relevant_texts or relevant_passage_ids):
            queries.append(EvalQueryWithGT(
                qid=queryid,
                query=query,
                query_type="oltp",
                relevant_texts=relevant_texts,
                relevant_passage_ids=relevant_passage_ids,
            ))
    
    print(f"  Loaded {len(queries)} MS MARCO queries with ground truth")
    if not any(q.relevant_passage_ids for q in queries):
        print("  ⚠ Warning: No passage IDs found. ID-based evaluation will not work.")
        print("  ⚠ Install ir_datasets for proper passage IDs: pip install ir-datasets")
    return queries


def load_hotpotqa_queries(limit: int = 50) -> List[EvalQueryWithGT]:
    """Load HotpotQA queries with supporting facts (OLAP)."""
    if not HAS_DATASETS:
        return []
    
    print("Loading HotpotQA queries with ground truth...")
    
    try:
        dataset = load_dataset(
            "hotpot_qa",
            "fullwiki",
            split="validation",
        )
    except Exception as e:
        print(f"  Error loading HotpotQA: {e}")
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
        relevant_passage_ids = []
        relevant_titles = []
        relevant_sent_ids = []
        
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
                            relevant_titles.append(sf_title)
                            relevant_sent_ids.append(sf_sent_id)
                            relevant_passage_ids.append(f"hotpot_ctx_{sf_title}")
        
        if question and (relevant_texts or relevant_passage_ids or relevant_titles):
            queries.append(EvalQueryWithGT(
                qid=f"hotpot_{len(queries)}",
                query=question,
                query_type="olap",
                relevant_texts=relevant_texts,
                relevant_passage_ids=relevant_passage_ids,
                relevant_titles=relevant_titles,
                relevant_sent_ids=relevant_sent_ids,
                answer=answer,
            ))
    
    print(f"  Loaded {len(queries)} HotpotQA queries with ground truth")
    return queries


def check_ground_truth_coverage(
    query: EvalQueryWithGT,
    qdrant_client: QdrantClient,
    collection: str,
) -> bool:
    """
    Check if at least one ground truth passage exists in Qdrant for this query.
    Returns True if any relevant passage is found.
    """
    # Check MS MARCO passage IDs
    for passage_id in query.relevant_passage_ids:
        passage_id_str = str(passage_id)
        try:
            # Try string match
            filter_condition = models.Filter(
                must=[
                    models.FieldCondition(
                        key="original_passage_id",
                        match=models.MatchValue(value=passage_id_str)
                    )
                ]
            )
            scroll_result = qdrant_client.scroll(
                collection_name=collection,
                scroll_filter=filter_condition,
                limit=1,
                with_payload=False,
            )
            points, _ = scroll_result
            if points:
                return True
        except Exception:
            # Try integer match if string fails
            try:
                if passage_id_str.isdigit():
                    filter_condition = models.Filter(
                        must=[
                            models.FieldCondition(
                                key="original_passage_id",
                                match=models.MatchValue(value=int(passage_id_str))
                            )
                        ]
                    )
                    scroll_result = qdrant_client.scroll(
                        collection_name=collection,
                        scroll_filter=filter_condition,
                        limit=1,
                        with_payload=False,
                    )
                    points, _ = scroll_result
                    if points:
                        return True
            except Exception:
                pass
    
    # Check HotpotQA titles
    for i, title in enumerate(query.relevant_titles):
        sent_id = None
        if query.relevant_sent_ids and i < len(query.relevant_sent_ids):
            sent_id = query.relevant_sent_ids[i]
        
        try:
            must_conditions = [
                models.FieldCondition(
                    key="original_title",
                    match=models.MatchValue(value=title)
                )
            ]
            if sent_id is not None:
                must_conditions.append(
                    models.FieldCondition(
                        key="original_sent_id",
                        match=models.MatchValue(value=sent_id)
                    )
                )
            
            filter_condition = models.Filter(must=must_conditions)
            scroll_result = qdrant_client.scroll(
                collection_name=collection,
                scroll_filter=filter_condition,
                limit=1,
                with_payload=False,
            )
            points, _ = scroll_result
            if points:
                return True
        except Exception:
            pass
    
    return False


def filter_queries_by_coverage(
    queries: Dict[str, List[EvalQueryWithGT]],
    qdrant_host: str,
    qdrant_port: int = 6333,
) -> Dict[str, List[EvalQueryWithGT]]:
    """
    Filter queries to only include those where ground truth passages exist in Qdrant.
    This ensures we only evaluate on queries where we can properly measure retrieval quality.
    """
    print("\nFiltering queries by ground truth coverage...")
    client = QdrantClient(host=qdrant_host, port=qdrant_port, check_compatibility=False)
    
    filtered_queries = {"oltp": [], "olap": []}
    total_queries = 0
    covered_queries = 0
    
    for query_type, query_list in queries.items():
        collection = "oltp_chunks" if query_type == "oltp" else "olap_chunks"
        
        for query in query_list:
            total_queries += 1
            if check_ground_truth_coverage(query, client, collection):
                filtered_queries[query_type].append(query)
                covered_queries += 1
            elif total_queries % 10 == 0:
                print(f"  Checked {total_queries} queries, {covered_queries} covered ({100*covered_queries/total_queries:.1f}%)")
    
    print(f"\nCoverage Summary:")
    print(f"  Total queries: {total_queries}")
    print(f"  Queries with covered ground truth: {covered_queries} ({100*covered_queries/total_queries:.1f}%)")
    print(f"  OLTP queries (covered): {len(filtered_queries['oltp'])}")
    print(f"  OLAP queries (covered): {len(filtered_queries['olap'])}")
    
    return filtered_queries


def load_queries_with_ground_truth(oltp_limit: int = 50, olap_limit: int = 50) -> Dict[str, List[EvalQueryWithGT]]:
    """Load queries with ground truth from datasets."""
    oltp_queries = load_msmarco_queries(limit=oltp_limit)
    olap_queries = load_hotpotqa_queries(limit=olap_limit)
    
    return {
        "oltp": oltp_queries,
        "olap": olap_queries,
    }


# =============================================================================
# Correctness Evaluation Functions
# =============================================================================

def text_similarity(text1: str, text2: str, method: str = "hybrid") -> float:
    """
    Compute text similarity using multiple methods.
    
    Methods:
    - "jaccard": Word-level Jaccard similarity (fast, simple)
    - "ngram": Character n-gram overlap (catches partial matches)
    - "hybrid": Combines word Jaccard + n-gram overlap (recommended)
    
    Args:
        text1: First text
        text2: Second text
        method: Similarity method ("jaccard", "ngram", or "hybrid")
    
    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not text1 or not text2:
        return 0.0
    
    text1_lower = text1.lower()
    text2_lower = text2.lower()
    
    if method == "jaccard":
        # Word-level Jaccard similarity (original method)
        t1 = set(text1_lower.split())
        t2 = set(text2_lower.split())
        
        if not t1 or not t2:
            return 0.0
        
        intersection = len(t1 & t2)
        union = len(t1 | t2)
        
        return intersection / union if union > 0 else 0.0
    
    elif method == "ngram":
        # Character 3-gram overlap (catches partial word matches)
        def get_ngrams(text, n=3):
            return set(text[i:i+n] for i in range(len(text) - n + 1))
        
        ngrams1 = get_ngrams(text1_lower, n=3)
        ngrams2 = get_ngrams(text2_lower, n=3)
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)
        
        return intersection / union if union > 0 else 0.0
    
    elif method == "hybrid":
        # Hybrid: Combine word Jaccard + n-gram overlap + length ratio
        # This is more robust for passage matching
        
        # 1. Word-level Jaccard (catches exact word matches)
        t1_words = set(text1_lower.split())
        t2_words = set(text2_lower.split())
        
        word_jaccard = 0.0
        if t1_words and t2_words:
            intersection = len(t1_words & t2_words)
            union = len(t1_words | t2_words)
            word_jaccard = intersection / union if union > 0 else 0.0
        
        # 2. Character 3-gram overlap (catches partial matches, typos, variations)
        def get_ngrams(text, n=3):
            return set(text[i:i+n] for i in range(len(text) - n + 1))
        
        ngrams1 = get_ngrams(text1_lower, n=3)
        ngrams2 = get_ngrams(text2_lower, n=3)
        
        ngram_sim = 0.0
        if ngrams1 and ngrams2:
            intersection = len(ngrams1 & ngrams2)
            union = len(ngrams1 | ngrams2)
            ngram_sim = intersection / union if union > 0 else 0.0
        
        # 3. Length ratio (penalize very different lengths)
        len1, len2 = len(text1_lower), len(text2_lower)
        length_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0.0
        
        # Weighted combination: 50% word Jaccard, 30% n-gram, 20% length ratio
        # This favors word matches but also catches partial matches
        hybrid_score = (0.5 * word_jaccard) + (0.3 * ngram_sim) + (0.2 * length_ratio)
        
        return hybrid_score
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'jaccard', 'ngram', or 'hybrid'")


def is_relevant_chunk(
    chunk,
    qdrant_client,
    collection: str,
    relevant_passage_ids: List[str],
    relevant_texts: List[str],
    relevant_titles: List[str] = None,
    relevant_sent_ids: List[int] = None,
    threshold: float = 0.25,
    use_text_first: bool = False,
    similarity_method: str = "hybrid",
) -> bool:
    """
    Check if retrieved chunk is relevant.
    
    If use_text_first=True: Uses text similarity as primary method (more lenient).
    Otherwise: Uses exact ID matching first, then falls back to text similarity.
    
    Fetches payload from Qdrant to get original_passage_id, original_title, original_sent_id.
    Supports both MS MARCO (passage_id-based) and HotpotQA (title-based) matching.
    """
    # Get chunk text for text similarity
    chunk_text = None
    if hasattr(chunk, 'text'):
        chunk_text = chunk.text if isinstance(chunk.text, str) else str(chunk.text)
    
    # TEXT-FIRST MODE: Prioritize text similarity (more lenient, higher recall)
    if use_text_first and relevant_texts and chunk_text:
        for rel_text in relevant_texts:
            similarity = text_similarity(chunk_text, rel_text, method=similarity_method)
            if similarity >= threshold:
                return True
        # If text similarity fails, still try ID matching as fallback
        # (fall through to ID matching below)
    
    # Fetch payload from Qdrant to get original IDs
    try:
        point = qdrant_client.retrieve(
            collection_name=collection,
            ids=[chunk.chunk_id],
            with_payload=True,
        )
        if point and len(point) > 0:
            payload = point[0].payload
        else:
            payload = {}
    except Exception:
        payload = {}
    
    # HotpotQA: Title-based matching (case-insensitive)
    if relevant_titles:
        original_title = payload.get("original_title")
        if original_title:
            retrieved_title_lower = original_title.lower()
            for i, gold_title in enumerate(relevant_titles):
                gold_title_lower = gold_title.lower()
                if retrieved_title_lower == gold_title_lower:
                    if relevant_sent_ids and i < len(relevant_sent_ids):
                        gold_sent_id = relevant_sent_ids[i]
                        original_sent_id = payload.get("original_sent_id")
                        if gold_sent_id is not None and original_sent_id is not None:
                            if original_sent_id == gold_sent_id:
                                return True
                    else:
                        return True
    
    # MS MARCO: Exact passage ID matching
    if relevant_passage_ids:
        original_passage_id = payload.get("original_passage_id")
        if original_passage_id and str(original_passage_id) in relevant_passage_ids:
            return True
    
    # Fallback to text similarity (if not already checked in text-first mode)
    if not use_text_first and relevant_texts and chunk_text:
        for rel_text in relevant_texts:
            if text_similarity(chunk_text, rel_text, method=similarity_method) >= threshold:
                return True
    
    return False


def reciprocal_rank(relevances: List[int]) -> float:
    """Compute reciprocal rank (1/position of first relevant doc)."""
    for i, rel in enumerate(relevances):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0


def dcg_at_k(relevances: List[int], k: int) -> float:
    """Compute DCG at k."""
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        dcg += relevances[i] / np.log2(i + 2)
    return dcg


def ndcg_at_k(relevances: List[int], k: int) -> float:
    """Compute Normalized DCG at k."""
    dcg = dcg_at_k(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)
    return dcg / idcg if idcg > 0 else 0.0


def calculate_correctness_metrics(
    retrieved_chunks: List,
    eval_query: EvalQueryWithGT,
    qdrant_client,
    collection: str,
    use_text_first: bool = False,
    similarity_threshold: float = 0.25,
    similarity_method: str = "hybrid",
) -> Dict[str, float]:
    """Calculate correctness metrics for a query."""
    # Check relevance for each retrieved chunk
    relevances = []
    for chunk in retrieved_chunks:
        is_rel = is_relevant_chunk(
            chunk,
            qdrant_client,
            collection,
            eval_query.relevant_passage_ids,
            eval_query.relevant_texts,
            eval_query.relevant_titles,
            eval_query.relevant_sent_ids,
            threshold=similarity_threshold,
            use_text_first=use_text_first,
            similarity_method=similarity_method,
        )
        relevances.append(1 if is_rel else 0)
    
    num_relevant_total = len(eval_query.relevant_passage_ids) + len(eval_query.relevant_titles)
    if num_relevant_total == 0:
        # Fallback: use text count
        num_relevant_total = len(eval_query.relevant_texts)
    
    num_relevant_found = sum(relevances)
    
    # Calculate metrics
    rel_at_5 = sum(relevances[:5])
    rel_at_10 = sum(relevances[:10])
    rel_at_20 = sum(relevances[:20])
    
    mrr = reciprocal_rank(relevances)
    
    # Recall@k: fraction of relevant passages found in top-k
    # Cap at 1.0 to handle cases where multiple chunks match same passage
    recall_at_5 = min(rel_at_5 / num_relevant_total, 1.0) if num_relevant_total > 0 else 0.0
    recall_at_10 = min(rel_at_10 / num_relevant_total, 1.0) if num_relevant_total > 0 else 0.0
    recall_at_20 = min(rel_at_20 / num_relevant_total, 1.0) if num_relevant_total > 0 else 0.0
    ndcg_at_10 = ndcg_at_k(relevances, 10)
    
    context_precision = num_relevant_found / len(relevances) if relevances else 0.0
    context_recall = num_relevant_found / num_relevant_total if num_relevant_total > 0 else 0.0
    
    return {
        "mrr": mrr,
        "recall_at_5": recall_at_5,
        "recall_at_10": recall_at_10,
        "recall_at_20": recall_at_20,
        "ndcg_at_10": ndcg_at_10,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "num_relevant_found": num_relevant_found,
        "num_relevant_total": num_relevant_total,
    }


def create_eval_configs() -> List[EvalConfig]:
    """Create all evaluation configurations to test."""
    configs = []
    
    # Baseline: Feature classifier, dense-only, no reranker
    configs.append(EvalConfig(
        name="baseline_feature_dense",
        classifier_type="feature",
        router_model="models/feature_router.pkl",
        use_hybrid=False,
        use_reranker=False,
        description="Baseline: Feature classifier, dense-only retrieval, no reranking"
    ))
    
    # Feature classifier + Hybrid retrieval
    configs.append(EvalConfig(
        name="feature_hybrid",
        classifier_type="feature",
        router_model="models/feature_router.pkl",
        use_hybrid=True,
        use_reranker=False,
        description="Feature classifier with hybrid retrieval (BM25 + dense)"
    ))
    
    # Feature classifier + Hybrid + OLTP reranker
    configs.append(EvalConfig(
        name="feature_hybrid_rerank_oltp",
        classifier_type="feature",
        router_model="models/feature_router.pkl",
        use_hybrid=True,
        use_reranker=True,
        oltp_reranker="tinybert",
        olap_reranker="minilm",
        description="Feature classifier, hybrid retrieval, with reranking"
    ))
    
    # Transformer classifier (MiniLM-L12) - available on cluster
    configs.append(EvalConfig(
        name="transformer_dense",
        classifier_type="transformer",
        classifier_model="classifier2/microsoft_MiniLM-L12-H384-uncased",
        use_hybrid=False,
        use_reranker=False,
        description="Transformer classifier (MiniLM-L12), dense-only retrieval"
    ))
    
    configs.append(EvalConfig(
        name="transformer_hybrid",
        classifier_type="transformer",
        classifier_model="classifier2/microsoft_MiniLM-L12-H384-uncased",
        use_hybrid=True,
        use_reranker=False,
        description="Transformer classifier (MiniLM-L12) with hybrid retrieval"
    ))
    
    configs.append(EvalConfig(
        name="transformer_hybrid_rerank",
        classifier_type="transformer",
        classifier_model="classifier2/microsoft_MiniLM-L12-H384-uncased",
        use_hybrid=True,
        use_reranker=True,
        oltp_reranker="tinybert",
        olap_reranker="minilm",
        description="Transformer classifier (MiniLM-L12), hybrid retrieval, with reranking"
    ))
    
    # Dense-only ablation (no hybrid)
    configs.append(EvalConfig(
        name="feature_dense_rerank",
        classifier_type="feature",
        router_model="models/feature_router.pkl",
        use_hybrid=False,
        use_reranker=True,
        oltp_reranker="tinybert",
        olap_reranker="minilm",
        description="Feature classifier, dense-only, with reranking (hybrid ablation)"
    ))
    
    return configs


def evaluate_config(
    config: EvalConfig,
    queries: Dict[str, List],  # Can be List[str] or List[EvalQueryWithGT]
    qdrant_host: str,
    qdrant_port: int = 6333,
    device: str = "cpu",
    use_ground_truth: bool = False,
    use_text_first: bool = False,
    similarity_threshold: float = 0.25,
    similarity_method: str = "hybrid",
) -> EvalResults:
    """Evaluate a single configuration."""
    print(f"\n{'='*70}")
    print(f"Evaluating: {config.name}")
    print(f"Description: {config.description}")
    print(f"{'='*70}")
    
    # Collect all queries (handle both string queries and EvalQueryWithGT)
    all_queries = []
    for query_type, query_list in queries.items():
        for query_item in query_list:
            if isinstance(query_item, EvalQueryWithGT):
                all_queries.append((query_item.query, query_type, query_item))
            else:
                all_queries.append((query_item, query_type, None))
    
    print(f"Total queries to process: {len(all_queries)}")
    
    # Initialize pipeline
    try:
        pipeline = QueryAwareRAGPipeline(
            router_model_path=config.router_model or "models/feature_router.pkl",
            classifier_type=config.classifier_type,
            classifier_model=config.classifier_model,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            device=device,
            use_hybrid=config.use_hybrid,
            use_query_expansion=True,  # Enable LLM-based query expansion for BM25
        )
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        print(f"   Skipping configuration: {config.name}")
        return None
    
    results = []
    latencies = []
    confidences = []
    routing_dist = {"oltp": 0, "olap": 0}
    
    # Correctness metrics (aggregated)
    all_mrr = []
    all_recall_at_5 = []
    all_recall_at_10 = []
    all_recall_at_20 = []
    all_ndcg_at_10 = []
    all_context_precision = []
    all_context_recall = []
    
    total_queries = len(all_queries)
    for query_idx, (query, expected_type, eval_query_gt) in enumerate(all_queries, 1):
        try:
            # Progress indicator
            if query_idx % 10 == 0 or query_idx == 1:
                print(f"  Processing query {query_idx}/{total_queries}...", flush=True)
            
            start_time = time.time()
            
            # Classify query first to determine query type
            query_type, confidence = pipeline.classify_query(query)
            
            # Select reranker model based on query type and config
            reranker_model = None
            if config.use_reranker:
                if query_type == "oltp":
                    # For OLTP queries, use oltp_reranker if specified
                    if config.oltp_reranker and config.oltp_reranker != "none":
                        reranker_model = config.oltp_reranker
                else:  # olap
                    # For OLAP queries, use olap_reranker if specified
                    if config.olap_reranker:
                        reranker_model = config.olap_reranker
            
            # Run pipeline with appropriate reranker
            result: RAGResult = pipeline.query(
                query=query,
                top_k=10,
                use_reranker=config.use_reranker,
                reranker_model=reranker_model,
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract top scores
            top_scores = [chunk.score for chunk in result.retrieved_chunks[:5]]
            
            # Calculate correctness metrics if ground truth available
            correctness_metrics = {}
            if use_ground_truth and eval_query_gt:
                collection = "oltp_chunks" if query_type == "oltp" else "olap_chunks"
                if query_idx % 20 == 0:
                    print(f"    Calculating correctness for query {query_idx}...", flush=True)
                correctness_metrics = calculate_correctness_metrics(
                    result.retrieved_chunks,
                    eval_query_gt,
                    pipeline.qdrant_client,
                    collection,
                    use_text_first=use_text_first,
                    similarity_threshold=similarity_threshold,
                    similarity_method=similarity_method,
                )
                
                # Aggregate for overall metrics
                all_mrr.append(correctness_metrics["mrr"])
                all_recall_at_5.append(correctness_metrics["recall_at_5"])
                all_recall_at_10.append(correctness_metrics["recall_at_10"])
                all_recall_at_20.append(correctness_metrics["recall_at_20"])
                all_ndcg_at_10.append(correctness_metrics["ndcg_at_10"])
                all_context_precision.append(correctness_metrics["context_precision"])
                all_context_recall.append(correctness_metrics["context_recall"])
            
            results.append(QueryResult(
                query=query,
                query_type=result.query_type,
                confidence=result.confidence,
                num_retrieved=len(result.retrieved_chunks),
                top_scores=top_scores,
                latency_ms=latency_ms,
                config_name=config.name,
                **correctness_metrics,  # Add correctness metrics if available
            ))
            
            latencies.append(latency_ms)
            confidences.append(result.confidence)
            routing_dist[result.query_type] += 1
            
        except Exception as e:
            print(f"⚠ Query failed: {query[:50]}... Error: {e}")
            continue
    
    if not results:
        print("❌ No successful queries")
        return None
    
    # Calculate aggregated correctness metrics
    correctness_agg = {}
    if use_ground_truth and all_mrr:
        correctness_agg = {
            "mrr": np.mean(all_mrr),
            "recall_at_5": np.mean(all_recall_at_5),
            "recall_at_10": np.mean(all_recall_at_10),
            "recall_at_20": np.mean(all_recall_at_20),
            "ndcg_at_10": np.mean(all_ndcg_at_10),
            "context_precision": np.mean(all_context_precision),
            "context_recall": np.mean(all_context_recall),
            "has_ground_truth": True,
        }
    else:
        correctness_agg = {
            "mrr": 0.0,
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "recall_at_20": 0.0,
            "ndcg_at_10": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "has_ground_truth": False,
        }
    
    return EvalResults(
        config=config,
        queries=results,
        avg_latency_ms=np.mean(latencies),
        avg_confidence=np.mean(confidences),
        routing_accuracy=routing_dist,
        num_queries=len(results),
        **correctness_agg,
    )


def print_summary(all_results: List[EvalResults]):
    """Print evaluation summary."""
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    # Check if any results have ground truth
    has_gt = any(r and r.has_ground_truth for r in all_results)
    
    if has_gt:
        # Table header with correctness metrics
        print(f"\n{'Config':<30} {'Latency':<10} {'Conf':<6} {'MRR':<6} {'R@10':<6} {'NDCG':<6} {'OLTP':<6} {'OLAP':<6}")
        print("-"*70)
        
        for result in all_results:
            if result is None:
                continue
            
            oltp_count = result.routing_accuracy.get("oltp", 0)
            olap_count = result.routing_accuracy.get("olap", 0)
            
            print(f"{result.config.name:<30} "
                  f"{result.avg_latency_ms:<10.1f} "
                  f"{result.avg_confidence:<6.3f} "
                  f"{result.mrr:<6.3f} "
                  f"{result.recall_at_10:<6.3f} "
                  f"{result.ndcg_at_10:<6.3f} "
                  f"{oltp_count:<6} "
                  f"{olap_count:<6}")
    else:
        # Table header without correctness metrics
        print(f"\n{'Config':<30} {'Latency (ms)':<15} {'Confidence':<12} {'OLTP':<8} {'OLAP':<8} {'Queries':<8}")
        print("-"*70)
        
        for result in all_results:
            if result is None:
                continue
            
            oltp_count = result.routing_accuracy.get("oltp", 0)
            olap_count = result.routing_accuracy.get("olap", 0)
            
            print(f"{result.config.name:<30} "
                  f"{result.avg_latency_ms:<15.2f} "
                  f"{result.avg_confidence:<12.3f} "
                  f"{oltp_count:<8} "
                  f"{olap_count:<8} "
                  f"{result.num_queries:<8}")
    
    print("\n" + "="*70)


def compare_configs(all_results: List[EvalResults], output_file: Optional[str] = None):
    """Compare configurations and generate report."""
    # Filter out None results
    valid_results = [r for r in all_results if r is not None]
    
    if not valid_results:
        print("❌ No valid results to compare")
        return
    
    # Find best configurations
    best_latency = min(valid_results, key=lambda x: x.avg_latency_ms)
    best_confidence = max(valid_results, key=lambda x: x.avg_confidence)
    
    print("\n" + "="*70)
    print("BEST CONFIGURATIONS")
    print("="*70)
    print(f"\n🏆 Fastest: {best_latency.config.name}")
    print(f"   Average Latency: {best_latency.avg_latency_ms:.2f} ms")
    print(f"   Description: {best_latency.config.description}")
    
    print(f"\n🏆 Highest Confidence: {best_confidence.config.name}")
    print(f"   Average Confidence: {best_confidence.avg_confidence:.3f}")
    print(f"   Description: {best_confidence.config.description}")
    
    # Ablation analysis
    print("\n" + "="*70)
    print("ABLATION ANALYSIS")
    print("="*70)
    
    # Compare feature vs transformer
    feature_results = [r for r in valid_results if r.config.classifier_type == "feature"]
    transformer_results = [r for r in valid_results if r.config.classifier_type == "transformer"]
    
    if feature_results and transformer_results:
        avg_feature_latency = np.mean([r.avg_latency_ms for r in feature_results])
        avg_transformer_latency = np.mean([r.avg_latency_ms for r in transformer_results])
        
        print(f"\n📊 Classifier Comparison:")
        print(f"   Feature-based: {avg_feature_latency:.2f} ms avg latency")
        print(f"   Transformer:   {avg_transformer_latency:.2f} ms avg latency")
        print(f"   Speedup: {avg_transformer_latency / avg_feature_latency:.2f}x slower")
    
    # Compare hybrid vs dense
    hybrid_results = [r for r in valid_results if r.config.use_hybrid]
    dense_results = [r for r in valid_results if not r.config.use_hybrid]
    
    if hybrid_results and dense_results:
        avg_hybrid_latency = np.mean([r.avg_latency_ms for r in hybrid_results])
        avg_dense_latency = np.mean([r.avg_latency_ms for r in dense_results])
        
        print(f"\n📊 Retrieval Method Comparison:")
        print(f"   Dense-only: {avg_dense_latency:.2f} ms avg latency")
        print(f"   Hybrid:    {avg_hybrid_latency:.2f} ms avg latency")
        print(f"   Overhead:  {avg_hybrid_latency - avg_dense_latency:.2f} ms")
        
        # Correctness comparison if ground truth available
        if hybrid_results[0].has_ground_truth:
            avg_hybrid_recall = np.mean([r.recall_at_10 for r in hybrid_results])
            avg_dense_recall = np.mean([r.recall_at_10 for r in dense_results])
            print(f"   Dense-only Recall@10: {avg_dense_recall:.3f}")
            print(f"   Hybrid Recall@10:     {avg_hybrid_recall:.3f}")
            if avg_hybrid_recall > avg_dense_recall:
                print(f"   ✅ Hybrid improves recall by {((avg_hybrid_recall / avg_dense_recall - 1) * 100):.1f}%")
    
    # Compare with/without reranker
    rerank_results = [r for r in valid_results if r.config.use_reranker]
    no_rerank_results = [r for r in valid_results if not r.config.use_reranker]
    
    if rerank_results and no_rerank_results:
        avg_rerank_latency = np.mean([r.avg_latency_ms for r in rerank_results])
        avg_no_rerank_latency = np.mean([r.avg_latency_ms for r in no_rerank_results])
        
        print(f"\n📊 Reranker Impact:")
        print(f"   Without reranker: {avg_no_rerank_latency:.2f} ms avg latency")
        print(f"   With reranker:   {avg_rerank_latency:.2f} ms avg latency")
        print(f"   Overhead:        {avg_rerank_latency - avg_no_rerank_latency:.2f} ms")
    
    # Save results
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results_dict = {
            "summary": {
                "best_latency": {
                    "config": best_latency.config.name,
                    "latency_ms": best_latency.avg_latency_ms,
                },
                "best_confidence": {
                    "config": best_confidence.config.name,
                    "confidence": best_confidence.avg_confidence,
                },
            },
            "configs": [asdict(result) for result in valid_results],
        }
        
        with open(output_path, "w") as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive RAG system evaluation")
    parser.add_argument("--qdrant-host", type=str, default="localhost", help="Qdrant host")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant port")
    parser.add_argument("--device", type=str, default="cpu", help="Device for models (cpu/cuda)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file (default: auto-generated based on settings)")
    
    parser.add_argument("--test-classifiers", action="store_true", help="Test only classifier comparison")
    parser.add_argument("--test-rerankers", action="store_true", help="Test only reranker ablation")
    parser.add_argument("--test-retrieval", action="store_true", help="Test only retrieval methods")
    
    parser.add_argument("--queries-file", type=str, default=None, help="JSON file with test queries")
    parser.add_argument("--use-ground-truth", action="store_true", help="Load queries with ground truth from MS MARCO and HotpotQA datasets")
    parser.add_argument("--oltp-limit", type=int, default=50, help="Number of OLTP queries to load (when using ground truth, default: 50)")
    parser.add_argument("--olap-limit", type=int, default=50, help="Number of OLAP queries to load (when using ground truth, default: 50)")
    
    parser.add_argument("--use-text-first", action="store_true", help="Use text similarity as primary relevance method (more lenient, higher recall)")
    parser.add_argument("--similarity-threshold", type=float, default=0.15, help="Text similarity threshold for relevance (0.0-1.0, default: 0.15). Lower = more lenient")
    parser.add_argument("--similarity-method", type=str, default="hybrid", choices=["jaccard", "ngram", "hybrid"], help="Text similarity method: 'jaccard' (word-level), 'ngram' (character n-grams), 'hybrid' (recommended, combines both)")
    
    args = parser.parse_args()
    
    # Load queries
    use_ground_truth = args.use_ground_truth
    if args.queries_file and Path(args.queries_file).exists():
        with open(args.queries_file, "r") as f:
            queries = json.load(f)
            use_ground_truth = False  # Custom queries don't have ground truth
    elif args.use_ground_truth and HAS_DATASETS:
        print("Loading queries with ground truth from datasets...")
        queries = load_queries_with_ground_truth(
            oltp_limit=args.oltp_limit,
            olap_limit=args.olap_limit,
        )
        if not queries["oltp"] and not queries["olap"]:
            print("⚠ Failed to load queries with ground truth, falling back to test queries")
            queries = TEST_QUERIES
            use_ground_truth = False
        else:
            # Filter to only queries where ground truth exists in Qdrant
            print("\nFiltering queries to only those with ground truth coverage in Qdrant...")
            original_query_counts = {
                "oltp": len(queries["oltp"]),
                "olap": len(queries["olap"]),
            }
            queries = filter_queries_by_coverage(
                queries,
                qdrant_host=args.qdrant_host,
                qdrant_port=args.qdrant_port,
            )
            filtered_query_counts = {
                "oltp": len(queries["oltp"]),
                "olap": len(queries["olap"]),
            }
            queries_filtered = True
            
            if not queries["oltp"] and not queries["olap"]:
                print("⚠ No queries with ground truth coverage found. Falling back to all queries.")
                queries = load_queries_with_ground_truth(
                    oltp_limit=args.oltp_limit,
                    olap_limit=args.olap_limit,
                )
                queries_filtered = False
                original_query_counts = None
                filtered_query_counts = None
    else:
        queries = TEST_QUERIES
        use_ground_truth = False
        queries_filtered = False
        original_query_counts = None
        filtered_query_counts = None
    
    # Get configurations to test
    all_configs = create_eval_configs()
    
    # Filter based on test flags
    if args.test_classifiers:
        all_configs = [c for c in all_configs if not c.use_reranker and not c.use_hybrid]
    elif args.test_rerankers:
        all_configs = [c for c in all_configs if c.classifier_type == "feature"]
    elif args.test_retrieval:
        all_configs = [c for c in all_configs if not c.use_reranker]
    
    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE RAG EVALUATION")
    print(f"{'='*70}")
    # Count queries
    if use_ground_truth:
        total_queries = sum(len(q) for q in queries.values())
    else:
        total_queries = sum(len(q) for q in queries.values())
    
    # Generate output filename if not specified
    if args.output is None:
        if use_ground_truth:
            if queries_filtered:
                args.output = f"results/comprehensive_eval_filtered_{filtered_query_counts['oltp']}oltp_{filtered_query_counts['olap']}olap.json"
            else:
                args.output = f"results/comprehensive_eval_{total_queries}queries.json"
        else:
            args.output = f"results/comprehensive_eval_{total_queries}queries.json"
    
    print(f"\nTesting {len(all_configs)} configurations")
    print(f"Total queries: {total_queries}")
    if queries_filtered:
        print(f"  (Filtered from {original_query_counts['oltp'] + original_query_counts['olap']} total queries)")
    print(f"Ground truth: {'Yes' if use_ground_truth else 'No'}")
    if queries_filtered:
        print(f"Queries filtered: Yes (only queries with ground truth coverage)")
    print(f"Qdrant host: {args.qdrant_host}:{args.qdrant_port}")
    print(f"Device: {args.device}")
    print(f"Output file: {args.output}")
    
    # Evaluate each configuration
    all_results = []
    for config in all_configs:
        result = evaluate_config(
            config=config,
            queries=queries,
            qdrant_host=args.qdrant_host,
            qdrant_port=args.qdrant_port,
            device=args.device,
            use_ground_truth=use_ground_truth,
            use_text_first=args.use_text_first,
            similarity_threshold=args.similarity_threshold,
            similarity_method=args.similarity_method,
        )
        all_results.append(result)
    
    # Print summary
    print_summary(all_results)
    
    # Compare and save
    compare_configs(all_results, args.output)
    
    # Add metadata about filtering if applicable
    if use_ground_truth and queries_filtered:
        try:
            # Load existing results and add metadata
            output_path = Path(args.output)
            if output_path.exists():
                with open(output_path, "r") as f:
                    results_dict = json.load(f)
                
                results_dict["metadata"] = results_dict.get("metadata", {})
                results_dict["metadata"]["queries_filtered"] = True
                results_dict["metadata"]["original_query_counts"] = original_query_counts
                results_dict["metadata"]["filtered_query_counts"] = filtered_query_counts
                results_dict["metadata"]["coverage_pct"] = {
                    "oltp": 100 * filtered_query_counts["oltp"] / original_query_counts["oltp"] if original_query_counts["oltp"] > 0 else 0,
                    "olap": 100 * filtered_query_counts["olap"] / original_query_counts["olap"] if original_query_counts["olap"] > 0 else 0,
                }
                
                with open(output_path, "w") as f:
                    json.dump(results_dict, f, indent=2, default=str)
        except Exception as e:
            print(f"⚠ Could not add filtering metadata: {e}")
    
    print("\n✅ Evaluation complete!")
    print(f"📄 Results saved to: {args.output}")


if __name__ == "__main__":
    main()

