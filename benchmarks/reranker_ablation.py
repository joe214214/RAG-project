#!/usr/bin/env python3
"""
Reranker Ablation Study

Systematically tests different reranker configurations to measure their
individual contribution to retrieval quality and latency.

Experiments:
- OLTP: None vs TinyBERT
- OLAP: None vs MiniLM vs BGE-base vs BGE-large

Output:
- JSON results with metrics for each configuration
- Bar charts comparing quality/latency tradeoffs
- Summary table for paper/report
"""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer
    from datasets import load_dataset
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment."""
    name: str
    query_type: str  # "oltp" or "olap"
    reranker: Optional[str]  # None, "tinybert", "minilm", "bge-base", "bge-large"
    description: str


ABLATION_CONFIGS = [
    # OLTP experiments
    AblationConfig("oltp_none", "oltp", None, "OLTP baseline (no reranker)"),
    AblationConfig("oltp_tinybert", "oltp", "tinybert", "OLTP + TinyBERT reranker"),
    
    # OLAP experiments  
    AblationConfig("olap_none", "olap", None, "OLAP baseline (no reranker)"),
    AblationConfig("olap_minilm", "olap", "minilm", "OLAP + MiniLM (22M params)"),
    AblationConfig("olap_bge_base", "olap", "bge-base", "OLAP + BGE-base (109M params)"),
    AblationConfig("olap_bge_large", "olap", "bge-large", "OLAP + BGE-large (335M params)"),
]


@dataclass
class AblationResult:
    """Results from a single ablation experiment."""
    config_name: str
    query_type: str
    reranker: Optional[str]
    
    # Quality metrics
    mrr: float
    recall_at_5: float
    recall_at_10: float
    precision_at_10: float
    ndcg_at_10: float
    
    # Latency metrics (ms)
    avg_retrieval_ms: float
    avg_rerank_ms: float
    avg_total_ms: float
    p95_total_ms: float
    
    # Counts
    num_queries: int


# =============================================================================
# Evaluation Functions
# =============================================================================

def text_similarity(text1: str, text2: str) -> float:
    """Jaccard similarity for text matching."""
    t1 = set(text1.lower().split())
    t2 = set(text2.lower().split())
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def is_relevant(retrieved_text: str, relevant_texts: List[str], threshold: float = 0.15) -> bool:
    """Check if retrieved text matches any relevant text."""
    for rel_text in relevant_texts:
        if text_similarity(retrieved_text, rel_text) >= threshold:
            return True
    return False


def dcg_at_k(relevances: List[int], k: int) -> float:
    """Compute DCG@k."""
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(relevances: List[int], k: int) -> float:
    """Compute NDCG@k."""
    dcg = dcg_at_k(relevances, k)
    idcg = dcg_at_k(sorted(relevances, reverse=True), k)
    return dcg / idcg if idcg > 0 else 0.0


def reciprocal_rank(relevances: List[int]) -> float:
    """Compute MRR."""
    for i, rel in enumerate(relevances):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0


# =============================================================================
# Data Loading
# =============================================================================

def load_queries(query_type: str, limit: int = 30) -> List[Dict]:
    """Load queries for the specified type."""
    queries = []
    
    if query_type == "oltp":
        print(f"  Loading MS MARCO queries (OLTP)...")
        try:
            dataset = load_dataset("microsoft/ms_marco", "v1.1", split="validation")
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
                    queries.append({"query": query, "relevant_texts": relevant_texts})
        except Exception as e:
            print(f"  Error loading MS MARCO: {e}")
    
    elif query_type == "olap":
        print(f"  Loading HotpotQA queries (OLAP)...")
        try:
            dataset = load_dataset("hotpot_qa", "fullwiki", split="validation")
            for item in dataset:
                if len(queries) >= limit:
                    break
                question = item.get("question", "")
                context = item.get("context", {})
                supporting_facts = item.get("supporting_facts", {})
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
                    queries.append({"query": question, "relevant_texts": relevant_texts})
        except Exception as e:
            print(f"  Error loading HotpotQA: {e}")
    
    print(f"  Loaded {len(queries)} queries")
    return queries


# =============================================================================
# Ablation Runner
# =============================================================================

class AblationRunner:
    """Runs ablation experiments."""
    
    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection: str = "oltp_chunks",
        device: str = "cuda",
    ):
        print(f"Connecting to Qdrant at {qdrant_host}:{qdrant_port}...")
        self.client = QdrantClient(
            host=qdrant_host, 
            port=qdrant_port,
            check_compatibility=False,
        )
        print(f"Loading embedding model on {device}...")
        self.embedder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device=device,
        )
        self.collection = collection
        self.device = device
        self._reranker_cache = {}
    
    def _get_reranker(self, reranker_name: Optional[str]):
        """Get or create a reranker instance."""
        if reranker_name is None:
            return None
        
        if reranker_name not in self._reranker_cache:
            print(f"  Loading reranker: {reranker_name}...")
            if reranker_name == "tinybert":
                from rerankers import OLTPReranker
                self._reranker_cache[reranker_name] = OLTPReranker(use_reranker=True)
            else:
                from rerankers import OLAPReranker
                self._reranker_cache[reranker_name] = OLAPReranker(model_name=reranker_name)
        
        return self._reranker_cache[reranker_name]
    
    def retrieve(self, query: str, top_k: int = 100) -> Tuple[List[Dict], float]:
        """Retrieve documents from Qdrant."""
        start = time.time()
        query_vec = self.embedder.encode(query)
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vec.tolist(),
            limit=top_k,
        )
        latency = (time.time() - start) * 1000
        
        docs = [
            {"id": str(hit.id), "score": hit.score, "text": hit.payload.get("text", "")}
            for hit in results.points
        ]
        return docs, latency
    
    def rerank(self, query: str, docs: List[Dict], reranker, top_k: int = 20) -> Tuple[List[Dict], float]:
        """Apply reranker if provided."""
        if reranker is None:
            return docs[:top_k], 0.0
        
        start = time.time()
        doc_tuples = [(d["id"], d["text"]) for d in docs[:top_k]]
        reranked = reranker.rerank(query, doc_tuples, top_k=top_k)
        latency = (time.time() - start) * 1000
        
        # Convert back to dict format
        result_docs = [
            {"id": doc_id, "score": score, "text": text}
            for doc_id, score, text in reranked
        ]
        return result_docs, latency
    
    def run_experiment(
        self,
        config: AblationConfig,
        queries: List[Dict],
        retrieve_k: int = 100,
        rerank_k: int = 20,
    ) -> AblationResult:
        """Run a single ablation experiment."""
        print(f"\n{'─' * 60}")
        print(f"Running: {config.name}")
        print(f"  {config.description}")
        print(f"{'─' * 60}")
        
        reranker = self._get_reranker(config.reranker)
        
        # Metrics accumulators
        mrrs = []
        recalls_5 = []
        recalls_10 = []
        precisions_10 = []
        ndcgs_10 = []
        retrieval_latencies = []
        rerank_latencies = []
        total_latencies = []
        
        for i, q in enumerate(queries):
            # Retrieve
            docs, ret_lat = self.retrieve(q["query"], retrieve_k)
            
            # Rerank
            docs, rerank_lat = self.rerank(q["query"], docs, reranker, rerank_k)
            
            # Compute relevance
            relevances = [
                1 if is_relevant(d["text"], q["relevant_texts"]) else 0
                for d in docs
            ]
            
            # Compute metrics
            mrrs.append(reciprocal_rank(relevances))
            recalls_5.append(sum(relevances[:5]) / len(q["relevant_texts"]) if q["relevant_texts"] else 0)
            recalls_10.append(sum(relevances[:10]) / len(q["relevant_texts"]) if q["relevant_texts"] else 0)
            precisions_10.append(sum(relevances[:10]) / 10)
            ndcgs_10.append(ndcg_at_k(relevances, 10))
            
            retrieval_latencies.append(ret_lat)
            rerank_latencies.append(rerank_lat)
            total_latencies.append(ret_lat + rerank_lat)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(queries)} queries...")
        
        return AblationResult(
            config_name=config.name,
            query_type=config.query_type,
            reranker=config.reranker,
            mrr=np.mean(mrrs),
            recall_at_5=np.mean(recalls_5),
            recall_at_10=np.mean(recalls_10),
            precision_at_10=np.mean(precisions_10),
            ndcg_at_10=np.mean(ndcgs_10),
            avg_retrieval_ms=np.mean(retrieval_latencies),
            avg_rerank_ms=np.mean(rerank_latencies),
            avg_total_ms=np.mean(total_latencies),
            p95_total_ms=np.percentile(total_latencies, 95),
            num_queries=len(queries),
        )
    
    def run_all(self, num_queries: int = 30) -> List[AblationResult]:
        """Run all ablation experiments."""
        results = []
        
        # Load queries once per type
        oltp_queries = load_queries("oltp", num_queries)
        olap_queries = load_queries("olap", num_queries)
        
        for config in ABLATION_CONFIGS:
            queries = oltp_queries if config.query_type == "oltp" else olap_queries
            if not queries:
                print(f"Skipping {config.name}: no queries loaded")
                continue
            
            result = self.run_experiment(config, queries)
            results.append(result)
        
        return results


# =============================================================================
# Visualization
# =============================================================================

def plot_results(results: List[AblationResult], output_dir: Path):
    """Generate visualization plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Separate OLTP and OLAP results
    oltp_results = [r for r in results if r.query_type == "oltp"]
    olap_results = [r for r in results if r.query_type == "olap"]
    
    # Plot 1: OLTP Quality Comparison
    if oltp_results:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        names = [r.reranker or "none" for r in oltp_results]
        mrrs = [r.mrr for r in oltp_results]
        latencies = [r.avg_total_ms for r in oltp_results]
        
        axes[0].bar(names, mrrs, color="steelblue")
        axes[0].set_ylabel("MRR")
        axes[0].set_title("OLTP: Quality (MRR)")
        
        axes[1].bar(names, latencies, color="coral")
        axes[1].set_ylabel("Latency (ms)")
        axes[1].set_title("OLTP: Latency")
        
        plt.tight_layout()
        plt.savefig(output_dir / "oltp_ablation.png", dpi=150)
        plt.close()
    
    # Plot 2: OLAP Quality Comparison
    if olap_results:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        names = [r.reranker or "none" for r in olap_results]
        recalls = [r.recall_at_10 for r in olap_results]
        latencies = [r.avg_total_ms for r in olap_results]
        
        axes[0].bar(names, recalls, color="seagreen")
        axes[0].set_ylabel("Recall@10")
        axes[0].set_title("OLAP: Quality (Recall@10)")
        axes[0].tick_params(axis='x', rotation=15)
        
        axes[1].bar(names, latencies, color="coral")
        axes[1].set_ylabel("Latency (ms)")
        axes[1].set_title("OLAP: Latency")
        axes[1].tick_params(axis='x', rotation=15)
        
        plt.tight_layout()
        plt.savefig(output_dir / "olap_ablation.png", dpi=150)
        plt.close()
    
    # Plot 3: Quality vs Latency Tradeoff
    if olap_results:
        plt.figure(figsize=(8, 6))
        
        for r in olap_results:
            label = r.reranker or "none"
            plt.scatter(r.avg_total_ms, r.recall_at_10, s=100, label=label)
            plt.annotate(label, (r.avg_total_ms + 20, r.recall_at_10), fontsize=9)
        
        plt.xlabel("Latency (ms)")
        plt.ylabel("Recall@10")
        plt.title("OLAP: Quality vs Latency Tradeoff")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "olap_tradeoff.png", dpi=150)
        plt.close()
    
    print(f"\nPlots saved to {output_dir}/")


def print_summary(results: List[AblationResult]):
    """Print a summary table for the paper."""
    print("\n" + "═" * 80)
    print("ABLATION STUDY RESULTS")
    print("═" * 80)
    
    print(f"\n{'Config':<20} {'Type':<6} {'Reranker':<12} {'MRR':<8} {'R@10':<8} {'P@10':<8} {'Latency':<10}")
    print("─" * 80)
    
    for r in results:
        reranker_name = r.reranker or "none"
        print(f"{r.config_name:<20} {r.query_type:<6} {reranker_name:<12} "
              f"{r.mrr:<8.4f} {r.recall_at_10:<8.4f} {r.precision_at_10:<8.4f} "
              f"{r.avg_total_ms:<10.1f}ms")
    
    # Summary insights
    print("\n" + "─" * 80)
    print("KEY FINDINGS:")
    
    oltp = [r for r in results if r.query_type == "oltp"]
    olap = [r for r in results if r.query_type == "olap"]
    
    if len(oltp) >= 2:
        baseline = next((r for r in oltp if r.reranker is None), None)
        best = max(oltp, key=lambda x: x.mrr)
        if baseline and best.reranker:
            improvement = (best.mrr - baseline.mrr) / baseline.mrr * 100 if baseline.mrr > 0 else 0
            latency_cost = best.avg_total_ms - baseline.avg_total_ms
            print(f"  OLTP: {best.reranker} improves MRR by {improvement:.1f}% at +{latency_cost:.0f}ms latency")
    
    if len(olap) >= 2:
        baseline = next((r for r in olap if r.reranker is None), None)
        best = max(olap, key=lambda x: x.recall_at_10)
        if baseline and best.reranker:
            improvement = (best.recall_at_10 - baseline.recall_at_10) / baseline.recall_at_10 * 100 if baseline.recall_at_10 > 0 else 0
            latency_cost = best.avg_total_ms - baseline.avg_total_ms
            print(f"  OLAP: {best.reranker} improves Recall@10 by {improvement:.1f}% at +{latency_cost:.0f}ms latency")
    
    print("═" * 80)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Reranker Ablation Study")
    parser.add_argument("--qdrant-host", default="localhost")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument("--collection", default="oltp_chunks")
    parser.add_argument("--num-queries", type=int, default=30)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--output-dir", default="results", help="Output directory for results")
    args = parser.parse_args()
    
    print("═" * 80)
    print("RERANKER ABLATION STUDY")
    print("═" * 80)
    print(f"\nQdrant: {args.qdrant_host}:{args.qdrant_port}")
    print(f"Collection: {args.collection}")
    print(f"Device: {args.device}")
    print(f"Queries per type: {args.num_queries}")
    
    # Run experiments
    runner = AblationRunner(
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        collection=args.collection,
        device=args.device,
    )
    
    results = runner.run_all(args.num_queries)
    
    # Output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON results
    results_dict = [asdict(r) for r in results]
    with open(output_dir / "reranker_ablation.json", "w") as f:
        json.dump({"experiments": results_dict}, f, indent=2)
    print(f"\nResults saved to {output_dir / 'reranker_ablation.json'}")
    
    # Generate plots
    plot_results(results, output_dir)
    
    # Print summary
    print_summary(results)
    
    print("\n✅ Ablation study complete!")


if __name__ == "__main__":
    main()

