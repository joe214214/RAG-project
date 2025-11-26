"""
MS MARCO Dataset Loader

Loads MS MARCO passages and queries for OLTP (factoid) evaluation.
- Passages: Document corpus for retrieval
- Queries: Factoid questions for OLTP pipeline testing
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
import random


@dataclass
class Passage:
    """A passage from the corpus."""
    id: str
    text: str
    title: Optional[str] = None


@dataclass  
class Query:
    """A query with relevance judgments."""
    id: str
    text: str
    relevant_passage_ids: List[str]
    query_type: str = "oltp"  # MS MARCO queries are factoid (OLTP)


class MSMARCOLoader:
    """
    Loader for MS MARCO dataset.
    
    MS MARCO contains:
    - ~8.8M passages for retrieval
    - ~500K training queries with relevance labels
    - ~6,980 dev queries for evaluation
    
    For this project, we use subsets for tractability.
    """
    
    def __init__(self, cache_dir: str = "data/datasets/msmarco"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_passages(self, max_passages: int = 100000) -> List[Passage]:
        """
        Load passages from MS MARCO.
        
        Args:
            max_passages: Maximum number of passages to load
            
        Returns:
            List of Passage objects
        """
        cache_file = self.cache_dir / f"passages_{max_passages}.json"
        
        # Try loading from cache
        if cache_file.exists():
            print(f"Loading passages from cache: {cache_file}")
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [Passage(**p) for p in data]
        
        print(f"Downloading MS MARCO passages (up to {max_passages})...")
        
        # Import here to avoid dependency issues if not installed
        from datasets import load_dataset
        
        # Load MS MARCO v1.1 (has passages)
        dataset = load_dataset(
            "ms_marco",
            "v1.1",
            split="train",
            streaming=True,
        )
        
        passages = []
        seen_texts = set()  # Deduplicate by text
        
        for i, example in enumerate(dataset):
            if len(passages) >= max_passages:
                break
            
            # MS MARCO has passages in the 'passages' field
            if 'passages' in example:
                for j, passage_text in enumerate(example['passages']['passage_text']):
                    if passage_text.strip() and passage_text not in seen_texts:
                        seen_texts.add(passage_text)
                        passages.append(Passage(
                            id=f"msmarco_{len(passages)}",
                            text=passage_text.strip(),
                            title=None,
                        ))
                        
                        if len(passages) >= max_passages:
                            break
            
            if i % 5000 == 0 and i > 0:
                print(f"  Processed {i} examples, collected {len(passages)} unique passages")
        
        print(f"Loaded {len(passages)} passages")
        
        # Cache for future use
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump([{"id": p.id, "text": p.text, "title": p.title} for p in passages], f)
        
        return passages
    
    def load_queries(self, split: str = "validation", max_queries: int = 1000) -> List[Query]:
        """
        Load queries from MS MARCO.
        
        Args:
            split: Dataset split ('validation' for dev set)
            max_queries: Maximum number of queries to load
            
        Returns:
            List of Query objects (all marked as OLTP)
        """
        cache_file = self.cache_dir / f"queries_{split}_{max_queries}.json"
        
        # Try loading from cache
        if cache_file.exists():
            print(f"Loading queries from cache: {cache_file}")
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [Query(**q) for q in data]
        
        print(f"Downloading MS MARCO queries (up to {max_queries})...")
        
        from datasets import load_dataset
        
        dataset = load_dataset(
            "ms_marco",
            "v1.1",
            split=split,
            streaming=True,
        )
        
        queries = []
        
        for i, example in enumerate(dataset):
            if len(queries) >= max_queries:
                break
            
            query_text = example.get('query', '')
            if not query_text.strip():
                continue
            
            # Get relevant passage indices (if available)
            relevant_ids = []
            if 'passages' in example and 'is_selected' in example['passages']:
                for j, is_selected in enumerate(example['passages']['is_selected']):
                    if is_selected:
                        relevant_ids.append(f"msmarco_{j}")
            
            queries.append(Query(
                id=f"msmarco_q_{len(queries)}",
                text=query_text.strip(),
                relevant_passage_ids=relevant_ids,
                query_type="oltp",  # All MS MARCO queries are factoid
            ))
        
        print(f"Loaded {len(queries)} queries")
        
        # Cache for future use
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump([{
                "id": q.id,
                "text": q.text,
                "relevant_passage_ids": q.relevant_passage_ids,
                "query_type": q.query_type,
            } for q in queries], f)
        
        return queries


# Testing
if __name__ == "__main__":
    print("Testing MS MARCO Loader...")
    
    loader = MSMARCOLoader()
    
    # Load small subset for testing
    passages = loader.load_passages(max_passages=1000)
    print(f"\nLoaded {len(passages)} passages")
    if passages:
        print(f"Sample passage: {passages[0].text[:200]}...")
    
    queries = loader.load_queries(max_queries=100)
    print(f"\nLoaded {len(queries)} queries")
    if queries:
        print(f"Sample query: {queries[0].text}")
    
    print("\nMS MARCO Loader test PASSED!")
