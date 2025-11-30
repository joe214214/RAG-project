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
        
        Uses sentence-transformers/msmarco corpus subset which includes passage_id
        for exact ID matching (research standard).
        
        Args:
            max_passages: Maximum number of passages to load
            
        Returns:
            List of Passage objects with actual passage IDs
        """
        cache_file = self.cache_dir / f"passages_{max_passages}.json"
        
        # Try loading from cache
        if cache_file.exists():
            print(f"Loading passages from cache: {cache_file}")
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [Passage(**p) for p in data]
        
        print(f"Downloading MS MARCO passages (up to {max_passages})...")
        print("Using sentence-transformers/msmarco corpus (includes passage_id for exact matching)")
        
        # Import here to avoid dependency issues if not installed
        from datasets import load_dataset
        
        # Use sentence-transformers/msmarco corpus which has passage_id and passage
        # This provides actual passage IDs for research-standard evaluation
        try:
            dataset = load_dataset(
                "sentence-transformers/msmarco",
                "corpus",
                split="train",
                streaming=True,
            )
        except Exception as e:
            print(f"Warning: Could not load sentence-transformers/msmarco corpus: {e}")
            print("Falling back to microsoft/ms_marco v1.1 (may not have passage_id)")
            # Fallback to original method
            dataset = load_dataset(
                "ms_marco",
                "v1.1",
                split="train",
                streaming=True,
            )
            passages = []
            seen_texts = set()
            for i, example in enumerate(dataset):
                if len(passages) >= max_passages:
                    break
                if 'passages' in example:
                    passage_texts = example['passages'].get('passage_text', [])
                    for passage_text in passage_texts:
                        if passage_text.strip() and passage_text not in seen_texts:
                            seen_texts.add(passage_text)
                            passage_id = f"msmarco_{len(passages)}"
                            passages.append(Passage(
                                id=passage_id,
                                text=passage_text.strip(),
                                title=None,
                            ))
                            if len(passages) >= max_passages:
                                break
                if i % 5000 == 0 and i > 0:
                    print(f"  Processed {i} examples, collected {len(passages)} unique passages")
            print(f"Loaded {len(passages)} passages")
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump([{"id": p.id, "text": p.text, "title": p.title} for p in passages], f)
            return passages
        
        passages = []
        seen_ids = set()  # Deduplicate by passage_id
        
        for i, example in enumerate(dataset):
            if len(passages) >= max_passages:
                break
            
            # sentence-transformers/msmarco corpus has: passage_id, passage
            passage_id = example.get('passage_id') or example.get('id')
            passage_text = example.get('passage') or example.get('passage_text') or example.get('text')
            
            if not passage_text or not passage_text.strip():
                continue
            
            # Use actual passage_id from dataset (research standard)
            if passage_id:
                passage_id = str(passage_id)
            else:
                # Fallback: construct ID (shouldn't happen with sentence-transformers/msmarco)
                passage_id = f"msmarco_{len(passages)}"
            
            # Deduplicate by passage_id
            if passage_id in seen_ids:
                continue
            seen_ids.add(passage_id)
            
            passages.append(Passage(
                id=passage_id,  # Actual passage ID from dataset
                text=passage_text.strip(),
                title=None,
            ))
            
            if i % 10000 == 0 and i > 0:
                print(f"  Processed {i} examples, collected {len(passages)} passages")
        
        print(f"Loaded {len(passages)} passages with actual passage IDs")
        
        # Cache for future use
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump([{"id": p.id, "text": p.text, "title": p.title} for p in passages], f)
        
        return passages
    
    def load_queries(self, split: str = "validation", max_queries: int = 1000) -> List[Query]:
        """
        Load queries from MS MARCO with relevant passage IDs.
        
        Uses sentence-transformers/msmarco queries + labeled-list subsets
        which provide query_id and positive passage IDs for exact matching.
        
        Args:
            split: Dataset split ('validation' for dev set, 'train' for training)
            max_queries: Maximum number of queries to load
            
        Returns:
            List of Query objects with actual passage IDs (all marked as OLTP)
        """
        cache_file = self.cache_dir / f"queries_{split}_{max_queries}.json"
        
        # Try loading from cache
        if cache_file.exists():
            print(f"Loading queries from cache: {cache_file}")
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [Query(**q) for q in data]
        
        print(f"Downloading MS MARCO queries (up to {max_queries})...")
        print("Using sentence-transformers/msmarco queries + labeled-list (includes passage IDs)")
        
        from datasets import load_dataset
        
        queries = []
        query_texts = {}  # query_id -> query text
        query_relevant_ids = {}  # query_id -> list of relevant passage IDs
        
        try:
            # Load queries (query_id, query)
            queries_dataset = load_dataset(
                "sentence-transformers/msmarco",
                "queries",
                split=split if split != "validation" else "dev",
                streaming=True,
            )
            
            for example in queries_dataset:
                query_id = str(example.get('query_id') or example.get('id'))
                query_text = example.get('query') or example.get('query_text')
                if query_id and query_text:
                    query_texts[query_id] = query_text.strip()
                    if len(query_texts) >= max_queries * 2:  # Load extra for matching
                        break
            
            # Load labeled-list (query_id, positive_ids) for relevant passage IDs
            labeled_dataset = load_dataset(
                "sentence-transformers/msmarco",
                "labeled-list",
                split=split if split != "validation" else "dev",
                streaming=True,
            )
            
            for example in labeled_dataset:
                query_id = str(example.get('query_id') or example.get('id'))
                positive_ids = example.get('positive_ids') or example.get('positive_id') or []
                
                if isinstance(positive_ids, (int, str)):
                    positive_ids = [positive_ids]
                
                if query_id:
                    # Convert all to strings for consistency
                    query_relevant_ids[query_id] = [str(pid) for pid in positive_ids]
                    if len(query_relevant_ids) >= max_queries:
                        break
            
            # Combine queries and relevant IDs
            for query_id, query_text in list(query_texts.items())[:max_queries]:
                relevant_ids = query_relevant_ids.get(query_id, [])
                queries.append(Query(
                    id=query_id,
                    text=query_text,
                    relevant_passage_ids=relevant_ids,
                    query_type="oltp",
                ))
                
        except Exception as e:
            print(f"Warning: Could not load sentence-transformers/msmarco queries: {e}")
            print("Falling back to microsoft/ms_marco v1.1 (may not have passage_id)")
            # Fallback to original method
            dataset = load_dataset(
                "ms_marco",
                "v1.1",
                split=split,
                streaming=True,
            )
            
            for i, example in enumerate(dataset):
                if len(queries) >= max_queries:
                    break
                
                query_text = example.get('query', '')
                if not query_text.strip():
                    continue
                
                queryid = example.get('queryid') or example.get('query_id')
                if not queryid:
                    queryid = f"msmarco_q_{len(queries)}"
                else:
                    queryid = str(queryid)
                
                relevant_ids = []
                if 'passages' in example:
                    passages_data = example['passages']
                    is_selected = passages_data.get('is_selected', [])
                    for j, is_selected_flag in enumerate(is_selected):
                        if is_selected_flag:
                            relevant_ids.append(f"msmarco_{j}")
                
                queries.append(Query(
                    id=queryid,
                    text=query_text.strip(),
                    relevant_passage_ids=relevant_ids,
                    query_type="oltp",
                ))
        
        print(f"Loaded {len(queries)} queries")
        print(f"  Queries with passage IDs: {sum(1 for q in queries if q.relevant_passage_ids)}")
        
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
