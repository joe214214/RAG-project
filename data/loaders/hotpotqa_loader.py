"""
HotpotQA Dataset Loader

Loads HotpotQA for OLAP (multi-hop) evaluation.
- Questions require reasoning over multiple documents
- Provides supporting facts for evaluation
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import random


@dataclass
class HotpotQuery:
    """A multi-hop query from HotpotQA."""
    id: str
    text: str  # The question
    answer: str
    supporting_facts: List[Dict]  # Title + sentence index pairs
    context_titles: List[str]
    query_type: str = "olap"  # All HotpotQA queries are multi-hop (OLAP)
    level: str = "medium"  # easy, medium, hard


@dataclass
class HotpotContext:
    """Context passage from HotpotQA."""
    id: str
    title: str
    sentences: List[str]
    
    @property
    def text(self) -> str:
        return " ".join(self.sentences)


class HotpotQALoader:
    """
    Loader for HotpotQA dataset.
    
    HotpotQA contains:
    - ~113K questions (90K train, 7.4K dev)
    - Multi-hop reasoning required
    - Supporting facts annotated
    
    Types:
    - Bridge: Requires finding intermediate entity
    - Comparison: Requires comparing two entities
    """
    
    def __init__(self, cache_dir: str = "data/datasets/hotpotqa"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_queries(
        self, 
        split: str = "validation",
        max_queries: int = 1000,
    ) -> List[HotpotQuery]:
        """
        Load multi-hop queries from HotpotQA.
        
        Args:
            split: Dataset split ('train' or 'validation')
            max_queries: Maximum queries to load
            
        Returns:
            List of HotpotQuery objects (all marked as OLAP)
        """
        cache_file = self.cache_dir / f"queries_{split}_{max_queries}.json"
        
        # Try loading from cache
        if cache_file.exists():
            print(f"Loading HotpotQA queries from cache: {cache_file}")
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [HotpotQuery(**q) for q in data]
        
        print(f"Downloading HotpotQA queries (up to {max_queries})...")
        
        from datasets import load_dataset
        
        # Load HotpotQA distractor setting (harder)
        dataset = load_dataset(
            "hotpot_qa",
            "distractor",
            split=split,
        )
        
        queries = []
        
        for i, example in enumerate(dataset):
            if len(queries) >= max_queries:
                break
            
            question = example.get('question', '')
            if not question.strip():
                continue
            
            # Extract supporting facts
            supporting_facts = []
            if 'supporting_facts' in example:
                sf = example['supporting_facts']
                if 'title' in sf and 'sent_id' in sf:
                    for title, sent_id in zip(sf['title'], sf['sent_id']):
                        supporting_facts.append({
                            "title": title,
                            "sent_id": sent_id
                        })
            
            # Extract context titles
            context_titles = example.get('context', {}).get('title', [])
            
            queries.append(HotpotQuery(
                id=example.get('id', f"hotpot_{i}"),
                text=question.strip(),
                answer=example.get('answer', ''),
                supporting_facts=supporting_facts,
                context_titles=context_titles if isinstance(context_titles, list) else [],
                query_type="olap",
                level=example.get('level', 'medium'),
            ))
        
        print(f"Loaded {len(queries)} HotpotQA queries")
        
        # Cache for future use
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump([{
                "id": q.id,
                "text": q.text,
                "answer": q.answer,
                "supporting_facts": q.supporting_facts,
                "context_titles": q.context_titles,
                "query_type": q.query_type,
                "level": q.level,
            } for q in queries], f)
        
        return queries
    
    def load_contexts(
        self,
        split: str = "validation",
        max_contexts: int = 10000,
    ) -> List[HotpotContext]:
        """
        Load context passages from HotpotQA.
        
        These are Wikipedia paragraphs that contain evidence for queries.
        """
        cache_file = self.cache_dir / f"contexts_{split}_{max_contexts}.json"
        
        if cache_file.exists():
            print(f"Loading HotpotQA contexts from cache: {cache_file}")
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [HotpotContext(**c) for c in data]
        
        print(f"Downloading HotpotQA contexts (up to {max_contexts})...")
        
        from datasets import load_dataset
        
        dataset = load_dataset(
            "hotpot_qa",
            "distractor",
            split=split,
        )
        
        contexts = {}  # Dedupe by title
        
        for example in dataset:
            if len(contexts) >= max_contexts:
                break
            
            context_data = example.get('context', {})
            titles = context_data.get('title', [])
            sentences_list = context_data.get('sentences', [])
            
            for title, sentences in zip(titles, sentences_list):
                if title not in contexts:
                    # Use title-based ID for exact matching with supporting facts
                    # Format: "hotpot_ctx_{title}" matches evaluation expectations
                    context_id = f"hotpot_ctx_{title}"
                    contexts[title] = HotpotContext(
                        id=context_id,  # Title-based ID for exact matching
                        title=title,
                        sentences=sentences if isinstance(sentences, list) else [sentences],
                    )
        
        context_list = list(contexts.values())
        print(f"Loaded {len(context_list)} unique contexts")
        
        # Cache
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump([{
                "id": c.id,
                "title": c.title,
                "sentences": c.sentences,
            } for c in context_list], f)
        
        return context_list


# Testing
if __name__ == "__main__":
    print("Testing HotpotQA Loader...")
    
    loader = HotpotQALoader()
    
    # Load queries
    queries = loader.load_queries(max_queries=100)
    print(f"\nLoaded {len(queries)} queries")
    if queries:
        q = queries[0]
        print(f"Sample query: {q.text}")
        print(f"Answer: {q.answer}")
        print(f"Level: {q.level}")
        print(f"Supporting facts: {len(q.supporting_facts)} facts")
    
    # Load contexts
    contexts = loader.load_contexts(max_contexts=500)
    print(f"\nLoaded {len(contexts)} contexts")
    if contexts:
        c = contexts[0]
        print(f"Sample context title: {c.title}")
        print(f"Text preview: {c.text[:200]}...")
    
    print("\nHotpotQA Loader test PASSED!")

