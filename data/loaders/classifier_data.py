"""
Classifier Training Data Loader

Creates mixed OLTP/OLAP training data for the query classifier.
Combines MS MARCO (OLTP) and HotpotQA (OLAP) queries.
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

from .msmarco_loader import MSMARCOLoader, Query as MSMARCOQuery
from .hotpotqa_loader import HotpotQALoader, HotpotQuery


@dataclass
class ClassifierExample:
    """A training example for OLTP/OLAP classifier."""
    id: str
    text: str
    label: int  # 0 = OLTP, 1 = OLAP
    label_str: str  # "oltp" or "olap"
    source: str  # "msmarco" or "hotpotqa"


class ClassifierDataLoader:
    """
    Creates balanced training data for OLTP/OLAP query classifier.
    
    Combines:
    - MS MARCO queries → OLTP (factoid)
    - HotpotQA queries → OLAP (multi-hop)
    """
    
    OLTP_LABEL = 0
    OLAP_LABEL = 1
    
    def __init__(self, cache_dir: str = "data/datasets/classifier"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.msmarco_loader = MSMARCOLoader()
        self.hotpotqa_loader = HotpotQALoader()
    
    def create_training_data(
        self,
        n_oltp: int = 500,
        n_olap: int = 500,
        seed: int = 42,
    ) -> List[ClassifierExample]:
        """
        Create balanced training dataset.
        
        Args:
            n_oltp: Number of OLTP examples
            n_olap: Number of OLAP examples
            seed: Random seed for reproducibility
            
        Returns:
            List of ClassifierExample objects
        """
        cache_file = self.cache_dir / f"train_{n_oltp}_{n_olap}.json"
        
        if cache_file.exists():
            print(f"Loading classifier training data from cache: {cache_file}")
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [ClassifierExample(**e) for e in data]
        
        random.seed(seed)
        
        print("Creating classifier training data...")
        
        # Load OLTP examples from MS MARCO
        print(f"Loading {n_oltp} OLTP examples from MS MARCO...")
        msmarco_queries = self.msmarco_loader.load_queries(max_queries=n_oltp * 2)
        random.shuffle(msmarco_queries)
        
        oltp_examples = [
            ClassifierExample(
                id=f"clf_oltp_{i}",
                text=q.text,
                label=self.OLTP_LABEL,
                label_str="oltp",
                source="msmarco",
            )
            for i, q in enumerate(msmarco_queries[:n_oltp])
        ]
        
        # Load OLAP examples from HotpotQA
        print(f"Loading {n_olap} OLAP examples from HotpotQA...")
        hotpot_queries = self.hotpotqa_loader.load_queries(max_queries=n_olap * 2)
        random.shuffle(hotpot_queries)
        
        olap_examples = [
            ClassifierExample(
                id=f"clf_olap_{i}",
                text=q.text,
                label=self.OLAP_LABEL,
                label_str="olap",
                source="hotpotqa",
            )
            for i, q in enumerate(hotpot_queries[:n_olap])
        ]
        
        # Combine and shuffle
        all_examples = oltp_examples + olap_examples
        random.shuffle(all_examples)
        
        print(f"Created {len(all_examples)} training examples "
              f"({len(oltp_examples)} OLTP, {len(olap_examples)} OLAP)")
        
        # Cache
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump([{
                "id": e.id,
                "text": e.text,
                "label": e.label,
                "label_str": e.label_str,
                "source": e.source,
            } for e in all_examples], f)
        
        return all_examples
    
    def split_data(
        self,
        examples: List[ClassifierExample],
        train_ratio: float = 0.8,
        seed: int = 42,
    ) -> Tuple[List[ClassifierExample], List[ClassifierExample]]:
        """
        Split data into train and validation sets.
        
        Args:
            examples: All examples
            train_ratio: Fraction for training
            seed: Random seed
            
        Returns:
            (train_examples, val_examples)
        """
        random.seed(seed)
        examples_copy = examples.copy()
        random.shuffle(examples_copy)
        
        split_idx = int(len(examples_copy) * train_ratio)
        train = examples_copy[:split_idx]
        val = examples_copy[split_idx:]
        
        print(f"Split: {len(train)} train, {len(val)} validation")
        return train, val
    
    def get_label_distribution(
        self, 
        examples: List[ClassifierExample]
    ) -> dict:
        """Get label distribution statistics."""
        oltp_count = sum(1 for e in examples if e.label == self.OLTP_LABEL)
        olap_count = sum(1 for e in examples if e.label == self.OLAP_LABEL)
        
        return {
            "total": len(examples),
            "oltp": oltp_count,
            "olap": olap_count,
            "oltp_ratio": oltp_count / len(examples) if examples else 0,
            "olap_ratio": olap_count / len(examples) if examples else 0,
        }
    
    def to_huggingface_format(
        self,
        examples: List[ClassifierExample],
    ) -> dict:
        """
        Convert to format suitable for HuggingFace training.
        
        Returns:
            dict with 'text' and 'label' lists
        """
        return {
            "text": [e.text for e in examples],
            "label": [e.label for e in examples],
        }


# Testing
if __name__ == "__main__":
    print("Testing Classifier Data Loader...")
    
    loader = ClassifierDataLoader()
    
    # Create small training set
    examples = loader.create_training_data(n_oltp=50, n_olap=50)
    print(f"\nCreated {len(examples)} examples")
    
    # Check distribution
    dist = loader.get_label_distribution(examples)
    print(f"Distribution: {dist}")
    
    # Split
    train, val = loader.split_data(examples)
    
    # Sample examples
    print("\nSample OLTP query:")
    oltp_sample = next(e for e in examples if e.label_str == "oltp")
    print(f"  {oltp_sample.text}")
    
    print("\nSample OLAP query:")
    olap_sample = next(e for e in examples if e.label_str == "olap")
    print(f"  {olap_sample.text}")
    
    # HuggingFace format
    hf_data = loader.to_huggingface_format(train)
    print(f"\nHuggingFace format: {len(hf_data['text'])} texts, {len(hf_data['label'])} labels")
    
    print("\nClassifier Data Loader test PASSED!")

