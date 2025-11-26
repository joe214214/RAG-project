#!/usr/bin/env python3
"""
Test script for data loaders.

Run on ecetesla0 to verify datasets download and load correctly.

Usage:
    python scripts/test_loaders.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_msmarco():
    """Test MS MARCO loader."""
    print("\n" + "="*60)
    print("Testing MS MARCO Loader (OLTP)")
    print("="*60)
    
    from data.loaders.msmarco_loader import MSMARCOLoader
    
    loader = MSMARCOLoader()
    
    # Load small subset
    print("\nLoading 100 passages...")
    passages = loader.load_passages(max_passages=100)
    print(f"✓ Loaded {len(passages)} passages")
    
    if passages:
        print(f"  Sample ID: {passages[0].id}")
        print(f"  Sample text: {passages[0].text[:100]}...")
    
    print("\nLoading 50 queries...")
    queries = loader.load_queries(max_queries=50)
    print(f"✓ Loaded {len(queries)} queries")
    
    if queries:
        print(f"  Sample query: {queries[0].text}")
        print(f"  Query type: {queries[0].query_type}")
    
    return True


def test_hotpotqa():
    """Test HotpotQA loader."""
    print("\n" + "="*60)
    print("Testing HotpotQA Loader (OLAP)")
    print("="*60)
    
    from data.loaders.hotpotqa_loader import HotpotQALoader
    
    loader = HotpotQALoader()
    
    # Load queries
    print("\nLoading 50 multi-hop queries...")
    queries = loader.load_queries(max_queries=50)
    print(f"✓ Loaded {len(queries)} queries")
    
    if queries:
        q = queries[0]
        print(f"  Sample query: {q.text}")
        print(f"  Answer: {q.answer}")
        print(f"  Level: {q.level}")
        print(f"  Query type: {q.query_type}")
    
    # Load contexts
    print("\nLoading 100 context passages...")
    contexts = loader.load_contexts(max_contexts=100)
    print(f"✓ Loaded {len(contexts)} contexts")
    
    if contexts:
        c = contexts[0]
        print(f"  Sample title: {c.title}")
        print(f"  Text preview: {c.text[:100]}...")
    
    return True


def test_classifier_data():
    """Test classifier data loader."""
    print("\n" + "="*60)
    print("Testing Classifier Data Loader")
    print("="*60)
    
    from data.loaders.classifier_data import ClassifierDataLoader
    
    loader = ClassifierDataLoader()
    
    # Create small training set
    print("\nCreating classifier training data (25 OLTP + 25 OLAP)...")
    examples = loader.create_training_data(n_oltp=25, n_olap=25)
    print(f"✓ Created {len(examples)} examples")
    
    # Check distribution
    dist = loader.get_label_distribution(examples)
    print(f"  OLTP: {dist['oltp']} ({dist['oltp_ratio']:.0%})")
    print(f"  OLAP: {dist['olap']} ({dist['olap_ratio']:.0%})")
    
    # Split
    train, val = loader.split_data(examples)
    print(f"  Train: {len(train)}, Val: {len(val)}")
    
    # Sample examples
    oltp_sample = next((e for e in examples if e.label_str == "oltp"), None)
    olap_sample = next((e for e in examples if e.label_str == "olap"), None)
    
    if oltp_sample:
        print(f"\n  OLTP example: {oltp_sample.text[:80]}...")
    if olap_sample:
        print(f"  OLAP example: {olap_sample.text[:80]}...")
    
    return True


def main():
    print("="*60)
    print("RAG System - Data Loader Tests")
    print("="*60)
    
    results = {}
    
    try:
        results['msmarco'] = test_msmarco()
    except Exception as e:
        print(f"\n✗ MS MARCO test failed: {e}")
        results['msmarco'] = False
    
    try:
        results['hotpotqa'] = test_hotpotqa()
    except Exception as e:
        print(f"\n✗ HotpotQA test failed: {e}")
        results['hotpotqa'] = False
    
    try:
        results['classifier'] = test_classifier_data()
    except Exception as e:
        print(f"\n✗ Classifier data test failed: {e}")
        results['classifier'] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed. Check errors above.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

