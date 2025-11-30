#!/usr/bin/env python3
"""
Test script for multi-granular chunking.

Run on ecetesla0 to verify chunking works with real datasets.

Usage:
    python scripts/test_chunking.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_basic_chunking():
    """Test basic chunking functionality."""
    print("\n" + "=" * 60)
    print("Test 1: Basic Chunking")
    print("=" * 60)
    
    from preprocess.chunking import create_chunker
    
    sample_text = """
# Introduction

This is a comprehensive introduction that explains the main concepts. 
We will cover multiple topics including data processing, analysis methods,
and key findings from our research. The goal is to provide readers with
a solid foundation for understanding the rest of the document.

# Methods

## Data Collection

Our data collection methodology involved gathering information from 
multiple sources. We conducted surveys with over 500 participants,
analyzed public datasets, and performed interviews with domain experts.
The process took approximately three months to complete.

## Analysis

We used both quantitative and qualitative analysis methods. Statistical
analysis was performed using Python with pandas and scikit-learn libraries.
The qualitative analysis followed standard thematic coding procedures.

# Results

The results demonstrate significant improvements across all metrics.
We observed a 25% increase in accuracy and 30% reduction in latency.
These findings were consistent across different test conditions.

# Conclusion

In conclusion, our approach shows promising results for real-world
applications. Future work should focus on scaling and edge cases.
"""
    
    chunker = create_chunker()
    result = chunker.chunk_document(sample_text, source_file="test_doc")
    
    print(f"\n✓ OLTP chunks created: {len(result['oltp'])}")
    print(f"✓ OLAP parent chunks: {len(result['olap_parents'])}")
    print(f"✓ OLAP child chunks: {len(result['olap_children'])}")
    
    # Verify structure
    assert len(result['oltp']) > 0, "Should have OLTP chunks"
    assert len(result['olap_parents']) > 0, "Should have OLAP parents"
    
    # Check metadata
    oltp_chunk = result['oltp'][0]
    assert oltp_chunk.chunk_type == "oltp", "Should be OLTP type"
    assert oltp_chunk.source_file == "test_doc", "Should have source file"
    
    print("\n✓ All basic tests passed!")
    return True


def test_with_msmarco():
    """Test chunking with MS MARCO passages."""
    print("\n" + "=" * 60)
    print("Test 2: Chunking MS MARCO Passages")
    print("=" * 60)
    
    from preprocess.chunking import create_chunker
    from data.loaders.msmarco_loader import MSMARCOLoader
    
    # Load small sample
    loader = MSMARCOLoader()
    passages = loader.load_passages(max_passages=50)
    
    print(f"Loaded {len(passages)} passages")
    
    # Convert to dict format
    passage_dicts = [{"id": p.id, "text": p.text} for p in passages]
    
    # Chunk
    chunker = create_chunker()
    result = chunker.chunk_passages(passage_dicts)
    
    print(f"\n✓ OLTP chunks: {len(result['oltp'])}")
    print(f"✓ OLAP parents: {len(result['olap_parents'])}")
    print(f"✓ OLAP children: {len(result['olap_children'])}")
    
    # Sample output
    if result['oltp']:
        sample = result['oltp'][0]
        print(f"\nSample OLTP chunk:")
        print(f"  ID: {sample.id}")
        print(f"  Text: {sample.text[:100]}...")
        print(f"  Source: {sample.source_file}")
    
    return True


def test_with_hotpotqa():
    """Test chunking with HotpotQA contexts."""
    print("\n" + "=" * 60)
    print("Test 3: Chunking HotpotQA Contexts")
    print("=" * 60)
    
    from preprocess.chunking import create_chunker
    from data.loaders.hotpotqa_loader import HotpotQALoader
    
    # Load small sample
    loader = HotpotQALoader()
    contexts = loader.load_contexts(max_contexts=50)
    
    print(f"Loaded {len(contexts)} contexts")
    
    # Convert to dict format
    context_dicts = [{"id": c.id, "text": c.text, "title": c.title} for c in contexts]
    
    # Chunk
    chunker = create_chunker()
    result = chunker.chunk_passages(context_dicts)
    
    print(f"\n✓ OLTP chunks: {len(result['oltp'])}")
    print(f"✓ OLAP parents: {len(result['olap_parents'])}")
    print(f"✓ OLAP children: {len(result['olap_children'])}")
    
    # Check hierarchical linking
    if result['olap_parents'] and result['olap_children']:
        parent = result['olap_parents'][0]
        if parent.child_ids:
            print(f"\nHierarchical linking check:")
            print(f"  Parent: {parent.id}")
            print(f"  Children: {parent.child_ids[:3]}...")
    
    return True


def test_deduplication():
    """Test that deduplication works."""
    print("\n" + "=" * 60)
    print("Test 4: Deduplication")
    print("=" * 60)
    
    from preprocess.chunking import create_chunker
    
    # Text with repetition
    repeated_text = """
This is a repeated sentence that appears multiple times in the document.
This is a repeated sentence that appears multiple times in the document.
This is a repeated sentence that appears multiple times in the document.

Some unique content here that should be preserved.

This is a repeated sentence that appears multiple times in the document.
This is a repeated sentence that appears multiple times in the document.
"""
    
    chunker = create_chunker()
    result = chunker.chunk_document(repeated_text, source_file="dedup_test")
    
    # Check for unique hashes
    hashes = [c.text_hash for c in result['oltp']]
    unique_hashes = set(hashes)
    
    print(f"Total OLTP chunks: {len(result['oltp'])}")
    print(f"Unique hashes: {len(unique_hashes)}")
    print(f"✓ Deduplication working: {len(unique_hashes) == len(result['oltp'])}")
    
    return True


def test_metadata_preservation():
    """Test that metadata is properly preserved."""
    print("\n" + "=" * 60)
    print("Test 5: Metadata Preservation")
    print("=" * 60)
    
    from preprocess.chunking import create_chunker
    
    text_with_headings = """
# Main Topic

Introduction paragraph here.

## Subtopic A

Content for subtopic A goes here with enough text to form a chunk.
This is important information about the subtopic that readers need.

## Subtopic B

Content for subtopic B with different information.
More details about this particular subtopic follow.
"""
    
    chunker = create_chunker()
    result = chunker.chunk_document(
        text_with_headings,
        source_file="metadata_test.md",
        heading_path=["Document"]
    )
    
    # Check OLAP chunks have heading paths
    for parent in result['olap_parents']:
        print(f"Parent: {parent.section}")
        print(f"  Heading path: {parent.heading_path}")
        print(f"  Source: {parent.source_file}")
        print(f"  Level: {parent.level}")
    
    # Verify
    assert any(p.heading_path for p in result['olap_parents']), "Should have heading paths"
    assert all(p.source_file == "metadata_test.md" for p in result['olap_parents']), "Should preserve source"
    
    print("\n✓ Metadata preservation working!")
    return True


def main():
    print("=" * 60)
    print("RAG System - Chunking Tests")
    print("=" * 60)
    
    results = {}
    
    try:
        results['basic'] = test_basic_chunking()
    except Exception as e:
        print(f"\n✗ Basic test failed: {e}")
        results['basic'] = False
    
    try:
        results['msmarco'] = test_with_msmarco()
    except Exception as e:
        print(f"\n✗ MS MARCO test failed: {e}")
        results['msmarco'] = False
    
    try:
        results['hotpotqa'] = test_with_hotpotqa()
    except Exception as e:
        print(f"\n✗ HotpotQA test failed: {e}")
        results['hotpotqa'] = False
    
    try:
        results['dedup'] = test_deduplication()
    except Exception as e:
        print(f"\n✗ Deduplication test failed: {e}")
        results['dedup'] = False
    
    try:
        results['metadata'] = test_metadata_preservation()
    except Exception as e:
        print(f"\n✗ Metadata test failed: {e}")
        results['metadata'] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All chunking tests passed!")
    else:
        print("\n⚠️  Some tests failed. Check errors above.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

