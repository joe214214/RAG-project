#!/usr/bin/env python3
"""Test query expansion functionality."""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.rag_pipeline import QueryAwareRAGPipeline

def main():
    print("=" * 60)
    print("QUERY EXPANSION TEST")
    print("=" * 60)
    
    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    print(f"\n1. API Key Check:")
    print(f"   Set: {bool(api_key)}")
    if api_key:
        print(f"   Value: {api_key[:20]}...{api_key[-10:]}")
    else:
        print("   ⚠️  OPENAI_API_KEY not set!")
        print("   Set it with: setenv OPENAI_API_KEY 'your-key'")
    
    # Test expansion
    print(f"\n2. Testing Query Expansion:")
    print(f"   Initializing pipeline...")
    
    try:
        pipeline = QueryAwareRAGPipeline(
            qdrant_host="ecetesla0",
            use_query_expansion=True,
        )
        
        # Test queries
        test_queries = [
            "machine learning algorithms",
            "what is artificial intelligence",
            "neural network training",
        ]
        
        for query in test_queries:
            print(f"\n   Query: '{query}'")
            expanded = pipeline._expand_query(query, top_n=3, use_llm=True)
            
            if expanded == query:
                print(f"   ⚠️  No expansion (returned original)")
            else:
                print(f"   ✅ Expanded: '{expanded}'")
                print(f"   Added terms: {expanded.replace(query, '').strip()}")
        
        print(f"\n✅ Query expansion test complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

