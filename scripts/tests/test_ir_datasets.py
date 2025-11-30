#!/usr/bin/env python3
"""Test ir_datasets integration."""

try:
    import ir_datasets
    print("✅ ir_datasets imported successfully")
    
    # Try loading MS MARCO
    try:
        dataset = ir_datasets.load("msmarco-passage/dev")
        print(f"✅ MS MARCO dataset loaded: {dataset}")
        print(f"   Queries: {dataset.queries_count()}")
        print(f"   Qrels: {dataset.qrels_count()}")
    except Exception as e:
        print(f"⚠ Could not load MS MARCO: {e}")
        
except ImportError as e:
    print(f"❌ ir_datasets not available: {e}")
    print("   Install with: pip install ir-datasets")

