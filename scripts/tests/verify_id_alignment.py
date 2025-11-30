#!/usr/bin/env python3
"""
Verify ID Alignment Between Ground Truth and Qdrant

This script checks if the ground truth passage IDs match what's stored in Qdrant.
It samples queries with mrr=0 and verifies if relevant passages exist.
"""

import argparse
import json
import re
import sys
import signal
from pathlib import Path
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from datasets import load_dataset

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_eval_results(results_file: str) -> Dict:
    """Load comprehensive evaluation results."""
    with open(results_file, 'r') as f:
        return json.load(f)


def find_failing_queries(results: Dict, num_samples: int = 10) -> List[Dict]:
    """Find queries with mrr=0 and num_relevant_total=1."""
    failing_queries = []
    
    for config in results.get('configs', []):
        for query_result in config.get('queries', []):
            if (query_result.get('mrr', 0) == 0 and 
                query_result.get('num_relevant_total', 0) == 1):
                failing_queries.append({
                    'query': query_result.get('query', ''),
                    'query_type': query_result.get('query_type', ''),
                    'config': config['config']['name'],
                    'num_relevant_total': query_result.get('num_relevant_total', 0),
                    'num_relevant_found': query_result.get('num_relevant_found', 0),
                })
                if len(failing_queries) >= num_samples:
                    break
        if len(failing_queries) >= num_samples:
            break
    
    return failing_queries


def get_ground_truth_ids(query: str, query_type: str, limit: int = 1) -> List[Dict]:
    """Get ground truth passage IDs for a query from datasets."""
    gt_ids = []
    
    try:
        if query_type == "oltp":
            # Skip sentence-transformers/msmarco for validation queries - they're likely not in training set
            # This avoids hanging on millions of training queries
            # Instead, use microsoft/ms_marco v1.1 validation split directly
            use_sentence_transformers = False  # Disabled to avoid hanging
            
            if use_sentence_transformers:
                try:
                    # Load queries to find query_id (use "train" split - sentence-transformers/msmarco only has train)
                    # Limit search to first 5K queries to avoid hanging on large datasets
                    queries_ds = load_dataset("sentence-transformers/msmarco", "queries", split="train", streaming=True)
                    query_id = None
                    checked_queries = 0
                    max_queries_to_check = 5000  # Small limit to avoid hanging
                    
                    for item in queries_ds:
                        checked_queries += 1
                        if checked_queries > max_queries_to_check:
                            break  # Stop searching after limit
                        
                        item_query = item.get("query") or item.get("query_text", "")
                        if item_query.lower() == query.lower():
                            query_id = str(item.get("query_id") or item.get("id"))
                            break
                        
                        # Progress indicator every 500 queries
                        if checked_queries % 500 == 0:
                            print(f"    Searched {checked_queries} queries...", end='\r')
                    
                        if query_id:
                            # Load labeled-list to get relevant passage IDs (use "train" split)
                            # Limit search to avoid hanging
                            labeled_ds = load_dataset("sentence-transformers/msmarco", "labeled-list", split="train", streaming=True)
                            positive_ids = []
                            checked_labels = 0
                            max_labels_to_check = 50000  # Limit for labeled-list search
                            
                            for item in labeled_ds:
                                checked_labels += 1
                                if checked_labels > max_labels_to_check:
                                    break
                                
                                if str(item.get("query_id") or item.get("id")) == query_id:
                                    pos_ids = item.get("positive_ids") or item.get("positive_id") or []
                                    if isinstance(pos_ids, (int, str)):
                                        pos_ids = [pos_ids]
                                    positive_ids = [str(pid) for pid in pos_ids]
                                    break
                            
                            # Get passage texts from corpus (for text matching fallback)
                            # Limit corpus search to avoid hanging
                            if positive_ids:
                                corpus_ds = load_dataset("sentence-transformers/msmarco", "corpus", split="train", streaming=True)
                                found_count = 0
                                checked_passages = 0
                                max_passages_to_check = 100000  # Limit corpus search
                                
                                for pid in positive_ids:
                                    if found_count >= limit:
                                        break
                                    # Find passage in corpus
                                    for passage in corpus_ds:
                                        checked_passages += 1
                                        if checked_passages > max_passages_to_check:
                                            break  # Stop if we've checked too many
                                        
                                        passage_id_val = passage.get("passage_id") or passage.get("id")
                                        if str(passage_id_val) == pid:
                                            passage_text = passage.get("passage") or passage.get("passage_text") or passage.get("text") or ""
                                            gt_ids.append({
                                                'passage_id': pid,  # Actual passage ID from dataset
                                                'text': passage_text[:200] + "..." if len(passage_text) > 200 else passage_text,
                                            })
                                            found_count += 1
                                            break
                                    
                                    if checked_passages > max_passages_to_check:
                                        break
                except Exception as e:
                    print(f"  Warning: Could not load sentence-transformers/msmarco: {e}")
                print(f"  Falling back to microsoft/ms_marco v1.1 (may not have passage_id)")
            
            # If sentence-transformers/msmarco didn't find the query, skip it (don't try fallback)
            # The fallback is too slow and the queries might not be in the training set anyway
            if gt_ids:
                return gt_ids
            
            # Use microsoft/ms_marco v1.1 validation split (queries are from validation set)
            # This is faster and more appropriate for validation queries
            if not gt_ids:
                dataset = load_dataset("microsoft/ms_marco", "v1.1", split="validation")
                for item in dataset:
                    if item.get("query", "").lower() == query.lower():
                        passages = item.get("passages", {})
                        if isinstance(passages, dict):
                            texts = passages.get("passage_text", [])
                            selected = passages.get("is_selected", [])
                            passage_ids = passages.get("passage_id", [])
                            
                            for idx, (text, sel) in enumerate(zip(texts, selected)):
                                if sel == 1:
                                    passage_id = str(passage_ids[idx]) if passage_ids and idx < len(passage_ids) else None
                                    gt_ids.append({
                                        'passage_id': passage_id,
                                        'text': text[:200] + "..." if len(text) > 200 else text,
                                    })
                                    if len(gt_ids) >= limit:
                                        break
                        break
        
        elif query_type == "olap":
            dataset = load_dataset("hotpot_qa", "fullwiki", split="validation")
            for item in dataset:
                if item.get("question", "").lower() == query.lower():
                    context = item.get("context", {})
                    supporting_facts = item.get("supporting_facts", {})
                    
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
                                        gt_ids.append({
                                            'title': sf_title,
                                            'sent_id': sf_sent_id,
                                            'text': sents[sf_sent_id][:200] + "..." if len(sents[sf_sent_id]) > 200 else sents[sf_sent_id],
                                        })
                                        if len(gt_ids) >= limit:
                                            break
                    break
    except Exception as e:
        print(f"  Error loading ground truth: {e}")
    
    return gt_ids


def check_qdrant_for_id(
    client: QdrantClient,
    collection: str,
    passage_id: Optional[str] = None,
    title: Optional[str] = None,
    sent_id: Optional[int] = None,
    text_content: Optional[str] = None,
) -> Dict:
    """Check if a passage exists in Qdrant by ID, title, or text content."""
    results = {
        'found': False,
        'chunk_id': None,
        'text': None,
        'payload': None,
        'match_method': None,  # 'id', 'title', 'text'
    }
    
    try:
        # For MS MARCO: search by original_passage_id
        if passage_id:
            passage_id_str = str(passage_id)
            
            # Extract leading numeric part (e.g., "9653" from "9653_passage_1")
            numeric_match = re.search(r'^(\d+)', passage_id_str)
            if numeric_match:
                passage_id_numeric = numeric_match.group(1)
            else:
                passage_id_numeric = passage_id_str
            
            # Try multiple ID format variations (exact matches only)
            id_variations = [
                passage_id_str,  # Full ID as-is
                passage_id_numeric,  # Just the numeric part
                f"msmarco_{passage_id_numeric}",  # With msmarco prefix
                f"msmarco_passage_{passage_id_numeric}",  # With msmarco_passage prefix
            ]
            
            # Also add variations without _passage_ suffix
            if "_passage_" in passage_id_str:
                base_id = passage_id_str.split("_passage_")[0]
                id_variations.extend([base_id, f"msmarco_{base_id}"])
            
            # Scroll through collection to find matching passage_id
            offset = 0
            scroll_limit = 1000
            checked_count = 0
            max_to_check = 10000  # Limit search to avoid infinite loops
            
            while checked_count < max_to_check:
                scroll_result = client.scroll(
                    collection_name=collection,
                    limit=scroll_limit,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                
                points, next_offset = scroll_result
                
                if not points:
                    break
                
                for point in points:
                    checked_count += 1
                    payload = point.payload
                    original_id = payload.get("original_passage_id")
                    
                    if not original_id:
                        continue
                    
                    original_id_str = str(original_id)
                    
                    # Try all ID variations with EXACT matching (not substring)
                    for id_var in id_variations:
                        # Exact match only - no substring matching to avoid false positives
                        if original_id_str == id_var:
                            results['found'] = True
                            results['chunk_id'] = str(point.id)
                            results['text'] = payload.get("text", "")[:200]
                            results['match_method'] = 'id'
                            results['payload'] = {
                                'original_passage_id': original_id,
                                'chunk_type': payload.get("chunk_type"),
                                'source_file': payload.get("source_file"),
                            }
                            return results
                    
                    # Special case: if both are numeric, check exact numeric match
                    # Extract leading numeric part from passage_id (e.g., "9653" from "9653_passage_1")
                    numeric_match = re.search(r'^(\d+)', passage_id_str)
                    if numeric_match:
                        passage_id_numeric = numeric_match.group(1)
                        # Only match if original_id is exactly this number (not a substring)
                        if original_id_str.isdigit() and original_id_str == passage_id_numeric:
                            results['found'] = True
                            results['chunk_id'] = str(point.id)
                            results['text'] = payload.get("text", "")[:200]
                            results['match_method'] = 'id'
                            results['payload'] = {
                                'original_passage_id': original_id,
                                'chunk_type': payload.get("chunk_type"),
                                'source_file': payload.get("source_file"),
                            }
                            return results
                
                if next_offset is None:
                    break
                offset = next_offset
        
        # For HotpotQA: search by original_title and original_sent_id
        elif title:
            offset = 0
            limit = 1000
            while True:
                scroll_result = client.scroll(
                    collection_name=collection,
                    limit=limit,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                
                points, next_offset = scroll_result
                
                for point in points:
                    payload = point.payload
                    original_title = payload.get("original_title")
                    original_sent_id = payload.get("original_sent_id")
                    
                    if original_title and original_title.lower() == title.lower():
                        if sent_id is None or original_sent_id == sent_id:
                            results['found'] = True
                            results['chunk_id'] = str(point.id)
                            results['text'] = payload.get("text", "")[:200]
                            results['match_method'] = 'title'
                            results['payload'] = {
                                'original_title': original_title,
                                'original_sent_id': original_sent_id,
                                'chunk_type': payload.get("chunk_type"),
                            }
                            return results
                
                if next_offset is None or len(points) == 0:
                    break
                offset = next_offset
        
        # Fallback: If no ID match and text_content provided, search by text similarity
        if not results['found'] and text_content:
            # Simple text matching: check if key phrases from ground truth appear in Qdrant chunks
            gt_text_lower = text_content.lower()
            gt_words = set(gt_text_lower.split()[:30])  # First 30 words as key phrases
            
            offset = 0
            scroll_limit = 1000
            checked_count = 0
            max_to_check = 10000  # Increased limit for text search
            
            while checked_count < max_to_check:
                scroll_result = client.scroll(
                    collection_name=collection,
                    limit=scroll_limit,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                
                points, next_offset = scroll_result
                
                if not points:
                    break
                
                for point in points:
                    checked_count += 1
                    payload = point.payload
                    chunk_text = payload.get("text", "")
                    if not chunk_text:
                        continue
                    
                    chunk_text_lower = chunk_text.lower()
                    chunk_words = set(chunk_text_lower.split())
                    
                    # Check overlap of key words
                    overlap = len(gt_words & chunk_words)
                    if overlap >= min(3, len(gt_words) * 0.2):  # At least 3 words or 20% overlap
                        # Check if first 150 chars are similar (more lenient)
                        gt_preview = gt_text_lower[:150]
                        chunk_preview = chunk_text_lower[:150]
                        if (gt_preview in chunk_text_lower or 
                            chunk_preview in gt_text_lower or
                            # Check if significant portion of ground truth text appears in chunk
                            len(set(gt_text_lower.split()[:10]) & chunk_words) >= 5):
                            results['found'] = True
                            results['chunk_id'] = str(point.id)
                            results['text'] = chunk_text[:200]
                            results['match_method'] = 'text'
                            results['payload'] = {
                                'original_passage_id': payload.get("original_passage_id"),
                                'chunk_type': payload.get("chunk_type"),
                                'source_file': payload.get("source_file"),
                                'text_overlap': overlap,
                            }
                            return results
                
                if next_offset is None:
                    break
                offset = next_offset
    
    except Exception as e:
        print(f"  Error checking Qdrant: {e}")
        results['error'] = str(e)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Verify ID alignment between ground truth and Qdrant")
    parser.add_argument("--results-file", type=str, default="results/comprehensive_eval.json",
                       help="Path to comprehensive evaluation results")
    parser.add_argument("--qdrant-host", type=str, default="localhost", help="Qdrant host")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant port")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of failing queries to check")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ID ALIGNMENT VERIFICATION")
    print("=" * 80)
    
    # Load results
    print(f"\nLoading results from: {args.results_file}")
    results = load_eval_results(args.results_file)
    
    # Find failing queries
    print(f"\nFinding queries with mrr=0 and num_relevant_total=1...")
    failing_queries = find_failing_queries(results, args.num_samples)
    print(f"Found {len(failing_queries)} failing queries to check")
    
    if not failing_queries:
        print("No failing queries found matching criteria.")
        return
    
    # Connect to Qdrant
    print(f"\nConnecting to Qdrant at {args.qdrant_host}:{args.qdrant_port}...")
    client = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)
    
    # Check each failing query
    print("\n" + "=" * 80)
    print("VERIFICATION RESULTS")
    print("=" * 80)
    
    found_count = 0
    not_found_count = 0
    
    for i, fq in enumerate(failing_queries[:args.num_samples], 1):
        print(f"\n[{i}/{len(failing_queries)}] Query: {fq['query'][:60]}...")
        print(f"  Type: {fq['query_type']}, Config: {fq['config']}")
        
        # Get ground truth IDs
        gt_ids = get_ground_truth_ids(fq['query'], fq['query_type'], limit=1)
        
        if not gt_ids:
            print("  ⚠ No ground truth IDs found in dataset")
            continue
        
        gt = gt_ids[0]
        collection = "oltp_chunks" if fq['query_type'] == "oltp" else "olap_chunks"
        
        # Check Qdrant
        if fq['query_type'] == "oltp":
            passage_id = gt.get('passage_id')
            queryid = gt.get('queryid', '')
            text_content = gt.get('text', '')
            print(f"  Ground truth passage_id: {passage_id}")
            print(f"  Query ID: {queryid}")
            if not passage_id:
                print(f"  ⚠ Warning: No passage_id extracted from dataset!")
                print(f"     This suggests the MS MARCO validation split may not include passage_id")
                print(f"     Will try text-based matching as fallback...")
            qdrant_result = check_qdrant_for_id(
                client, collection, 
                passage_id=passage_id,
                text_content=text_content if not passage_id else None,  # Use text matching if no ID
            )
        else:
            title = gt.get('title')
            sent_id = gt.get('sent_id')
            print(f"  Ground truth title: {title}, sent_id: {sent_id}")
            qdrant_result = check_qdrant_for_id(client, collection, title=title, sent_id=sent_id)
        
        if qdrant_result['found']:
            found_count += 1
            match_method = qdrant_result.get('match_method', 'unknown')
            print(f"  ✅ FOUND in Qdrant! (matched by: {match_method})")
            print(f"     Chunk ID: {qdrant_result['chunk_id']}")
            print(f"     Payload: {qdrant_result['payload']}")
            print(f"     Text preview: {qdrant_result['text'][:100]}...")
            
            # Check if text matches
            gt_text = gt.get('text', '').lower()
            qdrant_text = qdrant_result.get('text', '').lower()
            if match_method == 'text':
                print(f"  ✅ Text-based match (ID not available in dataset)")
            elif gt_text[:100] in qdrant_text or qdrant_text[:100] in gt_text:
                print(f"  ✅ Text content matches!")
            else:
                print(f"  ⚠ Text content may differ (ID matched but text differs)")
        else:
            not_found_count += 1
            print(f"  ❌ NOT FOUND in Qdrant")
            if 'error' in qdrant_result:
                print(f"     Error: {qdrant_result['error']}")
            print(f"     Ground truth text: {gt.get('text', '')[:100]}...")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total queries checked: {len(failing_queries)}")
    print(f"Found in Qdrant: {found_count} ({100*found_count/len(failing_queries):.1f}%)")
    print(f"Not found in Qdrant: {not_found_count} ({100*not_found_count/len(failing_queries):.1f}%)")
    
    if not_found_count > 0:
        print("\n⚠️  Some relevant passages are missing from Qdrant!")
        print("   This could explain low recall. Check:")
        print("   1. Were all documents ingested?")
        print("   2. Are IDs correctly stored in payloads?")
        print("   3. Is the ID format matching (e.g., '123' vs 'msmarco_passage_123')?")
    elif found_count == len(failing_queries):
        print("\n✅ All relevant passages exist in Qdrant!")
        print("   The issue may be with retrieval/ranking, not data alignment.")


if __name__ == "__main__":
    main()

