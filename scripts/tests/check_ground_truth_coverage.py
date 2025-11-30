#!/usr/bin/env python3
"""
Check what percentage of ground truth relevant passages exist in Qdrant.

This helps diagnose if low recall is due to missing data vs. poor retrieval.
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qdrant_client import QdrantClient, models
from scripts.comprehensive_evaluation import load_queries_with_ground_truth, EvalQueryWithGT


def check_passage_in_qdrant(
    client: QdrantClient,
    collection: str,
    passage_id: str = None,
    title: str = None,
    sent_id: int = None,
    text: str = None,
) -> Dict:
    """
    Check if a ground truth passage exists in Qdrant.
    
    Returns:
        {
            'found': bool,
            'match_method': str,  # 'id', 'title', 'text', or None
            'chunk_id': str or None,
            'text_preview': str or None,
        }
    """
    result = {
        'found': False,
        'match_method': None,
        'chunk_id': None,
        'text_preview': None,
    }
    
    # MS MARCO: Check by passage_id using filter
    if passage_id:
        passage_id_str = str(passage_id)
        
        # Use Qdrant filter to search by original_passage_id
        try:
            filter_condition = models.Filter(
                must=[
                    models.FieldCondition(
                        key="original_passage_id",
                        match=models.MatchValue(value=passage_id_str)
                    )
                ]
            )
            
            scroll_result = client.scroll(
                collection_name=collection,
                scroll_filter=filter_condition,
                limit=10,
                with_payload=True,
                with_vectors=False,
            )
            
            points, _ = scroll_result
            
            if points:
                point = points[0]
                result['found'] = True
                result['match_method'] = 'id'
                result['chunk_id'] = str(point.id)
                result['text_preview'] = point.payload.get("text", "")[:200]
                return result
        except Exception as e:
            # Fallback: try numeric match if string match fails
            try:
                # Try matching as integer if passage_id is numeric
                if passage_id_str.isdigit():
                    filter_condition = models.Filter(
                        must=[
                            models.FieldCondition(
                                key="original_passage_id",
                                match=models.MatchValue(value=int(passage_id_str))
                            )
                        ]
                    )
                    
                    scroll_result = client.scroll(
                        collection_name=collection,
                        scroll_filter=filter_condition,
                        limit=10,
                        with_payload=True,
                        with_vectors=False,
                    )
                    
                    points, _ = scroll_result
                    
                    if points:
                        point = points[0]
                        result['found'] = True
                        result['match_method'] = 'id'
                        result['chunk_id'] = str(point.id)
                        result['text_preview'] = point.payload.get("text", "")[:200]
                        return result
            except Exception:
                pass
    
    # HotpotQA: Check by title + sent_id using filter
    if title:
        title_lower = title.lower()
        
        try:
            # Build filter conditions
            must_conditions = [
                models.FieldCondition(
                    key="original_title",
                    match=models.MatchText(text=title_lower)  # Case-insensitive text match
                )
            ]
            
            # Add sent_id condition if specified
            if sent_id is not None:
                must_conditions.append(
                    models.FieldCondition(
                        key="original_sent_id",
                        match=models.MatchValue(value=sent_id)
                    )
                )
            
            filter_condition = models.Filter(must=must_conditions)
            
            scroll_result = client.scroll(
                collection_name=collection,
                scroll_filter=filter_condition,
                limit=10,
                with_payload=True,
                with_vectors=False,
            )
            
            points, _ = scroll_result
            
            if points:
                point = points[0]
                # Verify title match (case-insensitive)
                original_title = point.payload.get("original_title", "")
                if original_title.lower() == title_lower:
                    result['found'] = True
                    result['match_method'] = 'title+sent_id' if sent_id is not None else 'title'
                    result['chunk_id'] = str(point.id)
                    result['text_preview'] = point.payload.get("text", "")[:200]
                    return result
        except Exception as e:
            # Fallback: try exact match
            try:
                filter_condition = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="original_title",
                            match=models.MatchValue(value=title)  # Exact match
                        )
                    ]
                )
                
                if sent_id is not None:
                    filter_condition.must.append(
                        models.FieldCondition(
                            key="original_sent_id",
                            match=models.MatchValue(value=sent_id)
                        )
                    )
                
                scroll_result = client.scroll(
                    collection_name=collection,
                    scroll_filter=filter_condition,
                    limit=10,
                    with_payload=True,
                    with_vectors=False,
                )
                
                points, _ = scroll_result
                
                if points:
                    point = points[0]
                    result['found'] = True
                    result['match_method'] = 'title+sent_id' if sent_id is not None else 'title'
                    result['chunk_id'] = str(point.id)
                    result['text_preview'] = point.payload.get("text", "")[:200]
                    return result
            except Exception:
                pass
    
    # Fallback: Check by text similarity (if text provided)
    if text and not result['found']:
        text_lower = text.lower()
        text_words = set(text_lower.split()[:20])  # First 20 words
        
        offset = 0
        scroll_limit = 1000
        checked_count = 0
        max_to_check = 10000  # Smaller limit for text search
        
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
                chunk_words = set(chunk_text_lower.split()[:20])
                
                # Check word overlap
                overlap = len(text_words & chunk_words)
                if overlap >= min(5, len(text_words) * 0.3):  # At least 5 words or 30% overlap
                    # Check if significant portion of text appears
                    if text_lower[:100] in chunk_text_lower or chunk_text_lower[:100] in text_lower:
                        result['found'] = True
                        result['match_method'] = 'text'
                        result['chunk_id'] = str(point.id)
                        result['text_preview'] = chunk_text[:200]
                        return result
            
            if next_offset is None:
                break
            offset = next_offset
    
    return result


def check_coverage(
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    num_queries: int = 50,
):
    """Check ground truth coverage in Qdrant."""
    print("=" * 80)
    print("GROUND TRUTH COVERAGE CHECK")
    print("=" * 80)
    
    # Load queries with ground truth
    print("\nLoading queries with ground truth...")
    queries_dict = load_queries_with_ground_truth(oltp_limit=num_queries, olap_limit=num_queries)
    
    # Connect to Qdrant
    print(f"\nConnecting to Qdrant at {qdrant_host}:{qdrant_port}...")
    client = QdrantClient(host=qdrant_host, port=qdrant_port, check_compatibility=False)
    
    # Debug: Sample a few chunks to see ID format
    print("\nSampling chunks to check ID format...")
    try:
        sample_result = client.scroll(
            collection_name="oltp_chunks",
            limit=5,
            with_payload=True,
            with_vectors=False,
        )
        points, _ = sample_result
        if points:
            print("  Sample payload fields:")
            for i, point in enumerate(points[:3]):
                payload = point.payload
                print(f"    Chunk {i+1}:")
                print(f"      original_passage_id: {payload.get('original_passage_id')} (type: {type(payload.get('original_passage_id')).__name__})")
                print(f"      original_title: {payload.get('original_title')}")
                print(f"      original_sent_id: {payload.get('original_sent_id')}")
    except Exception as e:
        print(f"  Could not sample chunks: {e}")
    
    # Flatten queries
    all_queries = []
    for query_type, query_list in queries_dict.items():
        for query_item in query_list:
            all_queries.append((query_item, query_type))
    
    print(f"\nChecking coverage for {len(all_queries)} queries...")
    print("(This may take a few minutes...)\n")
    
    # Statistics
    total_relevant_passages = 0
    found_passages = 0
    queries_with_all_found = 0
    queries_with_some_found = 0
    queries_with_none_found = 0
    
    query_stats = []
    
    for idx, (query_item, query_type) in enumerate(all_queries):
        if (idx + 1) % 10 == 0:
            print(f"  Checking query {idx + 1}/{len(all_queries)}...")
        
        collection = "oltp_chunks" if query_type == "oltp" else "olap_chunks"
        
        # Count relevant passages for this query
        num_relevant = (
            len(query_item.relevant_passage_ids) +
            len(query_item.relevant_titles) +
            len(query_item.relevant_texts)
        )
        total_relevant_passages += num_relevant
        
        # Check each relevant passage
        found_count = 0
        
        # Check MS MARCO passage IDs
        for passage_id in query_item.relevant_passage_ids:
            result = check_passage_in_qdrant(
                client, collection, passage_id=passage_id
            )
            if result['found']:
                found_count += 1
                found_passages += 1
        
        # Check HotpotQA titles
        for i, title in enumerate(query_item.relevant_titles):
            sent_id = None
            if query_item.relevant_sent_ids and i < len(query_item.relevant_sent_ids):
                sent_id = query_item.relevant_sent_ids[i]
            
            result = check_passage_in_qdrant(
                client, collection, title=title, sent_id=sent_id
            )
            if result['found']:
                found_count += 1
                found_passages += 1
        
        # Check text-based (fallback) - similar to evaluation
        # This matches how comprehensive_evaluation.py works - uses text similarity
        if found_count == 0 and query_item.relevant_texts:
            # Use text similarity to find matching chunks (same as evaluation)
            from scripts.comprehensive_evaluation import text_similarity
            
            # Try to retrieve chunks using the query and check text similarity
            # This simulates what the evaluation does
            try:
                # Get embedding model to search
                from sentence_transformers import SentenceTransformer
                embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                query_vector = embedder.encode(query_item.query, normalize_embeddings=True)
                
                # Search Qdrant (like the evaluation does)
                search_results = client.query_points(
                    collection_name=collection,
                    query=query_vector.tolist(),
                    limit=20,  # Check top 20 retrieved chunks
                    with_payload=True,
                )
                
                # Check text similarity with retrieved chunks
                for hit in search_results.points:
                    chunk_text = hit.payload.get("text", "")
                    if chunk_text:
                        for rel_text in query_item.relevant_texts[:2]:  # Check first 2 texts
                            similarity = text_similarity(chunk_text, rel_text, method="hybrid")
                            if similarity >= 0.25:  # Same threshold as evaluation
                                found_count += 1
                                found_passages += 1
                                break  # Found at least one
                        if found_count > 0:
                            break
            except Exception as e:
                # If embedding search fails, skip text fallback
                pass
        
        # Track query-level stats
        if found_count == num_relevant and num_relevant > 0:
            queries_with_all_found += 1
        elif found_count > 0:
            queries_with_some_found += 1
        else:
            queries_with_none_found += 1
        
        query_stats.append({
            'query': query_item.query[:50],
            'query_type': query_type,
            'num_relevant': num_relevant,
            'found': found_count,
            'coverage': found_count / num_relevant if num_relevant > 0 else 0.0,
        })
    
    # Print results
    print("\n" + "=" * 80)
    print("COVERAGE RESULTS")
    print("=" * 80)
    
    print(f"\nOverall Coverage:")
    print(f"  Total relevant passages: {total_relevant_passages}")
    print(f"  Found in Qdrant: {found_passages}")
    print(f"  Coverage: {100*found_passages/total_relevant_passages:.1f}%" if total_relevant_passages > 0 else "  Coverage: N/A")
    
    print(f"\nQuery-level Coverage:")
    print(f"  Queries with all relevant passages found: {queries_with_all_found} ({100*queries_with_all_found/len(all_queries):.1f}%)")
    print(f"  Queries with some relevant passages found: {queries_with_some_found} ({100*queries_with_some_found/len(all_queries):.1f}%)")
    print(f"  Queries with no relevant passages found: {queries_with_none_found} ({100*queries_with_none_found/len(all_queries):.1f}%)")
    
    # Show queries with low coverage
    print(f"\n{'=' * 80}")
    print("QUERIES WITH LOW COVERAGE (<50%)")
    print(f"{'=' * 80}")
    low_coverage = [q for q in query_stats if q['coverage'] < 0.5 and q['num_relevant'] > 0]
    if low_coverage:
        for q in low_coverage[:10]:  # Show top 10
            print(f"\nQuery: {q['query']}...")
            print(f"  Type: {q['query_type']}")
            print(f"  Relevant passages: {q['num_relevant']}")
            print(f"  Found: {q['found']}")
            print(f"  Coverage: {100*q['coverage']:.1f}%")
    else:
        print("  No queries with low coverage found.")
    
    # Summary
    print(f"\n{'=' * 80}")
    print("DIAGNOSIS")
    print(f"{'=' * 80}")
    
    coverage_pct = 100*found_passages/total_relevant_passages if total_relevant_passages > 0 else 0
    
    if coverage_pct >= 80:
        print("✅ High coverage (≥80%): Low recall is likely due to retrieval/ranking issues, not missing data.")
    elif coverage_pct >= 50:
        print("⚠️  Moderate coverage (50-80%): Some missing data may contribute to low recall.")
    else:
        print("❌ Low coverage (<50%): Missing data is likely a major contributor to low recall.")
        print("   Consider re-ingesting data or checking ID alignment.")
    
    return {
        'total_relevant': total_relevant_passages,
        'found': found_passages,
        'coverage_pct': coverage_pct,
        'queries_with_all_found': queries_with_all_found,
        'queries_with_some_found': queries_with_some_found,
        'queries_with_none_found': queries_with_none_found,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check ground truth coverage in Qdrant")
    parser.add_argument("--qdrant-host", default="localhost", help="Qdrant host")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant port")
    parser.add_argument("--num-queries", type=int, default=50, help="Number of queries to check")
    args = parser.parse_args()
    
    check_coverage(
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        num_queries=args.num_queries,
    )

