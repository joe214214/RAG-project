#!/usr/bin/env python3
"""
Verify that HotpotQA chunks have original_title and MS MARCO chunks have original_passage_id.

Usage:
    python scripts/verify_hotpotqa_fields.py --qdrant-host ecetesla0
"""

import argparse
from qdrant_client import QdrantClient


def verify_hotpotqa_fields(qdrant_host: str = "localhost", qdrant_port: int = 6333, limit: int = 50):
    """Verify HotpotQA title fields and MS MARCO passage IDs in Qdrant collections."""
    print("=" * 70)
    print("Verifying Dataset Fields (HotpotQA + MS MARCO)")
    print("=" * 70)
    
    client = QdrantClient(host=qdrant_host, port=qdrant_port, check_compatibility=False)
    
    # Check olap_chunks (where HotpotQA OLAP chunks are)
    print("\n📚 Checking olap_chunks collection...")
    
    # Get total count first
    info = client.get_collection('olap_chunks')
    total_points = info.points_count
    
    # Check both beginning and end (HotpotQA was appended, so it's at the end)
    print(f"\nTotal points in collection: {total_points:,}")
    print(f"Checking first {limit} chunks AND last {limit} chunks...")
    print("-" * 70)
    
    # Check beginning
    result_start = client.scroll('olap_chunks', limit=limit)
    
    # Check end (scroll from offset)
    offset = max(0, total_points - limit)
    result_end = client.scroll('olap_chunks', limit=limit, offset=offset)
    
    hotpotqa_with_title = 0
    msmarco_count = 0
    
    # Check start
    print("\n📖 Checking START of collection (MS MARCO):")
    for p in result_start[0]:
        has_title = 'original_title' in p.payload and p.payload.get('original_title')
        has_passage_id = 'original_passage_id' in p.payload and p.payload.get('original_passage_id')
        
        if has_title:
            hotpotqa_with_title += 1
        elif has_passage_id and p.payload.get('original_passage_id', '').isdigit():
            msmarco_count += 1
    
    print(f"   MS MARCO chunks (numeric IDs): {msmarco_count}")
    print(f"   HotpotQA chunks (with title): {hotpotqa_with_title}")
    
    # Check end (where HotpotQA should be)
    print(f"\n📖 Checking END of collection (HotpotQA - last {limit} chunks):")
    hotpotqa_at_end = 0
    for p in result_end[0]:
        has_title = 'original_title' in p.payload and p.payload.get('original_title')
        has_passage_id = 'original_passage_id' in p.payload and p.payload.get('original_passage_id')
        
        if has_title:
            hotpotqa_at_end += 1
            if hotpotqa_at_end <= 5:  # Show first 5
                print(f"\n✅ HotpotQA chunk found:")
                print(f"   ID: {p.id}")
                print(f"   Title: {p.payload['original_title']}")
                print(f"   Has sent_id: {'original_sent_id' in p.payload}")
                if 'original_sent_id' in p.payload and p.payload['original_sent_id'] is not None:
                    print(f"   Sent ID: {p.payload['original_sent_id']}")
        elif has_passage_id and p.payload.get('original_passage_id', '').isdigit():
            msmarco_count += 1
    
    print(f"   HotpotQA chunks with title: {hotpotqa_at_end}")
    hotpotqa_with_title = hotpotqa_at_end
    
    print("\n" + "-" * 70)
    print(f"Summary:")
    print(f"   HotpotQA chunks with title (at end): {hotpotqa_with_title}")
    print(f"   MS MARCO chunks (numeric IDs): {msmarco_count}")
    
    # Check oltp_chunks too (check end where HotpotQA was appended)
    print("\n📚 Checking oltp_chunks collection...")
    info_oltp = client.get_collection('oltp_chunks')
    total_oltp = info_oltp.points_count
    offset_oltp = max(0, total_oltp - limit)
    result_oltp_end = client.scroll('oltp_chunks', limit=limit, offset=offset_oltp)
    
    hotpotqa_oltp = 0
    msmarco_oltp = 0
    
    print(f"   Checking last {limit} chunks (total: {total_oltp:,})...")
    for p in result_oltp_end[0]:
        has_title = 'original_title' in p.payload and p.payload.get('original_title')
        has_passage_id = 'original_passage_id' in p.payload and p.payload.get('original_passage_id')
        
        if has_title:
            hotpotqa_oltp += 1
            if hotpotqa_oltp <= 3:  # Show first 3
                print(f"\n   ✅ HotpotQA chunk:")
                print(f"      ID: {p.id}, Title: {p.payload['original_title']}")
        elif has_passage_id and p.payload.get('original_passage_id', '').isdigit():
            msmarco_oltp += 1
    
    print(f"   HotpotQA chunks with title (at end): {hotpotqa_oltp}")
    print(f"   MS MARCO chunks (numeric IDs): {msmarco_oltp}")
    
    # Collection stats
    print("\n" + "=" * 70)
    print("Collection Statistics:")
    print("-" * 70)
    
    for coll_name in ['oltp_chunks', 'olap_chunks']:
        info = client.get_collection(coll_name)
        print(f"\n{coll_name}:")
        print(f"   Total points: {info.points_count:,}")
    
    # Verify MS MARCO chunks (at start of collection)
    print("\n" + "=" * 70)
    print("Verifying MS MARCO Passage IDs")
    print("=" * 70)
    
    # Check olap_chunks start (where MS MARCO should be)
    print("\n📚 Checking START of olap_chunks (MS MARCO)...")
    result_msmarco_olap = client.scroll('olap_chunks', limit=20)
    msmarco_olap_found = 0
    msmarco_olap_samples = []
    
    for p in result_msmarco_olap[0]:
        has_passage_id = 'original_passage_id' in p.payload and p.payload.get('original_passage_id')
        has_title = 'original_title' in p.payload and p.payload.get('original_title')
        
        # MS MARCO has passage IDs (numeric or msmarco_X format), no title
        if has_passage_id and not has_title:
            passage_id = p.payload['original_passage_id']
            passage_id_str = str(passage_id)
            # Check if it's numeric (actual MS MARCO ID) or msmarco_X format (generated)
            is_numeric = passage_id_str.isdigit()
            is_msmarco_format = passage_id_str.startswith('msmarco_') and passage_id_str[8:].isdigit()
            
            if is_numeric or is_msmarco_format:
                msmarco_olap_found += 1
                if len(msmarco_olap_samples) < 3:
                    msmarco_olap_samples.append({
                        'id': p.id,
                        'passage_id': passage_id,
                        'is_numeric': is_numeric
                    })
    
    if msmarco_olap_found > 0:
        print(f"\n✅ Found {msmarco_olap_found} MS MARCO chunks in olap_chunks:")
        numeric_count = sum(1 for s in msmarco_olap_samples if s.get('is_numeric', False))
        for sample in msmarco_olap_samples:
            id_type = "numeric" if sample.get('is_numeric') else "generated"
            print(f"   Qdrant ID: {sample['id']}, Passage ID: {sample['passage_id']} ({id_type})")
        if numeric_count == 0:
            print("\n   ⚠️  WARNING: Using generated IDs (msmarco_X) instead of numeric passage IDs")
            print("   This may affect exact ID matching. Consider clearing cache and re-ingesting.")
    else:
        print("\n⚠️  No MS MARCO chunks detected (checking payload structure)...")
        if result_msmarco_olap[0]:
            sample = result_msmarco_olap[0][0]
            print(f"   Sample payload keys: {list(sample.payload.keys())}")
            if 'original_passage_id' in sample.payload:
                print(f"   original_passage_id value: {sample.payload['original_passage_id']} (type: {type(sample.payload['original_passage_id'])})")
    
    # Check oltp_chunks start (where MS MARCO should be)
    print("\n📚 Checking START of oltp_chunks (MS MARCO)...")
    result_msmarco_oltp = client.scroll('oltp_chunks', limit=20)
    msmarco_oltp_found = 0
    msmarco_oltp_samples = []
    
    for p in result_msmarco_oltp[0]:
        has_passage_id = 'original_passage_id' in p.payload and p.payload.get('original_passage_id')
        has_title = 'original_title' in p.payload and p.payload.get('original_title')
        
        if has_passage_id and not has_title:
            passage_id = p.payload['original_passage_id']
            passage_id_str = str(passage_id)
            is_numeric = passage_id_str.isdigit()
            is_msmarco_format = passage_id_str.startswith('msmarco_') and passage_id_str[8:].isdigit()
            
            if is_numeric or is_msmarco_format:
                msmarco_oltp_found += 1
                if len(msmarco_oltp_samples) < 3:
                    msmarco_oltp_samples.append({
                        'id': p.id,
                        'passage_id': passage_id,
                        'is_numeric': is_numeric
                    })
    
    if msmarco_oltp_found > 0:
        print(f"\n✅ Found {msmarco_oltp_found} MS MARCO chunks in oltp_chunks:")
        numeric_count = sum(1 for s in msmarco_oltp_samples if s.get('is_numeric', False))
        for sample in msmarco_oltp_samples:
            id_type = "numeric" if sample.get('is_numeric') else "generated"
            print(f"   Qdrant ID: {sample['id']}, Passage ID: {sample['passage_id']} ({id_type})")
        if numeric_count == 0:
            print("\n   ⚠️  WARNING: Using generated IDs (msmarco_X) instead of numeric passage IDs")
            print("   This may affect exact ID matching. Consider clearing cache and re-ingesting.")
    else:
        print("\n⚠️  No MS MARCO chunks detected (checking payload structure)...")
        if result_msmarco_oltp[0]:
            sample = result_msmarco_oltp[0][0]
            print(f"   Sample payload keys: {list(sample.payload.keys())}")
            if 'original_passage_id' in sample.payload:
                print(f"   original_passage_id value: {sample.payload['original_passage_id']} (type: {type(sample.payload['original_passage_id'])})")
    
    # Final summary
    print("\n" + "=" * 70)
    print("Final Verification Summary:")
    print("-" * 70)
    total_hotpotqa = hotpotqa_with_title + hotpotqa_oltp
    total_msmarco = msmarco_olap_found + msmarco_oltp_found
    
    print(f"✅ HotpotQA chunks with original_title: {total_hotpotqa}")
    print(f"✅ MS MARCO chunks with original_passage_id: {total_msmarco}")
    
    if total_hotpotqa > 0 and total_msmarco > 0:
        print("\n🎉 SUCCESS: Both datasets verified!")
        print("   ✅ HotpotQA: Ready for title-based evaluation")
        print("   ✅ MS MARCO: Ready for passage_id-based evaluation")
    elif total_hotpotqa > 0:
        print("\n✅ HotpotQA verified, but MS MARCO passage IDs not found")
        print("   This may be normal if MS MARCO uses different ID format")
    elif total_msmarco > 0:
        print("\n✅ MS MARCO verified, but HotpotQA titles not found")
        print("   Check end of collection where HotpotQA was appended")
    else:
        print("\n⚠️  WARNING: Neither dataset fields detected")
        print("   May need to check payload structure or re-ingest")
    
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify HotpotQA title fields")
    parser.add_argument("--qdrant-host", default="localhost", help="Qdrant host")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant port")
    parser.add_argument("--limit", type=int, default=50, help="Number of chunks to check")
    
    args = parser.parse_args()
    verify_hotpotqa_fields(args.qdrant_host, args.qdrant_port, args.limit)

