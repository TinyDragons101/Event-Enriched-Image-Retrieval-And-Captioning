import json
import csv

def debug_caption_mapping():
    """Debug why captions are empty in submission file"""
    
    caption_json_path = "assemble_result/cot_5_things_fact_more_event_llama.json"
    submission_csv_path = "final_csv_result/public_final_retrieval_merging_final_results.csv"
    
    print("=" * 80)
    print("DEBUG: Caption Mapping Analysis")
    print("=" * 80)
    
    # Load caption JSON
    print(f"\n1. Loading caption JSON: {caption_json_path}")
    try:
        with open(caption_json_path, "r", encoding="utf-8") as f:
            caption_data = json.load(f)
        print(f"   ✅ Loaded successfully")
        print(f"   📊 Type: {type(caption_data)}")
        print(f"   📊 Total entries: {len(caption_data)}")
        
        if isinstance(caption_data, dict):
            caption_keys = list(caption_data.keys())[:10]
            print(f"   📋 First 10 keys:")
            for i, key in enumerate(caption_keys, 1):
                caption_preview = str(caption_data[key])[:80] + "..." if len(str(caption_data[key])) > 80 else str(caption_data[key])
                print(f"      {i}. {key} -> {caption_preview}")
        elif isinstance(caption_data, list):
            print(f"   ⚠️  WARNING: Caption data is a LIST, not a DICT!")
            if caption_data:
                print(f"   First item: {caption_data[0]}")
    except FileNotFoundError:
        print(f"   ❌ ERROR: File not found!")
        return
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return
    
    # Load submission CSV
    print(f"\n2. Loading submission CSV: {submission_csv_path}")
    try:
        csv_query_ids = []
        with open(submission_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                csv_query_ids.append(row["query_id"])
        
        print(f"   ✅ Loaded successfully")
        print(f"   📊 Total rows: {len(csv_query_ids)}")
        print(f"   📋 First 10 query_ids:")
        for i, qid in enumerate(csv_query_ids[:10], 1):
            print(f"      {i}. {qid}")
    except FileNotFoundError:
        print(f"   ❌ ERROR: File not found!")
        return
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return
    
    # Mapping analysis for first 10 rows
    print("\n3. Mapping Analysis (First 10 rows)")
    print("-" * 80)
    
    if not isinstance(caption_data, dict):
        print("   ❌ Cannot map: caption_data is not a dictionary!")
        return
    
    matched = 0
    unmatched = 0
    
    for i, csv_qid in enumerate(csv_query_ids[:10], 1):
        found = csv_qid in caption_data
        status = "✅ FOUND" if found else "❌ NOT FOUND"
        
        if found:
            matched += 1
            caption = caption_data[csv_qid]
            caption_preview = caption[:80] + "..." if len(caption) > 80 else caption
            print(f"{i}. {status}")
            print(f"   Query ID: {csv_qid}")
            print(f"   Caption:  {caption_preview}")
        else:
            unmatched += 1
            print(f"{i}. {status}")
            print(f"   Query ID: {csv_qid}")
            
            # Try to find similar keys
            similar = [k for k in list(caption_data.keys())[:5] if csv_qid in k or k in csv_qid]
            if similar:
                print(f"   Similar keys in caption JSON: {similar}")
        
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY (First 10 rows)")
    print("=" * 80)
    print(f"Matched:   {matched}/10")
    print(f"Unmatched: {unmatched}/10")
    
    # Full dataset comparison
    print("\n" + "=" * 80)
    print("FULL DATASET COMPARISON")
    print("=" * 80)
    
    caption_query_set = set(caption_data.keys()) if isinstance(caption_data, dict) else set()
    csv_query_set = set(csv_query_ids)
    
    print(f"Caption JSON query count: {len(caption_query_set)}")
    print(f"CSV query count:          {len(csv_query_set)}")
    
    # Find overlaps and differences
    common_queries = caption_query_set & csv_query_set
    caption_only = caption_query_set - csv_query_set
    csv_only = csv_query_set - caption_query_set
    
    print(f"\n📊 Overlap Analysis:")
    print(f"   Common queries:        {len(common_queries)} ({len(common_queries)/len(csv_query_set)*100:.1f}% of CSV)")
    print(f"   Caption only:          {len(caption_only)}")
    print(f"   CSV only:              {len(csv_only)}")
    
    if len(common_queries) == 0:
        print(f"\n❌ CRITICAL: NO COMMON QUERIES!")
        print(f"   Caption and CSV are using COMPLETELY DIFFERENT query sets!")
        print(f"\n   Caption first 5: {list(caption_query_set)[:5]}")
        print(f"   CSV first 5:     {list(csv_query_set)[:5]}")
    elif len(common_queries) < len(csv_query_set):
        print(f"\n⚠️  WARNING: Partial overlap only!")
        print(f"   Some CSV queries missing from caption JSON")
        if csv_only:
            print(f"   Missing (first 5): {list(csv_only)[:5]}")
    else:
        print(f"\n✅ Perfect match! All CSV queries have captions.")
    
    # Check related input files
    print("\n" + "=" * 80)
    print("CHECKING RELATED INPUT FILES")
    print("=" * 80)
    
    related_files = [
        "private_test_final_elements_json/final_merge_result.json",
        "final_json_result/context_extraction_image_article.json",
        "result-hoang.json"
    ]
    
    for file_path in related_files:
        print(f"\nChecking: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                query_ids = set(data.keys())
                print(f"   ✅ Found {len(query_ids)} query_ids")
                overlap = query_ids & csv_query_set
                print(f"   📊 Overlap with CSV: {len(overlap)}/{len(csv_query_set)} ({len(overlap)/len(csv_query_set)*100:.1f}%)")
                
                if len(overlap) > len(common_queries):
                    print(f"   💡 This file has MORE overlap with CSV than caption JSON!")
            elif isinstance(data, list):
                if data and isinstance(data[0], dict) and 'query_id' in data[0]:
                    query_ids = set(item['query_id'] for item in data if 'query_id' in item)
                    print(f"   ✅ Found {len(query_ids)} query_ids (from list)")
                    overlap = query_ids & csv_query_set
                    print(f"   📊 Overlap with CSV: {len(overlap)}/{len(csv_query_set)} ({len(overlap)/len(csv_query_set)*100:.1f}%)")
                else:
                    print(f"   ⚠️  List format, no query_id field")
        except FileNotFoundError:
            print(f"   ⚠️  File not found")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    if unmatched > 0 or len(common_queries) < len(csv_query_set):
        print(f"\n" + "=" * 80)
        print("💡 RECOMMENDATION")
        print("=" * 80)
        if len(common_queries) == 0:
            print("Caption JSON was generated for a DIFFERENT test set!")
            print("You need to re-run caption generation for the PUBLIC test queries.")
        else:
            print("Caption JSON is incomplete or for partial dataset.")
            print("Check which input file was used for caption generation.")
    
    print("=" * 80)

if __name__ == "__main__":
    debug_caption_mapping()
