#!/usr/bin/env python3
"""
Debug tool for new_database phase
Analyzes matching results and verifies data consistency
"""

import json
import argparse
from pathlib import Path
import sys


def debug_matching_file(matching_file: str, 
                        crawled_dir: str = "crawled",
                        imgs_dir: str = "imgs",
                        origin_db: str = "./data/database.json"):
    """
    Debug a single matching JSON file
    
    Args:
        matching_file: Path to matching JSON file (e.g., "matching-01-no-threshold/article_id.json")
        crawled_dir: Directory containing crawled JSON files
        imgs_dir: Directory containing image files
        origin_db: Path to origin database JSON
    """
    
    matching_path = Path(matching_file)
    if not matching_path.exists():
        print(f"❌ Error: File not found: {matching_file}")
        return
    
    # Extract article_id from filename
    article_id = matching_path.stem
    print("=" * 80)
    print(f"🔍 DEBUGGING ARTICLE: {article_id}")
    print("=" * 80)
    
    # Load matching data
    with open(matching_path, 'r') as f:
        matching_data = json.load(f)
    
    print(f"\n📋 Matching Results:")
    print(f"   Total mappings: {len(matching_data)}")
    
    if not matching_data:
        print(f"   ⚠️  No mappings found (empty matching result)")
        return
    
    # Load crawled JSON
    crawled_path = Path(crawled_dir) / f"{article_id}.json"
    if crawled_path.exists():
        with open(crawled_path, 'r') as f:
            crawled_data = json.load(f)
        print(f"\n✅ Crawled JSON found: {crawled_path}")
        print(f"   Images in JSON: {len(crawled_data.get('images', []))}")
    else:
        print(f"\n❌ Crawled JSON not found: {crawled_path}")
        crawled_data = {"images": []}
    
    # Build image lookup
    crawled_images = {img["id"]: img for img in crawled_data.get("images", [])}
    
    # Load origin database (optional, for reference)
    origin_images = []
    if Path(origin_db).exists():
        with open(origin_db, 'r') as f:
            origin_db_data = json.load(f)
            if article_id in origin_db_data:
                origin_images = origin_db_data[article_id].get("images", [])
    
    print(f"\n📊 ANALYSIS:")
    print("-" * 80)
    
    # Analyze each mapping
    total_checked = 0
    found_in_json = 0
    missing_in_json = 0
    file_exists = 0
    file_missing = 0
    
    issues = []
    
    for origin_img_id, match_info in matching_data.items():
        total_checked += 1
        filename = match_info.get("filename", "") + ".jpg"
        score = match_info.get("score", 0.0)
        
        print(f"\n{total_checked}. Origin Image: {origin_img_id}")
        print(f"   ├─ Matched to: {filename}")
        print(f"   ├─ Score: {score:.4f}")
        
        # Check if matched image exists in crawled JSON
        if filename in crawled_images:
            found_in_json += 1
            img_info = crawled_images[filename]
            print(f"   ├─ ✅ Found in crawled JSON")
            print(f"   │  ├─ URL: {img_info.get('url', 'N/A')[:60]}...")
            print(f"   │  ├─ Alt: {img_info.get('alt', 'N/A')[:50]}")
            print(f"   │  ├─ Caption: {img_info.get('caption', 'N/A')}")
            print(f"   │  └─ Position: {img_info.get('position', 'N/A')}")
        else:
            missing_in_json += 1
            print(f"   ├─ ❌ NOT found in crawled JSON")
            issues.append({
                "origin_id": origin_img_id,
                "matched_filename": filename,
                "issue": "missing_in_json",
                "score": score
            })
        
        # Check if image file exists in imgs/
        img_path = Path(imgs_dir) / filename
        if img_path.exists():
            file_exists += 1
            file_size = img_path.stat().st_size / 1024  # KB
            print(f"   └─ ✅ Image file exists: {img_path} ({file_size:.1f} KB)")
        else:
            file_missing += 1
            print(f"   └─ ❌ Image file NOT found: {img_path}")
            if issues and issues[-1]["matched_filename"] == filename:
                issues[-1]["issue"] = "missing_both"
            else:
                issues.append({
                    "origin_id": origin_img_id,
                    "matched_filename": filename,
                    "issue": "missing_file",
                    "score": score
                })
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY:")
    print("=" * 80)
    print(f"Total mappings checked: {total_checked}")
    print(f"\n🔍 JSON Data:")
    print(f"   ✅ Found in crawled JSON: {found_in_json}")
    print(f"   ❌ Missing in JSON: {missing_in_json}")
    print(f"\n🖼️  Image Files:")
    print(f"   ✅ File exists: {file_exists}")
    print(f"   ❌ File missing: {file_missing}")
    
    if issues:
        print(f"\n⚠️  ISSUES FOUND: {len(issues)}")
        print("-" * 80)
        for idx, issue in enumerate(issues, 1):
            print(f"{idx}. {issue['origin_id']} → {issue['matched_filename']}")
            print(f"   Issue: {issue['issue']}, Score: {issue['score']:.4f}")
    else:
        print(f"\n✅ No issues found! All data is consistent.")
    
    # Additional info
    if origin_images:
        print(f"\n📌 Additional Info:")
        print(f"   Origin DB has {len(origin_images)} images for this article")
        if len(origin_images) != len(matching_data):
            print(f"   ⚠️  Mismatch: {len(origin_images)} origin vs {len(matching_data)} matched")
    
    # Recommendations
    if missing_in_json > 0 or file_missing > 0:
        print(f"\n💡 RECOMMENDATIONS:")
        if missing_in_json > 0:
            print(f"   1. Re-crawl article {article_id} to update images[] field")
            print(f"      python step_2_0_crawling.py --article-id {article_id}")
        if file_missing > 0:
            print(f"   2. Check crawling logs for download failures")
            print(f"   3. Verify image URLs are accessible")


def main():
    parser = argparse.ArgumentParser(
        description="Debug tool for new_database phase",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Debug a specific matching file
  python debug_new_database.py matching-01-no-threshold/abc123.json
  
  # Specify custom directories
  python debug_new_database.py matching-01-no-threshold/abc123.json \\
      --crawled-dir crawled --imgs-dir imgs
  
  # Debug with article ID directly
  python debug_new_database.py --article-id abc123
        """
    )
    
    parser.add_argument(
        'matching_file',
        nargs='?',
        help='Path to matching JSON file (e.g., matching-01-no-threshold/article_id.json)'
    )
    parser.add_argument(
        '--article-id',
        help='Article ID (alternative to providing full path)'
    )
    parser.add_argument(
        '--crawled-dir',
        default='crawled',
        help='Directory containing crawled JSON files (default: crawled)'
    )
    parser.add_argument(
        '--imgs-dir',
        default='imgs',
        help='Directory containing image files (default: imgs)'
    )
    parser.add_argument(
        '--origin-db',
        default='./data/database.json',
        help='Path to origin database JSON (default: ./data/database.json)'
    )
    parser.add_argument(
        '--matching-dir',
        default='matching-01-no-threshold',
        help='Directory containing matching results (default: matching-01-no-threshold)'
    )
    
    args = parser.parse_args()
    
    # Determine matching file path
    if args.article_id:
        matching_file = Path(args.matching_dir) / f"{args.article_id}.json"
    elif args.matching_file:
        matching_file = args.matching_file
    else:
        parser.print_help()
        print("\n❌ Error: Please provide either matching_file or --article-id")
        sys.exit(1)
    
    # Run debug
    debug_matching_file(
        matching_file=matching_file,
        crawled_dir=args.crawled_dir,
        imgs_dir=args.imgs_dir,
        origin_db=args.origin_db
    )


if __name__ == "__main__":
    main()
