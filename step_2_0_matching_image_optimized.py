"""
Optimized version: Batch processing on GPU
Estimated speedup: 50-100x faster
"""
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm
import os
import json
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

class ImageMatcher:
    def __init__(self, device='cuda:0'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🎯 Using device: {self.device}")
        
    def load_embeddings_batch(self, embedding_dir, id_list):
        """Load embeddings in batch to reduce I/O overhead"""
        embeddings = []
        valid_ids = []
        
        for img_id in id_list:
            emb_path = embedding_dir / (img_id + ".pt")
            if emb_path.exists():
                try:
                    emb = torch.load(emb_path, map_location=self.device)
                    embeddings.append(emb)
                    valid_ids.append(img_id)
                except:
                    continue
        
        if embeddings:
            return torch.stack(embeddings).to(self.device), valid_ids
        return None, []
    
    def compute_similarity_matrix(self, origin_embs, crawl_embs):
        """Compute cosine similarity matrix in one GPU operation"""
        # origin_embs: [N, D], crawl_embs: [M, D]
        # Output: [N, M] similarity matrix
        return F.cosine_similarity(
            origin_embs.unsqueeze(1),  # [N, 1, D]
            crawl_embs.unsqueeze(0),   # [1, M, D]
            dim=2
        )
    
    def match_images_fast(self, origin_ids, crawl_ids, 
                         origin_emb_dir, crawl_emb_dir):
        """Fast matching using batch GPU operations"""
        
        # Load all embeddings at once
        origin_embs, valid_origin_ids = self.load_embeddings_batch(
            origin_emb_dir, origin_ids
        )
        crawl_embs, valid_crawl_ids = self.load_embeddings_batch(
            crawl_emb_dir, crawl_ids
        )
        
        if origin_embs is None or crawl_embs is None:
            return {}
        
        # Compute similarity matrix in one shot
        sim_matrix = self.compute_similarity_matrix(origin_embs, crawl_embs)
        
        # Get best matches
        best_scores, best_indices = sim_matrix.max(dim=1)
        
        # Build mapping
        mapping = {}
        for i, origin_id in enumerate(valid_origin_ids):
            crawl_idx = best_indices[i].item()
            score = best_scores[i].item()
            
            mapping[origin_id] = {
                "filename": valid_crawl_ids[crawl_idx],
                "score": float(score)
            }
        
        return mapping


def process_key_optimized(key, origin_db, my_db, origin_emb, crawl_emb, 
                         matching_dir, matcher, skip_existing=True):
    """Optimized processing function"""
    
    json_matching_file = matching_dir / (key + ".json")
    
    if skip_existing and json_matching_file.exists():
        return None
    
    try:
        origin_images = origin_db[key]["images"]
        crawl_images = [img["id"].split(".")[0] for img in my_db[key]["images"]]
        
        # Fast GPU-based matching
        mapping = matcher.match_images_fast(
            origin_images, crawl_images,
            origin_emb, crawl_emb
        )
        
        # Save result
        with open(json_matching_file, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        
        return len(mapping)
    
    except Exception as e:
        tqdm.write(f"[ERROR] key={key}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Optimized image matching")
    parser.add_argument('--device', type=str, default='cuda:0',
                        help="GPU device")
    parser.add_argument('--output_matching_folder', type=str, 
                        default='matching-01-no-threshold',
                        help="Output folder")
    parser.add_argument('--origin_embedding', type=str, 
                        default='./embeddings/database',
                        help="Origin embeddings path")
    parser.add_argument('--new_embedding', type=str, 
                        default='./embeddings/maching_new_database_internvlg',
                        help="Crawled embeddings path")
    parser.add_argument('--origin_database_json', type=str, 
                        default='./data/database.json',
                        help="Origin database JSON")
    parser.add_argument('--crawled_folder', type=str, 
                        default='crawled',
                        help="Crawled data folder")
    parser.add_argument('--max_workers', type=int, default=4,
                        help="Number of parallel workers (reduce for GPU)")
    parser.add_argument('--skip_existing', action='store_true', default=True)
    parser.add_argument('--overwrite', action='store_true')
    
    args = parser.parse_args()
    
    if args.overwrite:
        args.skip_existing = False
    
    # Setup
    matching_dir = Path(args.output_matching_folder)
    matching_dir.mkdir(exist_ok=True)
    
    origin_emb = Path(args.origin_embedding)
    crawl_emb = Path(args.new_embedding)
    crawled_folder = Path(args.crawled_folder)
    
    # Initialize matcher (single GPU instance)
    matcher = ImageMatcher(device=args.device)
    
    # Load databases
    print("📖 Loading databases...")
    with open(args.origin_database_json, "r") as f:
        origin_db = json.load(f)
    
    my_db = {}
    for filename in os.listdir(crawled_folder):
        key = filename.split(".")[0]
        try:
            with open(crawled_folder / filename, "r") as f:
                my_db[key] = json.load(f)
        except:
            continue
    
    # Get intersection
    origin_keys = set(origin_db.keys())
    my_keys = set(my_db.keys())
    key_intersection = list(origin_keys & my_keys)
    
    print(f"📊 Processing {len(key_intersection)} queries...")
    
    # Process with reduced workers (avoid GPU contention)
    def process_wrapper(key):
        return process_key_optimized(
            key, origin_db, my_db, origin_emb, crawl_emb,
            matching_dir, matcher, args.skip_existing
        )
    
    # Use fewer workers to avoid GPU race conditions
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        results = list(tqdm(
            executor.map(process_wrapper, key_intersection),
            total=len(key_intersection),
            desc="Matching"
        ))
    
    # Stats
    processed = sum(1 for r in results if r is not None)
    skipped = len(results) - processed
    
    print(f"\n✅ Complete!")
    print(f"   - Processed: {processed}")
    print(f"   - Skipped: {skipped}")
    print(f"📁 Results: {matching_dir}")


if __name__ == '__main__':
    main()
