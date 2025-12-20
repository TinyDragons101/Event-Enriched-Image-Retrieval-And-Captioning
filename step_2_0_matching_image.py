from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os
import json
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import torch

def load_images_from_list(img_dir, img_id_list, ext=".jpg"):
    img_dict = {}
    for img_id in img_id_list:
        p = Path(img_id)
        if p.suffix == "":
            img_path = (img_dir / (img_id + ext))
        else:
            img_path = p if p.is_absolute() else (img_dir / img_id)
        if not img_path.exists():
            continue  # Skip if image file doesn't exist
        try:
            img = Image.open(img_path).convert("RGB")
            arr = np.array(img)
            img_dict[img_path.stem] = arr
        except Exception as e:
            continue
    return img_dict


def compare_img_arrays(arr1, arr2, threshold=0.01):
    if arr1.shape != arr2.shape:
        return False
    abs_diff = np.abs(arr1.astype(np.int32) - arr2.astype(np.int32))
    total_diff = np.sum(abs_diff)
    total_values = arr1.size * 255
    diff_ratio = total_diff / total_values
    return diff_ratio <= threshold


def load_embeddings_from_list(embedding_dir, id_list, device):
    embedding_dict = {}
    for id in id_list:
        embedding_path = embedding_dir / (id + ".pt")
        if not embedding_path.exists():
            continue
        try:
            embedding = torch.load(embedding_path, map_location=device).unsqueeze(0)
            embedding_dict[id] = embedding
        except Exception as e:
            continue
    return embedding_dict


def cosine_similarity(embedding1, embedding2):
    """Compute cosine similarity between two embeddings."""
    return torch.nn.functional.cosine_similarity(embedding1, embedding2)


def process_key(key, origin_db, my_db, origin_img, my_img, origin_embedding, my_embedding, matching_dir, device, threshold_pixel_diff, skip_existing=True):
    
    json_matching_file = matching_dir / (key + ".json")
        
    # Skip if file already exists and skip_existing is True
    if skip_existing and os.path.exists(json_matching_file):
        return None
    
    origin_images = origin_db[key]["images"]  # List of origin image ids (strings)
    my_images = [img_obj["id"] for img_obj in my_db[key]["images"]]  # List of my image ids
    my_ids = [filename.split(".")[0] for filename in my_images]

    origin_arrs = load_images_from_list(origin_img, origin_images, ext=".jpg")
    my_arrs = load_images_from_list(my_img, my_images, ext=".jpg")
    origin_embeddings = load_embeddings_from_list(origin_embedding, origin_images, device)
    my_embeddings = load_embeddings_from_list(my_embedding, my_ids, device)

    mapping = {}  # origin_id: new_id

    empty_tensor = torch.zeros(1, 768).to(device)

    for origin_id in origin_images:
        
        # Skip if this origin image does not have embedding
        if origin_id not in origin_embeddings:
            continue

        if origin_id not in mapping:
            mapping[origin_id] = None
        if mapping[origin_id] is not None: continue
        arr1 = origin_arrs.get(origin_id)
        if arr1 is None:
            continue
        
        found_match = False
        for my_filename, arr2 in my_arrs.items():
            my_id = my_filename.split(".")[0]
            if compare_img_arrays(arr1, arr2, threshold=threshold_pixel_diff):
                mapping[origin_id] = {
                    "filename": my_filename,
                    "score": 1.0,
                }
                found_match = True
                break
        if found_match: continue

        mx_cosine_similarity = 0
        for my_filename in my_arrs.keys():
            my_id = my_filename.split(".")[0]
            cur_cosine_similarity = cosine_similarity(
                origin_embeddings.get(origin_id, empty_tensor),
                my_embeddings.get(my_id, empty_tensor),
            ).item()
            if mx_cosine_similarity < cur_cosine_similarity:
                mx_cosine_similarity = cur_cosine_similarity
                mapping[origin_id] = {
                    "filename": my_filename,
                    "score": mx_cosine_similarity,
                }

    try:
        with open(json_matching_file, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)

    except Exception as e:
        tqdm.write(f"[ERROR WRITE JSON] key={key}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Match images between origin database and new crawled database")
    parser.add_argument('--device', type=str, 
                        default='cuda:7' if torch.cuda.is_available() else 'cpu', 
                        help="Torch device (e.g., 'cuda:0', 'cpu').")
    
    parser.add_argument('--threshold_pixel_diff', type=float, default=0.01,
                        help="Threshold for pixel difference in image comparison")
    
    parser.add_argument('--output_matching_folder', type=str, default='matching-01-no-threshold',
                        help="Output folder for matching results")
    
    parser.add_argument('--origin_embedding', type=str, default='./embeddings/database',
                        help="Path to origin embedding directory")
    
    parser.add_argument('--new_embedding', type=str, default='./embeddings/maching_new_database_internvlg',
                        help="Path to new embedding directory")
    
    parser.add_argument('--origin_img_folder', type=str, default='./data/database_images/database_images_compressed90',
                        help="Path to origin image folder")
    
    parser.add_argument('--new_img_folder', type=str, default='imgs',
                        help="Path to new image folder")
    
    parser.add_argument('--origin_database_json', type=str, default='./data/database.json',
                        help="Path to origin database JSON file")
    
    parser.add_argument('--crawled_folder', type=str, default='crawled',
                        help="Path to crawled folder containing new database files")
    
    parser.add_argument('--max_workers', type=int, default=64,
                        help="Maximum number of worker threads for parallel processing")
    
    parser.add_argument('--skip_existing', action='store_true', default=True,
                        help="Skip processing keys if output file already exists (default: True)")
    
    parser.add_argument('--overwrite', action='store_true', default=True,
                        help="Overwrite existing output files")

    args = parser.parse_args()
    
    # If overwrite is set, disable skip_existing
    if args.overwrite:
        args.skip_existing = False

    matching_dir = Path(args.output_matching_folder)
    os.makedirs(matching_dir, exist_ok=True)
    device = args.device
    print(f"Using device: {device}")

    origin_embedding = Path(args.origin_embedding)
    my_embedding = Path(args.new_embedding)
    origin_img = Path(args.origin_img_folder)
    my_img = Path(args.new_img_folder)
    crawled_folder = Path(args.crawled_folder)

    with open(args.origin_database_json, "r", encoding="utf-8") as f:
        origin_db = json.load(f)
    my_db = dict()

    filenames = list(os.listdir(crawled_folder))
    for idx, filename in enumerate(filenames):
        key = filename.split(".")[0]
        try:
            with open(crawled_folder / filename, "r", encoding="utf-8") as f:
                data = json.load(f)
                my_db[key] = data
        except Exception as e:
            continue

    # Get intersection of keys
    origin_keys = set(origin_db.keys())
    my_keys = set(my_db.keys())
    key_intersection = origin_keys & my_keys

    # Create a partial function to pass all necessary parameters
    def process_key_wrapper(key):
        return process_key(
            key, origin_db, my_db, origin_img, my_img, 
            origin_embedding, my_embedding, matching_dir, 
            device, args.threshold_pixel_diff, args.skip_existing
        )

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        results = list(tqdm(executor.map(process_key_wrapper, key_intersection), total=len(key_intersection), desc="Processing"))
    
    # Count successful results
    processed_count = sum(1 for r in results if r is not None)
    skipped_count = sum(1 for r in results if r is None)
    
    print(f"✅ Processing complete!")
    print(f"   - Processed: {processed_count} keys")
    print(f"   - Skipped (existing): {skipped_count} keys")
    print(f"📁 Results saved to: {matching_dir}")


if __name__ == '__main__':
    main()
