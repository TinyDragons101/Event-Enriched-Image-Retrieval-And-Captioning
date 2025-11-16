import argparse
import os
from tqdm import tqdm
import torch
from internvl import CustonInternVLRetrievalModel
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

BATCH_SIZE = 80
NUM_WORKERS = 8

def main():
    parser = argparse.ArgumentParser(description="Generate image embeddings using CustonInternVLRetrievalModel.")
    parser.add_argument("--part", type=int, default=None, help="Part number to process (1-based index).")
    parser.add_argument("--total_parts", type=int, default=4, help="Total number of parts to split workload.")
    parser.add_argument("--device", type=str, default="cuda:6", help="Torch device (e.g., 'cuda:0', 'cpu').")
    parser.add_argument("--input_folder", type=str, default="./data/database/database_origin/database_img/", help="Folder containing input images.")
    parser.add_argument("--output_folder", type=str, default="./embeddings/database/", help="Folder to save output embeddings.")

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    with torch.no_grad():
        embedding_model = CustonInternVLRetrievalModel(device=device)

        image_paths = [f for f in os.listdir(args.input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        image_paths.sort()

        if args.part is not None:
            assert 1 <= args.part <= args.total_parts, "part must be between 1 and total_parts"
            total = len(image_paths)
            split_size = total // args.total_parts
            start = (args.part - 1) * split_size
            end = total if args.part == args.total_parts else start + split_size
            image_subset = image_paths[start:end]
            print(f"🔹 Running Part {args.part}/{args.total_parts}: Processing {len(image_subset)} images")
        else:
            image_subset = image_paths
            print(f"🔹 Running Full Mode: Processing all {len(image_subset)} images")
            
        for i in tqdm(range(0, len(image_subset), BATCH_SIZE), desc="Embedding"):
            batch_files = image_subset[i:i+BATCH_SIZE]
            images, names = [], []
            
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                futures = [
                    executor.submit(
                        lambda name: (
                            os.path.splitext(name)[0],
                            Image.open(os.path.join(args.input_folder, name)).convert("RGB")
                        ),
                        name
                    )
                    for name in batch_files
                ]

                images, names = [], []
                for f in futures:
                    try:
                        name, img = f.result()
                        names.append(name)
                        images.append(img)
                    except Exception as e:
                        continue

            if not images:
                continue

            emb = embedding_model.encode_image(images, is_path=False)
            emb = emb.cpu()

            for name, e in zip(names, emb):
                torch.save(e, os.path.join(args.output_folder, f"{name}.pt"))

        
if __name__ == "__main__":
    main()