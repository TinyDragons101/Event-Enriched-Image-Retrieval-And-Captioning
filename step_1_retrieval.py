import os
import torch
from tqdm import tqdm
import csv
import json
import argparse
from torch.nn import functional as F
import torch.nn.functional as F
import os
import json

CUDA_VISIBLE_DEVICES=1

def main():
    parser = argparse.ArgumentParser(description="Image retrieval based on embeddings")
    parser.add_argument('--device', type=str, 
                        default='cuda:4', 
                        help="Torch device (e.g., 'cuda:0', 'cpu').")
    parser.add_argument('--model_type', type=str, default= "internvl", choices=['clip', 'internvl'],
                        help="Model type: 'clip' or 'internvl'")
    parser.add_argument('--pre_top_k', type=int, default=15,
                        help="Number of top similar items to retrieve per query")
    parser.add_argument('--coeff_path', type=str, default='./logit_scale.pt',
                        help="Path to coefficient tensor for internvl (required if model_type is internvl)")
    parser.add_argument('--database_folder', type=str, default='./embeddings/database_image_internVL_g/',
                        help="Path to folder containing database image embeddings (.pt files)")
    parser.add_argument('--query_folder', type=str, default='./embeddings/track_1_private_internvlg/',
                        help="Path to folder containing query image embeddings (.pt files)")
    parser.add_argument('--top_k', type=int, default=10,
                        help="Number of top similar items to retrieve per query")
    parser.add_argument('--db_chunk_size', type=int, default=100000,
                        help="Number of database embeddings to load per chunk")
    args = parser.parse_args()    
    
    database_folder = args.database_folder
    query_folder = args.query_folder
    pre_top_k = args.pre_top_k
    top_k = args.top_k

    device = args.device if torch.cuda.is_available() else 'cpu'

    # --- Step 1: Prepare database metadata ---
    db_files = [
        fname for fname in sorted(os.listdir(database_folder))
        if fname.endswith('.pt')
    ]
    if not db_files:
        raise FileNotFoundError(f"No .pt files found in database folder: {database_folder}")

    db_image_names = [fname.replace('.pt', '') for fname in db_files]
    print(f"Found {len(db_image_names)} database embeddings spread across chunks of size {args.db_chunk_size}")

    # --- Step 2: Load query embeddings ---
    query_embeddings = []
    query_names = []

    for fname in tqdm(sorted(os.listdir(query_folder)), desc="Loading query embeddings"):
        if fname.endswith('.pt'):
            path = os.path.join(query_folder, fname)
            embedding = torch.load(path, map_location=device).squeeze(0)
            query_embeddings.append(embedding)
            query_names.append(fname.replace('.pt', ''))

    query_embeddings = torch.stack(query_embeddings).to(device)
    print(f"Loaded {len(query_embeddings)} query embeddings of shape {query_embeddings.shape}")

    # --- Step 3: Compute cosine similarity chunk by chunk and keep full matrix ---
    coeff = None
    if args.model_type == 'internvl':
        if not os.path.exists(args.coeff_path):
            raise FileNotFoundError(f"Missing coefficient file: {args.coeff_path}")
        coeff = torch.load(args.coeff_path, map_location=device)
    elif args.model_type != 'clip':
        raise ValueError("Unsupported model_type. Choose from: ['clip', 'internvl']")

    cosine_similarity_chunks = []
    total_db = len(db_files)
    for start_idx in tqdm(range(0, total_db, args.db_chunk_size), desc="Processing database chunks"):
        chunk_files = db_files[start_idx:start_idx + args.db_chunk_size]
        chunk_embeddings = []
        for fname in chunk_files:
            path = os.path.join(database_folder, fname)
            embedding = torch.load(path, map_location=device).squeeze(0)
            chunk_embeddings.append(embedding)

        chunk_embeddings = torch.stack(chunk_embeddings).to(device)

        if args.model_type == 'clip':
            chunk_similarities = query_embeddings @ chunk_embeddings.T / 0.7  # temperature scaling
        else:  # internvl
            chunk_similarities = coeff * query_embeddings @ chunk_embeddings.T

        cosine_similarity_chunks.append(chunk_similarities.cpu())

        del chunk_embeddings
        del chunk_similarities
        if device.startswith('cuda'):
            torch.cuda.empty_cache()

    similarities = torch.cat(cosine_similarity_chunks, dim=1).to(device)
    del cosine_similarity_chunks

    topk_similarities, topk_indices = torch.topk(similarities, k=pre_top_k, dim=1)
    print(topk_indices.size())
    
    # --- Step 3.5: Save full similarity scores for pre_top_k to JSON ---
    similarity_dict = {}
    for i, query_name in enumerate(query_names):
        scores = {}
        for j, idx in enumerate(topk_indices[i]):
            db_image_id = db_image_names[idx]
            score = topk_similarities[i][j].item()
            scores[db_image_id] = round(score, 6)  
        similarity_dict[query_name] = scores

    similarity_output_path = './final_json_result/private_test_similarity_scores.json'
    os.makedirs(os.path.dirname(similarity_output_path), exist_ok=True)
    with open(similarity_output_path, 'w', encoding='utf-8') as f:
        json.dump(similarity_dict, f, indent=2)
    print(f"Similarity scores saved to: {similarity_output_path}")
    

    # --- Step 3.5: Find wrong samples from 3 heuristics and combine into a unique set (≤150 entries) ---
    all_query_stats = []
    for i in range(len(query_names)):
        query_name = query_names[i]
        sim_scores = topk_similarities[i]

        # Calculate metrics
        probs = F.softmax(sim_scores, dim=0)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        gap = sim_scores[0].item() - sim_scores[1].item()
        top1_sim = sim_scores[0].item()

        all_query_stats.append({
            "query_id": query_name,
            "entropy": round(entropy, 6),
            "gap": round(gap, 6),
            "low_top1_sim": round(top1_sim, 6)
        })

    entropy_sorted = sorted(all_query_stats, key=lambda x: x["entropy"], reverse=True)
    gap_sorted = sorted(all_query_stats, key=lambda x: x["gap"])
    top1_sorted = sorted(all_query_stats, key=lambda x: x["low_top1_sim"])

    top50_entropy = entropy_sorted[:50]
    top50_gap = gap_sorted[:50]
    top50_low_top1 = top1_sorted[:50]

    unique_samples_dict = {}
    for sample in top50_entropy + top50_gap + top50_low_top1:
        qid = sample["query_id"]
        if qid not in unique_samples_dict:
            unique_samples_dict[qid] = sample

    # Convert to final list
    unique_wrong_samples = list(unique_samples_dict.values())

    output_path = './final_json_result/temp_three_ways_wrong_samples_set.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(unique_wrong_samples, f, indent=2)

    print(f"✅ Final unique wrong samples saved to: {output_path} (total: {len(unique_wrong_samples)})")

    
    # --- Step 4: Map indices back to names, and embeddings ---
    all_top_matches = []
    all_top_matches_embeddings = [] 
    retrieval_results = {}
    for i, query_name in enumerate(query_names):
        top_matches = [db_image_names[idx] for idx in topk_indices[i]]
        
        top_matches_embeddings = torch.stack([
            torch.load(
                os.path.join(database_folder, f"{db_image_names[idx]}.pt"),
                map_location=device
            ).squeeze(0)
            for idx in topk_indices[i]
        ])
        all_top_matches.append(top_matches)
        all_top_matches_embeddings.append(top_matches_embeddings)
        

    retrieval_results = {}
    for query_name, top_matches in zip(query_names, all_top_matches):
        retrieval_results[query_name] = top_matches[:args.pre_top_k]

    
    # --- Step 6: Write to CSV ---
    output_file = './final_csv_result/temp_private_test_image_first_step_retrieval_results_with_caption.csv'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        header = ['query_id'] + [f'image_id_{i+1}' for i in range(pre_top_k)] + ['generated_caption']
        writer.writerow(header)

        for query_name, matches in retrieval_results.items():
            top_k_image_ids = matches[:pre_top_k]
            caption = "This is an image caption."  
            row = [query_name] + top_k_image_ids + [caption]
            writer.writerow(row)

    print(f"📄 CSV results (with image_ids only) saved to: {output_file}")


if __name__ == '__main__':
    main()