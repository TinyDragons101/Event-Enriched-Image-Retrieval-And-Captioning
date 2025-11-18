import json
import csv
import argparse
from internvl import CustonInternVLCaptionModel, CustonInternVLRetrievalModel
from tqdm import tqdm
import torch 
import os
import torch.nn.functional as F


def load_wrong_queries(wrong_sample_json_path):
    with open(wrong_sample_json_path, 'r', encoding='utf-8') as f:
        wrong_samples = json.load(f)
    return {entry["query_id"] for entry in wrong_samples}


def extract_rerank_inputs(csv_path, wrong_query_ids, pre_top_k):
    rerank_inputs = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_id = row['query_id']
            if query_id in wrong_query_ids:
                image_ids = [row[f'image_id_{i+1}'] for i in range(pre_top_k)]
                rerank_inputs.append({
                    "query_id": query_id,
                    "top_k_candidates": image_ids,
                })
    return rerank_inputs

    
def create_caption_json(rerank_inputs, output_path, device='cuda:7'):
    caption_model_query = CustonInternVLCaptionModel(model_name='OpenGVLab/InternVL2_5-4B', device=device)
    caption_model_db = CustonInternVLCaptionModel(model_name='OpenGVLab/InternVL2_5-8B', device=device)

    query_image_path = 'data/track1_private/query/'
    database_path = 'data/database_images/database_images_compressed90/'

    results = []

    for item in tqdm(rerank_inputs, desc="Generating captions"):
        query_id = item['query_id']
        query_image_file = os.path.join(query_image_path, f"{query_id}.jpg")
        try:
            query_caption = caption_model_query.generate__short_caption(query_image_file)
        except Exception as e:
            print(f"❌ Failed to caption query {query_image_file}: {e}")
            query_caption = ""

        top_k_captions = []
        for candidate_id in item['top_k_candidates']:
            candidate_image_file = os.path.join(database_path, f"{candidate_id}.jpg")
            try:
                caption = caption_model_db.generate__short_caption(candidate_image_file)
            except Exception as e:
                print(f"❌ Failed to caption db image {candidate_image_file}: {e}")
                caption = ""
            top_k_captions.append({
                "image_id": candidate_id,
                "caption": caption
            })

        results.append({
            "query_id": query_id,
            "query_caption": query_caption,
            "top_k_captions": top_k_captions
        })
        # Save final caption JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ Captions saved to: {output_path} (total: {len(results)})")


def rerank_embeddings(rerank_input_path, output_path):
    with open(rerank_input_path, 'r', encoding='utf-8') as f:
        rerank_inputs = json.load(f)

    # Image similarity scores JSON
    image_similarity_path = "final_json_result/private_test_similarity_scores.json"
    with open(image_similarity_path, 'r', encoding='utf-8') as f:
        image_similarity_dict = json.load(f)

    device = 'cuda:7' if torch.cuda.is_available() else 'cpu'
    model = CustonInternVLRetrievalModel(device=device)

    rerank_results = []
    for item in tqdm(rerank_inputs, desc="Reranking embeddings"):
        query_id = item['query_id']
        query_caption = item['query_caption']
        top_k_captions = item['top_k_captions']
        image_scores = image_similarity_dict.get(query_id, {})
        try:   
            query_embedding = model.encode_text(query_caption)  # shape (1, D)
        except Exception as e:
            print(f"❌ Failed to encode query caption for {query_id}: {e}")
            continue
        
        rerank_scores = []
        for candidate in top_k_captions:
            image_id = candidate['image_id']
            caption_text = candidate['caption']
            try:
                candidate_embedding = model.encode_text(caption_text)  # shape (1, D)
                coeff = torch.load("./logit_scale.pt", map_location= device)  # shape: []
                if coeff.dim() > 0:
                    coeff = coeff.squeeze()
                
                sim_score = (coeff * (query_embedding @ candidate_embedding.T)).item()
            except Exception as e:
                print(f"❌ Failed to encode caption for {image_id}: {e}")
                sim_score = 0.0

            image_sim = image_scores.get(image_id, 0.0)
            combined_score = round(sim_score + image_sim, 6)

            rerank_scores.append((image_id, combined_score))


        reranked_ids = [img_id for img_id, _ in rerank_scores]
        rerank_results.append({
            "query_id": query_id,
            "reranked_candidates": reranked_ids
        })
  
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(rerank_results, f, indent=2, ensure_ascii=False)
    print(f"✅ Reranked results saved to: {output_path} (total: {len(rerank_results)})")


def update_csv_with_rerank_results(csv_path, rerank_results_path, output_path):
    with open(rerank_results_path, 'r', encoding='utf-8') as f:
        rerank_results = json.load(f)

    rerank_dict = {item['query_id']: item['reranked_candidates'] for item in rerank_results}

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    pre_top_k = sum(1 for col in fieldnames if col.startswith('image_id_'))

    for row in rows:
        query_id = row['query_id']
        if query_id in rerank_dict:
            reranked = rerank_dict[query_id]
            for i in range(pre_top_k):
                key = f'image_id_{i+1}'
                row[key] = reranked[i] if i < len(reranked) else ""

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Updated CSV with reranked candidates saved to: {output_path}")


def create_context_json(csv_output_path):
    with open('./data/matching_articles.json', 'r', encoding='utf-8') as f:
        matching_image_article = json.load(f)
    with open('./data/database.json', 'r', encoding='utf-8') as df:
        database_data = json.load(df)

    new_rows = []
    context = {}

    with open(csv_output_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            query_id = row['query_id']
            image_id = row.get('image_id_1')

            if not image_id or image_id not in matching_image_article:
                print(f"Warning: image_id '{image_id}' not found for query_id '{query_id}'")
                continue
            
            article_id = matching_image_article[image_id]
            article_url = database_data[article_id]['url']

            context[query_id] = {
                "query_id": query_id,
                "image_id": image_id,
                "article_id": article_id,
                "article_url": article_url
            }
            generated_caption = row.get('generated_caption', '')
            new_row = {'query_id': query_id, 'generated_caption': generated_caption}

            for i in range(1, 11):  
                image_key = f'image_id_{i}'
                image_id = row.get(image_key)

                if image_id and image_id in matching_image_article:
                    article_id = matching_image_article[image_id]
                else:
                    article_id = ''  

                new_row[f'article_id_{i}'] = article_id
            new_rows.append(new_row)

    context_json_path = "final_json_result/context_extraction_image_article.json"
    new_csv_output_path = "./final_csv_result/private_final_retrieval_merging_final_results.csv"
    os.makedirs(os.path.dirname(new_csv_output_path), exist_ok=True)

    with open(context_json_path, 'w', encoding='utf-8') as out_json:
        json.dump(context, out_json, indent=2, ensure_ascii=False)

    fieldnames = ['query_id'] + [f'article_id_{i}' for i in range(1, 11)] + ['generated_caption']
    with open(new_csv_output_path, 'w', newline='', encoding='utf-8') as out_csv:
        writer = csv.DictWriter(out_csv, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_rows)

    print(f"New article CSV saved to {new_csv_output_path}")
    print(f"Context JSON saved to {context_json_path}")
    

def main():
    parser = argparse.ArgumentParser(description="Filter rerank queries from CSV using wrong-sample list")
    parser.add_argument('--device', type=str, 
                        default='cuda:4', 
                        help="Torch device (e.g., 'cuda:0', 'cpu').")
    parser.add_argument('--wrong_sample_json_path', type=str,
                        default='./final_json_result/temp_three_ways_wrong_samples_set.json',
                        help="Path to JSON file containing wrong sample query_ids")
    parser.add_argument('--csv_path', type=str,
                        default='./final_csv_result/temp_private_test_image_first_step_retrieval_results_with_caption.csv',
                        help="Path to CSV file with retrieval results")
    parser.add_argument('--pre_top_k', type=int, default=15,
                        help="Number of top-k candidates per query in the CSV")
    parser.add_argument('--rerank_input_path', type=str, default='rerank_inputs.json',
                        help="Path to save the filtered rerank inputs as JSON")
    parser.add_argument('--rerank_caption_output_path', type=str, default='./rerank_caption.json',
                        help="Path to save the filtered rerank inputs as JSON")
    parser.add_argument('--rerank_output_path', type=str, default='./rerank_results.json', help="Path to save the reranked results as JSON")
    parser.add_argument('--rerank_final_path', type=str, default='./final_csv_result/temp_final_rerank.csv',
                        help="Path to save the reranked results as CSV")
    parser.add_argument('--output_dir', type=str, default='./private_test_final_elements_json',
                        help="Directory to save the final caption JSON")


    args = parser.parse_args()
    wrong_query_ids = load_wrong_queries(args.wrong_sample_json_path)
    rerank_inputs = extract_rerank_inputs(args.csv_path, wrong_query_ids, args.pre_top_k)
    create_caption_json(rerank_inputs, args.rerank_caption_output_path, device=args.device)
    rerank_embeddings(args.rerank_caption_output_path, args.rerank_output_path)
    update_csv_with_rerank_results(args.csv_path, args.rerank_output_path, args.rerank_final_path)
    create_context_json(args.rerank_final_path)


if __name__ == '__main__':
    main()
