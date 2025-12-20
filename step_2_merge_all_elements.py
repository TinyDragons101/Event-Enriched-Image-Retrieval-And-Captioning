import json
import csv


def safe_load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load {path} due to {e}")
        return {}


def merge_function(generative_caption_path, query_first_article_path, entity_name_path, question_answer_path, new_database_path, output_path):
    generated_caption_data = safe_load_json(generative_caption_path)
    data_3 = safe_load_json(query_first_article_path)
    entity_name_data = safe_load_json(entity_name_path)
    question_answer_data = safe_load_json(question_answer_path)
    new_database = safe_load_json(new_database_path)
    
    query_ids = safe_load_json("final_json_result/context_extraction_image_article.json")
    
    # Convert new_database to a dictionary: {query_id: {...}}
    newdatabase_data = {}
    for value in new_database if isinstance(new_database, list) else []:
        newdatabase_data[value.get('query_id', '')] = {
            'position': value.get('article_position', ''),
            'content': value.get('article', ''),
            'crawl_caption': value.get('crawl_alt', '')
        }

    final_merge_result = []

    for query_id, caption_value in query_ids.items():
        batch = {
            'query_id': query_id,
            'question_answer': question_answer_data.get(query_id, ''),
            'name_entity_keyword': entity_name_data.get(query_id, ''),
            'generated_caption': generated_caption_data.get(query_id, ''),
            'article_summary': {
                'fact_summary': data_3.get(query_id, {}).get('summary', '')
            },
            'crawl_caption': newdatabase_data.get(query_id, {}).get("crawl_caption", ''),
            'article': newdatabase_data.get(query_id, {}).get("content", '')
        }
        final_merge_result.append(batch)

    # Save once after loop
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_merge_result, f, ensure_ascii=False, indent=2)
    print(f"Final merged result saved to: {output_path}")


def create_submission():
    final_caption_json = "assemble_result/cot_5_things_fact_more_event_llama.json"
    # For public test submission, use the public test file which already has correct format
    csv_path = "final_csv_result/public_final_retrieval_merging_final_results.csv"
    output_csv_path = "final_csv_result/submission_final.csv"

    # Load caption data with debug
    print(f"[DEBUG] Loading caption data from: {final_caption_json}")
    with open(final_caption_json, "r", encoding="utf-8") as f:
        caption_data = json.load(f)
    
    print(f"[DEBUG] Loaded {len(caption_data)} captions")
    print(f"[DEBUG] Caption data type: {type(caption_data)}")
    
    # Show sample caption keys
    if caption_data:
        sample_keys = list(caption_data.keys())[:5] if isinstance(caption_data, dict) else []
        print(f"[DEBUG] Sample caption keys: {sample_keys}")
        if sample_keys:
            print(f"[DEBUG] Sample caption value: {caption_data[sample_keys[0]][:100]}...")

    updated_rows = []
    csv_query_ids = []
    matched_count = 0
    unmatched_count = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames  # Already correct: article_id_1..10

        for idx, row in enumerate(reader):
            query_id = row["query_id"]
            csv_query_ids.append(query_id)
            
            new_caption = caption_data.get(query_id, "")
            
            if new_caption:
                matched_count += 1
            else:
                unmatched_count += 1
                if unmatched_count <= 5:  # Show first 5 unmatched
                    print(f"[DEBUG] No caption found for query_id: {query_id}")
            
            row["generated_caption"] = new_caption  
            updated_rows.append(row)
            
            # Show first few for debugging
            if idx < 3:
                print(f"[DEBUG] Row {idx}: query_id={query_id}, caption_length={len(new_caption)}")

    print(f"\n[DEBUG] CSV has {len(csv_query_ids)} queries")
    print(f"[DEBUG] Sample CSV query_ids: {csv_query_ids[:5]}")
    print(f"[DEBUG] Matched: {matched_count}, Unmatched: {unmatched_count}")

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    print(f"\n✅ Submission CSV saved to: {output_csv_path}")
