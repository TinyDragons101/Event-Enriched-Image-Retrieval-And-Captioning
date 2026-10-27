import json
import csv
import re

# ---------------------------------------------------------------------------
# Which crawled string becomes C_image's editorial caption
# ---------------------------------------------------------------------------
# step_2_0_create_result.py stores two strings per database image:
#
#   crawl_alt      the <img alt="..."> text
#   crawl_caption  the text of div.image__metadata (the figure caption)
#
# The shipped pipeline uses crawl_alt, and measurement on result-tanphuc.json
# (3000 records) says that is the right choice:
#
#   alt == caption                     38  ( 1.3%)
#   caption == alt + photo credit    2254  (75.1%)   e.g. "...the race." +
#                                                    "Matt King/Getty Images"
#   genuinely different               708  (23.6%)   caption = "<kicker> —<alt>"
#                                                    + credit; alt is the body
#   alt empty but caption present      11
#
# So the figure caption is the alt text wrapped in a kicker and a photo credit.
# Feeding it to the scorer would push agency names (Getty Images, AFP, Reuters)
# into exact_match_crawl, the highest-weighted feature (10.0).  'alt' therefore
# stays the default; 'merged' only fills the 11 records where alt is empty.
CAPTION_SOURCES = ("alt", "caption", "merged")

#: "<kicker> —" prefix the CNN template puts in front of the caption body
_KICKER = re.compile(r"^.{0,90}?\s*\u2014\s*")
#: trailing photo credit: one or more "Name/Agency" runs glued to the end
_CREDIT = re.compile(
    r"(?:[A-Z][\w.'&-]*(?:\s+[A-Z][\w.'&-]*)*/)+"
    r"[A-Z][\w.'&-]*(?:\s+[A-Z][\w.'&-]*)*\s*$"
)


def strip_caption_furniture(text: str) -> str:
    """Remove the kicker prefix and the trailing photo credit."""
    text = (text or "").strip()
    if not text:
        return ""
    text = _KICKER.sub("", text, count=1)
    text = _CREDIT.sub("", text).strip()
    return text


def pick_crawl_caption(record: dict, source: str = "alt") -> str:
    """The editorial-caption string handed to the prompt and to the scorer."""
    alt = (record.get("crawl_alt") or "").strip()
    cap = (record.get("crawl_caption") or "").strip()

    if source == "alt":
        return alt
    if source == "caption":
        return strip_caption_furniture(cap)
    if source == "merged":
        return alt or strip_caption_furniture(cap)
    raise ValueError(f"caption_source must be one of {CAPTION_SOURCES}, got {source!r}")


def safe_load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load {path} due to {e}")
        return {}


def merge_function(generative_caption_path, query_first_article_path, entity_name_path,
                   question_answer_path, new_database_path, output_path,
                   caption_source: str = "alt"):
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
            'crawl_caption': pick_crawl_caption(value, caption_source)
        }

    print(f"[merge] caption_source={caption_source}")
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


def load_caption_map(path) -> dict:
    """{query_id: caption} from either shape this repo writes.

    The LLM step saves a dict {query_id: caption}; CaptionNormalizer saves a
    LIST of records with 'query_id' / 'normalized_caption'.  create_submission()
    used to assume the dict shape, so feeding it normalized_captions.json
    raised AttributeError on .get().
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        out = {}
        for rec in data:
            caption = rec.get("normalized_caption")
            if caption is None:
                caption = rec.get("caption", rec.get("generated_caption", ""))
            out[rec["query_id"]] = caption
        return out
    raise TypeError(f"unsupported caption file shape in {path}: {type(data).__name__}")


def create_submission(
    final_caption_json_path: str = "assemble_result/cot_5_things_fact_more_event_llama.json",
    csv_path: str = "final_csv_result/public_final_retrieval_merging_final_results.csv",
    output_csv_path: str = "final_csv_result/submission_final.csv",
):
    """Write the submission CSV.

    FIXED: the parameter was hard-coded, so step_2_caption_process.py's calls
    (create_submission(final_caption_json_path=...)) raised TypeError and the
    normalized captions never reached the CSV -- which is why
    final_csv_result/submission_final.csv equals the raw drafts, not VG-Cap.
    """
    # Load caption data with debug
    print(f"[DEBUG] Loading caption data from: {final_caption_json_path}")
    caption_data = load_caption_map(final_caption_json_path)
    
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
