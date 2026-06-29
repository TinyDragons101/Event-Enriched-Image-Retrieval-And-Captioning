import argparse
from pathlib import Path
from jinja2 import Template
import json
from tqdm import tqdm
from llamassemblers import LLMAssembler
import time
from step_2_merge_all_elements import merge_function, create_submission
from step_2_caption_normalizer import CaptionNormalizer
import os

PROMPT_TEMPLATE_DIR = Path('./assemble_caption_prompt_template')
RESULT_DIR = Path('./assemble_result')
FEWSHOT_EXAMPLE_PATH = Path('./assemble_caption_prompt_template/test.json')
CAPTION_INPUT_PATH = Path('./public_test_final_elements_json/final_merge_result.json')
TEST_FEWSHOT_EXAMPLE_PATH = PROMPT_TEMPLATE_DIR / 'test.json'
TEST_CAPTION_INPUT_PATH = PROMPT_TEMPLATE_DIR / 'test.json'

merge_function(
    generative_caption_path='./public_test_final_elements_json/final_rerank_public_test_detail_top1_caption.json',
    query_first_article_path='./public_test_final_elements_json/reranking_query_first_article_question_answer.json',
    entity_name_path="./assemble_result/name_entity_llama.json",
    question_answer_path="./assemble_result/questions_answers_llama.json",
    new_database_path="./result-tanphuc.json",
    output_path=CAPTION_INPUT_PATH
)

assembler = LLMAssembler(device = 'cuda:5', model_type = 'llama3')
template_strats: dict[str, Template] = {}

RESULT_DIR.mkdir(exist_ok=True)

with open(CAPTION_INPUT_PATH, 'r') as f:
    inputs = json.load(f)

with open(FEWSHOT_EXAMPLE_PATH, 'r') as f:
    examples = json.load(f)

for path in PROMPT_TEMPLATE_DIR.rglob("*.j2"):
    with open(path, 'r', encoding='utf-8') as f:
        template_str = f.read()
        template = Template(template_str)
        template_strats[str(path.stem)] = template


def assemble(input: dict, strat: str) -> str:
    template = template_strats[strat]
    prompt = template.render(**input)
    enhanced_caption = assembler.assemble(prompt)
    return enhanced_caption


def model_qa(input: dict, strat: str) -> str:
    template = template_strats[strat]
    prompt = template.render(**input)
    enhanced_caption = assembler.question_answer(prompt)
    return enhanced_caption


def model_name_entity(input: dict, strat: str) -> str:
    template = template_strats[strat]
    prompt = template.render(**input)
    enhanced_caption = assembler.name_entity_extraction(prompt)
    return enhanced_caption


def compose_input(batch: dict, examples: list[dict]) -> dict:
    example_contexts = [None] * len(examples)
    return {
        "article": batch.get("article", ""),
        "generated_caption": batch.get("generated_caption", ""),
        "crawl_caption": batch.get("crawl_caption", ""),
        "question_answer": batch.get("question_answer", ""),
        "name_entity_keyword": batch.get("name_entity_keyword", ""),
        "related_phrases": batch.get("related_phrases", ""),
        "context": batch.get("context", ""),
        "summary": {
            "raw": batch.get("article_summary", {}).get("raw_summary", ""),
            "restruct": batch.get("article_summary", {}).get("restruct_summary", ""),
            "fact": batch.get("article_summary", {}).get("fact_summary", ""),
        },
        "examples": [
            {
                "generated_caption": example.get('generated_caption', ""),
                "crawl_caption": example.get('crawl_caption', ""),
                "context": example_context,
                "summary": example.get('article_summary', {}),
                "expected_output": example.get('expected_output', "")
            }
            for example, example_context in zip(examples, example_contexts)
        ],
    }


def normalize_captions(
    llm_output_path: Path,
    merge_data_path: Path,
    output_path: Path,
    max_words: int = 105,
    expected_length: int = 104
):
    """
    Bước normalize caption sau khi assemble
    Tương tự như sematicNorm + entityEnricker
    """
    print("STARTING CAPTION NORMALIZATION")
    
    normalizer = CaptionNormalizer(
        max_words=max_words,
        expected_length=expected_length
    )
    
    results = normalizer.normalize_batch(
        llm_output_path=str(llm_output_path),
        merge_data_path=str(merge_data_path),
        output_path=str(output_path),
        verbose=True
    )
    
    print(f"\nNormalized captions saved to: {output_path}")
    
    return results


def main(args):
    # Đường dẫn file output từ LLM
    llm_output_file = RESULT_DIR / f"{args.strategy}_{args.model_type}.json"
    
    # Nếu đã tồn tại file normalized, tạo submission luôn
    normalized_output_file = RESULT_DIR / f"{args.strategy}_{args.model_type}_normalized.json"
    
    if os.path.exists(normalized_output_file):
        print(f"Normalized file already exists: {normalized_output_file}")
        print("Creating submission...")
        create_submission()
        return
    
    # Nếu đã có file LLM output nhưng chưa normalize
    if os.path.exists(llm_output_file) and not args.qa and not args.name_entity:
        print(f"Found existing LLM output: {llm_output_file}")
        print("Skipping LLM generation, proceeding to normalization...")
        
        # Chạy normalization
        normalize_captions(
            llm_output_path=llm_output_file,
            merge_data_path=args.caption_input_path,
            output_path=normalized_output_file,
            max_words=args.max_words,
            expected_length=args.expected_length
        )
        
        # Tạo submission
        print("Creating submission...")
        create_submission()
        return

    CAPTION_INPUT_PATH = args.caption_input_path
    strat = args.strategy
    template_test = args.template_test
    qa = args.qa
    model_name = args.model_type
    name_entity = args.name_entity
    
    if template_test:
        with open(TEST_CAPTION_INPUT_PATH, 'r') as f:
            test_inputs = json.load(f)
        with open(TEST_FEWSHOT_EXAMPLE_PATH, 'r') as f:
            test_examples = json.load(f)
        template = template_strats[strat]
        prompt = template.render(**compose_input(test_inputs[0], test_examples))
        with open(PROMPT_TEMPLATE_DIR / f"{strat}.md", 'w') as f:
            f.write(prompt)
        return
    
    # ========================================
    # BƯỚC 1: ASSEMBLE CAPTION TỪ LLM
    # ========================================
    print("\n" + "="*60)
    print("STEP 1: GENERATING CAPTIONS WITH LLM")
    print("="*60)
    
    result = {}
    
    for batch in tqdm(inputs, desc="Processing batches"):
        input_data = compose_input(batch, examples)
        if qa:
            enhanced_caption = model_qa(input_data, strat)
        elif name_entity:
            enhanced_caption = model_name_entity(input_data, strat)
        else:
            enhanced_caption = assemble(input_data, strat)
        
        result[batch['query_id']] = enhanced_caption

        # Lưu file tạm sau mỗi batch
        with open(llm_output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\nLLM output saved to: {llm_output_file}")
    
    # ========================================
    # BƯỚC 2: NORMALIZE CAPTION
    # (Chỉ chạy cho caption chính, không chạy cho QA và name_entity)
    # ========================================
    if not args.qa and not args.name_entity:
        normalize_captions(
            llm_output_path=llm_output_file,
            merge_data_path=args.caption_input_path,
            output_path=normalized_output_file,
            max_words=args.max_words,
            expected_length=args.expected_length
        )
        
        # ========================================
        # BƯỚC 3: TẠO SUBMISSION
        # ========================================
        print("\n" + "="*60)
        print("STEP 3: CREATING SUBMISSION")
        print("="*60)
        create_submission()
        print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=str, choices=template_strats.keys(), required=True,
                        help="Strategy to use for caption assembly")
    parser.add_argument('--template_test', action='store_true',
                        help="Test and print loaded template")
    parser.add_argument('--qa', action='store_true',
                        help="Generate question-answer pairs")
    parser.add_argument('--name_entity', action='store_true',
                        help="Extract named entities")
    parser.add_argument('--model_type', type=str, default='llama',
                        help="Model type to use")
    parser.add_argument('--caption_input_path', type=Path, default=CAPTION_INPUT_PATH,
                        help="Path to the caption input JSON file")
    parser.add_argument('--max_words', type=int, default=105,
                        help="Maximum words for normalization")
    parser.add_argument('--expected_length', type=int, default=104,
                        help="Expected caption length after normalization")
    
    # Các arguments cũ (giữ lại để tương thích)
    parser.add_argument('--generate_caption_path', type=Path, 
                        default=RESULT_DIR / 'final_rerank_public_test_detail_top1_caption.json',
                        help="Path to the generated caption JSON file")
    parser.add_argument('--question_answer_path', type=Path, 
                        default=RESULT_DIR / 'question_answer.json',
                        help="Path to the question answer JSON file")
    parser.add_argument('--new_database_path', type=Path, 
                        default=RESULT_DIR / 'new_database.json',
                        help="Path to the new database JSON file")
    
    args = parser.parse_args()
    main(args)