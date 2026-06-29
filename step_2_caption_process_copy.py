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
COT_5_THINGS_FACT_MORE_EVENT_OUTPUT_PATH = RESULT_DIR / 'cot_5_things_fact_more_event_llama.json'
NORMALIZED_OUTPUT_PATH = RESULT_DIR / 'normalized_captions.json'

merge_function(
    generative_caption_path='./public_test_final_elements_json/final_rerank_public_test_detail_top1_caption.json',
    query_first_article_path='./public_test_final_elements_json/reranking_query_first_article_question_answer.json',
    entity_name_path="./assemble_result/name_entity_llama.json",
    question_answer_path="./assemble_result/questions_answers_llama.json",
    new_database_path="./result-tanphuc.json",
    output_path=CAPTION_INPUT_PATH
)

assembler = LLMAssembler(device = 'cuda:7', model_type = 'llama3')
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
    
    # Nếu đã tồn tại file normalized captions thì chỉ cần tạo submission
    if args.normalize_output and os.path.exists(NORMALIZED_OUTPUT_PATH):
        print(f"Normalized file already exists: {NORMALIZED_OUTPUT_PATH}")
        print("Creating submission with normalization...")
        create_submission(final_caption_json_path=NORMALIZED_OUTPUT_PATH)
        return
    
    CAPTION_INPUT_PATH = args.caption_input_path
    
    # Nếu đã tồn tại file cot_5_things_fact_more_event_output.json thì chỉ cần normalize captions rồi tạo submission
    if os.path.exists(COT_5_THINGS_FACT_MORE_EVENT_OUTPUT_PATH):
        print(f"Found existing LLM output: {COT_5_THINGS_FACT_MORE_EVENT_OUTPUT_PATH}")
        print("Skipping LLM generation, proceeding to normalization...")

        if args.normalize_output:
            normalize_captions(
                llm_output_path=COT_5_THINGS_FACT_MORE_EVENT_OUTPUT_PATH,
                merge_data_path=CAPTION_INPUT_PATH,
                output_path=NORMALIZED_OUTPUT_PATH,
                max_words=args.max_words,
                expected_length=args.expected_length
            )
            
            print("Creating submission with normalization...")
            create_submission(final_caption_json_path=NORMALIZED_OUTPUT_PATH)

        else:
            print("Creating submission without normalization...")
            create_submission(final_caption_json_path=COT_5_THINGS_FACT_MORE_EVENT_OUTPUT_PATH)
        
        return

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

        with open(RESULT_DIR / f"{strat}_{model_name}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    if not args.qa and not args.name_entity:
        if args.normalize_output:
            print("Normalizing captions...")
            normalize_captions(
                llm_output_path=RESULT_DIR / f"{strat}_{model_name}.json",
                merge_data_path=CAPTION_INPUT_PATH,
                output_path=NORMALIZED_OUTPUT_PATH,
                max_words=args.max_words,
                expected_length=args.expected_length
            )
            print("Creating submission with normalization...")
            create_submission(final_caption_json_path=NORMALIZED_OUTPUT_PATH)
            
        else:
            print("Creating submission without normalization...")
            create_submission(final_caption_json_path=RESULT_DIR / f"{strat}_{model_name}.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=str, choices=template_strats.keys(), required=True,
                        help="Strategy to use for caption assembly")
    parser.add_argument('--template_test', action='store_true',
                        help="Test and print loaded template")
    parser.add_argument('--qa', action='store_true',
                        help="Test and print loaded template strategies")
    parser.add_argument('--generate_caption_path', type=Path, default=RESULT_DIR / 'final_rerank_public_test_detail_top1_caption.json',
                        help="Path to the generated caption JSON file")
    parser.add_argument('--question_answer_path', type=Path, default=RESULT_DIR / 'question_answer.json',
                        help="Path to the question answer JSON file")
    parser.add_argument('--new_database_path', type=Path, default=RESULT_DIR / 'new_database.json',
                        help="Path to the new database JSON file")  
    parser.add_argument('--caption_input_path', type=Path, default=CAPTION_INPUT_PATH,
                        help="Path to the caption input JSON file")
    parser.add_argument('--name_entity', action='store_true',
                        help="Test and print loaded template strategies")
    parser.add_argument('--model_type',type=str, default='llama',
                        help="Test and print loaded template strategies")
    parser.add_argument('--normalize_output',action='store_true', default=True,
                        help="Whether to normalize captions after generation")
    parser.add_argument('--max_words', type=int, default=105,
                        help="Maximum words for normalization")
    parser.add_argument('--expected_length', type=int, default=104,
                        help="Expected caption length after normalization")
    
    args = parser.parse_args()
    main(args)
