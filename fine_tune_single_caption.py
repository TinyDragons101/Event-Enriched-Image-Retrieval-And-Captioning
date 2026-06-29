"""
Fine-tune Caption Generator - Single Sample Processing
=======================================================
File này dùng để generate caption cho một merge result đơn lẻ (1 bài báo + 1 ảnh)
Mục đích: Tạo training data cho fine-tuning model

Pipeline tương tự step_2_caption_process.py với strategy cot_5_things_fact_more_event
nhưng chỉ xử lý 1 sample duy nhất từ file input.

Usage:
    python fine_tune_single_caption.py \
        --input_file path/to/single_merge_result.json \
        --output_file path/to/output_caption.json \
        --model_type llama3 \
        --device cuda:0

Input JSON structure (single_merge_result.json):
{
    "query_id": "sample_001",
    "article": "Full article text...",
    "generated_caption": "Caption from VLM model...",
    "crawl_caption": "Original caption from article...",
    "question_answer": "Q1: ... A1: ...",
    "name_entity_keyword": "Entity1, Entity2, ...",
    "related_phrases": "Phrase1, Phrase2, ...",
    "context": "Additional context...",
    "article_summary": {
        "raw_summary": "...",
        "restruct_summary": "...",
        "fact_summary": "..."
    }
}
"""

import argparse
import json
from pathlib import Path
from jinja2 import Template
from llamassemblers import LLMAssembler
from step_2_caption_normalizer import CaptionNormalizer

# Paths
PROMPT_TEMPLATE_DIR = Path('./assemble_caption_prompt_template')
STRATEGY_TEMPLATE = 'cot_5_things_fact_more_event'
DEFAULT_OUTPUT_DIR = Path('./fine_tune_outputs')


class SingleCaptionGenerator:
    """
    Generator để tạo caption cho một sample đơn lẻ
    Sử dụng pipeline tương tự step_2_caption_process.py
    """

    def __init__(self, model_type='llama3', device='cuda:0'):
        """
        Initialize generator

        Args:
            model_type: Model type (llama3, qwen3, etc.)
            device: CUDA device
        """
        print(f"🚀 Initializing LLMAssembler with model_type={model_type}, device={device}")
        self.assembler = LLMAssembler(device=device, model_type=model_type)
        self.model_type = model_type

        # Load template
        template_path = PROMPT_TEMPLATE_DIR / f"{STRATEGY_TEMPLATE}.j2"
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")

        with open(template_path, 'r', encoding='utf-8') as f:
            template_str = f.read()
            self.template = Template(template_str)

        print(f"✅ Template loaded: {STRATEGY_TEMPLATE}.j2")

    def compose_input(self, merge_result: dict) -> dict:
        """
        Compose input data từ merge result
        Tương tự compose_input trong step_2_caption_process.py

        Args:
            merge_result: Dict chứa đầy đủ thông tin merge

        Returns:
            Dict với format phù hợp cho template
        """
        return {
            "article": merge_result.get("article", ""),
            "generated_caption": merge_result.get("generated_caption", ""),
            "crawl_caption": merge_result.get("crawl_caption", ""),
            "question_answer": merge_result.get("question_answer", ""),
            "name_entity_keyword": merge_result.get("name_entity_keyword", ""),
            "related_phrases": merge_result.get("related_phrases", ""),
            "context": merge_result.get("context", ""),
            "summary": {
                "raw": merge_result.get("article_summary", {}).get("raw_summary", ""),
                "restruct": merge_result.get("article_summary", {}).get("restruct_summary", ""),
                "fact": merge_result.get("article_summary", {}).get("fact_summary", ""),
            },
            "examples": []  # No few-shot examples for fine-tuning
        }

    def generate_caption(self, merge_result: dict) -> str:
        """
        Generate caption cho một merge result

        Args:
            merge_result: Dict chứa merge result

        Returns:
            Generated caption string
        """
        # Compose input
        input_data = self.compose_input(merge_result)

        # Render prompt từ template
        prompt = self.template.render(**input_data)

        # Generate caption bằng LLM
        print("🤖 Generating caption with LLM...")
        enhanced_caption = self.assembler.assemble(prompt)

        return enhanced_caption

    def save_prompt_for_inspection(self, merge_result: dict, output_path: Path):
        """
        Lưu prompt để kiểm tra (debug purpose)

        Args:
            merge_result: Dict chứa merge result
            output_path: Path để lưu prompt
        """
        input_data = self.compose_input(merge_result)
        prompt = self.template.render(**input_data)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(prompt)

        print(f"📝 Prompt saved to: {output_path}")


def normalize_caption(
    caption: str,
    merge_result: dict,
    max_words: int = 105,
    expected_length: int = 104,
    use_entity_scoring: bool = False,
    entity_config_path: str = None
) -> str:
    """
    Normalize caption sau khi generate

    Args:
        caption: Generated caption
        merge_result: Original merge result data
        max_words: Max words for truncation
        expected_length: Expected final length
        use_entity_scoring: Use entity heuristic scoring
        entity_config_path: Path to entity config

    Returns:
        Normalized caption
    """
    print("🔧 Normalizing caption...")

    normalizer = CaptionNormalizer(
        max_words=max_words,
        expected_length=expected_length,
        use_entity_scoring=use_entity_scoring,
        entity_config_path=entity_config_path
    )

    normalized = normalizer.process_single_caption(merge_result, caption)

    return normalized


def main(args):
    """Main function"""

    # Create output directory
    DEFAULT_OUTPUT_DIR.mkdir(exist_ok=True)

    # Load input merge result
    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"📂 Loading input from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        merge_result = json.load(f)

    # Validate input structure
    required_fields = ['query_id', 'article', 'generated_caption']
    missing_fields = [f for f in required_fields if f not in merge_result]
    if missing_fields:
        raise ValueError(f"Missing required fields in input: {missing_fields}")

    query_id = merge_result.get('query_id')
    print(f"✅ Loaded merge result for query_id: {query_id}")

    # Initialize generator
    generator = SingleCaptionGenerator(
        model_type=args.model_type,
        device=args.device
    )

    # Save prompt for inspection (optional)
    if args.save_prompt:
        prompt_path = DEFAULT_OUTPUT_DIR / f"{query_id}_prompt.txt"
        generator.save_prompt_for_inspection(merge_result, prompt_path)

    # Generate caption
    print("\n" + "="*60)
    print("GENERATING CAPTION")
    print("="*60)

    caption = generator.generate_caption(merge_result)

    print(f"\n📄 Raw LLM Output:\n{caption}\n")

    # Normalize if requested
    if args.normalize:
        normalized_caption = normalize_caption(
            caption=caption,
            merge_result=merge_result,
            max_words=args.max_words,
            expected_length=args.expected_length,
            use_entity_scoring=args.use_entity_scoring,
            entity_config_path=args.entity_config_path
        )

        print(f"✨ Normalized Caption:\n{normalized_caption}\n")
        final_caption = normalized_caption
    else:
        final_caption = caption

    # Prepare output
    output_data = {
        "query_id": query_id,
        "raw_llm_output": caption,
        "final_caption": final_caption,
        "normalized": args.normalize,
        "model_type": args.model_type,
        "strategy": STRATEGY_TEMPLATE
    }

    # Save output
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("="*60)
    print(f"✅ Caption saved to: {output_path}")
    print("="*60)

    # Print summary
    print(f"\n📊 Summary:")
    print(f"  - Query ID: {query_id}")
    print(f"  - Model: {args.model_type}")
    print(f"  - Strategy: {STRATEGY_TEMPLATE}")
    print(f"  - Normalized: {args.normalize}")
    print(f"  - Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate caption for a single merge result (fine-tuning purpose)"
    )

    # Required arguments
    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help="Path to input JSON file containing single merge result"
    )

    parser.add_argument(
        '--output_file',
        type=str,
        default=str(DEFAULT_OUTPUT_DIR / 'output_caption.json'),
        help="Path to save output caption JSON"
    )

    # Model arguments
    parser.add_argument(
        '--model_type',
        type=str,
        default='llama3',
        help="Model type: llama3, qwen3, or full names"
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help="CUDA device (e.g., cuda:0, cuda:1)"
    )

    # Normalization arguments
    parser.add_argument(
        '--normalize',
        action='store_true',
        default=True,
        help="Apply normalization to generated caption"
    )

    parser.add_argument(
        '--max_words',
        type=int,
        default=105,
        help="Maximum words for normalization"
    )

    parser.add_argument(
        '--expected_length',
        type=int,
        default=104,
        help="Expected caption length after normalization"
    )

    parser.add_argument(
        '--use_entity_scoring',
        action='store_true',
        default=False,
        help="Enable entity heuristic scoring for normalization"
    )

    parser.add_argument(
        '--entity_config_path',
        type=str,
        default=None,
        help="Path to entity scoring config JSON"
    )

    # Debug arguments
    parser.add_argument(
        '--save_prompt',
        action='store_true',
        default=False,
        help="Save rendered prompt to file for inspection"
    )

    args = parser.parse_args()
    main(args)
