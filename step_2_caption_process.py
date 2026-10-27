"""Stage-2 driver: merge context -> generate drafts -> normalize -> submission.

FIXES over the shipped version
------------------------------
* merge_function() and the template/JSON loading ran at IMPORT time, so merely
  importing this module (or asking for --help) rebuilt final_merge_result.json.
  They now run inside main().
* The script returned early whenever assemble_result/*.json existed, silently
  reusing another configuration's captions.  --force regenerates, and
  --run_suffix gives each configuration its own output files.
* create_submission() was called with a keyword the function did not accept, so
  the normalized captions never reached the CSV.  Fixed in
  step_2_merge_all_elements.py.
* The entity gate was enrichment-only: the prompt always received the raw
  keyword string.  --gate {none,prompt,enrichment,unified} now covers all four
  variants of Table 3(d).

For the ablation study use run_ablation.py instead: it keeps every run in its
own runs/<tag>/ directory.
"""
import argparse
import json
import os
from pathlib import Path

from jinja2 import Template
from tqdm import tqdm

from step_2_merge_all_elements import (merge_function, create_submission,
                                       CAPTION_SOURCES)
from step_2_caption_normalizer import CaptionNormalizer

PROMPT_TEMPLATE_DIR = Path('./assemble_caption_prompt_template')
RESULT_DIR = Path('./assemble_result')
FEWSHOT_EXAMPLE_PATH = PROMPT_TEMPLATE_DIR / 'test.json'
CAPTION_INPUT_PATH = Path('./public_test_final_elements_json/final_merge_result.json')
TEST_CAPTION_INPUT_PATH = PROMPT_TEMPLATE_DIR / 'test.json'

PROMPT_GATES = {"prompt", "unified"}
ENRICHMENT_GATES = {"enrichment", "unified"}

assembler = None


def run_merge(output_path: Path, caption_source: str = "alt") -> None:
    """Rebuild final_merge_result.json from the five stage-1 outputs."""
    merge_function(
        generative_caption_path='./public_test_final_elements_json/final_rerank_public_test_detail_top1_caption.json',
        query_first_article_path='./public_test_final_elements_json/reranking_query_first_article_question_answer.json',
        entity_name_path="./assemble_result/name_entity_llama.json",
        question_answer_path="./assemble_result/questions_answers_llama.json",
        new_database_path="./result-tanphuc.json",
        output_path=output_path,
        caption_source=caption_source,
    )


def load_templates() -> dict:
    templates = {}
    for path in PROMPT_TEMPLATE_DIR.rglob("*.j2"):
        templates[path.stem] = Template(path.read_text(encoding='utf-8'))
    return templates


def compose_input(batch: dict, examples: list, keyword_override: str | None = None) -> dict:
    """Render context for one query.

    keyword_override replaces the raw LLM keyword string by the gated set E*
    (Gate 1).  The shipped version had no such parameter, which is why the
    published run is enrichment-only gating.
    """
    example_contexts = [None] * len(examples)
    return {
        "article": batch.get("article", ""),
        "generated_caption": batch.get("generated_caption", ""),
        "crawl_caption": batch.get("crawl_caption", ""),
        "question_answer": batch.get("question_answer", ""),
        "name_entity_keyword": (
            keyword_override if keyword_override is not None
            else batch.get("name_entity_keyword", "")
        ),
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
                "expected_output": example.get('expected_output', ""),
            }
            for example, example_context in zip(examples, example_contexts)
        ],
    }


def select_entities(merge_data: list, args) -> dict:
    """E* per query, shared by both gates."""
    from step_2_entity_selectors import build_selector

    selector = build_selector("heuristic", config_path=args.entity_config_path,
                              budget=args.top_k)
    selections = {}
    for batch in tqdm(merge_data, desc="selecting entities"):
        chosen, _ = selector.select(batch)
        selections[batch["query_id"]] = chosen
    return selections


def normalize_captions(llm_output_path, merge_data_path, output_path, args,
                       selections=None):
    print("STARTING CAPTION NORMALIZATION")
    gate_on = args.gate in ENRICHMENT_GATES or args.use_entity_scoring
    print(f"Entity gate at enrichment: {'ENABLED' if gate_on else 'DISABLED'}")

    normalizer = CaptionNormalizer(
        max_words=args.max_words,
        expected_length=args.expected_length,
        use_entity_scoring=args.use_entity_scoring and selections is None,
        entity_config_path=args.entity_config_path,
        selection_map=selections if args.gate in ENRICHMENT_GATES else None,
        use_noun_phrases=not args.no_noun_phrases,
        preserve_selected=args.preserve_selected,
        pos_mask=args.pos_mask,
    )
    results = normalizer.normalize_batch(
        llm_output_path=str(llm_output_path),
        merge_data_path=str(merge_data_path),
        output_path=str(output_path),
        verbose=True,
    )
    print(f"\nNormalized captions saved to: {output_path}")
    return results


def main(args):
    global assembler

    RESULT_DIR.mkdir(exist_ok=True)
    caption_input_path = Path(args.caption_input_path)

    if args.merge or not caption_input_path.exists():
        run_merge(caption_input_path, args.caption_source)

    templates = load_templates()
    if args.strategy not in templates:
        raise SystemExit(f"unknown --strategy {args.strategy!r}; "
                         f"available: {sorted(templates)}")

    examples = json.loads(FEWSHOT_EXAMPLE_PATH.read_text(encoding='utf-8'))
    template = templates[args.strategy]

    # -- template preview, no model needed ---------------------------------
    if args.template_test:
        test_inputs = json.loads(TEST_CAPTION_INPUT_PATH.read_text(encoding='utf-8'))
        prompt = template.render(**compose_input(test_inputs[0], examples))
        out = PROMPT_TEMPLATE_DIR / f"{args.strategy}.md"
        out.write_text(prompt, encoding='utf-8')
        print(f"rendered prompt -> {out}")
        return

    inputs = json.loads(caption_input_path.read_text(encoding='utf-8'))
    if args.limit:
        inputs = inputs[: args.limit]

    if args.gate in PROMPT_GATES and args.reuse_drafts:
        raise SystemExit(
            "--reuse_drafts is invalid with --gate prompt/unified: these gates change "
            "the prompt, so the drafts must be regenerated.")

    suffix = f"_{args.run_suffix}" if args.run_suffix else ""
    draft_path = RESULT_DIR / f"{args.strategy}_{args.model_type}{suffix}.json"
    normalized_path = RESULT_DIR / f"normalized_captions{suffix}.json"

    # -- entity selection ---------------------------------------------------
    selections = None
    if args.gate != "none":
        selections = select_entities(inputs, args)
        (RESULT_DIR / f"selected_entities{suffix}.json").write_text(
            json.dumps(selections, indent=2, ensure_ascii=False), encoding='utf-8')

    # -- drafts -------------------------------------------------------------
    reuse_path = Path(args.reuse_drafts) if args.reuse_drafts else draft_path
    if reuse_path.exists() and not args.force:
        if args.gate in PROMPT_GATES:
            raise SystemExit(
                f"{reuse_path} already exists, but --gate {args.gate} changes the prompt, "
                f"so those drafts are stale. Pass --force, or --run_suffix to write new ones.")
        print(f"Reusing existing drafts: {reuse_path} (pass --force to regenerate)")
        result = json.loads(reuse_path.read_text(encoding='utf-8'))
        if args.limit:
            keep = {b['query_id'] for b in inputs}
            result = {k: v for k, v in result.items() if k in keep}
    else:
        # imported here so --help and --template_test work without transformers
        from llamassemblers import LLMAssembler
        print(f"Initializing LLMAssembler with model_type={args.model_type}")
        assembler = LLMAssembler(device=args.device, model_type=args.model_type)
        if args.seed is not None:
            from transformers import set_seed
            set_seed(args.seed)

        result = {}
        for i, batch in enumerate(tqdm(inputs, desc="Processing batches")):
            keyword_override = None
            if args.gate in PROMPT_GATES:
                keyword_override = ", ".join(selections.get(batch['query_id'], []))
            prompt = template.render(**compose_input(batch, examples, keyword_override))

            if args.qa:
                result[batch['query_id']] = assembler.question_answer(prompt)
            elif args.name_entity:
                result[batch['query_id']] = assembler.name_entity_extraction(prompt)
            else:
                result[batch['query_id']] = assembler.assemble(prompt)

            # checkpoint instead of rewriting the whole dict every query
            if (i + 1) % args.checkpoint_every == 0:
                draft_path.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                      encoding='utf-8')
        draft_path.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                              encoding='utf-8')

    if args.qa or args.name_entity:
        print(f"Saved to {draft_path}")
        return

    # normalize_batch reads drafts from disk: make sure this run's file holds
    # exactly the captions we are about to normalize
    draft_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')

    # -- normalization + submission ----------------------------------------
    if args.normalize_output:
        normalize_captions(draft_path, caption_input_path, normalized_path, args,
                           selections=selections)
        create_submission(final_caption_json_path=str(normalized_path),
                          output_csv_path=args.output_csv)
    else:
        create_submission(final_caption_json_path=str(draft_path),
                          output_csv_path=args.output_csv)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--strategy', type=str, default='cot_5_things_fact_more_event',
                   help="prompt template stem in assemble_caption_prompt_template/")
    p.add_argument('--template_test', action='store_true',
                   help="render the prompt for one example and exit")
    p.add_argument('--qa', action='store_true', help="run the QA-generation prompt")
    p.add_argument('--name_entity', action='store_true',
                   help="run the keyword-extraction prompt")
    p.add_argument('--merge', action='store_true',
                   help="rebuild final_merge_result.json before running")
    p.add_argument('--caption_source', default='alt', choices=list(CAPTION_SOURCES),
                   help="which crawled string becomes the editorial caption: 'alt' "
                        "(shipped default, the <img alt> text), 'caption' (the figure "
                        "caption, which also carries a kicker and a photo credit) or "
                        "'merged' (alt, falling back to a cleaned figure caption). "
                        "Only takes effect when the merge actually runs (--merge).")
    p.add_argument('--caption_input_path', type=Path, default=CAPTION_INPUT_PATH)
    p.add_argument('--model_type', type=str, default='llama3',
                   help="llama3 (Llama-3-8B) or qwen3 (Qwen2.5-14B)")
    p.add_argument('--device', type=str, default='cuda:5')
    p.add_argument('--seed', type=int, default=None,
                   help="decoding seed; the published run set none, so it was not reproducible")
    p.add_argument('--limit', type=int, default=None, help="use only the first N queries")
    p.add_argument('--checkpoint_every', type=int, default=50)
    p.add_argument('--reuse_drafts', type=str, default=None,
                   help="reuse this drafts JSON instead of generating; invalid with "
                        "--gate prompt/unified, which change the prompt")
    p.add_argument('--force', action='store_true',
                   help="regenerate drafts even if the output file already exists")
    p.add_argument('--run_suffix', type=str, default='',
                   help="suffix for every output file, so configurations cannot overwrite each other")
    p.add_argument('--output_csv', type=str,
                   default='final_csv_result/submission_final.csv')

    p.add_argument('--normalize_output', action='store_true', default=True)
    p.add_argument('--no_normalize_output', dest='normalize_output', action='store_false')
    p.add_argument('--max_words', type=int, default=105)
    p.add_argument('--expected_length', type=int, default=104)

    p.add_argument('--gate', default='enrichment',
                   choices=['none', 'prompt', 'enrichment', 'unified'],
                   help="where E* is applied: none=unfiltered baseline, "
                        "enrichment=the published VG-Cap run, unified=Section 3.6")
    p.add_argument('--top_k', type=int, default=10, help="K_e")
    p.add_argument('--use_entity_scoring', action='store_true', default=False,
                   help="legacy flag, equivalent to --gate enrichment")
    p.add_argument('--entity_config_path', type=str,
                   default='configs/entity_scoring_config.json',
                   help="use configs/entity_scoring_config_fixed.json for the repaired scorer")
    p.add_argument('--no_noun_phrases', action='store_true',
                   help="stop enrichment from appending ungated article noun phrases")
    p.add_argument('--preserve_selected', action='store_true',
                   help="pruning protects every mention of E* (Section 3.6 step 2)")
    p.add_argument('--pos_mask', action='store_true',
                   help="pruning also protects PROPN/NUM tokens")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.use_entity_scoring and args.gate == 'none':
        args.gate = 'enrichment'
    main(args)
