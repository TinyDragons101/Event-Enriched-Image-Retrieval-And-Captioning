"""T4: dump the fully rendered prompts for the supplementary material.

    python scripts/dump_prompts.py --n 3 --out supplementary/prompts
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jinja2 import Template
from run_ablation import compose_input, PROMPT_TEMPLATE_DIR, FEWSHOT_EXAMPLE_PATH, MERGE_DATA_PATH

SYSTEM_PROMPT = (
    "You are a professional journalist writing extended captions for major news outlets. "
    "Each caption should not only describe what is visible in the photo but clearly link it "
    "to the related news story. Your captions must answer: What does the image show? What is "
    "happening? Why does it matter? Use a journalistic tone, follow AP style, and ensure that "
    "the caption can stand on its own without needing to read the article."
)  # llama3.py:499-503


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", default="cot_5_things_fact_more_event")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", default="supplementary/prompts")
    ap.add_argument("--selected_entities", default=None,
                    help="runs/<tag>/selected_entities.json, to show a gated prompt too")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    template = Template((PROMPT_TEMPLATE_DIR / f"{args.strategy}.j2").read_text(encoding="utf-8"))
    merge = json.loads(MERGE_DATA_PATH.read_text(encoding="utf-8"))
    examples = json.loads(FEWSHOT_EXAMPLE_PATH.read_text(encoding="utf-8"))
    selections = json.loads(Path(args.selected_entities).read_text(encoding="utf-8")) \
        if args.selected_entities else {}

    (out_dir / "system_prompt.txt").write_text(SYSTEM_PROMPT + "\n", encoding="utf-8")
    (out_dir / "template.j2").write_text(
        (PROMPT_TEMPLATE_DIR / f"{args.strategy}.j2").read_text(encoding="utf-8"), encoding="utf-8")

    for batch in merge[: args.n]:
        qid = batch["query_id"]
        rendered = template.render(**compose_input(batch, examples))
        (out_dir / f"{qid}_user_prompt.txt").write_text(rendered, encoding="utf-8")
        if qid in selections:
            gated = template.render(**compose_input(batch, examples, ", ".join(selections[qid])))
            (out_dir / f"{qid}_user_prompt_gated.txt").write_text(gated, encoding="utf-8")

    print(f"-> {out_dir} (system prompt, template, {args.n} rendered examples)")
    print("Context-field order in the prompt: summary.fact, generated_caption, crawl_caption, "
          "name_entity_keyword, question_answer.")


if __name__ == "__main__":
    main()
