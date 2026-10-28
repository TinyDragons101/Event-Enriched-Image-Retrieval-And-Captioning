"""
Driver for the VG-Cap ablation study (Table 3 panels (b), (c), (d)).

Why a separate driver instead of more flags on step_2_caption_process.py:

  * that script writes to three hard-coded paths and returns early when they
    already exist, so a second configuration silently reuses the first one's
    captions -- fatal for an ablation table.  Here every run owns
    runs/<tag>/ and nothing is shared by accident;
  * the entity gate it exposes is enrichment-only.  compose_input() always
    renders the raw `name_entity_keyword` into the prompt, so the published
    0.6219 row is "enrichment-only gating", not the unified gate of
    Section 3.6.  --gate below makes all four variants runnable, which is
    exactly what Table 3(d) compares;
  * decoding is stochastic (do_sample=True) and no seed was ever set, so
    single runs were not reproducible.  --seed fixes that, per query.
    --temperature defaults to 0.3, the value the published drafts were made
    with (the code later switched to 0.6, which would not be comparable).

Gate semantics
--------------
  none         prompt: all keywords      enrichment: all article entities   (unfiltered baseline, 0.6166)
  prompt       prompt: E*                enrichment: all article entities
  enrichment   prompt: all keywords      enrichment: E*                     (= the published VG-Cap run)
  unified      prompt: E*                enrichment: E*                     (what Section 3.6 describes)

Examples
--------
  # Reference run: reproduce the published VG-Cap row and record its budget
  python run_ablation.py --tag vgcap_enrichment --gate enrichment --selector heuristic \
      --entity_config configs/entity_scoring_config.json

  # Table 3(d): the gate the paper actually claims
  python run_ablation.py --tag vgcap_unified --gate unified --selector heuristic \
      --entity_config configs/entity_scoring_config.json

  # Table 3(b) under a matched per-query budget, reusing the drafts when the
  # gate does not touch the prompt
  python run_ablation.py --tag sel_random_s1 --gate enrichment --selector random --seed 1 \
      --budget_match per_query --reference_selection runs/vgcap_enrichment/selected_entities.json \
      --reuse_drafts runs/vgcap_enrichment/llm_raw.json
"""
from __future__ import annotations

import argparse
import json
import csv
import os
import time
import zlib
from pathlib import Path
from statistics import mean

from jinja2 import Template
from tqdm import tqdm

from step_2_caption_normalizer import CaptionNormalizer
from step_2_entity_scorer import DEFAULT_FIXES, SPAN_PENALTY_MODES
from step_2_entity_selectors import build_selector, load_budget_map

PROMPT_TEMPLATE_DIR = Path("./assemble_caption_prompt_template")
FEWSHOT_EXAMPLE_PATH = PROMPT_TEMPLATE_DIR / "test.json"
MERGE_DATA_PATH = Path("./public_test_final_elements_json/final_merge_result.json")
RETRIEVAL_CSV = Path("./final_csv_result/public_final_retrieval_merging_final_results.csv")

PROMPT_GATES = {"prompt", "unified"}
ENRICHMENT_GATES = {"enrichment", "unified"}

DEFAULT_ENTITY_CONFIG = "configs/entity_scoring_config.json"

#: --fix_profile presets, applied on top of the config file's own `fixes` block.
#: 'legacy' is the published behaviour; 'all' repairs every known defect.
FIX_PROFILES = {
    "legacy": {},
    "all": {
        "parse_keyword_markdown": True,
        "word_boundary_match": True,
        "span_penalty": "paper",
        "substring_dedup": True,
        "first_mention_position": True,
        "drop_non_depictable": True,
    },
    # one switch at a time, for the "which defect mattered?" table
    "keyword_parse": {"parse_keyword_markdown": True},
    "word_boundary": {"word_boundary_match": True},
    "span_penalty": {"span_penalty": "paper"},
    "dedup": {"substring_dedup": True},
    "position": {"first_mention_position": True},
    "candidates": {"drop_non_depictable": True},
}

#: weight families addressed by --weight_scale / --weight_jitter
WEIGHT_FAMILIES = {
    "presence": ["exact_match_crawl", "exact_match_generated", "partial_match_crawl",
                 "partial_match_generated", "substring_in_qa", "substring_in_summary"],
    "position": ["early_position_bonus", "mid_position_bonus"],
    "type": ["type_person", "type_gpe", "type_org", "type_loc",
             "type_date", "type_event", "type_other"],
    "span": ["length_penalty", "length_penalty_factor"],
}


def atomic_write_json(path: Path, obj) -> None:
    """Write via a temp file + rename, so a kill mid-write never leaves a
    truncated llm_raw.json that --resume cannot parse."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def query_seed(seed: int, query_id: str) -> int:
    """Per-query seed: a draft depends only on (seed, query_id), not on how
    many queries ran before it, so --resume and job order cannot change it."""
    return (seed * 1_000_003 + zlib.crc32(query_id.encode("utf-8"))) % (2 ** 32)


def build_effective_config(args, run_dir: Path) -> str:
    """Apply --tau / --top_k / --weight_scale / --weight_jitter on top of the
    config file and write the result next to the run, so every sweep point is
    reproducible from its own directory."""
    cfg = json.loads(Path(args.entity_config).read_text(encoding="utf-8"))

    if args.tau is not None:
        cfg["thresholds"]["min_score"] = args.tau
    if args.top_k is not None:
        cfg["thresholds"]["top_k"] = args.top_k

    for spec in args.weight_scale or []:
        family, _, factor = spec.partition(":")
        if family not in WEIGHT_FAMILIES:
            raise SystemExit(f"unknown weight family '{family}'; pick from {sorted(WEIGHT_FAMILIES)}")
        for key in WEIGHT_FAMILIES[family]:
            if key in cfg["weights"]:
                cfg["weights"][key] = round(cfg["weights"][key] * float(factor), 6)

    if args.weight_jitter:
        import random as _random
        rng = _random.Random(args.jitter_seed)
        for key in list(cfg["weights"]):
            factor = 1.0 + rng.uniform(-args.weight_jitter, args.weight_jitter)
            cfg["weights"][key] = round(cfg["weights"][key] * factor, 6)

    fixes = dict(cfg.get("fixes", {}))
    fixes.update(FIX_PROFILES[args.fix_profile])
    for spec in args.fix or []:
        key, _, value = spec.partition("=")
        key, value = key.strip(), value.strip()
        if key not in DEFAULT_FIXES:
            raise SystemExit(f"unknown fix '{key}'; pick from {sorted(DEFAULT_FIXES)}")
        if key == "span_penalty":
            if value not in SPAN_PENALTY_MODES:
                raise SystemExit(f"--fix span_penalty= must be one of {SPAN_PENALTY_MODES}")
            fixes[key] = value
        else:
            fixes[key] = value.lower() not in ("", "0", "false", "no", "off")
    if args.span_penalty is not None:
        fixes["span_penalty"] = args.span_penalty
    if fixes:
        cfg["fixes"] = fixes

    cfg["_derived_from"] = args.entity_config
    path = run_dir / "effective_entity_config.json"
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# Prompt side
# ---------------------------------------------------------------------------
def compose_input(batch: dict, examples: list, keyword_override: str | None = None,
                  use_entities: bool = True, use_length_band: bool = True) -> dict:
    """Identical to step_2_caption_process.compose_input, except that the
    keyword field can be replaced by the gated entity set E* and that the two
    ablatable prompt blocks can be switched off (Table 3(a))."""
    example_contexts = [None] * len(examples)
    return {
        "use_entities": use_entities,
        "use_length_band": use_length_band,
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
                "generated_caption": example.get("generated_caption", ""),
                "crawl_caption": example.get("crawl_caption", ""),
                "context": example_context,
                "summary": example.get("article_summary", {}),
                "expected_output": example.get("expected_output", ""),
            }
            for example, example_context in zip(examples, example_contexts)
        ],
    }


def generate_drafts(args, merge_data, selections, out_path: Path) -> dict:
    """Run the LLM over every query and return {query_id: raw draft}."""
    from transformers import set_seed
    from llamassemblers import LLMAssembler

    # The ablation template is byte-identical to the original once both blocks
    # are on, but we still prefer the original file in that case so the
    # reference run cannot drift.
    full_prompt = args.prompt_entities and args.prompt_band
    stem = args.strategy if full_prompt else f"{args.strategy}_ablation"
    template_path = PROMPT_TEMPLATE_DIR / f"{stem}.j2"
    template = Template(template_path.read_text(encoding="utf-8"))
    print(f"[{args.tag}] prompt template: {template_path.name} "
          f"(entities={args.prompt_entities}, length_band={args.prompt_band}, "
          f"temperature={args.temperature})")
    examples = json.loads(FEWSHOT_EXAMPLE_PATH.read_text(encoding="utf-8"))

    assembler = LLMAssembler(device=args.device, model_type=args.model_type)

    drafts: dict = {}
    if args.resume and out_path.exists():
        drafts = json.loads(out_path.read_text(encoding="utf-8"))
        print(f"resuming from {out_path} with {len(drafts)} captions already done")

    for i, batch in enumerate(tqdm(merge_data, desc=f"[{args.tag}] generating")):
        qid = batch["query_id"]
        if qid in drafts:
            continue
        keyword_override = None
        if args.gate in PROMPT_GATES:
            keyword_override = ", ".join(selections.get(qid, []))
        prompt = template.render(**compose_input(
            batch, examples, keyword_override,
            use_entities=args.prompt_entities, use_length_band=args.prompt_band))
        set_seed(query_seed(args.seed, qid))
        drafts[qid] = assembler.assemble(prompt, temperature=args.temperature)

        # checkpoint periodically rather than re-serialising the whole dict
        # on every query, as the original loop did
        if (i + 1) % args.checkpoint_every == 0:
            atomic_write_json(out_path, drafts)

    atomic_write_json(out_path, drafts)
    return drafts


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def caption_stats(captions: dict, selections: dict, debug: dict) -> dict:
    """Mean length, mean entities per caption (the #Ent column) and, when a
    selection is available, the leakage rate of Table 4."""
    import spacy

    nlp = spacy.load("en_core_web_sm")
    lengths, entity_counts, leak_num, leak_den = [], [], 0, 0

    for qid, caption in tqdm(captions.items(), desc="stats"):
        lengths.append(len(caption.split()))
        doc = nlp(caption)
        entity_counts.append(len({ent.text.strip().lower() for ent in doc.ents}))

        rejected = debug.get(qid, {}).get("rejected_entities")
        if rejected:
            low = caption.lower()
            leak_num += sum(1 for e in rejected if e.strip() and e.lower() in low)
            leak_den += len(rejected)

    stats = {
        "n_captions": len(captions),
        "mean_length_words": round(mean(lengths), 2) if lengths else 0.0,
        "mean_entities_per_caption": round(mean(entity_counts), 3) if entity_counts else 0.0,
        "total_entity_mentions": sum(entity_counts),
    }
    if selections:
        stats["mean_entities_selected"] = round(
            mean(len(v) for v in selections.values()), 3
        )
    if leak_den:
        stats["leakage_rate"] = round(leak_num / leak_den, 4)
        stats["rejected_entities_total"] = leak_den
    return stats


def write_submission(captions: dict, csv_in: Path, csv_out: Path) -> None:
    """Submission CSV.

    Written here rather than reusing create_submission() because that helper
    reads a {query_id: caption} dict straight off disk, while the normalizer
    saves a list of records.
    """
    with csv_in.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    missing = 0
    for row in rows:
        caption = captions.get(row["query_id"], "")
        if not caption:
            missing += 1
        row["generated_caption"] = caption

    with csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"submission -> {csv_out} ({len(rows)} rows, {missing} without a caption)")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="run name; everything lands in runs/<tag>/")
    ap.add_argument("--gate", default="enrichment", choices=["none", "prompt", "enrichment", "unified"])
    ap.add_argument("--selector", default="heuristic",
                    choices=["heuristic", "article_order", "random", "presence_only",
                             "presence_score", "type_only", "learned"])
    ap.add_argument("--learned_coefficients", default=None,
                    help="JSON from scripts/train_learned_selector.py (--selector learned)")
    ap.add_argument("--entity_config", default=DEFAULT_ENTITY_CONFIG,
                    help="path to an entity scoring config JSON")
    ap.add_argument("--tau", type=float, default=None, help="override the admission threshold (Appendix A)")
    ap.add_argument("--top_k", type=int, default=None, help="override K_e (Appendix A)")
    ap.add_argument("--weight_scale", action="append", metavar="FAMILY:FACTOR",
                    help="scale one weight family, e.g. presence:0.5 (repeatable)")
    ap.add_argument("--weight_jitter", type=float, default=None,
                    help="multiply every weight by 1 +/- this fraction, e.g. 0.2")
    ap.add_argument("--jitter_seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=10, help="K_e when no per-query budget is given")
    ap.add_argument("--budget_match", default="fixed", choices=["fixed", "per_query"],
                    help="per_query copies VG-Cap's per-query entity count from --reference_selection")
    ap.add_argument("--reference_selection", default=None,
                    help="runs/<tag>/selected_entities.json of the reference run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N queries (subsampled ablations)")
    ap.add_argument("--no_noun_phrases", action="store_true",
                    help="stop enrichment from appending ungated article noun phrases")

    # -- scorer defect switches (see step_2_entity_scorer.DEFAULT_FIXES) ------
    ap.add_argument("--fix_profile", default="legacy", choices=sorted(FIX_PROFILES),
                    help="'legacy' = published behaviour (default); 'all' = every "
                         "repair on; the rest enable exactly one repair")
    ap.add_argument("--fix", action="append", metavar="KEY=VALUE",
                    help="override a single switch, e.g. --fix word_boundary_match=1 "
                         "(repeatable; applied after --fix_profile)")
    ap.add_argument("--span_penalty", default=None, choices=list(SPAN_PENALTY_MODES),
                    help="shorthand for --fix span_penalty=...  'paper' is Eq. (2)")
    ap.add_argument("--preserve_selected", action="store_true",
                    help="entity-aware pruning protects every mention of E* "
                         "(Section 3.6 step 2); off in the published run")
    ap.add_argument("--pos_mask", action="store_true",
                    help="pruning also protects PROPN/NUM tokens (the 'POS/NER "
                         "mask' of Section 3.6); off in the published run")

    ap.add_argument("--strategy", default="cot_5_things_fact_more_event")
    ap.add_argument("--no_prompt_entities", dest="prompt_entities", action="store_false",
                    help="drop the Keywords block from the prompt (Table 3(a))")
    ap.add_argument("--no_prompt_band", dest="prompt_band", action="store_false",
                    help="drop the 100-120 word instruction from the prompt (Table 3(a))")
    ap.add_argument("--no_normalizer", action="store_true",
                    help="skip SGN entirely; submit the cleaned drafts (ENRIC baseline row)")
    ap.add_argument("--model_type", default="llama3")
    ap.add_argument("--temperature", type=float, default=0.3,
                    help="decoding temperature; 0.3 produced both published draft sets")
    ap.add_argument("--device", default="cuda:5")
    ap.add_argument("--expected_length", type=int, default=104)
    ap.add_argument("--max_words", type=int, default=None,
                    help="SGN truncation trigger; defaults to expected_length + 1")
    ap.add_argument("--checkpoint_every", type=int, default=50)

    ap.add_argument("--reuse_drafts", default=None,
                    help="reuse this llm_raw.json instead of generating (valid when the gate does not change the prompt)")
    ap.add_argument("--resume", action="store_true", help="continue a partially finished generation")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow replacing the outputs of an existing run (including its drafts)")
    ap.add_argument("--skip_if_done", action="store_true",
                    help="exit quietly when runs/<tag>/stats.json already exists")
    ap.add_argument("--skip_stats", action="store_true")
    ap.add_argument("--out_dir", default="runs")
    args = ap.parse_args()
    if args.max_words is None:
        args.max_words = args.expected_length + 1

    if args.gate in PROMPT_GATES and not args.prompt_entities:
        raise SystemExit("--gate prompt/unified needs the Keywords block: drop --no_prompt_entities")
    if args.gate in PROMPT_GATES and args.reuse_drafts:
        raise SystemExit("--reuse_drafts is invalid with --gate prompt/unified: the gate changes the prompt")
    if args.budget_match == "per_query" and not args.reference_selection:
        raise SystemExit("--budget_match per_query requires --reference_selection")
    if args.preserve_selected and args.gate not in ENRICHMENT_GATES:
        # without an enrichment gate the normalizer falls back to ALL article
        # entities, so "preserve the selected set" would protect everything
        raise SystemExit("--preserve_selected needs --gate enrichment or unified")

    run_dir = Path(args.out_dir) / args.tag
    # Guard: a second call with the same tag used to overwrite everything,
    # including hours of generated drafts.
    finished = (run_dir / "stats.json").exists()
    if finished and args.skip_if_done:
        print(f"[{args.tag}] already finished ({run_dir / 'stats.json'}), skipping")
        return
    has_outputs = finished or (run_dir / "llm_raw.json").exists()
    if has_outputs and not (args.overwrite or args.resume):
        raise SystemExit(f"[{args.tag}] {run_dir} already has outputs; pass --resume to continue "
                         f"it, --skip_if_done to skip it, or --overwrite to replace it")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    entity_config = build_effective_config(args, run_dir) if args.gate != "none" else None
    timings: dict = {}

    merge_data = json.loads(MERGE_DATA_PATH.read_text(encoding="utf-8"))
    if args.limit:
        merge_data = merge_data[: args.limit]
    print(f"[{args.tag}] gate={args.gate} selector={args.selector} queries={len(merge_data)}")

    # -- entity selection, computed once and shared by both gates ------------
    selections: dict = {}
    debug_info: dict = {}
    t_select = time.perf_counter()
    if args.gate != "none":
        budget_map = load_budget_map(args.reference_selection) if args.budget_match == "per_query" else None
        selector = build_selector(
            args.selector,
            config_path=entity_config,
            budget=args.budget,
            budget_map=budget_map,
            seed=args.seed,
            coefficients_path=args.learned_coefficients,
        )
        for batch in tqdm(merge_data, desc=f"[{args.tag}] selecting entities"):
            chosen, debug = selector.select(batch)
            selections[batch["query_id"]] = chosen
            debug_info[batch["query_id"]] = debug

        (run_dir / "selected_entities.json").write_text(
            json.dumps(selections, indent=2, ensure_ascii=False), encoding="utf-8")
        (run_dir / "entity_debug.json").write_text(
            json.dumps(debug_info, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[{args.tag}] mean entities selected: "
              f"{mean(len(v) for v in selections.values()):.3f}")
    timings["entity_scoring_s"] = round(time.perf_counter() - t_select, 2)

    # -- drafts --------------------------------------------------------------
    drafts_path = run_dir / "llm_raw.json"
    t_gen = time.perf_counter()
    if args.reuse_drafts:
        source = json.loads(Path(args.reuse_drafts).read_text(encoding="utf-8"))
        drafts = {b["query_id"]: source[b["query_id"]] for b in merge_data if b["query_id"] in source}
        atomic_write_json(drafts_path, drafts)
        print(f"[{args.tag}] reused {len(drafts)} drafts from {args.reuse_drafts}")
    else:
        drafts = generate_drafts(args, merge_data, selections, drafts_path)
    timings["generation_s"] = round(time.perf_counter() - t_gen, 2)

    # -- normalization -------------------------------------------------------
    t_norm = time.perf_counter()
    if args.no_normalizer:
        # ENRIC baseline row: no SGN at all, only the newline/preamble cleanup
        # the submission CSV needs.
        from step_2_caption_normalizer import normalize_spaces
        cleaner = CaptionNormalizer.normalize_llm_output
        captions = {
            qid: normalize_spaces(cleaner(None, text).replace("\n", " "))
            for qid, text in drafts.items()
        }
        (run_dir / "normalized.json").write_text(
            json.dumps([{"query_id": q, "normalized_caption": c,
                         "caption_length": len(c.split())} for q, c in captions.items()],
                       indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[{args.tag}] SGN skipped (--no_normalizer)")
        timings["normalization_s"] = round(time.perf_counter() - t_norm, 2)
        write_submission(captions, RETRIEVAL_CSV, run_dir / "submission.csv")
        if not args.skip_stats:
            stats = caption_stats(captions, selections, debug_info)
            stats.update({"tag": args.tag, "gate": args.gate, "selector": args.selector,
                          "seed": args.seed, "timings": timings})
            (run_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
            print(json.dumps(stats, indent=2))
        return

    normalizer = CaptionNormalizer(
        max_words=args.max_words,
        expected_length=args.expected_length,
        use_entity_scoring=False,
        selection_map=selections if args.gate in ENRICHMENT_GATES else None,
        use_noun_phrases=not args.no_noun_phrases,
        preserve_selected=args.preserve_selected,
        pos_mask=args.pos_mask,
    )
    results = normalizer.normalize_batch(
        llm_output_path=str(drafts_path),
        merge_data_path=str(MERGE_DATA_PATH),
        output_path=str(run_dir / "normalized.json"),
        verbose=True,
        save_entity_scores=False,
    )
    captions = {r["query_id"]: r["normalized_caption"] for r in results}
    timings["normalization_s"] = round(time.perf_counter() - t_norm, 2)

    write_submission(captions, RETRIEVAL_CSV, run_dir / "submission.csv")

    if not args.skip_stats:
        stats = caption_stats(captions, selections, debug_info)
        stats.update({"tag": args.tag, "gate": args.gate, "selector": args.selector,
                      "seed": args.seed, "timings": timings})
        (run_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
