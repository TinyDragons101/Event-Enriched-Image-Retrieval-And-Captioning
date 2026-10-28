"""Which gate did a published result actually use?  (docs/CHECK_PUBLISHED_GATE.md)

Gate 1 = E* replaces the entity list in the PROMPT  -> changes the draft body.
Gate 2 = E* is the only source for ENRICHMENT       -> changes the appended tail.

Given the captions that were actually submitted (CSV or JSON), this script
measures three fingerprints. Tests 1-2 need only the standard library.

  1. Body test (Gate 1).  The normalizer only edits the END of a draft (pruning
     from the end, appending, final cut), so a caption built from a known draft
     file starts with exactly that draft's words. If ~all published captions
     start like the drafts in llama.json, those drafts were used, and their
     prompt carried the raw keyword list: Gate 1 was NOT applied.
  2. Tail test (Gate 2).  For captions longer than their draft, the extra words
     are what enrichment appended. If they contain selected entities (E*) but
     almost no rejected ones, Gate 2 was applied. Needs --debug (entity_debug.json
     of a run with the same scorer, e.g. runs/vgcap_enrichment_s0).
  3. Exact match.  Share of captions identical to each reconstructed run
     (--runs); ~100% with one run identifies the configuration.

    python scripts/check_published_gate.py --published path/to/submission_0.6219.csv \
        --drafts assemble_result/cot_5_things_fact_more_event_llama.json \
        --debug runs/vgcap_enrichment_s0/entity_debug.json \
        --runs runs/gate_none_s0 runs/vgcap_enrichment_s0
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

PREAMBLES = ["Here is the caption:", "Here is the rewritten caption:", "Caption:",
             "The caption is:", "Here's the caption:"]
BODY_WORDS = 20          # leading words compared in the body test


def clean_draft(text: str) -> str:
    """Same cleanup as CaptionNormalizer.normalize_llm_output + newline join."""
    text = text.strip()
    for p in PREAMBLES:
        if text.startswith(p):
            text = text[len(p):].strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    return " ".join(text.replace("\n", " ").split())


def load_captions(path: Path) -> dict[str, str]:
    """CSV with query_id + generated_caption, a run directory, normalized.json
    (list of records) or a {query_id: caption} JSON."""
    if path.is_dir():
        path = path / "normalized.json"
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8") as f:
            return {r["query_id"]: " ".join(r["generated_caption"].split()) for r in csv.DictReader(f)}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {r["query_id"]: " ".join(r["normalized_caption"].split()) for r in data}
    return {k: " ".join(str(v).split()) for k, v in data.items()}


def words(text: str) -> list[str]:
    return text.split()


def contains(haystack: str, needle: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(needle.lower())}(?!\w)", haystack.lower()) is not None


def usable(entity: str) -> bool:
    """Skip the ~300-char keyword blobs and letterless spans of the legacy scorer."""
    return 2 <= len(entity) <= 60 and any(c.isalpha() for c in entity)


def body_test(published: dict, drafts: dict) -> dict:
    common = [q for q in published if q in drafts]
    same = 0
    for q in common:
        p, d = words(published[q]), words(clean_draft(drafts[q]))
        n = min(BODY_WORDS, len(p), len(d))
        same += n > 0 and p[:n] == d[:n]
    return {"queries": len(common), "body_matches_drafts": same,
            "share": round(same / len(common), 4) if common else None}


def tail_test(published: dict, drafts: dict, debug: dict) -> dict:
    """Entities found in the appended tail, split by E* (selected) vs rejected."""
    n_tails = sel_hits = rej_hits = 0
    for q, cap in published.items():
        if q not in drafts or q not in debug:
            continue
        p, d = words(cap), words(clean_draft(drafts[q]))
        if len(p) <= len(d) or p[:min(BODY_WORDS, len(d))] != d[:min(BODY_WORDS, len(d))]:
            continue                      # no enrichment, or not built from these drafts
        tail = " ".join(p[len(d):])
        n_tails += 1
        info = debug[q]
        sel = [e for e in info.get("top_entities", []) if usable(e)]
        rej = [e for e in info.get("rejected_entities", []) if usable(e)
               and not any(contains(s, e) for s in sel)]   # 'Payne' inside 'Michelle Payne' is not a leak
        sel_hits += sum(contains(tail, e) for e in sel)
        rej_hits += sum(contains(tail, e) for e in rej)
    total = sel_hits + rej_hits
    return {"enriched_captions": n_tails, "selected_entities_in_tails": sel_hits,
            "rejected_entities_in_tails": rej_hits,
            "rejected_share": round(rej_hits / total, 4) if total else None}


def exact_match(published: dict, other: dict) -> float:
    common = [q for q in published if q in other]
    return round(sum(published[q] == other[q] for q in common) / len(common), 4) if common else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--published", required=True, help="the captions that were submitted (CSV / JSON / run dir)")
    ap.add_argument("--drafts", default="assemble_result/cot_5_things_fact_more_event_llama.json")
    ap.add_argument("--debug", default=None, help="entity_debug.json of a run with the same scorer")
    ap.add_argument("--runs", nargs="*", default=[], help="reconstructed runs to compare against")
    ap.add_argument("--calibrate", action="store_true",
                    help="also run the tail test on each --runs entry, to see both signatures")
    args = ap.parse_args()

    published = load_captions(Path(args.published))
    drafts = json.loads(Path(args.drafts).read_text(encoding="utf-8"))
    report = {"published": args.published, "n_captions": len(published),
              "test1_body": body_test(published, drafts)}
    debug = json.loads(Path(args.debug).read_text(encoding="utf-8")) if args.debug else None
    if debug:
        report["test2_tail"] = tail_test(published, drafts, debug)
    if args.runs:
        report["test3_exact_match"] = {r: exact_match(published, load_captions(Path(r))) for r in args.runs}
        if debug and args.calibrate:
            report["test2_tail_calibration"] = {r: tail_test(load_captions(Path(r)), drafts, debug)
                                                for r in args.runs}

    print(json.dumps(report, indent=2))

    # ---- verdict --------------------------------------------------------------
    print("\nVERDICT")
    share = report["test1_body"]["share"] or 0.0
    if share >= 0.95:
        print(f"  Gate 1: NOT applied. {share:.1%} of captions start word-for-word like {Path(args.drafts).name},")
        print("          whose prompt carried the raw keyword list, not E*.")
    elif share <= 0.05:
        print(f"  Gate 1: UNKNOWN. Only {share:.1%} match these drafts, so the result was built from")
        print("          other drafts. Find those drafts / that code before concluding anything.")
    else:
        print(f"  Gate 1: MIXED ({share:.1%} match). Several draft sets may have been merged; investigate.")
    tail = report.get("test2_tail")
    if tail and tail["rejected_share"] is not None:
        r = tail["rejected_share"]
        if r <= 0.10:
            print(f"  Gate 2: APPLIED. Only {r:.1%} of the entities found in appended tails were rejected by the scorer.")
        elif r >= 0.30:
            print(f"  Gate 2: NOT applied. {r:.1%} of the entities in appended tails were rejected by the scorer.")
        else:
            print(f"  Gate 2: UNCLEAR ({r:.1%} rejected). Compare with --runs ... --calibrate.")
    elif debug is None:
        print("  Gate 2: not tested (pass --debug runs/vgcap_enrichment_s0/entity_debug.json).")
    for run, m in report.get("test3_exact_match", {}).items():
        if m >= 0.99:
            print(f"  Exact: {m:.1%} identical to {run} -> the published result IS that configuration.")


if __name__ == "__main__":
    main()
