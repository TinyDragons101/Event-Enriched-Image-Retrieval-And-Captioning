"""Appendix D: pick qualitative examples with their score breakdown.

Four buckets the reviewers asked for:
  (i)   success        -- filtering removed entities and the caption stayed clean
  (ii)  wrongly kept   -- entity admitted by priors alone, no caption match
  (iii) wrongly removed-- a DATE/ORG mentioned early and often, yet rejected
  (iv)  malformed      -- caption ends in an entity fragment ("... the Tiger Woods.")

    python scripts/make_qualitative.py --run runs/vgcap_enrichment_s0 \
        --baseline runs/gate_none_s0 --per_bucket 3
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PRESENCE = ("exact_match_crawl", "exact_match_generated",
            "partial_match_crawl", "partial_match_generated")
# a caption that ends without a verb-bearing final clause: entity run-on
FRAGMENT = re.compile(r"(?:[A-Z][\w.'-]*\s+){0,4}[A-Z][\w.'-]*\s*\.\s*$")


def load_captions(run: Path) -> dict[str, str]:
    records = json.loads((run / "normalized.json").read_text(encoding="utf-8"))
    return {r["query_id"]: r["normalized_caption"] for r in records}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="the filtered run (VG-Cap)")
    ap.add_argument("--baseline", default=None, help="the unfiltered run, for side-by-side")
    ap.add_argument("--per_bucket", type=int, default=3)
    args = ap.parse_args()

    run = Path(args.run)
    captions = load_captions(run)
    baseline = load_captions(Path(args.baseline)) if args.baseline else {}
    debug = json.loads((run / "entity_debug.json").read_text(encoding="utf-8"))

    buckets: dict[str, list] = {"success": [], "wrongly_kept": [], "wrongly_removed": [], "malformed": []}

    for qid, info in debug.items():
        caption = captions.get(qid, "")
        if not caption:
            continue
        scored = info.get("scored_entities", [])
        selected = [e for e in scored if e.get("selected")]
        rejected = [e for e in scored if not e.get("selected")]

        # (iv) malformed enrichment
        if FRAGMENT.search(caption):
            buckets["malformed"].append((qid, caption, info))

        # (ii) admitted without any caption evidence -> priors only
        no_evidence = [e for e in selected
                       if not any(e.get("features", {}).get(f, 0.0) for f in PRESENCE)]
        if no_evidence:
            buckets["wrongly_kept"].append((qid, caption, info, [e["entity"] for e in no_evidence]))

        # (iii) rejected despite caption evidence -- the cap, not the threshold
        evidenced = [e for e in rejected
                     if any(e.get("features", {}).get(f, 0.0) for f in PRESENCE)]
        if evidenced:
            buckets["wrongly_removed"].append((qid, caption, info, [e["entity"] for e in evidenced]))

        # (i) clean success: removed something, caption ends in a normal sentence
        if rejected and not FRAGMENT.search(caption) and not no_evidence:
            buckets["success"].append((qid, caption, info))

    out_lines = ["# Appendix D -- qualitative examples", ""]
    for bucket, items in buckets.items():
        out_lines.append(f"## {bucket}  ({len(items)} queries matched)")
        out_lines.append("")
        for item in items[: args.per_bucket]:
            qid, caption, info = item[0], item[1], item[2]
            out_lines.append(f"### {qid}")
            if qid in baseline:
                out_lines.append(f"- **unfiltered:** {baseline[qid]}")
            out_lines.append(f"- **VG-Cap:** {caption}")
            if len(item) > 3:
                out_lines.append(f"- **flagged entities:** {', '.join(item[3])}")
            out_lines.append("- **score breakdown:**")
            out_lines.append("")
            out_lines.append("  | entity | type | score | selected | active features |")
            out_lines.append("  |---|---|---|---|---|")
            for e in sorted(info["scored_entities"], key=lambda e: -e["score"])[:12]:
                active = ", ".join(k for k, v in e.get("features", {}).items() if v)
                out_lines.append(f"  | {e['entity']} | {e['label']} | {e['score']} | "
                                 f"{'yes' if e['selected'] else 'no'} | {active} |")
            out_lines.append("")
        out_lines.append("")

    path = run / "qualitative.md"
    path.write_text("\n".join(out_lines), encoding="utf-8")
    for bucket, items in buckets.items():
        print(f"{bucket:18s} {len(items):5d} queries")
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
