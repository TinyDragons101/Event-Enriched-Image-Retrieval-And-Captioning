"""#Ent column of Table 3 and the entity statistics of Section 4.5, with ONE
frozen counting definition.

The script that produced 7.59 / 6.76 is lost, so every #Ent cell and every
number in Section 4.5 has to be recounted the same way, and the definition
stated in the paper.

    # #Ent for every run -> runs/<tag>/entity_count.json (collect_results.py picks it up)
    python scripts/count_entities.py --runs runs/*/

    # Section 4.5: unfiltered (A) vs VG-Cap (B), incl. the new_main.tex todo
    # (per-type removal rate, admissions, PERSON removals caused by the cap)
    python scripts/count_entities.py --pair runs/gate_none_s0 runs/vgcap_enrichment_s0 \
        --out analysis/entity_stats_s0.json

Definitions (--definition):
  all     unique lowercase spaCy entity surface forms per caption, all labels
          (the same as mean_entities_per_caption in stats.json)
  named   as `all`, minus numeric/temporal labels (DATE TIME CARDINAL ORDINAL
          PERCENT MONEY QUANTITY)
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev

NUMERIC_LABELS = {"DATE", "TIME", "CARDINAL", "ORDINAL", "PERCENT", "MONEY", "QUANTITY"}


def load_captions(run: Path) -> dict[str, str]:
    records = json.loads((run / "normalized.json").read_text(encoding="utf-8"))
    return {r["query_id"]: r["normalized_caption"] for r in records}


def entity_maps(captions: dict[str, str], nlp, definition: str) -> dict[str, dict[str, str]]:
    """query_id -> {lowercase surface form: spaCy label} under the definition."""
    qids = sorted(captions)
    out = {}
    for qid, doc in zip(qids, nlp.pipe((captions[q] for q in qids), batch_size=64)):
        ents = {}
        for ent in doc.ents:
            text = ent.text.strip().lower()
            if not text or (definition == "named" and ent.label_ in NUMERIC_LABELS):
                continue
            ents.setdefault(text, ent.label_)
        out[qid] = ents
    return out


def count_run(run: Path, nlp, definition: str) -> dict:
    maps = entity_maps(load_captions(run), nlp, definition)
    counts = [len(m) for m in maps.values()]
    result = {"definition": definition, "n_captions": len(counts),
              "mean_entities_per_caption": round(mean(counts), 3) if counts else 0.0,
              "total_entity_mentions": sum(counts)}
    (run / "entity_count.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _matches(caption_entity: str, candidate: str) -> bool:
    """Same entity when one form contains the other on word boundaries."""
    a, b = caption_entity, candidate.lower()
    if a == b:
        return True
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    return re.search(rf"(?<!\w){re.escape(short)}(?!\w)", long_) is not None


def removal_cause(entity: str, info: dict) -> str:
    """Why an entity of the unfiltered caption is missing from the filtered one.

    cap                 scored >= tau but outside the top K_e
    threshold           scored < tau
    selected_not_used   in E* yet absent (draft wording / enrichment order / truncation)
    not_candidate       never in the candidate set E
    """
    tau = info.get("threshold_used")
    tau = 3.0 if tau is None else float(tau)
    hits = [e for e in info.get("scored_entities", []) if _matches(entity, e["entity"])]
    if not hits:
        return "not_candidate"
    if any(e.get("selected") for e in hits):
        return "selected_not_used"
    if any(float(e.get("score", 0.0)) >= tau for e in hits):
        return "cap"
    return "threshold"


def compare(run_a: Path, run_b: Path, nlp, definition: str) -> dict:
    caps_a, caps_b = load_captions(run_a), load_captions(run_b)
    common = sorted(set(caps_a) & set(caps_b))
    ma = entity_maps({q: caps_a[q] for q in common}, nlp, definition)
    mb = entity_maps({q: caps_b[q] for q in common}, nlp, definition)
    debug_path = run_b / "entity_debug.json"
    debug = json.loads(debug_path.read_text(encoding="utf-8")) if debug_path.exists() else {}

    diffs, present, removed, admitted, causes = [], Counter(), Counter(), Counter(), Counter()
    for q in common:
        a, b = ma[q], mb[q]
        diffs.append(len(b) - len(a))
        for text, label in a.items():
            present[label] += 1
            if text not in b:
                removed[label] += 1
                if label == "PERSON" and q in debug:
                    causes[removal_cause(text, debug[q])] += 1
        for text, label in b.items():
            if text not in a:
                admitted[label] += 1

    n = len(diffs)
    person_removed = sum(causes.values())
    return {
        "definition": definition, "a": str(run_a), "b": str(run_b), "n_queries": n,
        "mean_entities_a": round(mean(len(ma[q]) for q in common), 3),
        "mean_entities_b": round(mean(len(mb[q]) for q in common), 3),
        "mentions_a": sum(len(m) for m in ma.values()),
        "mentions_b": sum(len(m) for m in mb.values()),
        "per_caption_diff_mean": round(mean(diffs), 3),
        "per_caption_diff_sd": round(pstdev(diffs), 3),
        "share_reduced": round(sum(d < 0 for d in diffs) / n, 4),
        "share_unchanged": round(sum(d == 0 for d in diffs) / n, 4),
        "share_increased": round(sum(d > 0 for d in diffs) / n, 4),
        "gross_removed": sum(removed.values()),
        "gross_admitted": sum(admitted.values()),
        "removed_by_type": dict(removed.most_common()),
        "admitted_by_type": dict(admitted.most_common()),
        "removal_rate_by_type": {k: round(removed[k] / present[k], 4)
                                 for k in sorted(present, key=lambda k: -removed[k]) if present[k]},
        "person_removal_causes": dict(causes.most_common()),
        "person_removals_share_cap": round(causes["cap"] / person_removed, 4) if person_removed else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=[], help="run directories to count")
    ap.add_argument("--pair", nargs=2, metavar=("UNFILTERED", "FILTERED"), help="Section 4.5 comparison")
    ap.add_argument("--definition", choices=["all", "named"], default="all")
    ap.add_argument("--out", default=None, help="JSON output for --pair")
    args = ap.parse_args()
    if not args.runs and not args.pair:
        raise SystemExit("pass --runs and/or --pair")

    import spacy
    nlp = spacy.load("en_core_web_sm")

    for run in args.runs:
        run = Path(run)
        if not (run / "normalized.json").exists():
            continue
        r = count_run(run, nlp, args.definition)
        print(f"{run.name:40s} #Ent={r['mean_entities_per_caption']:.3f}  mentions={r['total_entity_mentions']}")

    if args.pair:
        stats = compare(Path(args.pair[0]), Path(args.pair[1]), nlp, args.definition)
        print(json.dumps(stats, indent=2))
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(stats, indent=2), encoding="utf-8")
            print(f"-> {args.out}")


if __name__ == "__main__":
    main()
