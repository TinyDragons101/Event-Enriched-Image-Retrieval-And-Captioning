"""Table 4: entity-level grounding.

Computes the columns that need no human: entity precision / recall / F1 against
the reference captions, and the leakage rate.  Also emits the annotation sheet
for the two columns that do need humans (unsupported-entity rate, 1-5 ratings).

    python scripts/eval_entities.py --run runs/vgcap_enrichment_s0 \
        --gt data/public_test_gt.json --sample 200 --seed 0
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def entity_set(text: str, nlp) -> set[str]:
    """Deduplicated entity surface forms, lowercased -- as in Section 4.5."""
    return {ent.text.strip().lower() for ent in nlp(text).ents if ent.text.strip()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--sample", type=int, default=200, help="queries in the human-annotation sheet")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import spacy
    nlp = spacy.load("en_core_web_sm")

    run_dir = Path(args.run)
    records = json.loads((run_dir / "normalized.json").read_text(encoding="utf-8"))
    captions = {r["query_id"]: r["normalized_caption"] for r in records}
    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    debug = {}
    debug_path = run_dir / "entity_debug.json"
    if debug_path.exists():
        debug = json.loads(debug_path.read_text(encoding="utf-8"))

    tp = fp = fn = 0
    leak_hit = leak_total = 0
    per_query = {}

    for qid, caption in captions.items():
        if qid not in gt:
            continue
        reference = gt[qid]
        if isinstance(reference, list):
            reference = " ".join(reference)
        elif isinstance(reference, dict):
            reference = reference.get("caption") or " ".join(reference.get("captions", []))

        pred, ref = entity_set(caption, nlp), entity_set(reference, nlp)
        hit = len(pred & ref)
        tp += hit
        fp += len(pred - ref)
        fn += len(ref - pred)
        per_query[qid] = {"n_pred": len(pred), "n_ref": len(ref), "n_hit": hit}

        rejected = debug.get(qid, {}).get("rejected_entities", [])
        if rejected:
            low = caption.lower()
            leak_hit += sum(1 for e in rejected if e.strip() and e.lower() in low)
            leak_total += len(rejected)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    out = {"entity_precision": round(precision, 4),
           "entity_recall": round(recall, 4),
           "entity_f1": round(f1, 4),
           "n_queries_scored": len(per_query)}
    if leak_total:
        out["leakage_rate"] = round(leak_hit / leak_total, 4)
        out["rejected_entities_total"] = leak_total

    (run_dir / "entity_eval.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (run_dir / "entity_eval_per_query.json").write_text(
        json.dumps(per_query, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))

    # -- annotation sheet ---------------------------------------------------
    rng = random.Random(args.seed)
    sampled = rng.sample(sorted(per_query), min(args.sample, len(per_query)))
    sheet = run_dir / "annotation_sheet.csv"
    with sheet.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "entity", "entity_label", "caption",
                         "support (visual / contextual / unsupported)",
                         "factuality 1-5", "fluency 1-5", "informativeness 1-5", "notes"])
        for qid in sampled:
            caption = captions[qid]
            ents = list(nlp(caption).ents)
            if not ents:
                writer.writerow([qid, "", "", caption, "", "", "", "", "no entity found"])
            for i, ent in enumerate(ents):
                # caption + the 1-5 columns only on the first row of each query
                writer.writerow([qid, ent.text, ent.label_, caption if i == 0 else "",
                                 "", "", "", "", ""])
    print(f"\nannotation sheet -> {sheet} ({len(sampled)} queries)")
    print("Columns 'support' are per entity (2 annotators -> Cohen's kappa).")
    print("Columns 1-5 are per caption, first row of each query (3 raters -> FLEISS' kappa,")
    print("not Cohen's -- the manuscript currently says Cohen's for three raters).")


if __name__ == "__main__":
    main()
