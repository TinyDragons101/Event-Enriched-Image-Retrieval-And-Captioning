"""Appendix B: fit a logistic-regression selector on the Entity Heuristic
Scorer's own features.

Label: does the entity occur in the reference caption of a HELD-OUT query?
The held-out split must come from the training queries, never from the public
test set the paper reports on.

    python scripts/train_learned_selector.py \
        --merge public_test_final_elements_json/final_merge_result.json \
        --gt data/train_gt.json --n_queries 500 --out configs/learned_selector.json

Then:
    python run_ablation.py --tag learned --gate enrichment --selector learned \
        --learned_coefficients configs/learned_selector.json ...
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

FEATURES = [
    "exact_match_crawl", "exact_match_generated", "partial_match_crawl",
    "partial_match_generated", "substring_in_qa", "substring_in_summary",
    "early_position_bonus", "mid_position_bonus",
    "type_person", "type_gpe", "type_org", "type_loc",
    "type_date", "type_event", "type_other",
    "length_penalty",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merge", required=True, help="final_merge_result.json of the TRAINING queries")
    ap.add_argument("--gt", required=True, help="reference captions of those queries")
    ap.add_argument("--entity_config", default="configs/entity_scoring_config.json")
    ap.add_argument("--n_queries", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="configs/learned_selector.json")
    args = ap.parse_args()

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from step_2_entity_selectors import build_signals, make_scorer

    scorer = make_scorer(args.entity_config)
    merge = json.loads(Path(args.merge).read_text(encoding="utf-8"))
    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))

    rng = random.Random(args.seed)
    pool = [b for b in merge if b["query_id"] in gt]
    if not pool:
        raise SystemExit("no overlap between --merge and --gt")
    rng.shuffle(pool)
    pool = pool[: args.n_queries]
    print(f"training on {len(pool)} held-out queries")

    X, y = [], []
    for batch in pool:
        reference = gt[batch["query_id"]]
        if isinstance(reference, list):
            reference = " ".join(reference)
        elif isinstance(reference, dict):
            reference = reference.get("caption") or " ".join(reference.get("captions", []))
        reference = reference.lower()

        article = batch.get("article", "")
        entities = scorer.deduplicate_entities(
            scorer.extract_entities_from_article(article),
            scorer.parse_entity_string(batch.get("name_entity_keyword", "")),
        )
        entities = scorer.score_entities(entities, build_signals(batch), article)
        for entity in entities:
            X.append([entity.features.get(f, 0.0) for f in FEATURES])
            y.append(1 if entity.text.lower() in reference else 0)

    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
    print(f"{len(y)} entities, positive rate {y.mean():.3f}")

    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    auc = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
    model.fit(X, y)
    print(f"5-fold ROC-AUC: {auc.mean():.4f} +/- {auc.std():.4f}")

    coefficients = {f: round(float(w), 4) for f, w in zip(FEATURES, model.coef_[0])}
    payload = {
        "intercept": round(float(model.intercept_[0]), 4),
        "coefficients": coefficients,
        "threshold": 0.5,
        "cv_roc_auc_mean": round(float(auc.mean()), 4),
        "cv_roc_auc_std": round(float(auc.std()), 4),
        "n_queries": len(pool),
        "n_entities": int(len(y)),
        "positive_rate": round(float(y.mean()), 4),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"-> {args.out}")

    print("\nordering learned vs hand-set hierarchy (for the Appendix B discussion):")
    for f, w in sorted(coefficients.items(), key=lambda kv: -kv[1]):
        print(f"  {f:28s} {w:+.4f}   (hand-set {scorer.weights.get(f, 0.0):+.2f})")


if __name__ == "__main__":
    main()
