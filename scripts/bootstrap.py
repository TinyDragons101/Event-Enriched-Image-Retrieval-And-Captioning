"""Paired bootstrap over queries for two runs (Koehn 2004).

Answers the TODOs that ask which differences exceed the confidence interval.
Needs per_query_scores.json from scripts/evaluate.py in both runs.

    python scripts/bootstrap.py --a runs/gate_none_s0 --b runs/vgcap_enrichment_s0 \
        --metric cider --resamples 1000
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="baseline run directory")
    ap.add_argument("--b", required=True, help="run compared against the baseline")
    ap.add_argument("--metric", default="cider", choices=["cider", "clip"])
    ap.add_argument("--resamples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    def load(run: str) -> dict[str, float]:
        data = json.loads((Path(run) / "per_query_scores.json").read_text(encoding="utf-8"))
        return {qid: value[args.metric] for qid, value in data.items()}

    sa, sb = load(args.a), load(args.b)
    common = sorted(set(sa) & set(sb))
    if not common:
        raise SystemExit("the two runs share no query_id")

    diffs = [sb[q] - sa[q] for q in common]
    observed = sum(diffs) / len(diffs)

    rng = random.Random(args.seed)
    n = len(diffs)
    means = []
    for _ in range(args.resamples):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()

    lo = means[int(args.alpha / 2 * args.resamples)]
    hi = means[min(args.resamples - 1, int((1 - args.alpha / 2) * args.resamples))]
    # two-sided p: how often a resampled mean lands on the opposite side of zero
    if observed >= 0:
        wrong_side = sum(1 for m in means if m <= 0)
    else:
        wrong_side = sum(1 for m in means if m >= 0)
    p_value = min(1.0, 2 * wrong_side / args.resamples)

    result = {
        "metric": args.metric, "baseline": args.a, "system": args.b,
        "n_queries": n, "resamples": args.resamples,
        "mean_a": round(sum(sa[q] for q in common) / n, 4),
        "mean_b": round(sum(sb[q] for q in common) / n, 4),
        "delta": round(observed, 4),
        f"ci_{int((1-args.alpha)*100)}": [round(lo, 4), round(hi, 4)],
        "p_value_two_sided": round(p_value, 4),
        "significant": not (lo <= 0 <= hi),
    }
    print(json.dumps(result, indent=2))
    out = Path(args.b) / f"bootstrap_vs_{Path(args.a).name}_{args.metric}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
