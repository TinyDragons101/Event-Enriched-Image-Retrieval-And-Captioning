"""Collect runs/*/stats.json into one table, plus the LaTeX rows for Table 3.

Overall / CIDEr / CLIP are left blank: they come from the evaluator, not from
this repo.  Drop a scores.json of the form

    {"overall": 0.6219, "cider": 0.2651, "clip": 0.858}

into a run directory and it is merged in automatically.

    python scripts/collect_results.py                # all runs
    python scripts/collect_results.py --latex        # also print LaTeX rows
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

WEIGHTS = {"ap": 1, "r1": 2, "r10": 2, "clip": 3, "cider": 2}


def overall_score(ap: float, r1: float, r10: float, clip: float, cider: float, eps: float = 1e-8) -> float:
    """Weighted harmonic mean used by the EVENTA protocol (Eq. 3 of the paper)."""
    m = {"ap": ap, "r1": r1, "r10": r10, "clip": clip, "cider": cider}
    return sum(WEIGHTS.values()) / sum(w / (m[k] + eps) for k, w in WEIGHTS.items())


def load_runs(out_dir: Path) -> list[dict]:
    rows = []
    for run_dir in sorted(out_dir.iterdir()):
        stats_path = run_dir / "stats.json"
        if not stats_path.exists():
            continue
        row = json.loads(stats_path.read_text(encoding="utf-8"))
        row["tag"] = row.get("tag", run_dir.name)
        scores_path = run_dir / "scores.json"
        if scores_path.exists():
            row.update(json.loads(scores_path.read_text(encoding="utf-8")))
        # #Ent recounted by scripts/count_entities.py overrides stats.json, so
        # every row of the table uses the same frozen definition
        count_path = run_dir / "entity_count.json"
        if count_path.exists():
            row["mean_entities_per_caption"] = json.loads(
                count_path.read_text(encoding="utf-8"))["mean_entities_per_caption"]
        rows.append(row)
    return rows


def fold_random_draws(rows: list[dict]) -> list[dict]:
    """Average the five random draws into a single row with a std."""
    draws = [r for r in rows if "_random_d" in r["tag"]]
    if len(draws) < 2:
        return rows
    kept = [r for r in rows if "_random_d" not in r["tag"]]
    folded = {"tag": "sel_random (mean of %d draws)" % len(draws), "selector": "random",
              "gate": draws[0].get("gate"), "n_draws": len(draws)}
    for key in ("overall", "cider", "clip", "mean_length_words",
                "mean_entities_per_caption", "leakage_rate"):
        values = [r[key] for r in draws if r.get(key) is not None]
        if values:
            folded[key] = round(mean(values), 4)
            folded[key + "_std"] = round(pstdev(values), 4)
    return kept + [folded]


def fmt(value, digits=4):
    if value is None:
        return "--"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()

    rows = fold_random_draws(load_runs(Path(args.out_dir)))
    if not rows:
        raise SystemExit("no runs with stats.json found")

    header = f"{'run':<34}{'gate':<12}{'Overall':>9}{'CIDEr':>9}{'CLIP':>8}{'Len':>7}{'#Ent':>7}{'Leak':>8}"
    print(header)
    print("-" * len(header))
    for r in sorted(rows, key=lambda r: r["tag"]):
        print(f"{r['tag']:<34}{str(r.get('gate','')):<12}"
              f"{fmt(r.get('overall')):>9}{fmt(r.get('cider')):>9}{fmt(r.get('clip'),3):>8}"
              f"{fmt(r.get('mean_length_words'),1):>7}{fmt(r.get('mean_entities_per_caption'),2):>7}"
              f"{fmt(r.get('leakage_rate'),3):>8}")

    missing = [r["tag"] for r in rows if r.get("overall") is None]
    if missing:
        print(f"\nwaiting on scores.json for: {', '.join(missing)}")

    if args.latex:
        print("\n% --- Table 3 rows -------------------------------------------")
        for r in sorted(rows, key=lambda r: r["tag"]):
            print(f"{r['tag'].replace('_', ' ')} & {fmt(r.get('overall'))} & {fmt(r.get('cider'),3)} & "
                  f"{fmt(r.get('clip'),3)} & {fmt(r.get('mean_length_words'),0)} & "
                  f"{fmt(r.get('mean_entities_per_caption'),2)} \\\\")


if __name__ == "__main__":
    main()
