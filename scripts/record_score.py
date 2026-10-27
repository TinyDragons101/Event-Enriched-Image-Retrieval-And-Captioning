"""Write a leaderboard result back into runs/<tag>/scores.json.

With no public-test ground truth, CIDEr and Overall come from the web
leaderboard.  This records them so scripts/collect_results.py can build the
table, and recomputes Overall from the components as a consistency check.

    python scripts/record_score.py --run runs/repro_vgcap --cider 0.2651 --clip 0.858
    python scripts/record_score.py --run runs/repro_vgcap --overall 0.6219 --cider 0.2651 --clip 0.858
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

WEIGHTS = {"ap": 1, "r1": 2, "r10": 2, "clip": 3, "cider": 2}


def overall_score(ap, r1, r10, clip, cider, eps=1e-8):
    m = {"ap": ap, "r1": r1, "r10": r10, "clip": clip, "cider": cider}
    return sum(WEIGHTS.values()) / sum(w / (m[k] + eps) for k, w in WEIGHTS.items())


def main() -> None:
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--run", required=True)
    ap_.add_argument("--cider", type=float, required=True)
    ap_.add_argument("--clip", type=float, default=None,
                     help="omit to reuse the locally computed CLIPScore")
    ap_.add_argument("--ap", type=float, default=0.992)
    ap_.add_argument("--r1", type=float, default=0.985)
    ap_.add_argument("--r10", type=float, default=1.000)
    ap_.add_argument("--overall", type=float, default=None,
                     help="Overall as shown by the leaderboard; checked against the recomputed value")
    args = ap_.parse_args()

    run_dir = Path(args.run)
    path = run_dir / "scores.json"
    scores = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    clip = args.clip if args.clip is not None else scores.get("clip")
    if clip is None:
        raise SystemExit("no CLIPScore: pass --clip, or run scripts/evaluate.py first "
                         "(it works without ground truth)")

    recomputed = overall_score(args.ap, args.r1, args.r10, clip, args.cider)
    scores.update({"cider": args.cider, "clip": round(clip, 4), "ap": args.ap,
                   "r1": args.r1, "r10": args.r10,
                   "overall": round(args.overall if args.overall is not None else recomputed, 4),
                   "overall_recomputed": round(recomputed, 4),
                   "cider_source": "leaderboard"})
    path.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    print(json.dumps(scores, indent=2))

    if args.overall is not None and abs(args.overall - recomputed) > 0.001:
        print(f"\nWARNING: leaderboard Overall {args.overall} vs recomputed {recomputed:.4f} "
              f"(gap {abs(args.overall - recomputed):.4f}).")
        print("A gap this size usually means CIDEr was rounded; ask for more decimals before "
              "putting the number in a table -- this is exactly the ENRIC 0.257/0.6097 problem.")


if __name__ == "__main__":
    main()
