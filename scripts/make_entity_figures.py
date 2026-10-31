"""Fig. 2 of the paper (Section 4.5) from two runs, with the #Ent definition
of count_entities.py.

    python scripts/make_entity_figures.py --pair runs/gate_none_s0 runs/vgcap_s0 \
        --labels "No gate" "VG-Cap" --out figures/

Writes entity_gap_distribution.pdf (per-caption #Ent difference, B - A) and
entity_gap_by_type.pdf (gross removals by spaCy label).
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
from count_entities import entity_maps, load_captions  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", nargs=2, required=True, metavar=("UNFILTERED", "FILTERED"))
    ap.add_argument("--labels", nargs=2, default=["Unfiltered", "VG-Cap"])
    ap.add_argument("--definition", choices=["all", "named"], default="all")
    ap.add_argument("--top_types", type=int, default=8)
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import spacy
    nlp = spacy.load("en_core_web_sm")

    caps_a, caps_b = load_captions(Path(args.pair[0])), load_captions(Path(args.pair[1]))
    common = sorted(set(caps_a) & set(caps_b))
    ma = entity_maps({q: caps_a[q] for q in common}, nlp, args.definition)
    mb = entity_maps({q: caps_b[q] for q in common}, nlp, args.definition)

    diffs, removed = [], Counter()
    for q in common:
        diffs.append(len(mb[q]) - len(ma[q]))
        removed.update(label for text, label in ma[q].items() if text not in mb[q])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})

    lo, hi = min(diffs), max(diffs)
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    ax.hist(diffs, bins=range(lo, hi + 2), align="left", color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(mean(diffs), color="#C44E52", linestyle="--", linewidth=1,
               label=f"mean {mean(diffs):+.2f}")
    ax.set_xlabel(f"#Ent({args.labels[1]}) $-$ #Ent({args.labels[0]}) per caption")
    ax.set_ylabel("Queries")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "entity_gap_distribution.pdf")

    types = removed.most_common(args.top_types)
    other = sum(removed.values()) - sum(n for _, n in types)
    if other:
        types.append(("other", other))
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    ax.barh([t for t, _ in types][::-1], [n for _, n in types][::-1], color="#4C72B0")
    ax.set_xlabel(f"Gross removals ({sum(removed.values()):,})")
    fig.tight_layout()
    fig.savefig(out / "entity_gap_by_type.pdf")

    print(f"n={len(diffs)} mean diff {mean(diffs):+.3f} | removed {sum(removed.values())} "
          f"{dict(removed.most_common())} -> {out}")


if __name__ == "__main__":
    main()
