"""Gather every run's submission CSV, deduplicate them, and print a submit order.

The public-test ground truth is not available, so CIDEr and Overall can only
come from the leaderboard -- and the leaderboard has a submission quota while
the ablation study has ~40 configurations.  Two things save quota:

  * Deduplication.  Several configurations provably produce identical captions
    (the clearest case: 'w/o span penalty' equals VG-Cap, because the penalty
    never fires -- the scorer reads the feature key 'length_penalty' while the
    weights declare 'length_penalty_factor').  Identical CSVs need one
    submission, not five.
  * Priority.  Rows that decide a claim in the paper go first; sweep points
    that only fill an appendix go last.

    python scripts/make_submissions.py --out submissions
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

#: lower number submits earlier; matched on a substring of the run tag
PRIORITY = [
    (0, "repro_", "neo: tái lập số đã công bố"),
    (1, "abl_no_editorial", "kiểm tra leakage editorial caption -- tính hợp lệ của phương pháp"),
    (2, "sel_random", "sàn dưới của panel (b)"),
    (2, "sel_article_order", "baseline panel (b)"),
    (3, "sel_presence_only", "panel (b)"),
    (3, "sel_type_only", "panel (b)"),
    (4, "gate_", "panel (d)"),
    (4, "ladder_", "panel (a)"),
    (5, "abl_", "panel (c) còn lại"),
    (6, "len_", "phụ lục A: L*"),
    (7, "tau", "phụ lục A: lưới tau x K"),
    (7, "w_", "phụ lục A: nhiễu trọng số"),
    (8, "learned", "phụ lục B"),
]


def priority_of(tag: str) -> tuple[int, str]:
    for rank, needle, why in PRIORITY:
        if needle in tag:
            return rank, why
    return 9, ""


def caption_hash(csv_path: Path) -> str:
    """Hash of the captions only, so ordering or retrieval columns cannot hide a duplicate."""
    with csv_path.open("r", encoding="utf-8") as f:
        rows = sorted((r["query_id"], r.get("generated_caption", "")) for r in csv.DictReader(f))
    digest = hashlib.sha256()
    for qid, caption in rows:
        digest.update(qid.encode()); digest.update(b"\x00")
        digest.update(caption.encode()); digest.update(b"\x00")
    return digest.hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="submissions")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for run_dir in sorted(Path(args.runs).iterdir()):
        csv_path = run_dir / "submission.csv"
        if not csv_path.exists():
            continue
        stats = {}
        stats_path = run_dir / "stats.json"
        if stats_path.exists():
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
        scores = {}
        scores_path = run_dir / "scores.json"
        if scores_path.exists():
            scores = json.loads(scores_path.read_text(encoding="utf-8"))
        rank, why = priority_of(run_dir.name)
        entries.append({
            "tag": run_dir.name,
            "hash": caption_hash(csv_path),
            "priority": rank,
            "why": why,
            "len": stats.get("mean_length_words"),
            "n_ent": stats.get("mean_entities_per_caption"),
            "leak": stats.get("leakage_rate"),
            "clip_local": scores.get("clip"),
            "submitted_overall": scores.get("overall"),
            "source": str(csv_path),
        })

    # group identical caption sets
    by_hash: dict[str, list[dict]] = {}
    for entry in entries:
        by_hash.setdefault(entry["hash"], []).append(entry)

    manifest = []
    for digest, group in by_hash.items():
        group.sort(key=lambda e: (e["priority"], e["tag"]))
        primary = group[0]
        target = out_dir / f"{primary['tag']}.csv"
        shutil.copy(primary["source"], target)
        manifest.append({**primary, "file": str(target),
                         "duplicates": [e["tag"] for e in group[1:]]})

    manifest.sort(key=lambda e: (e["priority"], e["tag"]))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                           encoding="utf-8")

    print(f"{len(entries)} run -> {len(manifest)} CSV khác nhau "
          f"(tiết kiệm {len(entries) - len(manifest)} lượt submit)\n")
    header = f"{'#':>3}  {'file':<36}{'Len':>6}{'#Ent':>7}{'CLIP':>7}  {'ghi chú'}"
    print(header); print("-" * (len(header) + 20))
    for i, entry in enumerate(manifest, 1):
        done = "  [đã có điểm]" if entry["submitted_overall"] is not None else ""
        print(f"{i:>3}  {Path(entry['file']).name:<36}"
              f"{entry['len'] if entry['len'] is not None else '--':>6}"
              f"{entry['n_ent'] if entry['n_ent'] is not None else '--':>7}"
              f"{entry['clip_local'] if entry['clip_local'] is not None else '--':>7}"
              f"  {entry['why']}{done}")
        if entry["duplicates"]:
            print(f"     ^ caption trùng với: {', '.join(entry['duplicates'])} -> KHÔNG cần submit riêng")

    print(f"\n-> {out_dir}/  (manifest.json)")
    print("Sau khi có điểm:  python scripts/record_score.py --run runs/<tag> --overall .. --cider ..")


if __name__ == "__main__":
    main()
