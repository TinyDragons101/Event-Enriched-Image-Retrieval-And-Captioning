"""Regression test for the correctness pass.

Two things must hold before any number is trusted:

  1. LEGACY EQUIVALENCE.  With no `fixes` block (or fixes all off) the rewritten
     EntityHeuristicScorer must select exactly the same E* as the version that
     produced the published 0.6219 submission.  The reference implementation is
     backup_pre_fix/step_2_entity_scorer.py.bak.
  2. THE REPAIRS ACTUALLY FIRE.  The repaired scorer must stop admitting the
     markdown blobs and the letterless spans that the published run admitted.

    python scripts/test_fixes.py --n 300
    python scripts/test_fixes.py --n 3000 --reference runs/repro_vgcap/selected_entities.json
"""
from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from step_2_entity_scorer import EntityHeuristicScorer  # noqa: E402

MERGE = ROOT / "public_test_final_elements_json/final_merge_result.json"
OLD = ROOT / "backup_pre_fix/step_2_entity_scorer.py.bak"
FIXED_CFG = ROOT / "configs/entity_scoring_config_fixed.json"


def load_old_scorer():
    """Import the pre-fix scorer under its own module name."""
    if not OLD.exists():
        return None
    # .bak is not a recognised source suffix, so name the loader explicitly
    loader = importlib.machinery.SourceFileLoader("orig_entity_scorer", str(OLD))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EntityHeuristicScorer()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=300, help="queries to check")
    ap.add_argument("--merge", default=str(MERGE))
    ap.add_argument("--reference", default=None,
                    help="selected_entities.json of a published run, compared instead "
                         "of (or in addition to) the pre-fix module")
    args = ap.parse_args()

    data = json.loads(Path(args.merge).read_text(encoding="utf-8"))[: args.n]
    legacy = EntityHeuristicScorer()                       # all fixes off
    repaired = EntityHeuristicScorer(str(FIXED_CFG))
    failures = 0

    # -- 1. legacy equivalence against the pre-fix module --------------------
    old = load_old_scorer()
    if old is None:
        print(f"SKIP  pre-fix module not found at {OLD}")
    else:
        bad = [q["query_id"] for q in data
               if old.score_query(q)[0] != legacy.score_query(q)[0]]
        status = "PASS" if not bad else f"FAIL ({len(bad)} query)"
        print(f"[1] legacy == pre-fix module          {status}  ({len(data)} query)")
        if bad:
            failures += 1
            print("    vi du:", bad[:5])

    # -- 2. legacy equivalence against a published run -----------------------
    if args.reference:
        ref = json.loads(Path(args.reference).read_text(encoding="utf-8"))
        checked = [q for q in data if q["query_id"] in ref]
        bad = [q["query_id"] for q in checked
               if legacy.score_query(q)[0] != ref[q["query_id"]]]
        status = "PASS" if not bad else f"FAIL ({len(bad)} query)"
        print(f"[2] legacy == {Path(args.reference).parent.name:<22} {status}  ({len(checked)} query)")
        if bad:
            failures += 1
            print("    vi du:", bad[:5])

    # -- 3. the repairs change what they are supposed to change --------------
    def census(scorer):
        blobs = junk = total = 0
        for q in data:
            for e in scorer.score_query(q)[0]:
                total += 1
                blobs += len(e) > 150
                junk += not any(c.isalpha() for c in e)
        return blobs, junk, total

    b0, j0, t0 = census(legacy)
    b1, j1, t1 = census(repaired)
    print(f"[3] markdown blob trong E*            legacy {b0:5d}  ->  fixed {b1:5d}")
    print(f"[4] span khong co chu cai trong E*    legacy {j0:5d}  ->  fixed {j1:5d}")
    print(f"    tong so entity duoc chon          legacy {t0:5d}      fixed {t1:5d}")
    if b1 >= b0 and b0:
        failures += 1
        print("    FAIL: keyword parsing khong giam so blob")
    if j1 >= j0 and j0:
        failures += 1
        print("    FAIL: candidate filter khong loai duoc span khong chu cai")

    print()
    print("KET QUA:", "TAT CA PASS" if not failures else f"{failures} kiem tra FAIL")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
