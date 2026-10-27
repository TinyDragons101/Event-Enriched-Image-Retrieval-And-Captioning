"""Emit the Table 3(c) leave-one-family-out configs from the VG-Cap base config.

Generated rather than hand-written so the ablations can never drift away from
the default weights.

Every variant is emitted twice:

    ablation_<name>.json          on the PUBLISHED scorer (fixes off)
    ablation_<name>_fixed.json    on the REPAIRED scorer  (fixes on)

so panel (c) can be reported against whichever scorer the paper ends up
describing, without re-deriving the weights.

    python scripts/make_ablation_configs.py
"""
import json
from pathlib import Path

BASE = Path("configs/entity_scoring_config.json")
FIXED = Path("configs/entity_scoring_config_fixed.json")
OUT = Path("configs")

PRESENCE = ["exact_match_crawl", "exact_match_generated",
            "partial_match_crawl", "partial_match_generated",
            "substring_in_qa", "substring_in_summary"]
EDITORIAL = ["exact_match_crawl", "partial_match_crawl"]
POSITION = ["early_position_bonus", "mid_position_bonus"]
TYPE = ["type_person", "type_gpe", "type_org", "type_loc",
        "type_date", "type_event", "type_other"]

# name -> (comment, weights to zero, extra weights, extra thresholds, extra fixes)
VARIANTS = {
    "ablation_no_presence": (
        "Table 3(c) w/o contextual presence. Max attainable score drops to 5.0 "
        "(PERSON 3.0 + early position 2.0), so tau=3.0 becomes far more binding: "
        "report #Ent alongside the metrics.", PRESENCE, {}, {}, {}),
    "ablation_no_editorial": (
        "Table 3(c) w/o editorial-caption evidence. Validity check for the "
        "benchmark-leakage concern (crawl_caption carries the editorial caption "
        "of the retrieved database image, which R@1=0.985 makes near-identical "
        "to the query).", EDITORIAL, {}, {}, {}),
    "ablation_no_position": (
        "Table 3(c) w/o article position.", POSITION, {}, {}, {}),
    "ablation_no_type": (
        "Table 3(c) w/o entity-type prior. PERSON loses its free pass over tau, "
        "so the selected-entity count drops sharply; #Ent must be reported.",
        TYPE, {}, {}, {}),
    "ablation_span_penalty_code": (
        "Span penalty with the code's own semantics: -0.1 * |e| for |e| > 3. "
        "Driven by fixes.span_penalty, not by the old weight-key hack.",
        [], {}, {}, {"span_penalty": "code"}),
    "ablation_span_penalty_paper": (
        "Span penalty exactly as Eq. (2) of the paper: -0.1 * max(0, |e|-3).",
        [], {}, {}, {"span_penalty": "paper"}),
}


def emit(base_path: Path, suffix: str, note: str) -> None:
    base = json.loads(base_path.read_text())
    for name, (comment, zeroed, extra_w, extra_t, extra_f) in VARIANTS.items():
        cfg = json.loads(json.dumps(base))          # deep copy
        cfg["_comment"] = f"{comment} [{note}]"
        for key in zeroed:
            cfg["weights"][key] = 0.0
        cfg["weights"].update(extra_w)
        cfg["thresholds"].update(extra_t)
        if extra_f:
            cfg.setdefault("fixes", {}).update(extra_f)
        path = OUT / f"{name}{suffix}.json"
        path.write_text(json.dumps(cfg, indent=2) + "\n")
        print(f"wrote {path}")


def main() -> None:
    emit(BASE, "", "published scorer, fixes off")
    if FIXED.exists():
        emit(FIXED, "_fixed", "repaired scorer, fixes on")
    else:
        print(f"skip *_fixed.json: {FIXED} not found")


if __name__ == "__main__":
    main()
