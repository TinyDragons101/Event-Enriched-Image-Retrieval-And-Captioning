#!/usr/bin/env bash
# Panel (e) -- "which scorer defect mattered?"
#
# The published run carries six defects in the Entity Heuristic Scorer (see
# step_2_entity_scorer.DEFAULT_FIXES).  Each row below repairs exactly one of
# them, the last row repairs all six.  Everything else -- weights, tau, K_e,
# drafts, generator -- is held fixed, so each row isolates one defect.
#
#   legacy          the published VG-Cap run (0.6219)
#   keyword_parse   the LLM keyword list is parsed instead of comma-split, so it
#                   stops entering E as 1-3 markdown blobs
#   word_boundary   presence matching respects word boundaries ('One' no longer
#                   matches 'someone')
#   span_penalty    P_span of Eq. (2) is actually applied
#   dedup           substring-aware merge over ALL candidates
#   position        first-mention lookup instead of the start==0 sentinel
#   candidates      non-depictable labels and letterless spans are not candidates
#   all             every repair on (= configs/entity_scoring_config_fixed.json)
#
# With GATE=enrichment every row reuses the stored drafts: no GPU, ~15 min each.
#
#   bash scripts/run_fixes.sh [enrichment|unified]
set -euo pipefail

GATE="${1:-enrichment}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda:5}"
MODEL="${MODEL:-llama3}"
LIMIT="${LIMIT:-}"
DRAFTS="${DRAFTS:-assemble_result/cot_5_things_fact_more_event_llama.json}"
BASE_CFG="configs/entity_scoring_config.json"
OUT_DIR="${OUT_DIR:-runs}"

LIMIT_ARG=(); [[ -n "${LIMIT}" ]] && LIMIT_ARG=(--limit "${LIMIT}")
COMMON=(--seed "${SEED}" --device "${DEVICE}" --model_type "${MODEL}" --out_dir "${OUT_DIR}" "${LIMIT_ARG[@]}")
REUSE=(); [[ "${GATE}" == "enrichment" && -f "${DRAFTS}" ]] && REUSE=(--reuse_drafts "${DRAFTS}")

run() { echo; echo "=== $* ==="; python run_ablation.py "$@"; }

echo "gate=${GATE} seed=${SEED}  reuse_drafts=${#REUSE[@]}"

for PROFILE in legacy keyword_parse word_boundary span_penalty dedup position candidates all; do
  run --tag "fix_${PROFILE}_${GATE}_s${SEED}" --gate "${GATE}" --selector heuristic \
      --entity_config "${BASE_CFG}" --fix_profile "${PROFILE}" \
      "${COMMON[@]}" "${REUSE[@]}"
done

# The two normalizer claims of Section 3.6, on top of the repaired scorer.
run --tag "fix_all_gate2_${GATE}_s${SEED}" --gate "${GATE}" --selector heuristic \
    --entity_config "${BASE_CFG}" --fix_profile all \
    --preserve_selected --pos_mask "${COMMON[@]}" "${REUSE[@]}"

# Gate 2 without its noun-phrase leak: enrichment may append ONLY E*.
run --tag "fix_all_nonp_${GATE}_s${SEED}" --gate "${GATE}" --selector heuristic \
    --entity_config "${BASE_CFG}" --fix_profile all \
    --preserve_selected --pos_mask --no_noun_phrases "${COMMON[@]}" "${REUSE[@]}"

echo
echo "Xong. File can submit:"
ls -1 "${OUT_DIR}"/fix_*/submission.csv
