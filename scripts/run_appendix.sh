#!/usr/bin/env bash
# Appendix A sweeps.
#
# The L* sweep only re-runs the normalizer, so it reuses the reference drafts
# and costs no GPU.  The tau x K_e grid and the weight perturbations change the
# selected set; with GATE=enrichment they also reuse drafts and stay cheap.
#
# Usage:  bash scripts/run_appendix.sh [enrichment|unified]
set -euo pipefail

GATE="${1:-enrichment}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda:5}"
MODEL="${MODEL:-llama3}"
LIMIT="${LIMIT:-}"
REF_TAG="vgcap_${GATE}_s${SEED}"
DRAFTS="runs/${REF_TAG}/llm_raw.json"
BASE_CFG="configs/entity_scoring_config.json"

LIMIT_ARG=(); [[ -n "${LIMIT}" ]] && LIMIT_ARG=(--limit "${LIMIT}")
REUSE=(); [[ "${GATE}" == "enrichment" ]] && REUSE=(--reuse_drafts "${DRAFTS}")

if [[ ! -f "${DRAFTS}" ]]; then
  echo "missing ${DRAFTS} -- run scripts/run_panels.sh ${GATE} first" >&2
  exit 1
fi

run() { echo; echo "=== $* ==="; python run_ablation.py "$@"; }

# --- A.1  target length L* (normalizer only, no GPU) ------------------------
for L in 80 90 104 110 120; do
  run --tag "len_${L}_s${SEED}" --gate "${GATE}" --selector heuristic \
      --entity_config "${BASE_CFG}" --expected_length "${L}" --seed "${SEED}" \
      --device "${DEVICE}" --model_type "${MODEL}" "${LIMIT_ARG[@]}" \
      --reuse_drafts "${DRAFTS}"
done

# --- A.2  tau x K_e grid ----------------------------------------------------
for TAU in 2.0 3.0 4.0 5.0; do
  for K in 5 10 15 9999; do          # 9999 = no cap
    run --tag "tau${TAU}_k${K}_${GATE}_s${SEED}" --gate "${GATE}" --selector heuristic \
        --entity_config "${BASE_CFG}" --tau "${TAU}" --top_k "${K}" --seed "${SEED}" \
        --device "${DEVICE}" --model_type "${MODEL}" "${LIMIT_ARG[@]}" "${REUSE[@]}"
  done
done

# --- A.3  weight perturbation ----------------------------------------------
for FAMILY in presence position type; do
  for FACTOR in 0.5 1.5; do
    run --tag "w_${FAMILY}_x${FACTOR}_${GATE}_s${SEED}" --gate "${GATE}" --selector heuristic \
        --entity_config "${BASE_CFG}" --weight_scale "${FAMILY}:${FACTOR}" --seed "${SEED}" \
        --device "${DEVICE}" --model_type "${MODEL}" "${LIMIT_ARG[@]}" "${REUSE[@]}"
  done
done

for DRAW in 1 2 3 4 5 6 7 8 9 10; do
  run --tag "w_jitter_d${DRAW}_${GATE}_s${SEED}" --gate "${GATE}" --selector heuristic \
      --entity_config "${BASE_CFG}" --weight_jitter 0.2 --jitter_seed "${DRAW}" --seed "${SEED}" \
      --device "${DEVICE}" --model_type "${MODEL}" "${LIMIT_ARG[@]}" "${REUSE[@]}"
done

echo
echo "Done. Score each run with scripts/evaluate.py, then scripts/collect_results.py"
