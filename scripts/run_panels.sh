#!/usr/bin/env bash
# Sinh toàn bộ submission CSV cho Table 3 (panel a, b, c, d).
# Mỗi run để lại runs/<tag>/submission.csv -- đó là file đem lên leaderboard.
#
# TÁI DÙNG DRAFT
#   Repo đã có sẵn hai bộ draft LLM cho 3000 query, nên phần lớn cấu hình không
#   cần GPU. Phần sinh caption là thứ duy nhất ngẫu nhiên; normalizer và entity
#   scorer đều deterministic.
#
#     cot_5_things_fact_more_event_llama.json          97.4 tu  -> prompt CO length band
#     cot_5_things_fact_more_event_llama_version1.json 135.2 tu -> prompt KHONG length band
#                                                                  (khop dong ENRIC baseline, bai ghi 136)
#
#   Chi 4 cau hinh thuc su can sinh moi: ladder_band_only, ladder_entities_only,
#   gate_prompt, gate_unified. Cac gate prompt/unified doi prompt nen bat buoc sinh lai.
#
# Usage:  bash scripts/run_panels.sh [enrichment|unified]
set -euo pipefail

GATE="${1:-enrichment}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda:5}"
MODEL="${MODEL:-llama3}"
LIMIT="${LIMIT:-}"
BASE_CFG="${BASE_CFG:-configs/entity_scoring_config.json}"
# CFG_SUFFIX="_fixed" chay toan bo panel tren scorer da sua loi:
#   BASE_CFG=configs/entity_scoring_config_fixed.json CFG_SUFFIX=_fixed \
#     bash scripts/run_panels.sh enrichment
CFG_SUFFIX="${CFG_SUFFIX:-}"

DRAFTS_BAND="${DRAFTS_BAND:-assemble_result/cot_5_things_fact_more_event_llama.json}"
DRAFTS_NOBAND="${DRAFTS_NOBAND:-assemble_result/cot_5_things_fact_more_event_llama_version1.json}"

REF_TAG="vgcap_${GATE}${CFG_SUFFIX}_s${SEED}"
REF_DIR="runs/${REF_TAG}"
REF_SEL="${REF_DIR}/selected_entities.json"

LIMIT_ARG=(); [[ -n "${LIMIT}" ]] && LIMIT_ARG=(--limit "${LIMIT}")
COMMON=(--seed "${SEED}" --device "${DEVICE}" --model_type "${MODEL}" "${LIMIT_ARG[@]}")

run() { echo; echo "=== $* ==="; python run_ablation.py "$@"; }

# draft co san thi tai dung, khong thi sinh moi
band_drafts()   { [[ -f "${DRAFTS_BAND}"   ]] && echo "--reuse_drafts ${DRAFTS_BAND}"; }
noband_drafts() { [[ -f "${DRAFTS_NOBAND}" ]] && echo "--reuse_drafts ${DRAFTS_NOBAND}"; }
# cac gate doi prompt thi khong bao gio tai dung duoc
gate_drafts()   { [[ "${GATE}" == "enrichment" ]] && band_drafts; }

echo "gate=${GATE} seed=${SEED}"
echo "draft co band   : ${DRAFTS_BAND}   $([[ -f ${DRAFTS_BAND}   ]] && echo '[co]' || echo '[thieu -> se sinh moi]')"
echo "draft khong band: ${DRAFTS_NOBAND} $([[ -f ${DRAFTS_NOBAND} ]] && echo '[co]' || echo '[thieu -> se sinh moi]')"

# ---------------------------------------------------------------------------
# 0. Run tham chieu: VG-Cap. Sinh ra E* ma panel (b) can de khop ngan sach.
# ---------------------------------------------------------------------------
run --tag "${REF_TAG}" --gate "${GATE}" --selector heuristic \
    --entity_config "${BASE_CFG}" "${COMMON[@]}" $(gate_drafts)

# ---------------------------------------------------------------------------
# (a) Component ladder
# ---------------------------------------------------------------------------
run --tag "ladder_enric_s${SEED}" --gate none --no_prompt_entities --no_prompt_band \
    --no_normalizer "${COMMON[@]}" $(noband_drafts)

run --tag "ladder_sgn_s${SEED}" --gate none --no_prompt_entities --no_prompt_band \
    "${COMMON[@]}" $(noband_drafts)

run --tag "ladder_band_only_s${SEED}" --gate none --no_prompt_entities "${COMMON[@]}"

run --tag "ladder_entities_only_s${SEED}" --gate none --no_prompt_band "${COMMON[@]}"

# ---------------------------------------------------------------------------
# (d) Gate location. prompt/unified doi prompt -> bat buoc sinh moi.
# ---------------------------------------------------------------------------
run --tag "gate_none_s${SEED}" --gate none "${COMMON[@]}" $(band_drafts)

run --tag "gate_prompt_s${SEED}" --gate prompt --selector heuristic \
    --entity_config "${BASE_CFG}" "${COMMON[@]}"

run --tag "gate_unified_s${SEED}" --gate unified --selector heuristic \
    --entity_config "${BASE_CFG}" "${COMMON[@]}"

# ---------------------------------------------------------------------------
# (b) Alternative selectors, ngan sach khop theo tung query
# ---------------------------------------------------------------------------
for SELECTOR in article_order presence_only type_only; do
  run --tag "sel_${SELECTOR}_${GATE}_s${SEED}" --gate "${GATE}" --selector "${SELECTOR}" \
      --entity_config "${BASE_CFG}" --budget_match per_query --reference_selection "${REF_SEL}" \
      "${COMMON[@]}" $(gate_drafts)
done

for DRAW in 1 2 3 4 5; do
  run --tag "sel_random_d${DRAW}_${GATE}_s${SEED}" --gate "${GATE}" --selector random \
      --entity_config "${BASE_CFG}" --budget_match per_query --reference_selection "${REF_SEL}" \
      --seed "${DRAW}" --device "${DEVICE}" --model_type "${MODEL}" "${LIMIT_ARG[@]}" $(gate_drafts)
done

# ---------------------------------------------------------------------------
# (c) Leave one feature family out
#     'w/o span penalty' chinh la mac dinh hien tai: trong ban da cong bo
#     penalty khong bao gio chay (compute_score doc key 'length_penalty',
#     weights chi khai 'length_penalty_factor'). Hai dong span duoi day BAT no
#     len qua fixes.span_penalty = code | paper, khong phai tat.
# ---------------------------------------------------------------------------
for CFG in no_presence no_editorial no_position no_type span_penalty_code span_penalty_paper; do
  run --tag "abl_${CFG}${CFG_SUFFIX}_${GATE}_s${SEED}" --gate "${GATE}" --selector heuristic \
      --entity_config "configs/ablation_${CFG}${CFG_SUFFIX}.json" "${COMMON[@]}" $(gate_drafts)
done

echo
echo "Xong. Cac file can submit:"
ls -1 runs/*/submission.csv
