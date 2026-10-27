#!/usr/bin/env bash
# Score every run that does not have scores.json yet, then print the table.
#
#   GT=data/public_test_gt.json QUERY_IMAGES=data/track1_public/query \
#     bash scripts/score_all.sh
set -euo pipefail

GT="${GT:?set GT=path/to/ground_truth.json}"
QUERY_IMAGES="${QUERY_IMAGES:?set QUERY_IMAGES=path/to/query/images}"
DEVICE="${DEVICE:-cuda:0}"

for run in runs/*/; do
  [[ -f "${run}/normalized.json" ]] || continue
  if [[ -f "${run}/scores.json" ]]; then
    echo "skip ${run} (already scored)"
    continue
  fi
  echo "=== scoring ${run}"
  python scripts/evaluate.py --run "${run%/}" --gt "${GT}" \
      --query_images "${QUERY_IMAGES}" --device "${DEVICE}"
done

echo
python scripts/collect_results.py --latex
