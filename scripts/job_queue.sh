#!/usr/bin/env bash
# Run the job lines of a file with at most N running at once.
#
#   bash scripts/job_queue.sh jobs/G1_gpu.txt 4 180    # 4 GPU slots, starts 3 min apart
#   bash scripts/job_queue.sh jobs/C1_cpu_s0.txt 8      # 8 CPU slots
#
# Each non-empty line that does not start with '#' is a shell command. The
# `run` helper and the variables of vast_env.sh ($CFG, $BAND, $NOBAND, ...)
# are available. A new line starts as soon as a slot frees, so the GPU never
# idles between batches. DELAY staggers starts so model loads do not all hit
# host RAM at once. Safe to re-run: `run` passes --skip_if_done --resume.
set -uo pipefail

FILE=${1:?usage: job_queue.sh JOBFILE [N] [DELAY_SECONDS]}
N=${2:-4}
DELAY=${3:-0}
source "${VAST_ENV:-/workspace/vgcap/vast_env.sh}"

echo "[queue] $FILE  slots=$N  delay=${DELAY}s  start $(date '+%F %T')"
while IFS= read -r line || [[ -n $line ]]; do
  [[ -z ${line//[[:space:]]/} || $line =~ ^[[:space:]]*# ]] && continue
  while (( $(jobs -rp | wc -l) >= N )); do sleep 20; done
  eval "$line" < /dev/null &
  sleep "$DELAY"
done < "$FILE"
wait
echo "[queue] $FILE  done $(date '+%F %T')"
