# VG-Cap on vast.ai — Runbook (straight-line re-run)

Date: 2026-10-28 · Branch: `fix/vgcap-correctness` · Paper: `VGCap_ACIIDS2027/new_main.tex` · Companion notes: `docs/PAPER_NEW_MAIN_DIFF.md`, `docs/VALIDATION_NOTE.md`

Markers used below:
- ⛔ **STOP**: do not continue; fix first. Continuing here wastes paid hours.
- ✅ **CHECK**: expected output.
- 💰 **COST**: something that burns money if done wrong.

**What this runbook does.** The code, prompt and drafts behind the submitted 0.6166/0.6219 results are lost, so Table 3 is **re-run from the beginning with the current code**. Retrieval and the context files (`final_merge_result.json`) are inputs and are kept.

| Step | What | Where |
|---|---|---|
| 1 | The normal VG-Cap run, **Gates 1+2**, seed 0, together with No gate / Gate 1 only / Gate 2 only and the ENRIC/SGN rows | §6 |
| 2 | **Checkpoint:** compare the new VG-Cap with the submitted 0.6219 and with No gate | §7 |
| 3 | Every other P1 + P2 row of `new_main.tex` Table 3 (no P3), 3 seeds on P1 | §8 |

The submitted file is `D:\research\submission_0.6219.csv`; 0.6166 and 0.6123 are kept only as reference numbers.

---

## 0. Before renting (local, free)

### 0.1 Code changes already made (2026-10-27/28)

| # | Done in | Local check |
|---|---|---|
| P0-1 | Temperature **0.3** in `llama3.py` / `llamassemblers.py`; `run_ablation.py --temperature` (default 0.3, saved in `config.json`) | default resolves to 0.3 |
| P0-2 | Prompt wording restored (`100-120 words.`) in both `.j2` | original `.j2` == template of commit `2d47722` byte-for-byte |
| P0-3 | No-band branch keeps "at least 5 sentences" (template of commit `c004e52`) | rendered and compared |
| P0-4 | `run_ablation.py`: `--overwrite`, `--skip_if_done`, atomic `llm_raw.json`, per-query seed | tested |
| P0-5 | Selector debug has per-entity `features`; `configs/ablation_no_span(_fixed).json` | tested |
| new | `presence_score` selector, `scripts/count_entities.py`, `scripts/job_queue.sh`, `jobs/*.txt`, 4-label annotation sheet | tested |

`run_ablation.py` **refuses** to reuse a tag that already has drafts or results. The `run` helper (§4) passes `--skip_if_done --resume`, so job lists are safe to re-run.

### 0.2 Files to have ready on your PC

| What | Local path | Size |
|---|---|---|
| Repo (working tree incl. gitignored `public_test_final_elements_json/`) | `D:\research\Event-Enriched-Image-Retrieval-And-Captioning` | ~105 MB |
| Query images (3000 × `{query_id}.jpg`) | `D:\research\Track 1 - Public Set-20261026T004503Z-1-001\Track 1 - Public Set\images\pub_images` | ~1.4 GB |
| Submitted results (for the checkpoint, stay on the PC) | `D:\research\submission_0.6219.csv` (+ `_0.6166`, `_0.6123`) | — |
| Hugging Face token whose account has **accepted the Llama-3 licence** | — | — |
| SSH key for vast | `~/.ssh/id_ed25519_vast` (already in `~/.ssh/config` as `Host vast`) | — |

### 0.3 Instance to rent

| Field | Value | Why |
|---|---|---|
| GPU | **1× A100 80GB** (SXM or PCIe) | Fits 4 Llama-3-8B processes at once (~18 GB each). The paper says "one A100" |
| Max CUDA (offer filter) | **≥ 12.6** | `torch==2.7.0` wheels are cu126 |
| CPU quota | ≥ 16 | CPU jobs run next to generation |
| RAM | ≥ 64 GB | Each Llama load peaks ~16 GB host RAM |
| Disk holding `/workspace` | **≥ 100 GB** | Models ~17 GB, venv ~10 GB, runs ~5 GB |
| Type | **On-demand**, not interruptible | Interruptible can kill a 6 h generation |

### 0.4 Scorer config: **repaired config** (`configs/entity_scoring_config_fixed.json`)

The default `configs/entity_scoring_config.json` (legacy) turns the whole LLM keyword list into **one ~585-character "entity"**. That blob passes the threshold in **2998 of 3000 queries**, measured locally on `final_merge_result.json`. With Gate 1, `E*` is written into the prompt, so this blob would put almost the entire unfiltered keyword list back into the prompt, and Gate 1 would not filter anything.

The repaired config (same weights and thresholds; only the documented bug switches are on) produces **0** such blobs. It also matches what `new_main.tex` describes:
- keyword parsing;
- dedup that keeps the longer form;
- word-boundary matching;
- span penalty as in Eq. 2.

`vast_env.sh` (§4) therefore sets `CFG=configs/entity_scoring_config_fixed.json` and `CFG_SUFFIX=_fixed`.

### 0.5 Cost estimate

Measured: 6.1 h per 3000-query generation run, sequential. Assumed until §5.4 measures it: each run is 1.3–2.0× slower when 4 share the A100.

| Step | Runs | Time |
|---|---|---|
| Setup + pre-flight (§1–5) | — | ~2–2.5 h |
| Step 1 (`G1_core_s0.txt`, 4 at once) + CPU rows | 4 GPU + 2 CPU | ~8–12 h |
| Checkpoint (§7): CLIP, download, submit, compare | — | ~0.5–1 h (GPU idle) |
| Step 2 (`G2_rest.txt`, 4 slots) + CPU rows | 23 GPU + 3 CPU | ~46–70 h |
| Analysis, final download (§11) | — | ~1.5 h |
| **Total** | **27 GPU** | **~58–87 h ≈ $66–99** at $1.143/h |
| If you stop at the checkpoint | 4 GPU | ~11–16 h ≈ $13–18 |

Leaderboard: ~32 distinct CSVs (quota ~94 left). After §5.4, recompute with the measured `s/it × 3000 / 3600`.

---

## 1. Connect

```bash
ssh vast                      # Host entry in ~/.ssh/config; update HostName/Port when you rent a new instance
```

On the server, work inside **tmux** so jobs survive a dropped connection:

```bash
tmux new -s main              # or: tmux attach -t main
# new window: Ctrl+b c     next window: Ctrl+b n     detach: Ctrl+b d
nvidia-smi                    # ✅ 1x A100 80GB, "CUDA Version" >= 12.6, ~0 MiB used
df -h / /workspace            # ✅ the filesystem holding /workspace must be >= 100 GB
cat /sys/fs/cgroup/cpu.max 2>/dev/null || cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us /sys/fs/cgroup/cpu/cpu.cfs_period_us
                              # real CPU quota = quota / period, e.g. "3071999 100000" = 30.7 CPUs
```

⛔ STOP if the GPU is not an A100 80GB, CUDA < 12.6, or the filesystem holding `/workspace` is < 100 GB.
- Disk size **cannot be changed after renting**; set the disk slider explicitly.
- Destroy and re-rent; you pay only for the minutes used.

Keep everything big under `/workspace`. Point pip's cache and temp files there too, so the small root disk never fills:
```bash
mkdir -p /workspace/tmp /workspace/pip-cache
echo 'export TMPDIR=/workspace/tmp PIP_CACHE_DIR=/workspace/pip-cache' >> ~/.bashrc && source ~/.bashrc
```

⚠️ `nproc` and `free -g` show the **whole host**, not your share. Use the cgroup CPU quota when deciding how many CPU jobs to run in parallel.

---

## 2. Copy source and data (from your PC)

Rebuild the tarball **after** the latest local changes. In **Git Bash on your PC**:

```bash
cd /d/research
tar czf vgcap_repo.tgz Event-Enriched-Image-Retrieval-And-Captioning
tar cf  query_images.tar -C "Track 1 - Public Set-20261026T004503Z-1-001/Track 1 - Public Set/images" pub_images
ls -lh vgcap_repo.tgz query_images.tar        # ~60-105 MB and ~1.4 GB
scp vgcap_repo.tgz query_images.tar vast:/workspace/
```

On the **server**:

```bash
cd /workspace
tar xzf vgcap_repo.tgz && mv Event-Enriched-Image-Retrieval-And-Captioning vgcap
mkdir -p vgcap/data/track1_public
tar xf query_images.tar -C vgcap/data/track1_public && mv vgcap/data/track1_public/pub_images vgcap/data/track1_public/query
cd vgcap
ls data/track1_public/query | wc -l                                    # ✅ 3000
ls jobs/                                                               # ✅ C1_after_G1.txt C2_after_G2.txt G1_core_s0.txt G2_rest.txt
sha256sum public_test_final_elements_json/final_merge_result.json \
          final_csv_result/public_final_retrieval_merging_final_results.csv \
          assemble_result/cot_5_things_fact_more_event_llama.json | cut -c1-16
```
✅ Must print, in order: `a588dea79a676eb8`, `6308b803e2338f9f`, `9cd8af35ab95b694`.
- The first two are the **inputs** of every run: the contexts and the submission template.
- The third is only used by the CPU smoke test.

⛔ STOP on any mismatch: the copy is incomplete.

---

## 3. Python environment

Ubuntu 24.04 images (Python 3.12) refuse a system-wide `pip install` (PEP 668). Use a venv:

```bash
apt-get update && apt-get install -y python3-venv      # skip if `python3 -m venv` already works
python3 -m venv /workspace/venv
source /workspace/venv/bin/activate
pip install --upgrade pip
cd /workspace/vgcap
python --version                       # ✅ >= 3.10
```

💰 `requirements.txt` lists **`flash_attn` without a version**. pip would compile it from source, which takes 30–60+ min and often fails. **Exclude it**; Llama does not use it.

```bash
grep -v -i '^flash_attn' requirements.txt > /tmp/req.txt
pip install -r /tmp/req.txt                                   # installs torch 2.7.0 if the image has another version
pip install "spacy==3.8.*" jinja2 python-dotenv hf_transfer scikit-learn
python -m spacy download en_core_web_sm
python - <<'EOF'
import torch, transformers, spacy
nlp = spacy.load("en_core_web_sm")
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("transformers", transformers.__version__, "| spacy", spacy.__version__, "| en_core_web_sm", nlp.meta["version"])
EOF
```
✅ Expected: `torch 2.7.0 cuda True NVIDIA A100…`, `transformers 4.45.2`.

---

## 4. Models and the environment file

```bash
export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1
read -s -p "HF token: " HF_TOKEN; echo; export HF_TOKEN
printf 'HUGGING_FACE_TOKEN=%s\n' "$HF_TOKEN" > /workspace/vgcap/.env    # llama3.py reads this
chmod 600 /workspace/vgcap/.env

# Llama-3-8B-Instruct: safetensors only; "original/*" would add another ~16 GB of duplicate weights
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --exclude "original/*"
huggingface-cli download openai/clip-vit-base-patch32
ls $HF_HOME/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/   # record this revision hash
```
⛔ STOP on `401`/`403` for Llama: the HF account has not accepted the Meta licence, or the token is wrong.

Create one environment file and `source` it in **every** tmux window:

```bash
cat > /workspace/vgcap/vast_env.sh <<'EOF'
source /workspace/venv/bin/activate
cd /workspace/vgcap
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1          # never re-download mid-run
export DEVICE=cuda:0                                   # scripts default to cuda:5 -> wrong on a 1-GPU box
export CFG=configs/entity_scoring_config_fixed.json    # repaired scorer, see section 0.4
export CFG_SUFFIX=_fixed                               # (c) rows use configs/ablation_*_fixed.json
export BAND=assemble_result/cot_5_things_fact_more_event_llama.json   # CPU smoke test only
export QIMG=data/track1_public/query
mkdir -p logs analysis
# run <tag> <args...> : one run_ablation call, logged to logs/<tag>.log.
# --skip_if_done --resume make every job list safe to re-run: finished runs are
# skipped, a crashed generation continues where its checkpoint stopped.
run() { local tag=$1; shift
  python run_ablation.py --tag "$tag" --device "$DEVICE" --skip_if_done --resume "$@" >> "logs/$tag.log" 2>&1 \
    && echo "OK   $tag" || echo "FAIL $tag  (see logs/$tag.log)"; }
EOF
source /workspace/vgcap/vast_env.sh
```

---

## 5. Pre-flight: GO / NO-GO (~45 min)

### 5.1 Record the environment
```bash
{ date; git log -1 --format=%H; git status --short; nvidia-smi --query-gpu=name,driver_version --format=csv;
  pip freeze | grep -iE '^(torch|transformers|spacy|accelerate|tokenizers)=='; python -c "import spacy;print('en_core_web_sm',spacy.load('en_core_web_sm').meta['version'])";
  ls $HF_HOME/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/; echo "CFG=$CFG"; } > runs_env.txt
cat runs_env.txt
```

### 5.2 Scorer check (CPU, ~15–20 min)
```bash
mkdir -p backup_pre_fix && git show 9a23067:step_2_entity_scorer.py > backup_pre_fix/step_2_entity_scorer.py.bak
python scripts/test_fixes.py --n 3000 2>&1 | tee logs/test_fixes.log
```
✅ Expected:
- `[1] legacy == pre-fix module  PASS`: the refactor did not break the scorer code.
- `[3] markdown blob trong E*  legacy N -> fixed 0`: the repaired config removes the blobs this run depends on.
- `KET QUA: TAT CA PASS`.

⛔ STOP if `[1]` says SKIP or FAIL, or if `fixed` is not 0 in `[3]`.

### 5.3 CPU smoke test (while 5.2 runs, in another tmux window)
```bash
source /workspace/vgcap/vast_env.sh
run smoke_cpu --seed 0 --gate enrichment --selector heuristic --entity_config $CFG --limit 20 --reuse_drafts $BAND
ls runs/smoke_cpu/      # ✅ submission.csv stats.json normalized.json selected_entities.json entity_debug.json
```

### 5.4 GPU smoke test: Gates 1+2 prompt, speed, concurrency (decides the real cost)
```bash
run smoke_gpu --seed 0 --gate unified --selector heuristic --entity_config $CFG --limit 20
grep -o "[0-9.]*s/it\|[0-9.]*it/s" logs/smoke_gpu.log | tail -3
python - <<'EOF'
import json
d = json.load(open("runs/smoke_gpu/llm_raw.json")); s = json.load(open("runs/smoke_gpu/selected_entities.json"))
print("drafts:", len(d)); print(list(d.values())[0][:300])
print("E* sizes:", [len(v) for v in s.values()][:10], "| longest E* entity:", max((len(e) for v in s.values() for e in v), default=0), "chars")
EOF
```
✅ Expected:
- 20 drafts, each a news caption (often after "Here is the caption:").
- E\* sizes ≤ 10.
- Longest E\* entity **< 100 chars**. A ~500-char entity means the legacy config is in use: check `$CFG`.

💰 **Compute the real cost now:** hours per run = s/it × 3000 / 3600 (expect ~7 s/it ≈ 6 h). Then test concurrency: run 3 smoke runs at once and read their s/it while watching `nvidia-smi`:
```bash
for i in 2 3 4; do run smoke_gpu$i --seed $i --gate unified --selector heuristic --entity_config $CFG --limit 20 & sleep 60; done; wait
grep -ho "[0-9.]*s/it" logs/smoke_gpu[234].log | tail -3
```
- s/it barely rises → use **4 slots** in §6/§8.
- s/it doubles → use 2 slots.

Clean up: `rm -rf runs/smoke_*`.

---

## 6. STEP 1: the normal VG-Cap run (Gates 1+2) + seed-0 comparison rows

Window 1 (GPU, 4 runs together; starts are 4 min apart so the model loads don't all hit host RAM at once):
```bash
source /workspace/vgcap/vast_env.sh
bash scripts/job_queue.sh jobs/G1_core_s0.txt 4 240 2>&1 | tee logs/queue_G1.log
```

| Tag | Paper row | Prompt (Gate 1) | Enrichment (Gate 2) |
|---|---|---|---|
| `vgcap_s0` | **VG-Cap = Gates 1+2** | E\* | E\* |
| `gate_none_s0` | (d) No gate; (a) + band + entity list | all keywords | all article entities |
| `gate_prompt_s0` | (d) Gate 1 only | E\* | all article entities |
| `ladder_kw_noband_s0` | (a) + SGN / + entity list | all keywords, no band | all article entities |

Window 2 (CPU, when `gate_none_s0` and `ladder_kw_noband_s0` have `stats.json`):
```bash
source /workspace/vgcap/vast_env.sh
bash scripts/job_queue.sh jobs/C1_after_G1.txt 2 2>&1 | tee logs/queue_C1.log
```
This produces `gate_enrichment_s0` = (d) **Gate 2 only** (same drafts as No gate) and `ladder_enric_s0` = (a) ENRIC baseline (no SGN).

Monitor:
```bash
watch -n 60 nvidia-smi                                         # ✅ 4 python processes, total < 78 GB
tail -c 300 logs/vgcap_s0.log                                  # progress + s/it
python -c "import json;print(len(json.load(open('runs/vgcap_s0/llm_raw.json'))))"   # +50 per checkpoint
```
If anything crashes, re-run the same queue command: finished runs are skipped and the crashed one resumes from its checkpoint.

---

## 7. CHECKPOINT: compare the new VG-Cap with the submitted 0.6219

When `vgcap_s0`, `gate_none_s0`, `gate_prompt_s0` and `gate_enrichment_s0` all have `stats.json`:

**7.1 Server:** local CLIP, then pack.
```bash
source /workspace/vgcap/vast_env.sh
for t in vgcap_s0 gate_none_s0 gate_prompt_s0 gate_enrichment_s0 ladder_kw_noband_s0 ladder_enric_s0; do
  python scripts/evaluate.py --run runs/$t --query_images $QIMG --device $DEVICE > logs/clip_$t.log 2>&1 && echo "CLIP $t done"
done
python scripts/make_submissions.py --runs runs --out submissions
tar czf /workspace/vgcap_step1.tgz runs logs submissions runs_env.txt
```

**7.2 PC:** download, submit, record.
```bash
mkdir -p /d/research/vast_results && cd /d/research/vast_results
scp vast:/workspace/vgcap_step1.tgz . && tar xzf vgcap_step1.tgz
```
1. Submit `runs/vgcap_s0/submission.csv` **first**, then `gate_none_s0`, `gate_prompt_s0` and `gate_enrichment_s0`.
2. For each, record the leaderboard numbers; run this from the repo folder, with `runs` pointing to the downloaded copy:
```bash
python scripts/record_score.py --run <path>/runs/vgcap_s0 --overall <O> --cider <C> --clip <CLIP>
```

**7.3 PC:** compare with the submitted result. Standard library only.
```bash
cd /d/research
python - <<'EOF'
import csv, json, statistics as st
def caps(f): return [r["generated_caption"] for r in csv.DictReader(open(f, encoding="utf-8"))]
for name, f in [("submitted 0.6219", "submission_0.6219.csv"), ("new vgcap_s0", "vast_results/runs/vgcap_s0/submission.csv"),
                ("new gate_none_s0", "vast_results/runs/gate_none_s0/submission.csv")]:
    c = caps(f); L = [len(x.split()) for x in c]
    img = sum(x.startswith(("The image", "This image", "In the image", "In this image")) for x in c) / len(c)
    print(f"{name:18s} len {st.mean(L):.2f} | opens 'The/This image' {img:.1%}")
for t in ["vgcap_s0", "gate_none_s0", "gate_prompt_s0", "gate_enrichment_s0"]:
    try: s = json.load(open(f"vast_results/runs/{t}/scores.json")); print(f"{t:20s} Overall {s.get('overall')}  CIDEr {s.get('cider')}  CLIP {s.get('clip')}")
    except FileNotFoundError: print(t, "no scores.json yet")
print("reference: submitted VG-Cap 0.6219 | submitted unfiltered 0.6166 | repo-draft no-gate rebuild 0.6123")
EOF
```

**7.4 Decide before paying for Step 2 (~46–70 h).** The GPU idles meanwhile (~$1.1/h), so submit promptly.

| Result | Meaning | Action |
|---|---|---|
| `vgcap_s0` > `gate_none_s0` | The gates help under the re-run | **Go to Step 2** |
| `vgcap_s0` ≤ `gate_none_s0` | The method doesn't beat its own baseline under the current code | ⛔ **STOP**, talk to the supervisor before Step 2 |
| Gap to 0.6219 | Expected: the original prompt and drafts are lost (0.6219 opens with "The image…" in 52% of captions; the current prompt rarely does) | Record it for the paper sentence "re-run VG-Cap obtains X ± s vs 0.6219 submitted"; it is not a stop condition unless `vgcap_s0` is below 0.6123 |

---

## 8. STEP 2: all remaining P1 + P2 rows

Window 1 (GPU):
```bash
source /workspace/vgcap/vast_env.sh
bash scripts/job_queue.sh jobs/G2_rest.txt 4 240 2>&1 | tee logs/queue_G2.log
```

23 runs in this order:

| Group | Runs | Prio |
|---|---|---|
| Seeds 1–2 of VG-Cap, No gate, Gate 1 only | 6 | P1 |
| (b) article order ×3 seeds, random ×5 draws | 8 | P1 |
| (c) w/o editorial-caption match ×3 seeds | 3 | P1 |
| (a) ladder rows without the entity list | 2 | P2 |
| (b) presence only; (c) w/o presence, w/o type, presence required | 4 | P2 |

All (b)/(c) rows run through both gates, like VG-Cap.

Window 2 (CPU, when `gate_none_s1`, `gate_none_s2` and `ladder_nokw_noband_s0` have `stats.json`):
```bash
source /workspace/vgcap/vast_env.sh
bash scripts/job_queue.sh jobs/C2_after_G2.txt 3 2>&1 | tee logs/queue_C2.log
```
This produces `gate_enrichment_s1/s2` (Gate 2 only, seeds 1–2) and `ladder_nokw_noband_nosgn_s0`.

---

## 9. CLIP scoring (small; run any time during Step 2 and once at the end)

```bash
source /workspace/vgcap/vast_env.sh
for r in runs/*/; do t=${r%/}
  [ -f $t/stats.json ] || continue            # run not finished
  [ -f $t/scores.json ] && continue           # already scored
  python scripts/evaluate.py --run $t --query_images $QIMG --device $DEVICE > logs/clip_$(basename $t).log 2>&1 \
    && echo "CLIP $(basename $t) $(python -c "import json;print(json.load(open('$t/scores.json'))['clip'])")"
done
```
✅ `n_scored` = 3000 in each `scores.json`, and no "no image found" warning. Local CLIP is a screening number; the leaderboard CLIP is the official one.

---

## 10. Save results (after Step 1, during Step 2 at least daily, and at the end)

💰 vast deletes everything when the instance is destroyed, and `runs/` is gitignored.

Server:
```bash
cd /workspace/vgcap
python scripts/make_submissions.py --runs runs --out submissions      # dedups identical CSVs + submit list
tar czf /workspace/vgcap_runs_$(date +%m%d_%H%M).tgz runs logs submissions analysis runs_env.txt configs jobs
```
PC:
```bash
cd /d/research/vast_results
scp "vast:/workspace/vgcap_runs_*.tgz" .
for f in vgcap_runs_*.tgz; do tar tzf "$f" > /dev/null && echo "OK $f" || echo "CORRUPT $f"; done
```

---

## 11. Analysis on the server, then shutdown

### 11.1 Entity counts and Section 4.5 statistics (CPU, ~15 min)
```bash
source /workspace/vgcap/vast_env.sh
python scripts/count_entities.py --definition all --runs runs/*/                    # #Ent of every row -> entity_count.json
python scripts/count_entities.py --definition all --pair runs/gate_none_s0 runs/vgcap_s0 \
       --out analysis/entity_stats_s0.json                                            # Section 4.5 + Fig. 2 numbers
```
Then run §9 once more for runs without `scores.json`, and §10.

### 11.2 Final check before destroying the instance
```bash
for r in runs/*/; do [ -f $r/stats.json ] || echo "UNFINISHED $r"; done
for r in runs/*/; do [ -f $r/scores.json ] || echo "NO CLIP $r"; done
python - <<'EOF'
import csv, glob
for f in sorted(glob.glob("runs/*/submission.csv")):
    rows = list(csv.DictReader(open(f, encoding="utf-8")))
    bad = sum(1 for r in rows if not r["generated_caption"].strip() or "\n" in r["generated_caption"])
    print(("OK " if len(rows) == 3000 and bad == 0 else "BAD"), f, len(rows), "empty/newline:", bad)
EOF
```
✅ No UNFINISHED and no BAD rows; CLIP present for every run. Then do the final §10 → download → `tar tzf` OK on your PC.

💰 **Only then destroy the instance.** A stopped instance still bills storage.

---

## 12. Not in this session (Session B, needs new code)

| Paper item | Needs |
|---|---|
| App. A sensitivity (τ × K_e, weights, L\*) | train-subset mode + `gt_train.csv` |
| App. B learned selector | train-subset context (InternVL2.5-8B + Llama) |
| App. C latency / peak memory | profiling script |
| Table 4 P/R/F1 | `gt_train.csv` subset; human annotation separately |
| R3.2 masked context | `--mask_rejected` option |

---

## 13. Cost traps and fixes

| Trap | Avoid by |
|---|---|
| Legacy scorer config: Gate 1 prompt gets the whole keyword list back | `CFG=configs/entity_scoring_config_fixed.json` (§0.4); check in §5.4 |
| `pip install -r requirements.txt` compiles `flash_attn` for an hour | Install with `flash_attn` excluded (§3) |
| Scripts default to `cuda:5` | `DEVICE=cuda:0` in `vast_env.sh`; `run` always passes `--device $DEVICE` |
| Llama download includes `original/` (+16 GB) | `--exclude "original/*"` |
| SSH drop kills jobs | Always run inside tmux |
| Re-running a finished tag | Refused by the code; job lists skip finished runs |
| Running GPU jobs one by one / GPU idle between batches | `job_queue.sh` with 4 slots (measure in §5.4 first) |
| Starting Step 2 before the checkpoint | §7.4 decides; Step 2 is ~80% of the cost |
| Destroying before download | §11.2 checklist, then `tar tzf` on the PC |
| Interruptible instance preempted | Rent on-demand |
| Leaderboard quota (~94 left) | Submit only `submissions/` (deduplicated) |

---

## 14. Paper row → run (`new_main.tex` Table 3, P1 + P2)

Report mean ± std over seeds 0–2 for P1 rows. Random = mean of 5 draws.

| Panel | Row (line) | Prio | Run tag(s) |
|---|---|---|---|
| (a) | ENRIC baseline (351) | P2 | `ladder_enric_s0` (entity list, no band, no SGN) · literal "no list": `ladder_nokw_noband_nosgn_s0` |
| (a) | + SGN (352) | P2 | `ladder_kw_noband_s0` · literal "no list": `ladder_nokw_noband_s0` |
| (a) | + prompt band (353) | P2 | `ladder_nokw_band_s0` |
| (a) | + entity list (unfiltered) (354) | P2 | `ladder_kw_noband_s0` |
| (a) | + band + entity list (355) | — | `gate_none_s0/1/2` |
| (a) | + groundedness selection, VG-Cap (356) | — | `vgcap_s0/1/2` |
| (b) | Article order (359) | P1 | `sel_article_order_s0/1/2` |
| (b) | Random, mean of 5 (360) | P1 | `sel_random_d1..d5_s0` |
| (b) | Contextual presence only (362) | P2 | `sel_presence_score_s0` |
| (c) | w/o contextual presence (365) | P2 | `abl_no_presence_s0` |
| (c) | w/o editorial-caption match (366) | P1 | `abl_no_editorial_s0/1/2` |
| (c) | w/o entity-type prior (368) | P2 | `abl_no_type_s0` |
| (c) | Presence required (370) | P2 | `sel_presence_required_s0` |
| (d) | No gate (373) | — | `gate_none_s0/1/2` |
| (d) | Gate 1 only (374) | P1 | `gate_prompt_s0/1/2` |
| (d) | Gate 2 only (375) | P1 | `gate_enrichment_s0/1/2` |
| (d) | Gates 1+2, VG-Cap (376) | P1 | `vgcap_s0/1/2` |
| — | #Ent of every row; §4.5 statistics | P2 | `scripts/count_entities.py` (§11.1) |

Not run (P3): w/o article position, w/o span penalty, type prior only.
