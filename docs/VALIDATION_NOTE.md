# VG-Cap — Validation note (branch `fix/vgcap-correctness`)

Date: 2026-10-27 · Branch HEAD: `1ce3c7f` · Paper source of truth: `VGCap_ACIIDS2027/main.tex`

What was validated:
- Every script needed to produce the paper numbers: `run_ablation.py`, `step_2_entity_scorer.py`, `step_2_entity_selectors.py`, `step_2_caption_normalizer.py`, `configs/*`, `scripts/*`.
- The runbooks in the repo: `VGCAP_TODO.md`, `VGCAP_ACTION.md`, `VGCAP_RUN.md`, `docs/paper_todo_summary.md`.
- All `\todo` items in `VGCap_ACIIDS2027/main.tex`.

Nothing was executed except pure-Python data checks (section 1.3). Every GPU or spaCy run is planned for the server.

---

## 0. Summary

| Area | Verdict |
|---|---|
| Input data for draft reuse (drafts, merge file, retrieval CSV) | ✅ Consistent. 3000/3000 IDs aligned, same order, no empty drafts, every preamble is handled |
| Driver `run_ablation.py` (4 gates, selectors, fix switches, seeds) | ✅ Logic is right. ⚠️ **No overwrite guard, non-atomic checkpoint, seed not resume-safe** |
| Table 3 (a)(b)(c)(d) scripts | ✅ Every row has a command. ⚠️ No "w/o span" config for the fixed scorer. The `unified` path costs about 19 GPU runs |
| Appendix A script | ❌ **`run_appendix.sh unified` aborts on its first command.** The L\* sweep does not shift the prompt band, although the paper says it does |
| Appendix D (`make_qualitative.py`) | ❌ **Wrong output.** The selector debug file has no `features`, so every bucket is mis-filled and no score breakdown is available |
| Regression test `test_fixes.py` | ❌ **It does not prove legacy equivalence.** Check [1] is skipped because its backup file is gitignored and absent. Check [2] compares the new code with its own output |
| Evaluator / bootstrap | ⚠️ No Overall CI (the paper asks for it). Per-query CIDEr needs ground truth. CLIP model/protocol not yet validated against 0.858 |
| Seeds (paper: mean of 3 seeds) | ⚠️ Possible, but reuse runs are identical across seeds. There is no seed aggregation (mean ± std) |
| Table 4, App. B, App. C, R3.2 masking, figures | ❌ Missing code. See section 4 |
| `requirements.txt` | ⚠️ Missing `spacy`, `jinja2`, `python-dotenv`, `scikit-learn`, `numpy`. spaCy model version not pinned |

### 0b. Team answers (2026-10-27) and what they change

| Question | Answer | Consequence |
|---|---|---|
| Public-test GT / scoring tool | None. Only the submission page and the Overall formula in the paper | CIDEr, CLIP and Overall come **only** from the leaderboard. No per-query scores, so the paper's "paired bootstrap over queries" for CIDEr/Overall cannot be done on the test set. Rewrite the statistical protocol: std over 3 seeds from leaderboard numbers, plus bootstrap on local CLIP or on train-subset CIDEr |
| Leaderboard quota | ~100 submissions, 5–6 used → **~94 left** | The full plan is ~85–90 CSVs, which nearly exhausts it. Budget table in `VGCAP_ACTION.md` B1. **Appendix A moves to a train subset scored locally** (0 submissions) |
| Official CLIP model | Unknown | Local CLIP is a screening proxy only. Calibrate it on the 4 anchors (local vs leaderboard CLIP) |
| Train GT | Yes: `gt_train.csv` on Drive | Enables App. A (local CIDEr), App. B, Table 4 P/R/F1 (on a train subset), and a direct SAR.Q5 check (GT vs editorial caption). Needs Stage-2 context + drafts for ~500 train queries (N11) |
| Original #Ent script | Lost | Recount **every** §4.5 number and #Ent cell with one frozen definition (N9); redraw Fig. 2 |
| Paper source of truth | `VGCap_ACIIDS2027/main.tex` | `VGCAP_ACTION.md` Part C is now renumbered against it. Do not edit the repo copy |

Two facts traced from git and the old scripts:
- **`version1.json` drafts (ENRIC 0.6097, +SGN 0.6124) were generated with the Keywords block.** The file is byte-identical to `llama.json` at commit `c004e52`, whose template has Keywords and no 100–120 band ("at least 5 sentences"). So the published ladder's 0.6124 → 0.6166 step changes **only the length band**, not entity conditioning. The ladder (a) needs a 2×2 rebuild (keywords × band); see `VGCAP_TODO.md` §2 and N13.
- **`generated_caption` is very likely InternVL2.5-8B.** That is the default in `step_2_create_caption_query.py`, and its log loads an InternLM2 language model; the 1B/4B variants and InternVL3-1B use Qwen2.

**Decisions to make before any GPU time** (unchanged from `VGCAP_ACTION.md` Part D):
1. Unified gate or rewrite §3.6.
2. Legacy or fixed scorer.
3. Retrieval re-ranking: sort, or state that it is a no-op.

Run `gate_unified_s0` first (one 6 h GPU run) together with `run_fixes.sh` (CPU only). Their results settle decisions 1 and 2.

---

## 1. Where each phase runs

### 1.1 Server (GPU / heavy CPU), the only place the pipeline runs
These steps need the server:
- All `run_ablation.py` runs. Even draft-reuse runs need spaCy over 3000 articles, about 10–15 min each.
- Generation: Llama-3-8B, about 6 h per run sequentially.
- CLIPScore.
- Profiling.
- Train-split context building.

### 1.2 Local machine: analysis and summary only (no GPU, no project venv)
Local work reads the `runs/` folder copied back from the server. The scripts below import only the standard library at the top level, so a plain Python ≥ 3.9 is enough. No `pip install` is needed.

| Task | Command (local) | Needs |
|---|---|---|
| Table of all runs + LaTeX rows | `python scripts/collect_results.py --latex` | `runs/*/stats.json`, `scores.json` |
| Deduplicate CSVs + submit order | `python scripts/make_submissions.py --runs runs --out submissions` | `runs/*/submission.csv` |
| Record leaderboard score + Overall consistency check | `python scripts/record_score.py --run runs/<tag> --overall .. --cider .. --clip ..` | — |
| Paired bootstrap (CLIP; CIDEr if GT) | `python scripts/bootstrap.py --a runs/A --b runs/B --metric clip` | `per_query_scores.json` |
| CIDEr-D locally, **only if GT exists** | `python scripts/evaluate.py --run runs/<tag> --gt <gt.json> --clip <value from server>` | GT file. Passing `--clip` skips torch |
| Post-run validation *(new, section 4 N2)* | `python scripts/validate_run.py runs/<tag>` | stdlib only |
| Seed aggregation, paper-ordered tables *(new, N8)* | `python scripts/aggregate_results.py` | stdlib only |
| Annotation pack + κ *(new, N10)* | `python scripts/annotation_kit.py build/score` | stdlib only |
| Figures *(new, N9)* | `python scripts/make_figures.py` | `matplotlib` (the only optional local install) |

**Not local:** `run_ablation.py`, `eval_entities.py`, `train_learned_selector.py`, `test_fixes.py`, and CLIP scoring. They need spaCy, sklearn or torch, so run them on the server.

Sync pattern, server → local, after each phase:
```bash
# on server: pack only what analysis needs (llm_raw.json is also kept: it is the expensive artifact)
tar czf runs_phaseN.tgz runs/*/{config.json,env.json,effective_entity_config.json,stats.json,scores.json,per_query_scores.json,submission.csv,selected_entities.json,entity_debug.json,normalized.json,llm_raw.json}
# local
scp server:.../runs_phaseN.tgz . && tar xzf runs_phaseN.tgz
```

### 1.3 Data checks already done (pure Python, local)
| Check | Result |
|---|---|
| Drafts `..._llama.json` / `..._version1.json` | 3000 / 3000 entries, 0 empty |
| IDs: drafts = merge = retrieval CSV | identical sets; merge order == CSV order |
| Retrieval CSV columns | `query_id, article_id_1..10, generated_caption` |
| Preambles not covered by `normalize_llm_output` | 0 / 3000 in both files. About 2780 start with "Here is the caption:", which is handled |
| Trailing notes ("Note:", "(Word count…)") | 0 |
| Raw draft length (whitespace split, preamble included) | 101.5 / 139.3 words. Minus the 4-word preamble this is 97.5 / 135.3, matching the docs (97.4 / 135.2) |
| Merge records with empty `crawl_caption` | 19 |

SHA-256 prefixes of the reused inputs. Verify them on the server before any reuse run; the preflight in N1 does this automatically:
```
9cd8af35ab95b694  assemble_result/cot_5_things_fact_more_event_llama.json
66eb1611c34917c5  assemble_result/cot_5_things_fact_more_event_llama_version1.json
a588dea79a676eb8  public_test_final_elements_json/final_merge_result.json   (gitignored! copy it manually)
6308b803e2338f9f  final_csv_result/public_final_retrieval_merging_final_results.csv
782f89011ac332c7  logit_scale.pt
```

---

## 2. Paper TODO ↔ code coverage

Line numbers refer to `VGCap_ACIIDS2027/main.tex`.

> The repo's own `main.tex` is a different copy: 596 lines, 68 `\todo`, with `\aciids` macros. The team confirmed the paper folder copy (579 lines, 61 `\todo`) is the source of truth. `VGCAP_ACTION.md` Part C has been renumbered against it.

| Paper line | TODO | Code status | Where it runs |
|---|---|---|---|
| 150 | Re-draw Fig. 1 | Manual. Depends on decision 1 (if gate = enrichment only, the arrow scorer → prompt is wrong) | local |
| 187 | λ, softmax temperature, 3 detector thresholds | **Text fix.** Code has no λ and no temperature: fusion = `logit_scale·cos_text + cos_img`. The detector is a precomputed list of 141 queries (`final_json_result/temp_three_ways_wrong_samples_set.json`, fields `entropy/gap/low_top1_sim`). Re-ranking is never sorted. Optional re-run: `step_1_rerank.py --sort_reranked --fusion_lambda` | server (GPU, InternVL) |
| 192 | "w/o editorial evidence" row | ✅ `abl_no_editorial_*` (CPU with gate enrichment) | server |
| 261 | Release prompt template | ✅ `scripts/dump_prompts.py`. ⚠️ The instruction quoted in the paper does not match `cot_5_things_fact_more_event*.j2:34-35`: no "strictly", no "Each sentence must add" | server or local with jinja2 |
| 273 | R3.2 variant masking rejected entities in summary/QA | ❌ **Missing**, see N6 | server (GPU) |
| 289 | CIDEr to 4 decimals | Manual (leaderboard decimals) + `record_score.py` | local |
| 295 | ReCap 0.61619 vs 0.5987 | Manual (literature) | — |
| 298 | 3 seeds, std, bootstrap CI | ⚠️ Partial: `--seed` exists. No seed aggregation (N8). Reuse runs do not vary with the seed, so new drafts per seed are needed (section 5, phase 5). No public GT → per-query bootstrap of CIDEr/Overall is impossible on test; **paper text must change** (section 0b) | server + local |
| 327 | CI of CIDEr **and Overall** gain over ENRIC | ❌ Not possible on test (no GT, leaderboard gives totals only). Report seed std instead; N7 applies only to the train subset | local |
| 340-345 | Table 3(a): #Ent of first two rows + 2 new rows | ⚠️ ENRIC/+SGN rows already had keywords (section 0b). "+ unfiltered entities only" = the existing 0.6124 row. The 2 new rows become "no keywords, no band" and "no keywords, band" (2 GPU runs, N13). #Ent needs a new definition (N9) | server |
| 348-351 | Table 3(b) selectors | ✅ `sel_*` with `--budget_match per_query`. ⚠️ The paper caption still says "matched budget K_e=10"; change it to per-query matched budget | server |
| 354-358 | Table 3(c) leave-one-out | ✅ `abl_*`. ⚠️ "w/o span" semantics (U5) | server |
| 361-362 | Table 3(d) gate location | ✅ `gate_prompt`, `gate_unified` (GPU each) | server |
| 394-401 | Table 4 P/R/F1, Unsup., Leak, human 1–5 | ⚠️ P/R/F1 only on a train subset with `gt_train.csv` (`eval_entities.py`); change "over all 3,000 queries". Leak in `stats.json` (buggy matching, U4). Annotation pack and κ missing (N10) | server + local + humans |
| 420 | Fig. 2a axis label | ❌ No figure script in the repo (N9) | local |
| 432 | Fig. 3 labels | Manual image edit | local |
| 483-486 | App. A τ × K_e | ✅ `run_appendix.sh` A.2 (CPU, gate enrichment). Run on the **train subset** and score locally to save quota (needs `run_ablation.py` to accept a train merge file and drafts) | server |
| 502-505 | App. A weight ×0.5/×1.5, ±20 % × 10 | ✅ A.3, same train-subset plan | server |
| 515-519, 525 | App. A L\* sweep + plot | ⚠️ Prompt band not shifted (U3). No plot (N9) | server + local |
| 530 | App. B learned selector | ⚠️ Script exists, but no train-split context file exists (N11). Label matching is buggy (U8) | server (GPU) |
| 544-554 | App. C latency + peak memory | ❌ `stats.json` only has whole-run wall time; no per-query time, no peak memory, no retrieval stage (N4) | server (GPU) |
| 559 | App. D 6–8 examples with score breakdown | ❌ `make_qualitative.py` gives wrong buckets (U2) | server → local |
| 564 | Repo URL, seeds, annotations | Manual | — |

---

## 3. UPDATE section — fixes to existing code

Ordered by severity. "Pub-safe" means the fix does not change any published number, so the legacy reproduction still holds.

### 3.1 Safety and saving (do these first; they protect the GPU hours)

**U1. No overwrite guard; non-atomic checkpoint; resume breaks the seed.** Pub-safe.
- [run_ablation.py:378-380](../run_ablation.py#L378-L380) uses `mkdir(exist_ok=True)`. A second call with the same `--tag` silently overwrites everything. Without `--resume`, [run_ablation.py:209](../run_ablation.py#L209) regenerates and overwrites `llm_raw.json`, losing a 6 h draft.
- `run_panels.sh` always re-runs `gate_unified` even if you already ran it in the decision step.
- [run_ablation.py:228,230](../run_ablation.py#L228) write the checkpoint in place. A kill during the write leaves truncated JSON, and `--resume` then crashes in `json.loads`.
- [run_ablation.py:205](../run_ablation.py#L205) calls `set_seed` once before the loop. After a resume the RNG stream differs, so "seed 1" is not reproducible.

```python
# run_ablation.py — new flags
ap.add_argument("--overwrite", action="store_true", help="allow replacing a finished run")
ap.add_argument("--skip_if_done", action="store_true", help="exit 0 if runs/<tag>/stats.json exists")
...
done = run_dir / "stats.json"
if done.exists() and args.skip_if_done:
    print(f"[{args.tag}] already done, skipping"); return
if (done.exists() or (run_dir / "llm_raw.json").exists()) and not (args.overwrite or args.resume):
    raise SystemExit(f"{run_dir} already has outputs; pass --resume, --skip_if_done or --overwrite")

def atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)          # atomic on POSIX

# generate_drafts: per-query seed -> order/resume/batch independent
import zlib
for i, batch in enumerate(...):
    ...
    set_seed(args.seed * 1_000_003 + zlib.crc32(qid.encode()))
    drafts[qid] = assembler.assemble(prompt)
```
Then pass `--skip_if_done` from the `run()` helper in `run_panels.sh`, `run_fixes.sh` and `run_appendix.sh`.

**U1b. Provenance is not saved.** Pub-safe. `config.json` stores the arguments only. Add `runs/<tag>/env.json` containing:
- `git rev-parse HEAD` and `git status --porcelain`
- python, torch, transformers and spacy versions; `spacy.load(...).meta["version"]` for `en_core_web_sm`
- GPU name, `CUDA_VISIBLE_DEVICES`, hostname
- the HF model revision
- sha256 of `--reuse_drafts`, `MERGE_DATA_PATH` and `RETRIEVAL_CSV`

Without these, a number in the paper cannot be traced to code.

**U1c. Silent data gaps.** Pub-safe.
- [run_ablation.py:422](../run_ablation.py#L422) silently drops queries missing from the reuse file. Assert that `len(drafts) == len(merge_data)`.
- [step_2_caption_normalizer.py:347](../step_2_caption_normalizer.py#L347) uses `query_dict.get(query_id, {})`, so a missing record means an empty article and no enrichment, with no warning. Count these and fail if any exist.
- With `--limit`, `write_submission` still writes 3000 rows with empty captions. Mark `"limited": true` in `stats.json`. Never submit or locally score such a CSV as a full run: CIDEr's IDF depends on the evaluated subset.

**U1d. Save paths and weights.** This pipeline trains no network weights. What must be saved:

| Artifact | Path | Note |
|---|---|---|
| LLM / VLM weights | HF cache | Set `HF_HOME=/data/<big disk>/hf` on the server. Pin `revision=` (commit hash) in `llama3.py:52-60` for `meta-llama/Meta-Llama-3-8B-Instruct` (gated: needs `HUGGING_FACE_TOKEN` in `.env`) and for the InternVL models |
| Re-rank coefficient | `logit_scale.pt` | committed |
| Scorer config of each run | `runs/<tag>/effective_entity_config.json` | already written, but only when gate ≠ none |
| Learned selector "weights" | `configs/learned_selector.json` | commit it with the train-query ID list (N11) |
| Drafts (expensive) | `runs/<tag>/llm_raw.json`, `assemble_result/*.json` | `runs/` is **gitignored**. Back it up after every GPU run. Make `assemble_result/*.json` read-only (`chmod a-w`) |

### 3.2 Correctness of results

**U2. `make_qualitative.py` output is wrong (App. D).** Pub-safe. [step_2_entity_selectors.py:154-162](../step_2_entity_selectors.py#L154-L162): `_pack()` writes `scored_entities` without `features`. `make_qualitative.py` reads `e.get("features", {})`. As a result, every selected entity looks "admitted without evidence", the "wrongly removed" bucket is always empty, and no per-entity score breakdown is available.
```python
# _pack(): add the breakdown
{"entity": e.text, "label": e.label, "score": round(e.score, 2),
 "features": {k: round(v, 2) for k, v in e.features.items()},
 "selected": id(e) in selected_ids}
```

**U3. `run_appendix.sh` breaks for `unified`, and the L\* sweep does not match the paper.**
- [scripts/run_appendix.sh:31-36](../scripts/run_appendix.sh#L31-L36) always passes `--reuse_drafts`. `run_ablation.py` rejects that for `--gate unified`, and `set -e` then aborts the whole script. Use `"${REUSE[@]}"` as A.2 and A.3 do.
- The tags `len_${L}_s${SEED}` lack `${GATE}`, so runs of the two gates collide.
- The paper (Table A2 caption) says "prompt band shifted accordingly". The template hard-codes `100-120` ([cot_5_things_fact_more_event_ablation.j2:34](../assemble_caption_prompt_template/cot_5_things_fact_more_event_ablation.j2#L34)), so the current sweep is "normalizer only". Two options:
  - change the caption text, at no cost; or
  - implement N5 and regenerate drafts: 4 GPU runs.

**U4. Leakage and #Ent definitions (Table 3 #Ent, Table 4 Leak).**
- [run_ablation.py:253](../run_ablation.py#L253) and [scripts/eval_entities.py](../scripts/eval_entities.py) test leakage with raw substring `e.lower() in caption`, so `One` matches "someone".
- Under the legacy scorer, `rejected_entities` contains the ~300-char keyword blobs. They never match, so they inflate the denominator and deflate Leak. The published 0.0808 was computed over 221,659 rejected entities, blobs included.
- Fix: word-boundary match, and exclude spans > 150 chars or letterless spans from the denominator. Report both numbers once, then keep one.
- `#Ent` ([run_ablation.py:248](../run_ablation.py#L248)) is "unique lowercase, all spaCy labels", which does not reproduce the paper's 7.59 / 6.76 (see `VGCAP_TODO.md` §12). Freeze one definition in code (N9) and state it in §4.5.

**U5. Panel (c) "w/o span penalty" semantics.**
- Legacy scorer: the span penalty never fires, so "w/o span" equals the VG-Cap reference row. No run is needed; reuse the row and say so.
- Fixed scorer (`CFG_SUFFIX=_fixed`): `ablation_span_penalty_*_fixed.json` turn the penalty **on**, which equals `fix_all`, and **no "w/o span" config exists**.
- Add a variant to [scripts/make_ablation_configs.py](../scripts/make_ablation_configs.py):
```python
"ablation_no_span": ("Table 3(c) w/o span penalty.", [], {}, {}, {"span_penalty": "off"}),
```
  and add `no_span` to the panel (c) loop in `run_panels.sh`.

**U6. `test_fixes.py` does not prove legacy equivalence.**
- Check [1] needs `backup_pre_fix/step_2_entity_scorer.py.bak`, which is gitignored and not in the repo, so the check prints SKIP.
- Check [2] compares against `runs/repro_vgcap/selected_entities.json`, which **the new code itself** produced. That comparison is circular.
- The "300/300 identical" in the commit message was measured with mocked NER.
- Fix: restore the pre-fix scorer from git and re-run with real spaCy on the server:
```bash
mkdir -p backup_pre_fix
git show 9a23067:step_2_entity_scorer.py > backup_pre_fix/step_2_entity_scorer.py.bak
python scripts/test_fixes.py --n 3000        # [1] must PASS before trusting any legacy row
```

**U7. Evaluator assumptions** ([scripts/evaluate.py](../scripts/evaluate.py)). Validate these before trusting any local number:
- CLIP model defaults to `openai/clip-vit-base-patch32`, truncated at 77 tokens. The 104-word captions are cut. It is unknown whether EVENTA used the same model. Gate: `repro_vgcap` must give CLIP ≈ 0.858 (`--expect '{"clip":0.858}' --tolerance 0.003`). If not, find the official setting before scoring 80 runs.
- The CIDEr tokenizer is a regex, while pycocoevalcap uses PTB. IDF is computed over the evaluated set, so subset scores are not comparable.
- AP/R1/R10 are hard-coded (0.992/0.985/1.000). This is only valid while retrieval is unchanged; it is wrong if decision 3 = `--sort_reranked`.
- [scripts/score_all.sh](../scripts/score_all.sh) requires `GT` (`${GT:?}`), so it cannot run in leaderboard-only mode. It also skips any run that already has `scores.json`, even if the run was regenerated. Compare mtimes, or store the submission hash in `scores.json`.

**U8. `train_learned_selector.py` label bug.** The label is `entity.text.lower() in reference`, a raw substring with the same `one`/`someone` problem. Use a word-boundary regex. With the legacy config, the features also inherit the blob and position bugs. Train on the fixed config if decision 2 = fixed.

### 3.3 Setup and config

**U9. `requirements.txt` is incomplete** (server env). Add:
```
spacy==3.8.*            # pin the exact version used for the published run if known
jinja2
python-dotenv           # llama3.py:5 imports dotenv
numpy
scikit-learn            # App. B only
# then: python -m spacy download en_core_web_sm-<version>   (pin! NER drives E*, so a model change changes results)
```

**U10. Doc drift** (low):
- `configs/entity_scoring_config.json` `_comment` points to `vgcap_span_penalty_on.json`. The real file is `ablation_span_penalty_code.json`. (The same wrong name in `VGCAP_TODO.md` and the stale `create_submission()` entry are now fixed in the md files.)

**U11. Minor script robustness.**
- `run_panels.sh` and `run_fixes.sh` expand `"${LIMIT_ARG[@]}"` under `set -u`. This fails on bash < 4.4; fine on a modern Linux server.
- With `GATE=unified`, `run_panels.sh` regenerates **every** `sel_*` and `abl_*` run (about 19 GPU runs, ≈ 115 h sequential). Make the cost explicit before launching.

---

## 4. NEW CODE section — for the paper TODOs that have no code yet

| ID | File (new) | Covers paper TODO | Runs on | Pub-safe |
|---|---|---|---|---|
| N1 | `scripts/preflight.py` | auto safety check before any phase | server (+ local) | yes |
| N2 | `scripts/validate_run.py` | auto result check after every run | server + local | yes |
| N3 | env capture in `run_ablation.py` (U1b) | App. E reproducibility | server | yes |
| N4 | `scripts/profile_efficiency.py` | App. C Table (544-554) | server GPU | yes |
| N5 | `--band_min/--band_max` in template + driver | App. A L\* sweep "band shifted" | server GPU | yes |
| N6 | `--mask_rejected` in `compose_input` | R3.2 (273) | server GPU | yes |
| N7 | `bootstrap.py --metric overall` | 298, 327 CI of Overall | local | yes |
| N8 | `scripts/aggregate_results.py` | 298 mean ± std over 3 seeds; paper-ordered LaTeX | local | yes |
| N9 | `scripts/count_entities.py` + `scripts/make_figures.py` | #Ent column, Fig. 2a/2b (420), L\* plot (525) | server (spaCy) → local (plot) | yes |
| N10 | `scripts/annotation_kit.py` | Table 4 Unsup., κ, human 1–5 (380, 401) | local | yes |
| N11 | `scripts/build_train_context.py` | App. B (530) | server GPU | yes |
| N12 | batched generation (optional) | wall time of every GPU phase | server GPU | changes sampling, new runs only |
| N13 | ladder (a) prompt variants | 340–345, 371 (R3.6/R4.2) | server GPU | yes |
| N14 | train-subset mode + SAR.Q5 check | App. A/B, Table 4 P/R/F1, 192 | server (+ local scoring) | yes |

### N1 `scripts/preflight.py` (exit ≠ 0 blocks the phase)
Checks:
1. Required files exist and their sha256 matches section 1.3. This catches a stale or partial `final_merge_result.json`, which is gitignored and copied by hand.
2. Query IDs of drafts = merge = retrieval CSV, with the same order and 3000 each.
3. Every `configs/*.json` loads. Weight keys must be a subset of the feature names the scorer actually reads. This would have caught `length_penalty_factor` vs `length_penalty`.
4. `en_core_web_sm` version equals the pinned one.
5. `HUGGING_FACE_TOKEN` is present for GPU phases. `torch.cuda.mem_get_info()` reports ≥ 20 GB free on `$DEVICE`.
6. Free disk ≥ 5 GB for `runs/`.
7. Planned tags do not collide with finished runs unless `--skip_if_done` is set.
8. `git status --porcelain` is clean. Otherwise warn that the run is not traceable.

### N2 `scripts/validate_run.py runs/<tag> [--anchor repro]`
Post-run gates. All are stdlib checks over files the run already writes:

| Check | Fail condition |
|---|---|
| `submission.csv` | ≠ 3000 rows; query_id set or order ≠ retrieval CSV; any `article_id_*` cell differs from the template |
| captions | empty, contains `\n`, starts with a preamble, does not end in `.!?` |
| length | `stats.mean_length_words` outside `expected_length ± 1.5` (SGN runs); max > `expected_length` |
| selection (gate ≠ none) | `selected_entities.json` missing or wrong size; any query with > `top_k` entities; an E\* entity missing from that query's `entity_debug` candidates |
| per-query budget runs | `len(E*)` ≠ reference budget for any query |
| consistency | `config.json` tag ≠ dir name; `stats.json` missing; `env.json` git SHA differs between runs of one table (warn) |
| anchors (`--anchor repro`) | `repro_vgcap` Len ≠ 104.01 ±0.05, `repro_unfiltered` Len ≠ 104.02 ±0.05, changed captions between them ≠ 2119 ±5 |
| duplicates | caption hash equals another run that should differ (e.g. an `abl_*` identical to the reference, except the legacy `no_span`) |

Writes `runs/<tag>/validation.json`. `collect_results.py` should print ✗ next to any run without a passing validation.

### N4 `scripts/profile_efficiency.py --n 100 --device cuda:0`
For N sampled queries, time each stage with `torch.cuda.synchronize()` and `torch.cuda.reset_peak_memory_stats()` / `max_memory_allocated()`:
1. Retrieval: InternVL-14B-224px query encoding plus similarity over the DB embeddings.
2. Re-ranking: InternVL2.5 captioning plus fusion, for uncertain queries only. Report both "per uncertain query" and "amortized over 3000".
3. Entity scoring (CPU).
4. Generation: `LLMAssembler.assemble`. Also log the **generated token count**, which answers "cost of the 1,500-token budget".
5. Normalization (CPU).

Output `runs/efficiency/efficiency.json` plus LaTeX rows for Table C. Reuse the model classes in `internvl.py` and `llama3.py`; do not re-implement them.

### N5 Prompt band as a parameter (L\* sweep)
```jinja
{% if use_length_band %}The caption must be between {{ band_min }}-{{ band_max }} words in length.
```
Add `band_min`/`band_max` to `compose_input` (defaults 100/120), so the default render stays byte-identical. Add `--band_min/--band_max` to `run_ablation.py`, and refuse `--reuse_drafts` when the band differs from 100–120. Sweep: L\* ∈ {80, 90, 110, 120} with band (L\*−4, L\*+16), the same offsets as 104 → 100–120. That is 4 GPU runs.

### N6 Masked-context variant (R3.2)
`--mask_rejected {none,summary_qa,all_context}` in `compose_input`. Before rendering, delete every rejected entity (word-boundary, longest first) from `summary.fact` and `question_answer`. With `all_context`, also delete them from `crawl_caption` and `generated_caption`. Only valid with `--gate prompt|unified`, because it needs E\* and changes the prompt (1 GPU run per variant). Report Leak next to the gate rows.

### N7 Bootstrap of Overall
In `scripts/bootstrap.py`, add `--metric overall`. Each resample draws the same query indices for A and B, computes mean CIDEr and mean CLIP per system, and then `overall_score(ap, r1, r10, clip, cider)` with the fixed retrieval metrics. The delta CI comes from those values. This needs per-query CIDEr, so it needs **GT**. Without GT, only the CLIP CI can be reported; say so in the paper.

### N8 `scripts/aggregate_results.py`
- Group tags by stripping `_s<seed>` / `_d<draw>`, then report mean ± std (pstdev) over seeds or draws.
- A `PAPER_ROWS` map (tag → table, row label, order) emits `tables/table3.tex`, `tableA1.tex` and `tableA2.tex` in paper order. CIDEr is printed to 4 decimals.
- Missing cells are printed as `\todo{}`.
- Runs without a passing `validation.json` are refused.

### N9 Entity counting + figures
- `count_entities.py --a runs/gate_none_s0 --b runs/vgcap_enrichment_s0` (server, spaCy) applies **one** frozen #Ent definition. It writes `#Ent` for every run, per-caption differences, gross removed/added, and removals by label to `analysis/entity_stats.json`.
- `make_figures.py` (local, matplotlib) regenerates `entity_gap_distribution.pdf` with the correct 0.6219 label, `entity_gap_by_type.pdf`, and the L\* plot from `analysis/*.json`.

### N10 `scripts/annotation_kit.py`
- `build`: samples 200 queries with a fixed seed. Puts the captions of ENRIC, Unfiltered and VG-Cap in **random order with system names hidden** (the paper promises "blind, random order"). Writes `annotation/sheet_<annotator>.csv` plus a private `annotation/key.json`.
- `score`: reads the filled sheets. Computes the Unsup. rate per system, Cohen's κ for the 2 entity annotators, **Fleiss' κ** for the 3 raters (the paper currently says Cohen), and mean Fact./Flu./Inf.
- Write the guideline `annotation/GUIDELINE.md` (visual / contextual / unsupported, with examples) at the same time.

### N11 Train-split context for App. B
- The learned selector needs `final_merge_result.json`-shaped records for **held-out training queries**: InternVL caption, editorial caption, Llama summary, QA and keywords. None exists.
- Write `build_train_context.py --n 500 --seed 0`. It samples train query IDs, saves the list to `configs/learned_selector_train_ids.json`, and calls the existing Stage-2 context functions with output paths parameterized. The current scripts hard-code public-test paths.
- Then run `train_learned_selector.py` (with U8 fixed) and `run_ablation.py --selector learned` on the public test once.

### N12 Batched generation (optional)
- Add `--batch_size` to `generate_drafts`: left padding, `generator(prompts, batch_size=B)`, keeping the per-query seeding from U1. Alternatively use vLLM.
- Expected speed-up is 5–10× on an A100 for 8B. This is an estimate, not measured.
- Use it only for **new** runs. It changes the samples, so never mix batched and sequential drafts within one comparison.

### N13 Ladder (a) prompt variants
- The no-band branch of `cot_5_things_fact_more_event_ablation.j2` drops the sentence instruction entirely. The `version1.json` prompt (commit `c004e52`) had "at least 5 sentences". Add that line to the no-band branch so a no-band run is comparable with the 0.6124 row.
- In `run_panels.sh`:
  - replace `ladder_entities_only` (keywords, no band), which duplicates the 0.6124 row, with `ladder_no_kw_no_band` (`--no_prompt_entities --no_prompt_band`, generated);
  - keep `ladder_band_only` (no keywords, band);
  - drop `--no_prompt_entities` from `ladder_enric`/`ladder_sgn`, where it only mislabels `config.json`.

### N14 Train-subset mode + SAR.Q5
- `scripts/prepare_train_gt.py` (local or server, stdlib): convert `gt_train.csv` → `data/train_gt.json` (`{query_id: caption}`), and sample N held-out IDs with a fixed seed → `data/train_subset_ids.json`.
- `run_ablation.py`: add `--merge` and `--retrieval_csv` flags (today both paths are hard-coded constants), so the whole driver runs on the train-subset merge file from N11. Every App. A config then reuses one set of train drafts (~1 GPU hour for 500 queries, estimate) and is scored locally with `evaluate.py --gt data/train_gt.json`. CIDEr IDF is computed over the subset, so compare only within the subset.
- `scripts/check_editorial_overlap.py` (SAR.Q5): for train images, compare each GT caption with that image's editorial caption (`crawl_caption`). Report exact-match rate, containment rate and mean unigram/bigram overlap. High overlap means references may derive from editorial captions, and the "w/o editorial evidence" row becomes a required validity check.

---

## 5. Runtime estimate (server, one A100 80 GB)

Measured inputs (from repo logs and docs): sequential generation **6 h 05 / run**; draft-reuse run **10–15 min** on 8 CPU cores. CLIP scoring and train-context times below are **estimates**.

| Phase | Runs | GPU | CPU | Wall time (sequential) |
|---|---|---|---|---|
| P0 preflight, U6 test, 4 anchor runs (`repro_*`) | 4 | 0 | ✓ | ~1 h |
| P1 `run_fixes.sh enrichment` (panel e) | 10 | 0 | ✓ | ~2–2.5 h |
| P2 decision run `gate_unified_s0` | 1 | ✓ | | ~6 h |
| P3 `run_panels.sh enrichment` (with `--skip_if_done`) | 22 (18 CPU + 3 new GPU) | ✓ | ✓ | ~18 h GPU + 3–5 h CPU |
| P3b panel (c) on fixed scorer (if decision 2 = fixed) | 7 | 0 | ✓ | ~1.5 h |
| P4 App. A on a 500-query train subset (N14): one draft set + 37 CPU runs, scored locally | 1 + 37 | ✓ | ✓ | ~1 h GPU + ~2 h CPU (estimate); **0 leaderboard submissions** |
| P4b L\* with band shift (N5) | 4 | ✓ | | ~24 h |
| P5 true 3 seeds: new band and no-band drafts for s1, s2 + the 4 GPU configs × 2 | 4 + 8 | ✓ | | ~72 h |
| P6 R3.2 masking (N6) | 1–2 | ✓ | | 6–12 h |
| P7 App. B: train context 500 q (N11) + selector | 1 | ✓ | ✓ | ~2–4 h (estimate) |
| P8 App. C profiling (N4) | — | ✓ | | ~1 h |
| P9 CLIP scoring of every unique CSV (~80) | — | ✓ | | ~4–7 h (estimate) |
| **Total GPU, minimal paper (P2+P3+P8+P9)** | | | | **≈ 30–35 h** |
| **Total GPU, everything incl. 3 seeds, L\* band, R3.2** | | | | **≈ 140–150 h** (÷5–10 for generation with N12) |

Notes:
- CPU-only runs are independent and can run in parallel: `xargs -P 4`, one spaCy process each.
- P3 under `GATE=unified` instead: about 19 GPU runs ≈ 115 h. Decide after P2.
- Human annotation (Table 4) takes 1–2 weeks of calendar time. Start the guideline now, in parallel with everything else.

---

## 6. Run order checklist

```
[ ] U1, U1b, U1c, U2, U3, U5, U6, U9 merged (all pub-safe)        local edit -> push -> pull on server
[ ] N1 preflight + N2 validate_run written                        local edit
[ ] SERVER P0: preflight; test_fixes [1] PASS with real spaCy; 4 anchors; validate_run --anchor repro
[ ] SERVER P0: evaluate.py repro_vgcap --expect clip 0.858 (U7)  -> if FAIL, stop: fix CLIP protocol first
[ ] submit 4 anchor CSVs -> 0.6097 / 0.6124 / 0.6166 / 0.6219 ?   local: record_score.py  (quota: ~94 left)
[ ] calibrate local CLIP vs leaderboard CLIP on the 4 anchors
[ ] copy gt_train.csv from Drive -> server data/; N14 prepare_train_gt + SAR.Q5 overlap check
[ ] N11 train context for ~500 queries (also feeds App. A, App. B, Table 4 P/R/F1)
[ ] SERVER P1 + P2 (in parallel: CPU + GPU)                       -> decisions 1 and 2
[ ] SERVER P3 (+P3b)                                              -> validate_run on all; sync runs/ to local
[ ] LOCAL: make_submissions -> submit -> record_score -> aggregate_results -> tables/*.tex
[ ] SERVER P4, P8, P9; P5/P4b/P6/P7 as time allows
[ ] LOCAL: count_entities output -> make_figures; annotation_kit build/score; paper text fixes (VGCAP_ACTION Part C)
[ ] back up runs/ (gitignored) after every GPU phase
```
