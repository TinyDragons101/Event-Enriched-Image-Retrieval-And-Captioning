# `new_main.tex` vs `main.tex`: changes and impact on the pipeline

Date: 2026-10-27 · Files: `VGCap_ACIIDS2027/main.tex` (579 lines, 61 `\todo`) → `VGCap_ACIIDS2027/new_main.tex` (598 lines, 68 `\todo`).
Line numbers below refer to **`new_main.tex`**.

## 0. What `new_main.tex` is

- Ignoring line endings (`new_main.tex` uses CRLF), it is **identical to `Event-Enriched-Image-Retrieval-And-Captioning/main.tex`** (the copy in the code repo), except for the title:
  - old: *Grounding What Matters: Selective Entity Conditioning…*
  - new: *VG-Cap: Entity-Gated Generation for Event-Enriched News Image Captioning*
- It differs from `VGCap_ACIIDS2027/main.tex` in ~200 lines (all marked `\aciids{…}` or `\prio{…}`).
- The original `VGCAP_ACTION.md` Part C was written against this text. Its line numbers were later remapped to the older `main.tex` and now need remapping to `new_main.tex`.
- ⛔ The LaTeX build will fail: `figures/overview.png` and `figures/qualitative_results.png` are referenced but **missing** from `VGCap_ACIIDS2027/figures/`.

## 1. What the supervisor changed

| Area | Change | Lines |
|---|---|---|
| Framing | New title; module renamed **Entity-Gated Length-Controlled Generation** (was Two-Stage Length Control) with Step 1 / Step 2; "Gate 1" (prompt) and "Gate 2" (normalization) made explicit; "peripheral entities" introduced | 112–136, 162, 266–284 |
| Scorer | **Four factors**: presence, position, type, **span length**; type order written as **PERSON > GPE > EVENT > ORG/LOC > DATE/other**; Table 1 splits GPE and EVENT and asks for the EVENT weight (`\todo`) | 112, 134, 220–225, 229, 243, 260 |
| Step 2 text | Pruning "preserves every mention of E\*" + POS/NER mask; enrichment "using only entities from E\*" | 186 (Alg. 1), 284 |
| Table 3(a) | Rows renamed: `+ prompt band`, `+ entity list (unfiltered)`, `+ prompt band + entity list`; caption says rows above VG-Cap use the original SGN | 341, 351–356 |
| Table 3(b) | "**same cap K_e=10**, applied through **both gates**"; rows: article order (P1), random (P1), type prior only (P3), **contextual presence only** (P2, new) | 341, 358–362 |
| Table 3(c) | Renamed "Scorer variants"; adds **presence required** (P2, moved here from (b)) | 364–370 |
| Table 3(d) | No gate 0.6166 / Gate 1 only (P1) / Gate 2 only (P1) / **Gates 1+2 (VG-Cap) 0.6219** | 372–376 |
| Priorities | P1 / P2 / P3 tags. P1 rows + unfiltered + VG-Cap "with 3 seeds for the bootstrap CI" | 380 |
| Table 4 | Annotation now **4 labels**: visual / contextual / **peripheral** / **hallucinated**; columns Vis./Peri./Hall. replace Unsup. | 397–419 |
| §4.5 | New `\todo`: per-type **removal rate**, admissions, and **share of PERSON removals caused by the cap**; Fig. 2b axis label fix | 438–443 |

## 2. Conflicts with the code and published runs (send to supervisor)

These statements in `new_main.tex` disagree with what the code ran or what the data allow. Each needs either a paper edit or a specific run.

| # | Line | `new_main.tex` says | Fact (code / git / data) | Resolution |
|---|---|---|---|---|
| 1 | 224, 229 (`\todo`), 243, 260 | EVENT weight "strictly between 2.0 and 2.5"; order PERSON > GPE > EVENT | Code: **EVENT = 2.5 = GPE** (as the old Table 1 said) | Answer the `\todo`: EVENT = 2.5. Write PERSON > GPE = EVENT > ORG/LOC; fix Fig. 1. Do **not** change the code (it would change 0.6219) |
| 2 | 376 | Gates 1+2 (VG-Cap) = 0.6219 | **Corrected 2026-10-28:** the submitted 0.6166 and 0.6219 CSVs differ from the first words (different drafts), so 0.6219 is **not** Gate 2 only; the authors state it used Gates 1+2. Its code, prompt and drafts are lost (its captions open "The image captures…" in 52% of cases, the current prompt in ~2%) | Keep 0.6219 as the submitted result; Table 3 is re-run with the current code (`docs/VAST_RUNBOOK.md`), and the paper states "re-run VG-Cap obtains X ± s vs 0.6219 submitted" |
| 3 | 351–355 | ENRIC and +SGN rows have no entity list; "+ entity list (unfiltered)" is a new row | `version1.json` (ENRIC 0.6097, SGN 0.6124) was generated **with** the keyword list (git `c004e52`). So the 0.6124 row **already is** "+ entity list, no band", and 0.6124 → 0.6166 adds only the band | Relabel: ENRIC/SGN rows *include* the entity list; new rows = "w/o entity list". Or add the runs in §3 that remove the list from ENRIC/SGN |
| 4 | 186, 284 | Gate 2 pruning preserves E\*; POS mask; enrichment only from E\* | Published run: pruning protects only spaCy entities in the caption, no POS mask, enrichment falls back to article noun chunks | Either soften the text or adopt the strict Gate 2 run (§3, `vgcap_gate2strict`) if it scores ≥ 0.6219 |
| 5 | 112, 134, 216–225, 229 | Four factors incl. span length | Span penalty **never fires** in the published run | Three factors, or switch to the fixed scorer (`--span_penalty paper`) |
| 6 | 282 | "The LLM receives four inputs, namely the query image…" + `\todo` | Llama-3-8B is text-only; the image enters via `generated_caption` from **InternVL2.5-8B** | Answer the `\todo` with that |
| 7 | 282 | Quoted instruction "must strictly contain… Each sentence must add…" | Real text (`.j2`): "The caption must be between 100-120 words. The caption should contain 3–5 concise sentences, each adding new factual information…" | Quote the real text |
| 8 | 303 | τ_dec = 0.6; re-ranking captions use InternVL2.5-8B | Published drafts used **0.3** (git); re-ranking captions the query with **InternVL2.5-4B** | 0.3; 4B for query / 8B for database |
| 9 | 199 | "the top-1 re-ranked candidate determines A" | Re-ranking never sorted (141/141 unchanged) | Say re-ranking left top-1 unchanged, or re-run with `--sort_reranked` |
| 10 | 112, 130, 467 | CLIP 0.833 → 0.858 "at an unchanged caption length" vs ENRIC | ENRIC = 136 words, VG-Cap = 104 | Compare with the unfiltered row (both 104) |
| 11 | 380 | P1 rows "with 3 seeds for the bootstrap CI" | No public GT, and the leaderboard returns totals only. A per-query bootstrap of CIDEr/Overall is impossible on test | Report mean ± std over 3 seeds; bootstrap only CLIP (local) or a train subset |
| 12 | 399 | Entity overlap on "all 3,000 queries" with the reference caption | No public-test GT | Train subset with `gt_train.csv`, or drop the columns |
| 13 | 443 | PERSON removals "or from the generator, whose sampled captions differ even for identical inputs" | Unfiltered and VG-Cap reuse the **same** drafts (if the anchors confirm), so the generator cannot cause removals | Attribute to the cap or to enrichment order. The new `\todo` (share caused by the cap) is computable from `entity_debug.json` |
| 14 | 162 + figures | `figures/overview.png`, `figures/qualitative_results.png` | Files missing | Add the new figures |

## 3. Can the pipeline still run?

> **Superseded 2026-10-28.** The run plan is now the straight-line re-run in `docs/VAST_RUNBOOK.md`: VG-Cap = Gates 1+2 with the repaired scorer config, then a checkpoint against 0.6219, then all P1+P2 rows; job lists `jobs/G1_core_s0.txt`, `C1_after_G1.txt`, `G2_rest.txt`, `C2_after_G2.txt`. The scenario E/U discussion below is kept only for history.

**Yes.** Nothing in `new_main.tex` changes retrieval, data, models or the prompt. Session A GPU batch 1 in `docs/VAST_RUNBOOK.md` still matches what the new paper needs:
- `gate_unified` → "Gates 1+2", P1
- `gate_prompt` → "Gate 1 only", P1
- `ladder_band_only` and `ladder_nokw_noband` → the P2 ladder rows

What must change before or while running:

### Code (small; CPU side only; not done yet)

| # | Change | For |
|---|---|---|
| K1 | New selector `presence_score`: rank by the presence-family score only (no position, no type), cap K_e | (b) "Contextual presence only", P2 |
| K2 | `eval_entities.py` annotation sheet: labels visual / contextual / peripheral / hallucinated; agreement script reports Vis./Peri./Hall. rates | Table 4 |
| K3 | Entity-count script (N9): per-type removal rate, admissions per type, PERSON removals caused by the cap (selected-by-score but ranked > K_e) | §4.5 `\todo` |

### Runbook run list (Session A)

| Run | Command difference | Paper row | GPU? |
|---|---|---|---|
| (b) fixed cap | `--budget_match fixed --budget 10` for `article_order`, `random` ×5, `type_only`, `presence_score` | 359–362 as now written ("same cap K_e=10") | no (Gate 2) |
| (b) matched budget | keep `--budget_match per_query` runs as a supplementary check (fixed 10 vs VG-Cap's ~6.8 is not budget-matched) | appendix / response letter | no |
| (c) presence required | `--selector presence_only --budget_match fixed --budget 10` | 370 | no |
| Strict Gate 2 | legacy scorer + `--preserve_selected --pos_mask --no_noun_phrases`, tag `vgcap_gate2strict_enrichment_s0` | decides conflict #4 | no |
| ENRIC/SGN without entity list | reuse `runs/ladder_nokw_noband_s0/llm_raw.json` with `--no_normalizer` | only if the supervisor keeps the ladder as written (conflict #3) | no |
| Seeds (P1 only) | s1, s2 band drafts (`vgcap_enrichment_s1/2`), `gate_prompt_s1/2`, `gate_unified_s1/2`; then on CPU re-run `gate_none`, `article_order`, `random`, `abl_no_editorial` on the new drafts. **Drop** `ladder_sgn_s1/2` (ENRIC/SGN are P2) | 380 | 6 runs |

### If VG-Cap becomes unified (decision after `gate_unified_s0`)

The caption of (b) says selectors are "applied through both gates". Under unified gating every (b) and (c) row changes the prompt, so each needs generation:
- P1: 1 + 5 random → 6 GPU runs
- P2: ~5 runs
- P3: ~3 runs

At 4 concurrent runs, that is about 3–4 extra GPU batches (~30–45 h, roughly $35–50 at $1.14/h). Under Gate 2 they all stay on CPU.

## 4. Updates still to apply to the notes

- `VGCAP_ACTION.md` Part C: renumber against `new_main.tex` and add conflicts #1, #2 (now in the paper), #3, #11, #13 and #14.
- `docs/VAST_RUNBOOK.md` Phase C and Phase E: the run list of §3.
- `VGCAP_TODO.md` §2 panels (b)/(c)/(d) and Table 4: the new rows, labels and P1/P2/P3 priorities.
