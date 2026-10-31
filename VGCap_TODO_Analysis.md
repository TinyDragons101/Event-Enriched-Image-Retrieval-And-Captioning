# VG-Cap — `main-28-10.tex` vs `main.tex`: What Still Needs Filling or Confirming

Analysis only — no edits made to either `.tex` file. `main.tex` = older MMM 2027 revision (has a full appendix and an entity‑grounding/human‑eval section). `main-28-10.tex` = newer ACIIDS 2027 trim (appendix and grounding section removed, title/keywords updated). This doc has three parts:

1. Structural diffs between the two versions (context for why some `main.tex` TODOs are gone)
2. Every live `\todo{...}` left in `main-28-10.tex` — these are the things to **fill in**
3. `\revise{...}` / `\aciids{...}` passages that carry claims/numbers worth a final **confirm pass** before submission

---

## 1. Structural differences (main.tex → main-28-10.tex)

| Removed / changed in `main-28-10.tex` | Was in `main.tex` |
|---|---|
| Title: `VG-Cap: Entity-Gated Generation for Event-Enriched News Image Captioning` | `Grounding What Matters: Selective Entity Conditioning for Event-Enriched News Image Captioning` (now a commented-out line) |
| Keywords drop "Trustworthy multimedia" | Keywords included it |
| Abstract/intro no longer promise "entity-level grounding annotation"; shortened to budget-matched baselines + gate ablations | Abstract/intro promised Table `tab:grounding` (overlap, manual annotation, leakage, human ratings) too |
| Contribution #3 reduced to one generic sentence about Overall/CIDEr | Contribution #3 described the full empirical study incl. entity-level evaluation, leakage, human ratings |
| Related Work "hallucination" paragraph drops the sentence pointing to `sec:grounding` | Pointed to a dedicated grounding/human-eval section |
| §Entity Heuristic Scorer "Design rationale" no longer cites Appendices (sensitivity sweep, learned logistic-regression selector) | Cited `app:sensitivity` and `app:learned` |
| §Step 2 (SGN) drops the clause about Appendix sweeping $L^{*}$ | Had "...and Appendix~\ref{app:sensitivity} sweeps $L^{*}$ for our generator." |
| §Unified entity gating drops the "leakage rate" sentence (no Table `tab:grounding` to report it in) | Defined leakage rate and pointed to `tab:grounding` |
| **Table `tab:ablation`(b)** has 3 rows (Article order, Random, Contextual presence only) | Had 4 rows — also included **"Type prior only" (P3)** |
| **Table `tab:ablation`(c)** has 4 rows (w/o contextual presence, w/o editorial-caption match, w/o entity-type prior, Presence required) | Had 6 rows — also included **"w/o article position" (P3)** and **"w/o span penalty" (P3)** |
| Priority-note parbox under the ablation table drops the whole "P3, optional, move to Appendix" sentence | Explained why the P3 rows were deprioritized/could move to appendix |
| **Entire §6 "Entity-Level Grounding and Caption Quality" removed**, incl. Table `tab:grounding` (P/R/F1 overlap, Vis./Peri./Hall. annotation, Leakage, Human ratings) | Present, fully templated with `\todo{}` placeholders |
| §Entity Selection Analysis drops the pointer to `tab:grounding` and to `app:qualitative` over-pruned cases | Had both pointers |
| §Qualitative Analysis drops the pointer to `app:qualitative` for more successes/failures | Had it |
| §Limitations: title shortened from **"Limitations and Broader Impact"** to **"Limitations"**; the entire Broader Impact content (reputational risk of misattributed names, human-editor-in-the-loop, multi-source corroboration for PERSON, provenance exposure) is gone; appendix references removed | Present as a distinct, detailed paragraph |
| Conclusion shortened: drops the specific "most removals fall on DATE/ORG" sentence, the "mask rejected entities in all channels" + "fluency-aware rewriting" detail, and the `\todo{Replace the last clause by the measured result of Table~\ref{tab:grounding}}` | Present, more detailed |
| **Entire Appendix removed**: Sensitivity Analysis ($\tau$/$K_e$ sweep, weight-perturbation, $L^{*}$ sweep), Learned Selector Variant (logistic-regression baseline), Efficiency table (latency/GPU memory), Additional Qualitative/Failure Cases, Reproducibility statement (code/prompt/seed release) | Present, all still `\todo{}`-templated |
| Fig.~\ref{fig:pipeline} caption todo (rename "Two-Stage Length Control" block) now **commented out** | Was a live `\todo{}` |
| Fig.~\ref{fig:qualitative} caption todo (relabel in-figure scores) now **commented out** | Was a live `\todo{}` |

**Needs your confirmation:** was cutting the grounding section, the two P3 ablation rows, the Broader Impact paragraph, and the whole appendix a deliberate page-budget decision for the ACIIDS 2027 venue (shorter page limit than MMM), or should any of that be folded back in condensed form (e.g., a one-paragraph reproducibility statement, or re-adding "Type prior only" since it's cheap to run)?

---

## 2. Live `\todo{}` items in `main-28-10.tex` — things to fill in

### Abstract (line 110)
- Once Table `tab:ablation`(b) is filled, replace the abstract's last sentence with the actual outcome (currently just says "we analyze its source with budget-matched selection baselines and gate ablations").

### §3.2 Article Retrieval and Re-ranking (line 197)
- Report **λ** (fusion weight), the **softmax temperature**, and the **three uncertainty-detector thresholds** (entropy, top-1/top-2 gap, top-1 similarity).

### §3.2 Context Construction (line 202)
- R-SAR.Q5: confirm with the OpenEvents-V1 release that reference captions were **not** derived from editorial captions; report the "w/o editorial evidence" ablation row either way.
  - ⚠️ Note: the table (panel c) currently names this row **"w/o editorial-caption match,"** not "w/o editorial evidence" — reconcile the wording before filling it in.

### Table `tab:weights` (lines 227, 241)
- Report the actual **EVENT weight** used in the code (must be strictly between 2.0 and 2.5), or correct Fig.~\ref{fig:pipeline}'s ordering if EVENT and GPE are both meant to be 2.5. Two placeholders tied to the same unresolved number (caption `\todo{}` + table cell `\todo{}`).

### §3.4 Entity-Gated Length-Controlled Generation (line 278)
- R3.2 (optional): consider adding a variant that masks rejected entities inside the summary/QA text, reported next to the existing gating rows.

### §3.4 Step 1: bounded generation (line 280)
- State **how the query image $q$ reaches $G_\theta$** given that Llama-3-8B-Instruct is text-only (e.g., via its generated description), or name the actual multimodal generator used.
- Release the **full prompt template** (system/user turns, context-field order) as supplementary material.

### §4.1 Data and metrics (line 298)
- Report **CIDEr to four decimals** in Table `tab:sota`; flag that ENRIC's rounded 0.257 recomputes to 0.6089, not the listed 0.6097 — resolve the discrepancy.

### §4.1 Baseline provenance (line 304)
- R4.3: ReCap's own paper reports **0.61619** for its complete model vs. the 0.5987 listed here. Identify exactly which ReCap configuration/table the 0.5987 comes from (leaderboard submission vs. post-challenge variant), cite precisely, and consider adding 0.61619 as its own row.

### §4.1 Statistical protocol (line 307)
- Confirm/fill the number of seeds — currently literally `\todo{3}` (i.e., "3" is a provisional guess flagged for confirmation, not yet verified).
- SAR: re-run Table `tab:ablation` rows with multiple seeds; report std and bootstrap CIs — **all numbers currently come from a single run.**

### §4.2 Comparison with SOTA (line 336)
- SAR: add the **bootstrap CI** for VG-Cap's CIDEr and Overall gains over ENRIC.

### Table `tab:ablation` — empty cells to run and fill
- **(a) Component ladder:** `#Ent` for "ENRIC baseline (reproduced)" and "+ semantic normalizer (SGN)"; all 5 metrics for "+ prompt band" (alone) and "+ entity list (unfiltered)" (alone) — i.e., the two single-component rows that isolate what panel (a)'s combined row already shows together. **(P2)**
- **(b) Alternative selectors ($K_e{=}10$):** Article order (first 10 mentions) **(P1)**, Random (mean of 5 draws) **(P1)**, Contextual presence only **(P2)** — all 5 metrics each.
- **(c) Scorer variants:** w/o contextual presence **(P2)**, w/o editorial-caption match **(P1)**, w/o entity-type prior **(P2)**, Presence required (caption match needed) **(P2)** — all 5 metrics each.
- **(d) Gate location:** Gate 1 only (prompt) **(P1)**, Gate 2 only (normalization) **(P1)** — all 5 metrics each.
- The priority note itself says: run the P1 rows (plus the already-filled unfiltered/VG-Cap rows) with 3 seeds for bootstrap CIs; P2 rows strengthen the analysis but aren't load-bearing for the main claim.

### §4.3 Ablation Study discussion (line 381)
- R3.6/R4.2: once the two single-component ladder rows exist, write the discussion of how the $0.6124\to0.6166$ gain splits between the prompt band and the unfiltered entity list.

### §4.3 "Which gate matters?" (line 386)
- R1.2/R4.2/SAR.Q3: fill panels (b)–(d) and state which pairwise differences exceed the bootstrap CI.

### Fig.~\ref{fig:entity-stats} caption (line 405)
- SAR: regenerate panel (a) — its axis label currently reads **0.6218** instead of **0.6219**.
- Regenerate panel (b) — its axis label reads **"Net entities removed"** but the bars actually sum to the **gross** removals (7,698); fix the label or the data.

### §4.4 Entity Selection Analysis (line 409)
- Add, **per entity type**, the removal rate (removed / present in unfiltered captions) and the admission count; report what share of PERSON removals is attributable to the $K_e$ cap vs. generator stochasticity.

---

## 3. `\revise{}` / `\aciids{}` passages worth a final confirm pass

These aren't blank placeholders — they're already-written claims/numbers tagged as changed for this revision or for the ACIIDS 2027 consistency pass. Worth a last read-through against the actual code/results before submission:

- **Abstract & Intro (aciids):** the four-factor scorer description, the "fixed retrieval backbone" framing, and the exact numbers (Overall 0.6219, CIDEr 0.265, CLIP 0.833→0.858, 104 words) — these now appear in multiple places (abstract, intro, contributions, Table `tab:sota`, conclusion); confirm they're all still consistent with each other after the trims above.
- **§3.3 Entity Heuristic Scorer, "Design rationale" (revise):** the claim "$\tau$ is non-binding for short PERSON spans... for DATE, at least 2.0 of additional evidence is required" and the worked example (15.0 / 3.5 / 1.0 scores) — arithmetic worth re-checking against Table `tab:weights` once the EVENT weight TODO above is resolved.
- **§3.1 Retrieval (aciids):** the re-ranking fusion formula $s_{\mathrm{rerank}}=\lambda s_{\mathrm{img}}+(1-\lambda)s_{\mathrm{text}}$ — confirm this matches the code now that λ itself is still a pending TODO.
- **§4.1 "Baseline provenance" (revise):** the claim that "recomputing every row from its rounded components reproduces the listed Overall Scores to within ±0.001" — double check this still holds given the ENRIC 0.6089-vs-0.6097 discrepancy flagged in the Data/metrics TODO above (it's arguably a counterexample to this exact claim).
- **§4.3 Ablation narrative (aciids):** "This last step changes both gates at once... panel (d) separates them" — confirm this framing still reads correctly now that panel (d) has only 2 unfilled rows instead of being folded with the dropped grounding discussion.
- **§4.4 Entity Selection Analysis (aciids):** the mention-level counts (7,698 removed / 5,215 admitted / net −2,483; DATE 1,957; ORG 1,464; PERSON 1,433) — these numbers are untouched between the two files, so they should already be final, but worth one source-data re-check since the Fig.~\ref{fig:entity-stats} caption TODO flags a labeling bug in the figure built from this same data.
- **§5 Limitations (revise):** now a single condensed paragraph — confirm it still covers every limitation you want on record, since the Broader Impact material (misattribution risk, human-in-the-loop deployment, PERSON corroboration, provenance exposure) was cut rather than condensed.
- **Conclusion (aciids):** confirm the shortened future-work list ("learn the scorer from data, verify entity grounding directly and cross-modally, extend the gate to all context channels") still matches what §5 Limitations promises, since some specifics (fluency-aware rewriting, masking all context channels) were trimmed from the sentence structure, not just the wording.
