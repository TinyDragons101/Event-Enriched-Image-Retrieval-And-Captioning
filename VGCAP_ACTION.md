# VG-Cap — Chạy gì, sửa gì

Cập nhật 2026-10-27. File này là **việc phải làm**, theo thứ tự.

| File | Dùng khi |
|---|---|
| **`VGCAP_ACTION.md`** (file này) | chạy gì, sửa gì — bắt đầu từ đây |
| `VGCAP_FIXES.md` | đã vá lỗi gì, switch nào bật/tắt |
| `VGCAP_FLOW.md` | code chạy thế nào |
| `VGCAP_TODO.md` | vì sao cần từng thí nghiệm |

---

# PHẦN A — Chuẩn bị

```bash
cd phucpv/Eventa
source ./.venv/bin/activate
pip install spacy jinja2 scikit-learn
python -m spacy download en_core_web_sm
nvidia-smi                      # theo NOTE.md: không dùng cuda:1
export DEVICE=cuda:5            # mọi script đều đọc biến này
```

File đã có sẵn trong repo, **không cần sinh lại**:

| File | Vai trò |
|---|---|
| `assemble_result/cot_5_things_fact_more_event_llama.json` | 3000 draft, prompt **có** length band (97,4 từ) |
| `assemble_result/cot_5_things_fact_more_event_llama_version1.json` | 3000 draft, prompt **không** band (135,2 từ) |
| `public_test_final_elements_json/final_merge_result.json` | context đầu vào |
| `final_csv_result/public_final_retrieval_merging_final_results.csv` | khung submission |

---

# PHẦN B — Thứ tự chạy

## B0. Test hồi quy — BẮT BUỘC, 2 phút, 0 GPU

Chứng minh bản vá không làm lệch số đã công bố.

```bash
python scripts/test_fixes.py --n 3000 \
    --reference runs/repro_vgcap/selected_entities.json
```

Phải thấy `KET QUA: TAT CA PASS`. Nếu `[1]` hoặc `[2]` FAIL thì **dừng lại** —
scorer đã lệch khỏi bản đã công bố, mọi số sau đó vô nghĩa.

> Đã chạy trên NER giả lập: 300/300 query khớp E\*. Chạy lại với spaCy thật.

## B1. Dựng evaluator cục bộ — chặn TẤT CẢ phần còn lại

Không có ground truth public test thì CIDEr/Overall chỉ lấy được từ leaderboard,
mà có ~40 cấu hình cần chấm. Phải chấm cục bộ.

```bash
# 1) Kiểm evaluator có tái dựng được số đã công bố không
python scripts/evaluate.py --run runs/repro_vgcap \
    --gt data/public_test_gt.json \
    --query_images data/track1_public/query \
    --device $DEVICE \
    --expect 0.6219 --tolerance 0.002
```

| Kết quả | Nghĩa là |
|---|---|
| PASS | evaluator tin được → chạy tiếp B2 |
| lệch đều mọi dòng | lỗi ở evaluator (công thức/tokenizer) |
| chỉ vài dòng lệch | lỗi ở pipeline tái dựng |

**Nếu không có `data/public_test_gt.json`:** chỉ CLIPScore tính được cục bộ;
CIDEr/Overall phải submit. Khi đó dùng `scripts/make_submissions.py` để khử trùng
lặp và xếp thứ tự ưu tiên submit:

```bash
python scripts/make_submissions.py --runs runs --out submissions
```

## B2. Panel (e) — lỗi nào mới là nguyên nhân? 0 GPU, ~2 giờ CPU

**Đây là bảng quan trọng nhất.** Nó trả lời thẳng câu reviewer hỏi ở panel (b):
0.6219 đến từ thiết kế scorer hay chỉ từ việc giới hạn còn 10 entity?

```bash
DEVICE=$DEVICE bash scripts/run_fixes.sh enrichment
```

Sinh 10 run, tất cả tái dùng draft nên **không tốn GPU**:

| tag | Nội dung |
|---|---|
| `fix_legacy_*` | bản đã công bố (mốc 0.6219) |
| `fix_keyword_parse_*` | sửa lỗi parse markdown |
| `fix_word_boundary_*` | sửa lỗi khớp chuỗi con |
| `fix_span_penalty_*` | bật P_span đúng Eq. (2) |
| `fix_dedup_*` | dedup theo chuỗi con |
| `fix_position_*` | first-mention thay cho sentinel `start==0` |
| `fix_candidates_*` | loại `'1'`, `'One'`, `'first'` |
| `fix_all_*` | bật tất cả |
| `fix_all_gate2_*` | + pruning bảo toàn E\* + POS mask |
| `fix_all_nonp_*` | + enrichment chỉ được nối E\* |

Chấm điểm rồi xem bảng:

```bash
GT=data/public_test_gt.json QUERY_IMAGES=data/track1_public/query \
  bash scripts/score_all.sh
python scripts/collect_results.py --latex
```

**Đọc kết quả thế nào:**
- `fix_all` **cao hơn** `fix_legacy` → bản vá cải thiện thật, chuyển bài sang dùng
  scorer đã vá, chạy lại Table 3 với `BASE_CFG=..._fixed.json`.
- `fix_all` **ngang** `fix_legacy` → scorer đúng hay sai không quan trọng, nghĩa là
  gain đến từ **budget** chứ không từ thiết kế. Phải viết lại bài theo hướng
  budget-aware gating. Đây là kết quả khó nghe nhưng vẫn công bố được.
- `fix_all` **thấp hơn** → phải giải thích vì sao entity rác lại giúp ích (nhiều khả
  năng vì chúng ngắn, làm enrichment ít phá câu hơn).

## B3. Table 3 — panel (a)(b)(c)(d)

```bash
DEVICE=$DEVICE bash scripts/run_panels.sh enrichment
```

Phần lớn tái dùng draft. **Chỉ 4 cấu hình cần GPU** (~6h mỗi cái):
`ladder_band_only`, `ladder_entities_only`, `gate_prompt`, `gate_unified`.

Chạy panel (c) trên scorer đã vá:

```bash
python scripts/make_ablation_configs.py
BASE_CFG=configs/entity_scoring_config_fixed.json CFG_SUFFIX=_fixed \
  DEVICE=$DEVICE bash scripts/run_panels.sh enrichment
```

> `gate_unified` là dòng quyết định đóng góp "unified gating" còn hay mất.
> Chạy nó trước tiên trong 4 dòng cần GPU.

## B4. Ba seed + bootstrap

```bash
for S in 0 1 2; do
  python run_ablation.py --tag vgcap_s$S --gate enrichment --seed $S --device $DEVICE
  python run_ablation.py --tag none_s$S  --gate none       --seed $S --device $DEVICE
done
python scripts/bootstrap.py --a runs/none_s0 --b runs/vgcap_s0 --metric cider --resamples 1000
```

## B5. Table 4 — Entity-level grounding

```bash
# Leakage da co san trong stats.json cua moi run (repro_vgcap: 0.0808)
python scripts/eval_entities.py --run runs/vgcap_enrichment_s0 \
    --gt data/public_test_gt.json --sample 200 --seed 0
```

Sinh P/R/F1 + bảng annotation cho 200 query. Phần nhãn tay (visual / contextual /
peripheral / hallucinated) và điểm 1–5 cần người, 1–2 tuần.

## B6. Phụ lục

> Cần chạy **B3 trước**: `run_appendix.sh` đọc `runs/vgcap_<gate>_s<seed>/llm_raw.json`,
> và `eval_entities.py` / `make_qualitative.py` đọc các run do `run_panels.sh` sinh ra.

```bash
DEVICE=$DEVICE bash scripts/run_appendix.sh enrichment    # A: L*, tau x K_e, weight
python scripts/train_learned_selector.py \
    --merge public_test_final_elements_json/final_merge_result.json \
    --gt data/train_gt.json --n_queries 500 --out configs/learned_selector.json   # B
python scripts/make_qualitative.py --run runs/vgcap_enrichment_s0 \
    --baseline runs/gate_none_s0 --per_bucket 3                                   # D
python scripts/dump_prompts.py --n 3 --out supplementary/prompts                  # E
```

Phụ lục C (efficiency) lấy từ `timings` trong `runs/*/stats.json`.

## B7. Gom lại

```bash
python scripts/collect_results.py --latex > table3_rows.tex
```

---

# PHẦN C — Sửa `main.tex`

Đi từ trên xuống. Cột "Sửa thành" là nội dung đúng theo code.

## C1. Bắt buộc sửa — hiện đang sai sự thật

| Dòng | Đang viết | Sửa thành |
|---|---|---|
| 110, 126, 133, 160, 167, 181, 265, 274–276, 282, 339, 384, 389, 465 | E\* gate **cả prompt lẫn normalizer** ("unified gating") | Chỉ gate ở **enrichment**. Hạ contribution 2 xuống *"gating the enrichment step of length control"*. **Trừ khi** `gate_unified` ở B3 cho kết quả tốt hơn |
| 110, 126, 132, 178, 213–216, 218, 223, 227, 245 | "**four** factors", có span penalty, Eq. (2) có `+P_span(e)` | "**three** factors"; Eq. (2) bỏ `+P_span(e)`; bỏ bullet *Span penalty*; bỏ dòng Penalty ở Table 1. **Trừ khi** dùng `--span_penalty paper` |
| 222, 227, 241, 258 | `PERSON > GPE > EVENT > ORG/LOC` | `PERSON 3.0 > GPE = EVENT 2.5 > ORG = LOC 2.0 > DATE 1.0 > other 0.5`. Điền ô `\todo{}` ở Table 1 = **2.5** |
| 197 | `s_rerank = λ·s_img + (1−λ)·s_text`; "three detector thresholds" | Không có λ: `combined = logit_scale·cos(text) + cos(image)`. Detector là **quota cố định**: top-50 entropy ∪ top-50 gap ∪ top-50 top1-sim, ≤150 query. Không có softmax temperature |
| 197 | "the top-1 re-ranked candidate determines A" | Re-ranking **không đổi thứ hạng** (141/141 query thứ tự ra = thứ tự vào). Retrieval thực chất là image-only top-1. **Trừ khi** chạy lại với `--sort_reranked` |
| 167, 181, 267, 280 | `G_θ(q, ...)`, "The LLM receives **four** inputs, namely the query image, ..." | Llama-3-8B text-only. Bỏ `q` khỏi Eq. (3) và Alg. 1: `G_θ(C_article, C_image, E*; L_min, L_max)`. "**three** inputs"; ảnh vào qua `generated_caption` của InternVL2.5-8B |
| 110, 128, 465 | "raises CLIPScore over its ENRIC backbone from 0.833 to 0.858 **at an unchanged caption length**" | ENRIC = **136 từ**, VG-Cap = 104. Sửa thành *"...over the unfiltered-entity configuration at an identical mean length of 104 words"* |
| 272, 280 | draft nằm trong band; "needs little or no enrichment" | Đo thật: chỉ **40,5%** draft trong band 100–120; **57,2%** dưới 100; **71%** phải enrich. Viết đúng rằng Step 2 là bắt buộc, không phải phụ trợ |
| 282 | pruning "preserves every mention of E\*"; "POS/NER mask keeps proper nouns and numbers" | Code chỉ bảo vệ entity spaCy **trong caption**, chạy **trước khi** tính E\*, không có POS mask. **Trừ khi** chạy với `--preserve_selected --pos_mask` |
| 282, 278 | enrichment dùng "**only** entities from E\*" | Hết E\* thì nối **noun chunks toàn bài**, ngoài gate. Thêm kênh rò thứ ba vào câu leakage ở dòng 278. **Trừ khi** chạy `--no_noun_phrases` |
| 176, 202, 208 | E = spaCy NER ∪ article keywords | Keyword list là markdown, bị `split(',')` → vào E dưới dạng 1–3 blob; **47% query** có blob trong E\*. Thực tế E ≈ chỉ spaCy. **Trừ khi** chạy `--fix_profile all` |
| 208 | dedup "keeps the longer surface form (Joe Biden over Biden)" | Chỉ đúng cho nhánh keyword. `'Michelle Payne'` và `'Payne'` cùng tồn tại |
| 220 | "exact matches outweigh partial ones, which **only absorb morphological variation**" | Là **chuỗi con thô**: `'One'` khớp `"someone"` → 14.0 điểm, cao hơn `'Michelle Payne'` 8.5 |
| 221 | vị trí ảnh trong bài tìm bằng URL/HTML marker + fuzzy | `article_position` **bị bỏ rơi** ở `step_2_merge_all_elements.py:27`, không tới scorer. Bỏ câu này |
| 280 | câu prompt trích nguyên văn | Chép đúng từ `cot_5_things_fact_more_event.j2:34-35` — bản thật không có "strictly", không có "Each sentence must add" |
| 339, 353, 371, 384 | "unfiltered entity list" = "all article entities" | `name_entity.j2:17` đã chỉ thị LLM *"Prioritize entities that appear visually in the image"*. Gọi đúng là **LLM-extracted, image-prioritized keyword list** — baseline mạnh hơn, gain của VG-Cap có giá trị hơn |
| 441 | PERSON removals "come from ... the generator, whose sampled captions differ even for identical inputs" | **Không thể**: hai cấu hình dùng chung một file draft, truncation cũng độc lập với E\*. Sửa thành *"every removal is attributable to enrichment on the 71% of drafts shorter than L\*"* — câu này mạnh hơn và khớp §4.7 |
| 371–374 | Table 3(d) có "Gate 2 only" và "Gates 1+2" | Hiện là **cùng một run**. 0.6219 thuộc dòng *Gate 2 only* |
| 128, 353–354, 423, 441 | #Ent 7.59 → 6.76; 22.759 → 20.276 mentions | `runs/*/stats.json` cho **12,725 → 11,004** (38.176 → 33.011). Chốt một cách đếm rồi vẽ lại Fig. 2 |

## C2. Điền được ngay, không cần chạy gì

| Dòng | Việc |
|---|---|
| 413 | Table 4 cột **Leak = 0.0808** (`runs/repro_vgcap/stats.json`, trên 221.659 entity bị loại) |
| 298 | CIDEr 4 chữ số: ENRIC cần **0.2577** mới ra 0.6097; dòng 0.6166 cần **0.2639**; dòng 0.6124 cần **0.2598** |
| 304 | Câu *"within ±0.001"* **đúng** — tôi recompute cả 7 dòng Table 1, lệch lớn nhất 0.0010 |
| 298 | Câu *"CIDEr = 0.2651 gives 0.6219 exactly"* **đúng** → 0.621916 |
| 437 | Fig. 2a nhãn trục `0.6218` → `0.6219`; Fig. 2b đổi "Net entities removed" → "Gross removals" |
| 449 | Fig. 3 nhãn `0.6166`/`0.62` → `Unfiltered`/`VG-Cap` |
| 160 | Vẽ lại Fig. 1: mũi tên từ scorer **chỉ trỏ vào normalizer**, bỏ ô span penalty, sửa thứ tự type prior |
| 459 | Limitations: thêm noun-chunk leak và nhãn spaCy sai (`Michelle Payne`→ORG, `Suffragettes`→PERSON) |
| 582 | Repo URL cho Appendix E |
| 304 | ReCap tự báo 0.61619 vs 0.5987 trong bảng — **giá trị cao nhất trên công sức**. Nếu 0.61619 đúng thì lead của VG-Cap chỉ còn 0.006 và ReCap có CLIP cao hơn |

## C3. Đã kiểm — ĐÚNG, giữ nguyên

τ=3.0, K_e=10, max_words=105, expected_length=104, các trọng số presence/position
còn lại, thứ tự 5 thao tác SGN, danh sách preamble, `temperature=0.6` / `top_p=0.9`
/ `min_new=100` / `max_new=1500`, InternVL-14B-224px + InternVL2\_5-8B,
Llama-3-8B-Instruct, spaCy `en_core_web_sm`, `pre_top_k=15`, công thức Overall,
worked example §3.4 (15.0 / 3.5 / 1.0), "τ non-binding for short PERSON spans".

---

# PHẦN D — Ba quyết định phải chốt trước khi tốn giờ GPU

| # | Quyết định | Nếu chọn "có" | Nếu chọn "không" |
|---|---|---|---|
| 1 | Chạy `--gate unified`? | Giữ được đóng góp "unified gating"; phải chạy lại **toàn bộ** kể cả Table 1 | Viết lại §3.6 thành enrichment-only; giữ nguyên mọi số |
| 2 | Chuyển sang scorer đã vá? | §3.4 mô tả đúng thứ đang chạy; chạy lại Table 3 (phần lớn 0 GPU) | Phải thừa nhận trong Limitations rằng E thực chất chỉ là spaCy và presence là chuỗi con thô |
| 3 | Bật `--sort_reranked` / `--caption_source merged`? | §3.2 và §3.3 đúng | Phải viết rằng re-ranking là no-op và retrieval là image-only |

Cả ba câu "có" đều kéo theo **sinh lại 3000 caption (~6h GPU)**.

---

# PHẦN E — Nếu chỉ có một ngày

```bash
python scripts/test_fixes.py --n 3000 --reference runs/repro_vgcap/selected_entities.json   # 2 phut
bash scripts/run_fixes.sh enrichment                                                        # ~2h CPU
python run_ablation.py --tag gate_unified_s0 --gate unified --selector heuristic --device $DEVICE  # ~6h GPU
```

Song song đó sửa toàn bộ **C2** (không cần chạy gì) và các dòng của **C1** không
phụ thuộc kết quả: 222/227/241 (EVENT), 197 (λ + detector), 167/181/267/280 (ảnh),
110/128/465 (unchanged length), 280 (trích prompt), 220 (chuỗi con), 221 (vị trí ảnh),
441 (generator), 413 (leakage).

Ba lệnh trên đủ để biết **có giữ được đóng góp "unified gating" không** và
**gain đến từ scorer hay từ budget** — hai câu hỏi quyết định bài này.
