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
# backup_pre_fix/ bị gitignore và KHÔNG có trong repo -> khôi phục scorer gốc từ git trước
mkdir -p backup_pre_fix
git show 9a23067:step_2_entity_scorer.py > backup_pre_fix/step_2_entity_scorer.py.bak
python scripts/test_fixes.py --n 3000
```

Phải thấy dòng `[1] legacy == pre-fix module ... PASS` và `KET QUA: TAT CA PASS`.
`[1]` báo `SKIP` nghĩa là chưa khôi phục file `.bak` → **chưa chứng minh được gì**.
Nếu `[1]` FAIL thì **dừng lại**: scorer đã lệch khỏi bản đã công bố, mọi số sau đó vô nghĩa.

> ⚠️ Kiểm tra `[2]` (`--reference runs/repro_vgcap/selected_entities.json`) **không** phải
> phép so với bản đã công bố: file đó do chính code mới sinh ra, nên so sánh là vòng tròn.
> Chỉ `[1]` (so với module gốc) mới có giá trị.
>
> Con số "300/300 query khớp E\*" trước đây đo bằng NER **giả lập**. Phải chạy lại với spaCy
> thật trên server.

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

> **Đã xác nhận với nhóm (2026-10-27):**
> - **Không có GT public test, không có công cụ chấm.** Chỉ có trang nộp bài và công thức Overall
>   trong bài. Lệnh `--gt data/public_test_gt.json` ở trên **không chạy được**. CIDEr, CLIP và
>   Overall chính thức **chỉ** lấy từ leaderboard.
> - **Leaderboard có quota khoảng 100 lần, đã dùng 5–6 → còn khoảng 94.** Toàn bộ kế hoạch nếu
>   nộp hết là khoảng 85–90 CSV, gần cạn quota. Xem "Ngân sách nộp bài" bên dưới.
> - **Không biết model CLIP chính thức.** CLIP cục bộ (`openai/clip-vit-base-patch32`, cắt ở 77
>   token) chỉ là số sàng lọc. Hiệu chỉnh nó bằng 4 run neo: so CLIP cục bộ với CLIP leaderboard
>   của chính 4 CSV đó. Lệch đều → dùng được để xếp hạng trước khi nộp; lệch không đều → chỉ tin
>   số leaderboard.
> - **Có GT của tập train** (`gt_train.csv`, đang ở drive, chưa có trong repo). Chép lên server ở
>   `data/gt_train.csv`. Dùng cho Phụ lục A, Phụ lục B, cột P/R/F1 của Table 4 và kiểm tra SAR.Q5.
>
> **Ngân sách nộp bài (~94 lượt):**
>
> | Ưu tiên | Nhóm | Số CSV | Ghi chú |
> |---|---|---|---|
> | 1 | 4 run neo | 4 | phải nộp trước tiên |
> | 2 | `run_fixes.sh` | ≤10 | khử trùng lặp trước |
> | 3 | Table 3 (a)(b)(c)(d) | ~21 | khử trùng lặp; dòng "w/o span" (scorer cũ) = run tham chiếu |
> | 4 | 3 seed | ~8–12 | chỉ các dòng chính |
> | 5 | Phụ lục B, R3.2 | 2–3 | |
> | — | Phụ lục A (37 run) | **0** | chạy trên tập con train, chấm CIDEr cục bộ bằng `gt_train.csv`; TODO dòng 525 của bài cũng yêu cầu dùng held-out train |
>
> Luôn chạy `make_submissions.py` để bỏ CSV trùng, và giữ lại ~10 lượt dự phòng cho nộp lại khi có lỗi.

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
>
> ⚠️ `run_panels.sh` **không bỏ qua run đã xong**: nếu đã chạy `gate_unified_s0` ở Phần E,
> script sẽ sinh lại và **ghi đè** `runs/gate_unified_s0/llm_raw.json` (mất 6h). Trước khi
> chạy, chép `runs/gate_unified_s0` ra chỗ khác hoặc comment dòng đó trong script.
> Với `run_panels.sh unified`, mọi dòng `sel_*` và `abl_*` đều phải sinh lại: khoảng
> **19 run GPU (~115h)**, không phải 4.

## B4. Ba seed + bootstrap

```bash
for S in 0 1 2; do
  python run_ablation.py --tag vgcap_s$S --gate enrichment --seed $S --device $DEVICE
  python run_ablation.py --tag none_s$S  --gate none       --seed $S --device $DEVICE
done
python scripts/bootstrap.py --a runs/none_s0 --b runs/vgcap_s0 --metric cider --resamples 1000
```

> ⚠️ Chi phí: các lệnh trên **không** có `--reuse_drafts`, nên mỗi lệnh sinh draft mới:
> 6 run × ~6h = **~36h GPU**. Đây là cách đúng để có 3 seed thật. Ngược lại,
> `SEED=$S bash scripts/run_panels.sh` chỉ đổi 4 run sinh mới; 18 run tái dùng draft cho
> caption **y hệt nhau** giữa các seed, nên không đo được phương sai.
>
> ⚠️ `--metric cider` cần CIDEr **từng query**, tức là cần GT public test, mà nhóm đã xác nhận
> **không có**. Trên public test chỉ chạy được `--metric clip` (CLIP cục bộ, không phải CLIP
> chính thức). CI của CIDEr/Overall (TODO dòng 298, 327) không làm được như bài viết; đổi giao
> thức thống kê theo C1 (dòng 298, 327).
>
> ⚠️ Tránh `--resume` cho các run seed: `set_seed` chỉ gọi một lần trước vòng lặp, nên sau khi
> resume luồng ngẫu nhiên đã khác và kết quả không tái lập được.

## B5. Table 4 — Entity-level grounding

```bash
# Leakage da co san trong stats.json cua moi run (repro_vgcap: 0.0808)
python scripts/eval_entities.py --run runs/vgcap_enrichment_s0 \
    --gt data/public_test_gt.json --sample 200 --seed 0
```

Sinh P/R/F1 + bảng annotation cho 200 query. Phần nhãn tay (visual / contextual /
peripheral / hallucinated) và điểm 1–5 cần người, 1–2 tuần.

> ⚠️ Không có GT public test (B1), nên P/R/F1 **không tính được trên 3000 query test**. Tính trên
> tập con train bằng `gt_train.csv` (cần context và caption của các query train đó, như Phụ lục B),
> rồi sửa câu "over all 3,000 queries" ở dòng 380 của bài.
>
> SAR.Q5 kiểm được luôn bằng `gt_train.csv`: so caption GT của ảnh train với editorial caption
> (`crawl_caption`) của chính ảnh đó. Trùng nhiều → reference có thể được dẫn xuất từ editorial
> caption, và dòng "w/o editorial evidence" trở thành phép kiểm bắt buộc.
> Leakage hiện đo bằng chuỗi con thô (`'One'` khớp "someone"), và mẫu số chứa các blob keyword
> ~300 ký tự không bao giờ xuất hiện trong caption, nên tỉ lệ bị kéo thấp. Sửa cách đo
> (khớp theo ranh giới từ, bỏ blob khỏi mẫu số) rồi mới đưa số vào Table 4.
>
> ⚠️ `annotation_sheet.csv` là một file **cho từng run**, có ghi rõ caption của hệ nào. Bài hứa
> chấm "blind, random order", nên phải trộn caption của 3 hệ và giấu tên hệ trước khi phát.

## B6. Phụ lục

> Cần chạy **B3 trước**: `run_appendix.sh` đọc `runs/vgcap_<gate>_s<seed>/llm_raw.json`,
> và `eval_entities.py` / `make_qualitative.py` đọc các run do `run_panels.sh` sinh ra.

```bash
DEVICE=$DEVICE bash scripts/run_appendix.sh enrichment    # A: L*, tau x K_e, weight
python scripts/train_learned_selector.py \
    --merge data/train_final_merge_result.json \
    --gt data/train_gt.json --n_queries 500 --out configs/learned_selector.json   # B
python scripts/make_qualitative.py --run runs/vgcap_enrichment_s0 \
    --baseline runs/gate_none_s0 --per_bucket 3                                   # D
python scripts/dump_prompts.py --n 3 --out supplementary/prompts                  # E
```

> ✅ **GT train có** (`gt_train.csv` trên drive, nhóm xác nhận). Chép lên server, xem các cột
> (query/image id, caption, article id) rồi chuyển sang JSON `{query_id: caption}` mà các script
> đang đọc. Nếu ảnh train đã gắn sẵn article thì Stage 2 cho tập train không cần chạy retrieval.
>
> ⚠️ **Phụ lục B:** `data/train_final_merge_result.json` **chưa tồn tại**. Bản cũ của lệnh này
> truyền `public_test_final_elements_json/final_merge_result.json` (tập public test) cùng GT
> của tập train: hai tập không có query chung, nên script dừng với lỗi "no overlap". Về nguyên
> tắc cũng sai, vì bài yêu cầu query **train** tách riêng. Phải chạy lại toàn bộ bước dựng
> context Stage 2 (caption InternVL, summary, QA, keyword, editorial caption) cho ~500 query
> train trên server (cần GPU) để có file này. Nhãn trong `train_learned_selector.py` cũng đang
> dùng chuỗi con thô; sửa sang khớp theo ranh giới từ.
>
> ⚠️ **Phụ lục A:**
> - `run_appendix.sh unified` dừng ngay ở lệnh đầu tiên: phần L\* luôn truyền `--reuse_drafts`,
>   mà `run_ablation.py` từ chối cờ này với `--gate unified`.
> - Tag `len_${L}_s${SEED}` không có tên gate, nên hai gate ghi đè lên nhau.
> - Bài ghi Table A2 là *"prompt band shifted accordingly"*, nhưng template cố định `100-120`,
>   nên sweep hiện tại chỉ đổi normalizer. Hoặc sửa caption của bảng, hoặc tham số hoá band trong
>   template và sinh lại draft (4 run GPU).
>
> ⚠️ **Phụ lục D:** `make_qualitative.py` đọc `features` của từng entity, nhưng
> `step_2_entity_selectors.py` (`_pack`) không ghi `features` vào `entity_debug.json`. Kết quả:
> mọi entity được chọn bị xếp vào "wrongly kept", nhóm "wrongly removed" luôn rỗng, và không có
> bảng điểm từng entity. Sửa `_pack` trước khi chạy.

Phụ lục C (efficiency): `timings` trong `runs/*/stats.json` chỉ là **tổng thời gian cả run**.
Bảng của bài cần thời gian **mỗi query**, **peak GPU memory**, và cả giai đoạn retrieval/re-ranking,
nên phải viết thêm script đo riêng trên server.

## B7. Gom lại

```bash
python scripts/collect_results.py --latex > table3_rows.tex
```

---

# PHẦN C — Sửa `main.tex`

> **Bản chính: `VGCap_ACIIDS2027/main.tex`** (nhóm xác nhận 2026-10-27). Mọi số dòng dưới đây theo
> bản đó. `main.tex` trong repo là một bản khác (có macro `\aciids`, abstract viết "four factors…
> span length"); **không sửa bản repo**. Đoạn nào trong bản repo muốn giữ thì chép sang bản chính.

Đi từ trên xuống. Cột "Sửa thành" là nội dung đúng theo code.

## C1. Bắt buộc sửa — hiện đang sai sự thật

| Dòng | Đang viết | Sửa thành |
|---|---|---|
| 100, 115, 122, 150, 171, 176, 267–270, 273, 406, 447 | E\* gate **cả prompt lẫn enrichment** ("unified gate", "gates both pathways", "gate 1: prompt", "differ only in the entity set supplied to the generator") | Chỉ gate ở **enrichment**. Hạ contribution 2 (dòng 122) xuống *"gating the enrichment step of length control"*. **Trừ khi** `gate_unified` ở B3 cho kết quả tốt hơn |
| 361–363 | Table 3(d): 0.6219 nằm ở dòng *Unified gating (VG-Cap)* | 0.6219 là run **enrichment-only**, phải chuyển sang dòng *Enrichment-only gating*. Dòng *Unified* điền bằng kết quả `gate_unified` |
| 168, 203–208, 216, 233 | Eq. (2) và Alg. 1 có `+P_span(e)`; câu "The span penalty discourages…"; caption Table 1 "except the span penalty"; dòng *Penalty* | Penalty **chưa bao giờ chạy**. Bỏ `P_span` khỏi Eq. (2) và Alg. 1, bỏ câu ở dòng 208 và dòng Penalty. **Trừ khi** dùng `--span_penalty paper`. (Bản chính không có cụm "four factors"; abstract đã liệt kê đúng 3 yếu tố) |
| 187 | `s_rerank = λ·s_img + (1−λ)·s_text`; todo "λ, softmax temperature, three detector thresholds" | Không có λ, không có temperature: `combined = logit_scale·cos(text) + cos(image)`. Detector là danh sách cố định 141 query có ba trường `entropy / gap / low_top1_sim` (`final_json_result/temp_three_ways_wrong_samples_set.json`); ngưỡng hoặc quota thật phải đọc `step_1_retrieval.py` (C3) |
| 187 | "the highest-scoring candidate determines the retrieved article" | Re-ranking **không đổi thứ hạng** (141/141 query thứ tự ra = thứ tự vào). Retrieval thực chất là image-only top-1. **Trừ khi** chạy lại với `--sort_reranked` |
| 171, 255 | `G_θ(q, C_article, C_image, E*; …)` | Llama-3-8B chỉ nhận chữ. Bỏ `q`: `G_θ(C_article, C_image, E*; L_min, L_max)`. Ảnh vào qua `generated_caption` (InternVL2.5-8B, xem C3) |
| 292 | "re-ranking descriptions use InternVL2.5-8B" | Ảnh **query** ở bước re-ranking dùng **InternVL2.5-4B**, ảnh database dùng 8B. Thêm: mô tả ảnh query trong prompt (`generated_caption`) do InternVL2.5-8B sinh |
| 100, 117, 447 | "raises CLIPScore over its ENRIC backbone from 0.833 to 0.858 **at an unchanged caption length**"; dòng 117 "while the mean caption length stays at 104 words" | ENRIC = **136 từ**, VG-Cap = 104. Sửa thành *"...over the unfiltered-entity configuration at an identical mean length of 104 words"* |
| 265 | "the prompt band acts mainly through its lower bound, avoiding drafts that would need heavy enrichment" | Đo thật: chỉ **40,5%** draft trong band 100–120; **57,2%** dưới 100; **71%** phải enrich. Viết đúng rằng enrichment là bước bắt buộc với đa số draft |
| 174, 263, 270 | pruning "protecting E\* and named entities"; "spaCy POS/NER mask that protects proper nouns, numbers"; "selected entity is protected during truncation" | Code chỉ bảo vệ entity spaCy **trong caption**, chạy **trước khi** tính E\*, không có POS mask. **Trừ khi** chạy với `--preserve_selected --pos_mask` |
| 176, 263, 270, 273, 441 | enrichment chỉ nối "e ∈ E\*"; "rejected entity cannot be re-introduced by enrichment"; leakage chỉ qua "summary, QA pairs, or editorial caption" | Hết E\* thì nối **noun chunks toàn bài**, ngoài gate. Thêm kênh rò này vào dòng 273 và Limitations (441). **Trừ khi** chạy `--no_noun_phrases` |
| 166, 198 | E = spaCy NER ∪ keyword list | Keyword list là markdown, bị `split(',')` → vào E dưới dạng 1–3 blob; **47% query** có blob trong E\*. Thực tế E ≈ chỉ spaCy. **Trừ khi** chạy `--fix_profile all` |
| 166, 198 | dedup "keeps the longer surface form (Joe Biden over Biden)" | Chỉ đúng cho nhánh keyword. `'Michelle Payne'` và `'Payne'` cùng tồn tại |
| 210 | "exact matches outweigh partial ones, which only absorb morphological variation" | Là **chuỗi con thô**: `'One'` khớp `"someone"` → 14.0 điểm, cao hơn `'Michelle Payne'` 8.5 |
| 211 | vị trí ảnh trong bài tìm bằng URL/HTML marker + fuzzy | `article_position` **bị bỏ rơi** ở `step_2_merge_all_elements.py`, không tới scorer. Bỏ câu này |
| 261 | câu prompt trích nguyên văn ("must strictly contain… Each sentence must add…") | Chép đúng từ `cot_5_things_fact_more_event.j2:34-35`: bản thật không có "strictly", không có "Each sentence must add" |
| 100, 113, 344, 395, 406 | baseline nhận "all named entities of the retrieved article" | `name_entity.j2:17` đã chỉ thị LLM *"Prioritize entities that appear visually in the image"*. Gọi đúng là **LLM-extracted, image-prioritized keyword list**: baseline mạnh hơn, gain của VG-Cap có giá trị hơn |
| 340–345, 371 | Ladder (a): "Adding entities with the prompt band reaches 0.6166"; hai dòng mới "+ prompt band only", "+ unfiltered entities only" | **Kiểm từ git:** `version1.json` (dòng ENRIC 0.6097 và +SGN 0.6124) sinh bằng template có **khối Keywords**, chỉ không có band. Vậy 0.6124 → 0.6166 chỉ đổi **length band**, không thêm entity. Dòng "+ unfiltered entities only" chính là dòng 0.6124 hiện có. Xem TODO §2 panel (a) cho ladder mới |
| 424 | PERSON removals "come from the ten-entity cap discarding secondary actors" | **Chưa kiểm.** Với gate ở enrichment, một PERSON bị loại có thể do vượt cap, hoặc do caption đã đủ 104 từ trước khi tới lượt nó. Kiểm bằng `entity_debug.json` của run tái dựng rồi mới giữ hoặc sửa |
| 117, 344–345, 406, 420, 424, 436 | #Ent 7.59 → 6.76; 22.759 → 20.276; −0.83, s.d. 2.44; 54.5 / 17.3 / 28.2%; 7.698 bị loại, 5.215 thêm; DATE 1.957, ORG 1.464, PERSON 1.433 | **Script đếm gốc đã mất** (nhóm xác nhận). Chốt **một** định nghĩa, ghi rõ trong §4.5, đếm lại **toàn bộ** các số này và mọi dòng #Ent bằng cùng một script, rồi vẽ lại Fig. 2. Tham khảo: cách đếm hiện tại của `stats.json` cho 12,725 → 11,004 (38.176 → 33.011) |
| 298, 327 | "paired bootstrap over queries" cho CIDEr/Overall | Leaderboard chỉ trả điểm tổng, không có điểm từng query, và không có công cụ chấm. Đổi giao thức: std qua 3 seed bằng số leaderboard; bootstrap trên CLIP cục bộ (ghi rõ là đo cục bộ) hoặc trên CIDEr của tập con train có GT |
| 380, 394–396 | P/R/F1 "over all 3,000 queries"; "Cohen's κ" cho cả 3 rater | Không có GT public test → tính P/R/F1 trên tập con train (`gt_train.csv`) và nói rõ. 2 annotator: Cohen's κ; 3 rater: **Fleiss' κ** |
| 491–492, 515–519 | Table A2: L\* sweep "prompt band shifted accordingly" | Sweep rẻ chỉ đổi normalizer. Hoặc sửa caption, hoặc tham số hoá band và sinh lại draft |

## C2. Điền được ngay, không cần chạy gì

| Dòng | Việc |
|---|---|
| 396 | Table 4 cột Leak: **chưa điền được**. Số 0.0808 (`runs/repro_vgcap/stats.json`, trên 221.659 entity bị loại) đo bằng chuỗi con thô và mẫu số có blob keyword, xem B5. Đo lại rồi mới điền |
| 289, 312–319 | CIDEr 4 chữ số: tính ngược thì ENRIC cần **0.2577** mới ra 0.6097; dòng 0.6166 cần **0.2639**; dòng 0.6124 cần **0.2598**. Đây là số **để kiểm tra**, không được đưa vào bảng như số đo. Số 4 chữ số thật phải lấy từ leaderboard |
| 295 | Câu *"within ±0.001"* **đúng**: recompute cả 7 dòng Table 2, lệch lớn nhất 0.0010 |
| 289 | Câu *"CIDEr = 0.2651 gives 0.6219 exactly"* **đúng** → 0.621916 |
| 420 | Fig. 2a nhãn trục `0.6218` → `0.6219`. Caption đã ghi "Gross removals"; kiểm nhãn bên trong ảnh Fig. 2b cho khớp. Số liệu của hình phải đếm lại (C1) |
| 432 | Fig. 3 nhãn `0.6166`/`0.62` → `Unfiltered`/`VG-Cap` |
| 150 | Vẽ lại Fig. 1: mũi tên từ scorer **chỉ trỏ vào normalizer** (trừ khi chọn unified), bỏ ô span penalty |
| 441 | Limitations: thêm noun-chunk leak và nhãn spaCy sai (`Michelle Payne`→ORG, `Suffragettes`→PERSON) |
| 261 | Release prompt template: `scripts/dump_prompts.py` |
| 564 | Repo URL cho Appendix E |
| 295 | ReCap tự báo 0.61619 vs 0.5987 trong bảng: **giá trị cao nhất trên công sức**. Nếu 0.61619 đúng thì lead của VG-Cap chỉ còn 0.006 và ReCap có CLIP cao hơn |

## C3. Đã kiểm — ĐÚNG, giữ nguyên

τ=3.0, K_e=10, max_words=105, expected_length=104, các trọng số presence/position
còn lại, thứ tự 5 thao tác SGN, danh sách preamble, `top_p=0.9`
/ `min_new=100` / `max_new=1500`, InternVL-14B-224px,
Llama-3-8B-Instruct, spaCy `en_core_web_sm`, `pre_top_k=15`, công thức Overall,
worked example §3.4 (15.0 / 3.5 / 1.0), "τ non-binding for short PERSON spans".

Thứ tự type prior trong bản chính **đã đúng**: dòng 229 ghi "GPE, EVENT 2.5", dòng 246 ghi
"PERSON outranks GPE/EVENT, ORG/LOC, and DATE". (Lỗi `PERSON > GPE > EVENT` chỉ có trong bản repo.)

⚠️ **`temperature` KHÔNG đúng như bài ghi (0.6).** Cả hai bộ draft đã công bố được commit cùng code
có `temperature=0.3` trong `Llama.assemble()` (commit `c004e52` cho `version1.json`, `2d47722` cho
`llama.json`). Giá trị chỉ đổi thành 0.6 ở `9a23067`, sau khi draft đã có. Sửa dòng 292 của bài
thành 0.3. Trước mọi run GPU mới, đưa temperature về 0.3 (hoặc thành cờ, mặc định 0.3), nếu không
run mới sẽ không so được với draft cũ. Template hiện tại cũng thêm chữ "in length" so với template
đã sinh `llama.json`; bỏ đi cho khớp. Xem `docs/VAST_RUNBOOK.md` §0.1.

Đã tra thêm (2026-10-27):
- **`generated_caption` gần như chắc chắn là InternVL2.5-8B.** `step_2_create_caption_query.py`
  mặc định `OpenGVLab/InternVL2_5-8B`, và log của script nạp phần ngôn ngữ **InternLM2**, đúng là
  phần ngôn ngữ của InternVL2.5-8B (bản 1B/4B và InternVL3-1B dùng Qwen2).
- **`version1.json` có khối Keywords.** File này giống hệt từng byte với `llama.json` ở commit
  `c004e52`, mà template ở commit đó **có** khối Keywords, **không** có band 100–120 (thay bằng
  "at least 5 sentences"). Ở `2d47722` file cũ được đổi tên thành `version1.json` và template chỉ
  đổi đúng các dòng về độ dài/số câu. Hệ quả cho ladder (a): xem C1 dòng 340–345.

Còn chưa xác nhận:
- Detector của Stage 1 (ngưỡng hay quota top-50 mỗi tiêu chí): chưa đối chiếu với
  `step_1_retrieval.py`. File danh sách có **141** query.

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
mkdir -p backup_pre_fix && git show 9a23067:step_2_entity_scorer.py > backup_pre_fix/step_2_entity_scorer.py.bak   # xem B0
python scripts/test_fixes.py --n 3000                                                       # 2 phut, [1] phai PASS
bash scripts/run_fixes.sh enrichment                                                        # ~2h CPU
python run_ablation.py --tag gate_unified_s0 --gate unified --selector heuristic --device $DEVICE  # ~6h GPU
```

Song song đó sửa toàn bộ **C2** (không cần chạy gì) và các dòng của **C1** không
phụ thuộc kết quả: 222/227/241 (EVENT), 197 (λ + detector), 167/181/267/280 (ảnh),
110/128/465 (unchanged length), 280 (trích prompt), 220 (chuỗi con), 221 (vị trí ảnh),
441 (generator), 413 (leakage).

Ba lệnh trên đủ để biết **có giữ được đóng góp "unified gating" không** và
**gain đến từ scorer hay từ budget** — hai câu hỏi quyết định bài này.
