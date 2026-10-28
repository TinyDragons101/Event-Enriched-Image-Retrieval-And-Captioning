# VG-Cap — Bản vá đúng đắn (correctness pass)

Cập nhật 2026-10-27. Tài liệu này ghi **đã sửa gì**, **bật/tắt bằng cách nào**, và
**chạy lệnh gì** để điền các ô TODO của `main.tex`.

> `VGCAP_FLOW.md` = code chạy thế nào · `VGCAP_TODO.md` = chạy cái gì và vì sao
> · `VGCAP_RUN.md` = chạy ra sao · **file này = đã sửa gì**.

---

## 0. Nguyên tắc: không có fix nào tự động bật

Mọi sửa lỗi đều là **switch, mặc định TẮT**. Config không có khối `fixes` sẽ chạy
y hệt bản đã công bố.

```
configs/entity_scoring_config.json         as-run,  Overall 0.6219
configs/entity_scoring_config_fixed.json   đã vá toàn bộ (chỉ khác khối "fixes")
```

Hai file có **weights và thresholds giống hệt nhau**, nên mọi chênh lệch điểm là do
bản vá, không phải do tinh chỉnh lại trọng số.

Kiểm chứng bất cứ lúc nào:

```bash
# backup_pre_fix/ bị gitignore, KHÔNG có trong repo -> khôi phục scorer gốc từ commit trước bản vá
mkdir -p backup_pre_fix
git show 9a23067:step_2_entity_scorer.py > backup_pre_fix/step_2_entity_scorer.py.bak
python scripts/test_fixes.py --n 3000      # dòng [1] phải PASS, không được SKIP
```

Đã chạy tại chỗ (300 query, spaCy **giả lập**): 300/300 query khớp E\* với module gốc.
⚠️ Kết quả này **chưa đủ**: NER giả lập không chứng minh được hành vi với spaCy thật. Phải chạy
lại trên server.

⚠️ Cờ `--reference runs/repro_vgcap/selected_entities.json` (kiểm tra `[2]`) **không** so với bản
đã công bố: file đó do chính code mới sinh ra, nên phép so là vòng tròn. Chỉ kiểm tra `[1]` có giá trị.

---

## 1. Sáu lỗi của scorer

| Switch | Lỗi | Ở đâu | Ảnh hưởng tới bài |
|---|---|---|---|
| `parse_keyword_markdown` | `parse_entity_string` tách danh sách keyword bằng dấu phẩy, nhưng LLM xuất **markdown bullet list** → cả danh sách vào $\mathcal{E}$ dưới dạng 1–3 blob ~300 ký tự. 47% query có blob lọt vào $\mathcal{E}^*$ | `step_2_entity_scorer.py` | §3.4 *"union of spaCy NER and the article keywords"* |
| `word_boundary_match` | presence feature dùng chuỗi con thô: `'One'` khớp `"someone"`, nên `'One'`=14.0 còn `'Michelle Payne'`=8.5 | nt | §3.4 *"exact matches outweigh partial ones"* |
| `span_penalty` | `compute_score` tra key `length_penalty`, weights chỉ khai `length_penalty_factor` → $P_{span}$ **chưa bao giờ chạy** | nt | Eq. (2), "four factors" |
| `substring_dedup` | dedup chỉ exact-lowercase giữa các entity spaCy → `'Michelle Payne'` và `'Payne'` cùng tồn tại | nt | §3.4 *"keeps the longer surface form"* |
| `first_mention_position` | `start == 0` bị dùng làm sentinel "không có vị trí" → mọi keyword mất position bonus | nt | §3.4 bullet *Article position* |
| `drop_non_depictable` | `'1'`, `'10'`, `'One'`, `'first'` là candidate hợp lệ | nt | §3.4 *Candidate construction* |

Ba giá trị của `span_penalty`:

| | Công thức | Ghi chú |
|---|---|---|
| `off` | 0 | **bản đã công bố** |
| `code` | $-0.1\,\lvert e\rvert$ khi $\lvert e\rvert>3$ | ý định của cấu trúc code |
| `paper` | $-0.1\max(0,\lvert e\rvert-3)$ | **đúng Eq. (2)** |

---

## 2. Hai lỗi của normalizer (Gate 2)

| Switch | Bài nói | Code thật |
|---|---|---|
| `--preserve_selected` | §3.6 bước (2): pruning *"preserves every mention of an entity in $\mathcal{E}^*$"* | chỉ bảo vệ entity spaCy **trong caption**, và chạy **trước khi** $\mathcal{E}^*$ được tính. Switch này đảo thứ tự hai bước |
| `--pos_mask` | §3.6: *"a spaCy POS/NER mask keeps other proper nouns and numbers intact"* | không có POS mask, chỉ `doc.ents` |
| `--no_noun_phrases` | §3.6: *"enriched using **only** entities from $\mathcal{E}^*$"* | hết $\mathcal{E}^*$ thì nối **noun chunks toàn bài**, ngoài gate. Chính là `"the Tiger Woods."` ở §4.7 |

---

## 3. Bốn lỗi khiến pipeline không chạy được

| Lỗi | Đã sửa |
|---|---|
| `create_submission()` không nhận tham số, nhưng `step_2_caption_process.py` gọi `create_submission(final_caption_json_path=...)` → `TypeError`. Đây là lý do `submission_final.csv` = draft thô chứ không phải VG-Cap | thêm tham số + `load_caption_map()` đọc được cả hai dạng (dict của LLM và list của normalizer) |
| `merge_function()` và việc nạp template chạy ngay lúc **import** → chỉ `--help` cũng dựng lại `final_merge_result.json` | chuyển vào `main()`; thêm `--merge` |
| Script **im lặng tái dùng** caption của cấu hình trước nếu file đã tồn tại — chí mạng cho bảng ablation | thêm `--force`, `--run_suffix`, `--reuse_drafts` (chặn khi `--gate prompt/unified` vì gate đó đổi prompt) |
| `import transformers` ở module level → không xem được prompt nếu chưa cài | import LLM lazy; `--template_test` và `--help` chạy được trên laptop |

---

## 4. Lệnh chạy cho từng TODO của bài

Tất cả dùng `run_ablation.py`; mỗi run nằm gọn trong `runs/<tag>/`.
Với `--gate enrichment` thì **tái dùng được draft đã lưu → 0 GPU, ~15 phút/run**.

### 4.1 Table 3(d) — Gate location (TODO R1.2/R4.2, P1)

```bash
bash scripts/run_panels.sh enrichment      # gate none / enrichment
# hai dòng này đổi prompt nên bắt buộc sinh lại (~6h GPU mỗi dòng)
python run_ablation.py --tag gate_prompt_s0  --gate prompt  --selector heuristic
python run_ablation.py --tag gate_unified_s0 --gate unified --selector heuristic
```

### 4.2 Table 3(b) — Budget-matched selectors (P1)

```bash
# chạy sau khi đã có runs/vgcap_enrichment_s0/selected_entities.json
bash scripts/run_panels.sh enrichment
```

### 4.3 Table 3(c) — Leave-one-family-out

```bash
python scripts/make_ablation_configs.py        # sinh 12 config
bash scripts/run_panels.sh enrichment          # đã bao gồm panel (c)
# chạy panel (c) trên scorer đã vá:
BASE_CFG=configs/entity_scoring_config_fixed.json CFG_SUFFIX=_fixed \
  bash scripts/run_panels.sh enrichment
```

⚠️ Dòng "w/o span penalty":
- Scorer **cũ**: penalty không bao giờ chạy, nên dòng này **chính là** run VG-Cap tham chiếu; không
  cần chạy riêng, chỉ cần ghi rõ trong bài.
- Scorer **đã vá**: cả hai config `ablation_span_penalty_*_fixed.json` đều **bật** penalty (trùng
  `fix_all`). Chưa có config tắt penalty trên nền đã vá; cần thêm `ablation_no_span`
  (`fixes.span_penalty = "off"`) vào `make_ablation_configs.py`.

### 4.4 Panel (e) mới — lỗi nào mới là nguyên nhân?

```bash
bash scripts/run_fixes.sh enrichment
```

Sinh 10 dòng: `legacy`, từng fix một, `all`, `all + preserve_selected + pos_mask`,
và `all + không noun-phrase`. Toàn bộ tái dùng draft → **0 GPU**.

Đây là bảng quan trọng nhất: nó trả lời câu *"0.6219 đến từ thiết kế scorer hay từ
việc giới hạn budget?"* mà reviewer đang hỏi ở panel (b).

### 4.5 Appendix A — Sensitivity

⚠️ Chỉ dùng `enrichment`: `run_appendix.sh unified` dừng ngay ở lệnh đầu (phần L\* luôn truyền
`--reuse_drafts`). Sweep L\* hiện **không** dịch length band trong prompt, trái với caption
Table A2 của bài ("prompt band shifted accordingly").

```bash
bash scripts/run_appendix.sh enrichment
# hoặc từng điểm:
python run_ablation.py --tag tau4_k15 --gate enrichment --tau 4.0 --top_k 15 \
    --reuse_drafts assemble_result/cot_5_things_fact_more_event_llama.json
python run_ablation.py --tag len90 --gate enrichment --expected_length 90 \
    --reuse_drafts assemble_result/cot_5_things_fact_more_event_llama.json
```

### 4.6 Statistical protocol — 3 seeds (TODO SAR)

Bản đã công bố **không set seed**. Giờ có `--seed` (cả hai driver).

```bash
for S in 0 1 2; do
  python run_ablation.py --tag vgcap_s$S --gate enrichment --seed $S
done
python scripts/bootstrap.py --a runs/gate_none_s0 --b runs/vgcap_s0 --metric cider
```

⚠️ Vòng lặp trên không tái dùng draft, nên mỗi lệnh sinh mới (~6h GPU, tổng ~18h).
`--metric cider` cần CIDEr từng query, tức là cần GT public test (hiện chưa có); không có GT
thì chỉ dùng được `--metric clip`.

### 4.7 Table 4 — Leakage: có số nhưng chưa dùng được

`runs/<tag>/stats.json` có `leakage_rate` (repro_vgcap: 0.0808). ⚠️ Số này khớp entity bằng chuỗi
con thô (`'One'` khớp "someone") và mẫu số có các blob keyword ~300 ký tự (không bao giờ xuất hiện
trong caption) → bị kéo thấp. Sửa cách đo trước khi đưa vào Table 4.
P/R/F1 và annotation sheet: `python scripts/eval_entities.py --run ... --gt ...` (cần GT).

---

## 5. Pipeline gốc (không dùng ablation driver)

```bash
# VG-Cap như đã công bố
python step_2_caption_process.py --gate enrichment

# VG-Cap với scorer đã vá + Gate 2 đúng như §3.6
python step_2_caption_process.py --gate enrichment \
    --entity_config_path configs/entity_scoring_config_fixed.json \
    --preserve_selected --pos_mask --no_noun_phrases \
    --run_suffix fixed --seed 0

# unified gating (§3.6) — đổi prompt nên phải sinh lại
python step_2_caption_process.py --gate unified --run_suffix unified --seed 0
```

---

## 5b. Hai chỗ ở Stage 1 / merge

### `crawl_caption` — đo xong thì **code đang đúng**

`step_2_0_create_result.py` lưu hai chuỗi cho mỗi ảnh database: `crawl_alt`
(thuộc tính `alt`) và `crawl_caption` (text của `div.image__metadata`).
Pipeline dùng `crawl_alt`. Đo trên cả 3000 bản ghi của `result-tanphuc.json`:

| | |
|---|---|
| `alt` == `caption` | 38 (1,3%) |
| `caption` = `alt` + credit ảnh | 2.254 (75,1%) |
| khác hẳn | 708 (23,6%) — `caption` = `"<kicker> —<alt><credit>"` |
| `alt` rỗng nhưng `caption` có | 11 |

Nghĩa là figure caption chính là alt text bọc thêm kicker và credit ảnh.
Dùng nó sẽ đẩy `Getty Images`, `AFP`, `Reuters` vào `exact_match_crawl` —
feature trọng số cao nhất (10.0). Kết quả ba chế độ:

| `--caption_source` | rỗng | độ dài TB | còn dính tên hãng ảnh |
|---|---|---|---|
| `alt` (mặc định, như đã công bố) | 19 | 22,0 từ | 2,5% |
| `caption` (đã bóc kicker + credit) | 91 | 22,0 từ | **14,5%** |
| `merged` (alt, rỗng thì dùng caption đã bóc) | **9** | 21,9 từ | 2,6% |

**Kết luận: `merged` tốt hơn `alt` một chút** (10 bản ghi được cứu, nhiễu không
tăng); `caption` thô thì tệ hơn hẳn. Mặc định vẫn là `alt` để tái dựng 0.6219.

```bash
python step_2_caption_process.py --merge --caption_source merged --gate enrichment
```

Với bài: §3.3 nên viết đúng là *the `alt` text of the image element, which on this
corpus carries the editorial caption*, và nói rõ figure caption chỉ thêm kicker +
photo credit. Không được mô tả nó như hai nguồn khác nhau.

### `step_1_rerank.py` — re-rank cũ là **no-op hoàn toàn**

Code dựng `(image_id, combined_score)` rồi ghi
`[img_id for img_id, _ in rerank_scores]` mà **không sort**. Đối chiếu
`rerank_caption.json` với `rerank_results.json`:

```
141/141 query  thu tu RA == thu tu VAO   (100%)
141/141 query  top-1 khong doi           (100%)
```

Tức là toàn bộ bước text re-ranking chạy InternVL trên 141×15 candidate rồi
**vứt điểm đi**. Mọi số retrieval đã công bố (AP 0.992, R@1 0.985) là top-1 thuần
theo image similarity.

Đã thêm hai cờ, cả hai mặc định giữ hành vi cũ:

```bash
# thực sự re-rank
python step_1_rerank.py --sort_reranked

# đúng công thức §3.2: lambda*s_img + (1-lambda)*s_text (bỏ logit scale)
python step_1_rerank.py --sort_reranked --fusion_lambda 0.5
```

Tiện thể sửa một lỗi hiệu năng: `torch.load("logit_scale.pt")` nằm trong vòng lặp
trong cùng nên bị đọc lại từ đĩa 141×15 lần; nay nạp một lần.

**Cảnh báo:** bật `--sort_reranked` sẽ đổi bài báo được truy hồi cho tối đa 141
query → đổi `final_merge_result.json` → **phải sinh lại toàn bộ caption**. Đừng
bật nếu còn muốn so với 0.6219.

Với bài: §3.2 hiện mô tả một bước re-ranking không có tác dụng. Hoặc chạy lại với
`--sort_reranked`, hoặc nói thẳng rằng detector chọn ≤150 query nhưng re-ranking
không đổi top-1 nên retrieval thực chất là image-only.

---

## 6. Cần thầy quyết

1. **Chưa chạy thật trên 3000 query với spaCy thật.** Máy soạn bản vá không có spaCy
   nên mọi smoke test dùng NER giả lập. Phải chạy `scripts/test_fixes.py` rồi
   `scripts/run_fixes.sh` trên máy có GPU trước khi tin bất kỳ con số nào.
2. **Bật `--sort_reranked` hay không.** Bật thì §3.2 đúng nhưng phải sinh lại toàn
   bộ caption; không bật thì phải viết lại §3.2 cho đúng rằng re-ranking là no-op.
3. **Bật `--caption_source merged` hay không.** Lợi ích nhỏ (10 bản ghi), nhưng cũng
   phải sinh lại caption.
4. **Trọng số EVENT = 2.5**, bằng GPE. Code không sai; **bài** mới sai khi ghi
   `PERSON > GPE > EVENT > ORG/LOC`. Sửa ở `main.tex`, không sửa code.

---

## 7. File đã đổi

```
step_2_entity_scorer.py        viết lại: 6 switch, spaCy lazy-load
step_2_caption_normalizer.py   + preserve_selected, pos_mask
step_2_caption_process.py      viết lại: --gate, --seed, --force, --run_suffix,
                               --reuse_drafts, bỏ side effect lúc import
step_2_merge_all_elements.py   create_submission() nhận tham số + load_caption_map()
                               + pick_crawl_caption() / --caption_source
step_1_rerank.py               + --sort_reranked, --fusion_lambda; nạp logit_scale 1 lần
step_2_entity_selectors.py     bỏ nguy cơ cộng đôi span penalty
run_ablation.py                + --fix_profile, --fix, --span_penalty,
                               --preserve_selected, --pos_mask
scripts/make_ablation_configs.py  sinh thêm bản *_fixed
scripts/run_panels.sh          BASE_CFG/CFG_SUFFIX; sửa tên config span penalty
scripts/run_fixes.sh           MỚI — panel (e)
scripts/test_fixes.py          MỚI — test hồi quy
configs/entity_scoring_config_fixed.json   MỚI
backup_pre_fix/                bản gốc của mọi file bị sửa -- bị gitignore, KHÔNG có trong repo;
                               khôi phục bằng: git show 9a23067:<file> > backup_pre_fix/<file>.bak
```
