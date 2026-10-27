# VG-Cap — Danh sách việc cần làm (bản revised MMM 2027)

Cập nhật: 2026-10-27. Nguồn: 19 `\todo` trong manuscript + đối chiếu code trong repo.

---

## 0. Blocker — đã rõ tình hình

Cập nhật 2026-10-27 sau khi kiểm tra thực tế.

### B1. Không có ground truth public test → CIDEr chỉ lấy được từ leaderboard

Repo không có reference caption của public test, và chưa rõ ban tổ chức có phát hành không.
Hệ quả chia làm hai:

| Chỉ số | Tính cục bộ được? | Cách lấy |
|---|---|---|
| **CLIPScore** | ✅ có | reference-free, chỉ cần ảnh query — `scripts/evaluate.py --query_images ...` |
| **CIDEr, Overall** | ❌ không | phải submit CSV lên leaderboard |
| **Len, #Ent, Leakage** | ✅ có | `stats.json` của mỗi run, không cần gì thêm |
| **Entity P/R/F1 (Table 4)** | ❌ không | cần reference caption |

Leaderboard **không giới hạn số lần submit**, nên hướng đi là: sinh hết CSV → submit từng cái →
`scripts/record_score.py` ghi điểm ngược lại → `scripts/collect_results.py --latex`.

Ba lựa chọn cho cột Entity P/R/F1: (1) chạy trên tập con **train split** có GT và báo cáo riêng —
cũng là điều phụ lục A của bài đã hứa; (2) bỏ cột, chỉ giữ unsupported-rate và human eval;
(3) hỏi ban tổ chức EVENTA.

### B2. Không set seed → đã xử lý

`llama3.py:128` dùng `do_sample=True, temperature=0.6, top_p=0.9` và không có `set_seed` ở đâu.
`run_ablation.py --seed` đã gọi `transformers.set_seed`. Lưu ý: seed chỉ ảnh hưởng 4 run sinh mới;
18 run tái dùng draft sẽ cho caption y hệt nhau giữa các seed.

### B3 (mới). Caption của run 0.6219 không có trong repo

`normalized_captions.json` chưa từng được commit, và `submission_final.csv` là draft thô chưa qua
SGN (trùng 100% với `llama.json`, độ dài TB 97.4 từ, 25.6% caption dài hơn 104 từ).

**Không chặn đường**, vì repo có sẵn **hai bộ draft** cho phép tái dựng cả 4 dòng đã công bố mà
không cần GPU:

```
llama.json      97.4 tu  -> 0.6166 va 0.6219
version1.json  135.2 tu  -> ENRIC baseline (bai ghi 136) va +SGN (0.6124)
```

Đây thành **bước neo**: tái dựng 4 dòng, kiểm `Len`/`#Ent` ngay tại chỗ (không cần submit), rồi
submit để kiểm Overall. Bốn điểm neo phân biệt được lỗi evaluator (cả 4 lệch cùng hệ số) với lỗi
pipeline (chỉ vài dòng lệch).

## 1. Quyết định phải chốt TRƯỚC khi tốn giờ GPU

**Code hiện tại là enrichment-only gating, không phải unified gating như §3.6 mô tả.**
`compose_input` (`step_2_caption_process.py:77`) luôn truyền `name_entity_keyword` thô vào prompt,
bất kể `--use_entity_scoring`; scorer chỉ được gọi trong normalizer. Template lại dùng
`{{ name_entity_keyword }}` ở mục "Keywords ... must be included".

⇒ Con số **0.6219 đã công bố là enrichment-only**. Dòng "Unified gating (VG-Cap)" ở panel (d)
hiện đang trùng với dòng "Enrichment-only gating".

Hai lựa chọn:

| | Giữ `enrichment` | Chuyển sang `unified` |
|---|---|---|
| Khớp số đã công bố | ✅ | ❌ phải chạy lại **tất cả**, kể cả Table 1 |
| Khớp §3.6 của bài | ❌ phải viết lại §3.6 | ✅ |
| Chi phí panel (b)+(c) | **0 giờ GPU** (tái dùng draft, đã kiểm) | ~78 giờ GPU |

**Đề xuất:** chạy `--gate unified` cho đúng **một** dòng VG-Cap trước (6h) để biết unified hơn hay
kém enrichment-only. Hơn → chuyển hẳn. Không hơn → sửa §3.6 cho khớp code, giữ toàn bộ số cũ.

---

## 2. Table 3 — bốn panel làm gì

Chạy tất cả: `bash scripts/run_panels.sh enrichment` (hoặc `unified`).

### Panel (a) — Component ladder (6 dòng)
**Câu hỏi:** mỗi thành phần đóng góp bao nhiêu khi cộng dồn?

| Dòng | Trạng thái | Chạy bằng |
|---|---|---|
| ENRIC baseline (reproduced) | ✅ 0.6097 — thiếu ô `#Ent` | — |
| + semantic normalizer (SGN) | ✅ 0.6124 — thiếu ô `#Ent` | — |
| + prompt band only | ❌ **TODO R3.6/R4.2** | prompt band, không entity |
| + unfiltered entities only | ❌ **TODO R3.6/R4.2** | entity, không prompt band |
| + unfiltered entities, prompt band | ✅ 0.6166 | `--gate none` |
| + heuristic selector (VG-Cap) | ✅ 0.6219 | `--gate enrichment --selector heuristic` |

Hai dòng mới tách tác dụng của *prompt band* khỏi *entity conditioning* — ladder gốc cộng cả hai
cùng lúc nên không biết cái nào có công.

⚠️ Ô `#Ent` của hai dòng đầu điền được ngay từ caption đã có (`stats.json` tự tính).

### Panel (b) — Alternative selectors, matched budget (4 dòng)
**Câu hỏi:** cái làm nên kết quả là *công thức chấm điểm*, hay chỉ là việc *giới hạn số entity*?

Đây là confound thật: VG-Cap vừa giảm entity (7.59→6.76) vừa đổi *entity nào*. Panel (b) giữ nguyên
ngân sách, chỉ đổi cách chọn.

| Dòng | Chọn thế nào | Vai trò | Lệnh |
|---|---|---|---|
| Article order (first 10) | 10 entity nhắc sớm nhất | Baseline hợp lý, chỉ dùng front-loading | `--selector article_order` |
| Random 10 (mean of 5 draws) | Bốc ngẫu nhiên | **Sàn dưới.** ≈0.62 ⇒ bài hỏng | `--selector random --seed 1..5` |
| Caption match only | Bắt buộc khớp editorial/generated caption | Cô lập tín hiệu *presence* | `--selector presence_only` |
| Type prior only | Chỉ xếp hạng theo loại entity | Cô lập tín hiệu *type* | `--selector type_only` |

⚠️ **"Matched budget $K_e=10$" như bài đang viết là KHÔNG matched.** Article-order và Random luôn lấy
đủ 10, VG-Cap trung bình chỉ 6.76 (vì còn ngưỡng τ). Đã xử lý bằng `--budget_match per_query`: mỗi
baseline nhận đúng số entity VG-Cap chọn cho *chính query đó*.

⚠️ Nếu "Caption match only" một mình đã ra ~0.62 thì thông điệp bài phải đổi: không cần hệ 3 họ
đặc trưng, chỉ cần "entity nào có trong caption thì giữ".

### Panel (c) — Leave one feature family out (5 dòng)
**Câu hỏi:** trong ba họ đặc trưng, họ nào thực sự tạo ra kết quả?

| Dòng | Config | Ghi chú |
|---|---|---|
| w/o contextual presence | `ablation_no_presence.json` | Điểm tối đa còn 5.0 → τ rất chặt |
| ├ w/o editorial-caption evidence only | `ablation_no_editorial.json` | **Quan trọng nhất cả bảng** |
| w/o article position | `ablation_no_position.json` | |
| w/o entity-type prior | `ablation_no_type.json` | PERSON mất "vé miễn phí" qua τ → #Ent tụt mạnh |
| w/o span penalty | — xem bên dưới | |

**Vì sao dòng "w/o editorial evidence" là kiểm tra tính hợp lệ, không phải ablation thường:**
`crawl_caption` chính là editorial caption, mang trọng số cao nhất (10.0). Query image là biến thể
của ảnh database, retrieval R@1 = 0.985 → editorial caption lấy về gần như luôn là caption gốc của
chính bức ảnh query. Nếu reference caption của OpenEvents-V1 cũng dẫn xuất từ editorial caption thì
scorer đang đọc gần-ground-truth. **Chạy dòng này trước mọi dòng ablation khác.**

**Về dòng "w/o span penalty" — span penalty là dead code:**
features sinh key `length_penalty`, weights khai báo key `length_penalty_factor`
→ `self.weights.get('length_penalty', 0.0)` trả về **0.0**. Eq. (2) chưa bao giờ chạy.
⇒ Dòng "w/o span penalty" sẽ ra **y hệt** VG-Cap. Hai config thay thế để bật nó lên:
- `ablation_span_penalty_on.json` — theo ngữ nghĩa code: `−0.1 × số_từ`
- `ablation_span_penalty_paper.json` — theo đúng Eq. (2): `−0.1 × max(0, số_từ − 3)`

Kiểm chứng trên span 5 từ "the Australian Grand Prix circuit": **10.5** (hiện tại) / **10.0** / **10.3**.

⚠️ Tắt một họ làm đổi số entity được chọn (τ giữ nguyên) → lại rơi vào đúng confound panel (b) muốn
loại. Rõ nhất ở "w/o entity-type prior". **Phải báo cáo cột `#Ent` cho từng dòng** và giải thích,
hoặc hiệu chỉnh τ cho mỗi biến thể về `#Ent` ≈ 6.76.

### Panel (d) — Gate location (3 dòng)
**Câu hỏi:** chặn entity ở đâu thì có tác dụng — lúc viết prompt, lúc enrichment, hay cả hai?

| Dòng | Prompt | Enrichment | Lệnh |
|---|---|---|---|
| Prompt-only gating | E* | tất cả entity bài báo | `--gate prompt` |
| Enrichment-only gating | keyword thô | E* | `--gate enrichment` ← **số 0.6219 hiện tại** |
| Unified gating (VG-Cap) | E* | E* | `--gate unified` ← **đúng như §3.6** |

Thêm `--gate none` = baseline unfiltered (0.6166).

Đây là panel chứng minh luận điểm "unified gate là cần thiết": nếu enrichment-only ≈ unified thì
đóng góp "unified gating" không có giá trị thực nghiệm.

⚠️ Cả ba dòng đều đụng prompt (trừ enrichment) nên **không tái dùng draft được** → mỗi dòng 1 lần
sinh (~6h).

---

## 3. Table 4 — Entity-level grounding (toàn bộ TBD)

| Cột | Cần gì | Chi phí |
|---|---|---|
| Entity P/R/F1 | So entity của caption với entity của reference caption, 3000 query | Cần B1 (ground truth) |
| Unsupported-entity rate | 200 query × 2 annotator, nhãn: visually supported / contextually supported / unsupported | **Người**, 1–2 tuần |
| Leakage rate | % entity bị loại nhưng vẫn xuất hiện trong caption cuối | ✅ `stats.json` tự tính |
| Human 1–5 (Fact./Flu./Inf.) | 3 rater, cùng 200 query, blind + random order | **Người** |

⚠️ Bài đang viết "Cohen's κ" nhưng dùng 3 rater → phải là **Fleiss' κ**. Chỗ 2 annotator
(unsupported rate) thì Cohen's κ đúng.

⚠️ Khâu người mất **lịch tuần**, không phải giờ máy → khởi động sớm nhất có thể, song song với B1.

---

## 4. Phụ lục

| Phụ lục | Nội dung | Chi phí |
|---|---|---|
| A — Sensitivity τ × K_e (15 ô) | Đổi prompt → phải sinh lại | Đắt; cân nhắc subsample 500–1000 query |
| A — Weight perturbation (7 ô) | ×0.5 / ×1.5 từng họ + random ±20% | Đắt |
| A — Length sweep L* (12 ô) | Chỉ đổi `--expected_length` | ✅ **Rẻ**, tái dùng draft |
| B — Learned selector | Logistic regression trên cùng feature, label = entity có trong reference caption | Cần B1 + 1 lần sinh |
| C — Efficiency | Latency + peak memory từng stage | 1 pass đo |
| D — Qualitative 6–8 ví dụ | Dùng `runs/<tag>/entity_debug.json` (có score breakdown từng entity) | ✅ Rẻ |
| E — Reproducibility | Release code, prompt, seed, annotation | ✅ Rẻ |

---

## 5. Kiểm kê 19 TODO

### Nhóm A — làm ngay, 0 GPU

| # | Việc | Ghi chú |
|---|---|---|
| T7 | ReCap 0.61619 vs 0.5987 trong bảng | **Giá trị cao nhất trên công sức.** ReCap tự báo cáo 0.61619; nếu đúng thì lead của VG-Cap chỉ còn 0.006 và ReCap có CLIP cao hơn (0.872 vs 0.858) → claim "best Overall" phụ thuộc hoàn toàn vào dòng này |
| T6 | CIDEr 4 chữ số | ENRIC cần CIDEr ≈ **0.2577** mới ra 0.6097 (0.257 → 0.6089). Dòng 0.6166 recompute ra **0.6156** |
| T2 | λ, softmax temperature, 3 ngưỡng detector | Đọc từ `step_1_rerank.py` |
| T4 | Release prompt template | Có sẵn `assemble_caption_prompt_template/*.j2` |
| T13 | Fig.2a nhãn trục 0.6218 → 0.6219 | |
| T14 | Fig.3 nhãn "0.6166"/"0.62" → tên method | |
| T1 | Vẽ lại Fig.1 overview | TODO còn ghi "check which figure correct" |
| T19 | Reproducibility, repo URL | |
| — | Ô `#Ent` trống ở Table 3(a) | Đếm từ caption đã có |

### Nhóm B — rẻ, chỉ CPU (tái dùng draft đã lưu)

| # | Việc |
|---|---|
| T11(d) | Panel (d) dòng enrichment-only — tái dùng draft VG-Cap |
| T15 | Sweep `L*` (80/90/104/110/120) |
| T12 | Leakage rate |
| T18 | Ví dụ định tính + score breakdown |
| T5 | Biến thể `--no_noun_phrases` (chặn rò rỉ qua noun phrase) |
| T17 | Bảng efficiency |
| T11(b)(c) | **Nếu chọn gate `enrichment`** → cũng chỉ CPU |

### Nhóm C — đắt, cần GPU (~6h/config)

| # | Việc | Số lần sinh |
|---|---|---|
| T3 | Dòng "w/o editorial evidence" | 1 — **ưu tiên cao nhất nhóm này** |
| T11(b) | Panel (b) | 8 (random cần 5 draw) |
| T11(c) | Panel (c) | 5 |
| T10 | 2 dòng mới của ladder | 2 |
| T11(d) | Panel (d) prompt-only + unified | 2 |
| T16 | Learned selector | 1 |
| T8, T9 | 3 seeds + bootstrap CI | ×3 toàn bộ |

### Nhóm D — tổ chức, khởi động sớm

| # | Việc |
|---|---|
| T12 | 200 query × 2 annotator + 3 rater, viết guideline, tuyển người |

---

## 6. Thứ tự đề xuất

1. **Hôm nay** — T7 (ReCap), T6 (4 chữ số), thêm seeding, các sửa text/hình nhóm A.
2. **Ngày 2–3** — dựng evaluator cục bộ, validate lại 0.6219/0.6097. Song song: khởi động T12
   (viết guideline, chuẩn bị annotator).
3. **Ngày 3–4** — batch hoá generation. Log cho thấy đang chạy tuần tự
   (`You seem to be using the pipelines sequentially on GPU`); vLLM hoặc batched `generate`
   thường cho 5–10× trên A100 80GB với Llama-3-8B → **6h còn ~40–60 phút**. Đòn bẩy lớn nhất.
4. **Ngày 4–5** — quét hết nhóm B. Nếu chốt gate `enrichment` thì panel (b)(c) nằm luôn ở đây.
5. **Tuần 2** — nhóm C theo thứ tự rủi ro: w/o editorial → panel (b) → panel (c) → 3 seeds.
6. **Cắt giảm nếu không kịp** — grid τ×K đầy đủ và learned selector chạy trên subsample 500–1000
   query (khai báo rõ trong bài); giữ full 3000 cho Table 1 và ladder chính.

---

## 7. Rủi ro cần theo dõi

- **Nếu đến hạn panel (b) và Table 4 vẫn trống → phải sửa abstract.** Câu
  *"Controlled ablations against simple selection baselines, together with entity-level grounding
  analysis, indicate that the gain comes from which entities are selected"* hiện chưa có dữ liệu nào
  trong bài chống lưng. R1/R4 hỏi đúng chỗ đó.
- **Abstract đang gộp hai so sánh khác nhau.** "raises CLIPScore over its ENRIC backbone from 0.833
  to 0.858 **at an unchanged caption length**" — so với ENRIC thì độ dài *có* đổi (136 → 104 từ);
  chỉ so với baseline unfiltered-entities mới là không đổi. Điều này vô tình làm yếu chính luận điểm
  "không phải do độ dài".
- **CLIPScore nhảy 0.833 → 0.858** chỉ với ~0.83 entity/caption khác biệt, trong khi bước trước đó
  (thêm toàn bộ entity + prompt band) không làm CLIP nhúc nhích. Cần kiểm lại pipeline của hai run
  và giải thích cơ chế, nếu không reviewer sẽ nghi là lỗi.
- **`\showrevisionsfalse` sẽ `\PackageError`** khi còn `\todo` — cơ chế tốt, không placeholder nào
  lọt vào camera-ready. Nhớ bỏ `\includepdf` revision log khi nộp bản cuối.

---

## 8. Phụ lục: file đã thêm vào repo

| File | Nội dung |
|---|---|
| `configs/entity_scoring_config.json` | VG-Cap as-run, tường minh |
| `configs/ablation_*.json` (6) | Panel (c), sinh tự động |
| `scripts/make_ablation_configs.py` | Bộ sinh config trên |
| `step_2_entity_selectors.py` | 5 selector panel (b), `PaperSpanPenaltyScorer`, budget matching |
| `run_ablation.py` | Driver: 4 gate, seed, output tách theo run, thống kê |
| `scripts/run_panels.sh` | Chạy trọn panel (b)(c)(d) |
| `scripts/collect_results.py` | Gom `runs/*/stats.json` → bảng + dòng LaTeX |
| `step_2_caption_normalizer.py` | Sửa nhỏ: `selection_map`, `use_noun_phrases` |

**Chưa test end-to-end** — máy local không có `spacy`/`torch`. Mới chỉ smoke-test logic selector với
spaCy giả lập (pass) + compile + CLI. Lần chạy thật đầu tiên trên server nên dùng `--limit 20`.

### Lỗi phụ phát hiện khi đọc code (chưa sửa)
- `create_submission()` trong `step_2_merge_all_elements.py` **không nhận tham số**, nhưng
  `step_2_caption_process.py:171` gọi `create_submission(final_caption_json_path=...)` → TypeError.
  Bản `_copy` mới có tham số.
- Cả hai bản `create_submission` đọc dict `{query_id: caption}`, trong khi `normalize_batch` lưu
  **list** các record → `.get()` trên list sẽ lỗi. `run_ablation.py` tự ghi submission nên không dính.
- `calculate_position_features` coi `entity.start == 0` là "không có thông tin vị trí", nên entity
  đứng đầu bài báo bị mất bonus; entity parse từ keyword LLM luôn `start=0` nên **không bao giờ**
  được position bonus.
- Vòng lặp sinh caption cũ ghi lại toàn bộ dict JSON sau **mỗi** query (3000 lần ghi một dict đang
  lớn dần). `run_ablation.py` checkpoint mỗi 50 query.

---

## 9. Bảng ánh xạ: TODO -> lệnh chạy

Mọi TODO cần số đều đã có đường chạy. Thứ tự bắt buộc: **(1) ground truth -> (2) validate
evaluator -> (3) chạy run -> (4) chấm điểm -> (5) gom bảng.**

### Bước 0 — validate evaluator trước khi tin bất kỳ số mới nào

```bash
# tái tạo con số đã công bố từ caption cũ; sai thì dừng lại, đừng chạy tiếp
python scripts/evaluate.py --run runs/vgcap_enrichment_s0 \
    --gt data/public_test_gt.json --query_images data/track1_public/query \
    --expect '{"overall":0.6219,"cider":0.2651,"clip":0.858}'
```

⚠️ Thang đo CIDEr phải khớp: bản cài trong `scripts/evaluate.py` trả về thang x10 (giống hệt
reference = 10.0). Nếu `--expect` báo lệch đúng hệ số 10 thì chia/nhân lại rồi ghi rõ trong bài.

### Table 3

| Panel | TODO | Lệnh |
|---|---|---|
| (a) 2 dòng mới + dòng ENRIC/SGN | T10 | `bash scripts/run_panels.sh` (đã bao gồm) |
| (b) 4 selector | T11 | `--selector article_order\|random\|presence_only\|type_only --budget_match per_query` |
| (c) 5 dòng leave-one-out | T11, T3 | `--entity_config configs/ablation_*.json` |
| (d) 3 vị trí gate | T11 | `--gate prompt\|enrichment\|unified` |

```bash
bash scripts/run_panels.sh enrichment     # (b),(c) tái dùng draft -> ~0 GPU
bash scripts/run_panels.sh unified        # đúng §3.6 -> mỗi dòng ~6h
SEED=1 bash scripts/run_panels.sh         # T8: lặp cho seed 1, 2
```

### Table 4

| Cột | TODO | Lệnh |
|---|---|---|
| Entity P/R/F1 + Leak | T12 | `python scripts/eval_entities.py --run runs/<tag> --gt <gt>` |
| Unsup. + Human 1-5 | T12 | cùng lệnh trên sinh `annotation_sheet.csv` cho người chấm |

### Phụ lục

| Phụ lục | TODO | Lệnh |
|---|---|---|
| A — L* sweep (rẻ) | T15 | `bash scripts/run_appendix.sh` (phần A.1) |
| A — tau x K_e, weight | T15 | cùng script, phần A.2/A.3; hoặc `--tau --top_k --weight_scale --weight_jitter` |
| B — learned selector | T16 | `python scripts/train_learned_selector.py` rồi `--selector learned --learned_coefficients ...` |
| C — efficiency | T17 | `stats.json` của mỗi run đã có khối `timings` |
| D — qualitative | T18 | `python scripts/make_qualitative.py --run <vgcap> --baseline <unfiltered>` |
| E — prompt template | T4 | `python scripts/dump_prompts.py --n 3 --out supplementary/prompts` |

### Thống kê

| Việc | TODO | Lệnh |
|---|---|---|
| Bootstrap CI | T8, T9 | `python scripts/bootstrap.py --a runs/<baseline> --b runs/<vgcap> --metric cider` |
| Chấm hàng loạt | — | `GT=... QUERY_IMAGES=... bash scripts/score_all.sh` |
| Gom bảng + LaTeX | — | `python scripts/collect_results.py --latex` |

### Còn lại phải làm tay (không script hoá được)

| TODO | Việc |
|---|---|
| T7 | Đọc paper ReCap, xác định cấu hình đứng sau 0.5987 vs 0.61619 |
| T6 | Điền CIDEr 4 chữ số vào Table 1/3 sau khi có evaluator |
| T2 | Chép lambda, temperature, 3 ngưỡng từ `step_1_rerank.py` vào bài |
| T1, T13, T14 | Vẽ lại hình overview, sửa nhãn Fig.2a/Fig.3 |
| T12 | Tuyển annotator, chấm 200 query |
| T19 | Thêm URL repo vào bài |

---

## 10. Những gì KHÔNG tự sửa (cố ý)

Ba chỗ dưới đây tôi để nguyên hành vi cũ, vì sửa sẽ làm con số 0.6219 không tái lập được nữa.
Chọn sửa hay không là quyết định của nhóm, và mỗi lựa chọn kéo theo việc chạy lại:

1. **Span penalty chết** (`length_penalty` vs `length_penalty_factor`) — giữ nguyên để tái lập.
   Bật bằng `configs/ablation_span_penalty_on.json` hoặc `..._paper.json`.
2. **Prompt không qua gate** — giữ nguyên. Bật bằng `--gate unified`.
3. **Noun phrase bỏ qua gate** trong `enrich_caption` — giữ nguyên. Tắt bằng `--no_noun_phrases`.

Cả ba đều có đường chạy để đo tác động trước khi quyết định.

---

## 11. Chi phí thực tế (đo 2026-10-27, không phải ước lượng)

Sau khi phát hiện hai bộ draft có sẵn, bức tranh chi phí đổi hẳn.

### Table 3 — 22 run

| Run | Cách chạy | GPU |
|---|---|---|
| `vgcap_<gate>` (tham chiếu) | tái dùng `llama.json` | 0 |
| `ladder_enric`, `ladder_sgn` | tái dùng `version1.json` | 0 |
| `ladder_band_only` | **sinh mới** | ~6h |
| `ladder_entities_only` | **sinh mới** | ~6h |
| `gate_none` | tái dùng `llama.json` | 0 |
| `gate_prompt` | **sinh mới** (đổi prompt) | ~6h |
| `gate_unified` | **sinh mới** (đổi prompt) | ~6h |
| `sel_*` (3 selector + 5 draw random) | tái dùng `llama.json` | 0 |
| `abl_*` (6 config) | tái dùng `llama.json` | 0 |

**18 run không cần GPU, 4 run cần — tổng ~24h GPU** (không phải ~130h như ước lượng ban đầu).

### Phụ lục A — 37 run, 0 GPU

Sweep `L*`, lưới `τ × K_e`, nhiễu trọng số đều tái dùng draft.

### Tốc độ đo được trên máy local (macOS, 8 nhân, không GPU)

| Việc | Thời gian |
|---|---|
| Smoke test 20 query | 7.5 giây |
| Một run full 3000 query | ~10–15 phút |
| 18 run của Table 3 | ~3–5 giờ |
| 37 run phụ lục A | ~8–10 giờ |

So sánh: sinh caption bằng Llama-3-8B mất **6h05/run** (đo từ log, chạy tuần tự không batch).

### Môi trường local tối thiểu

```bash
pip install spacy jinja2 tqdm
python -m spacy download en_core_web_sm
```

**Không cần torch/transformers** khi tái dùng draft: `from transformers import set_seed` và
`LLMAssembler` nằm trong `generate_drafts()` (`run_ablation.py:151-152`), chỉ nạp khi thực sự sinh
caption. Đã kiểm với Python 3.15 + spaCy 3.8.17.

---

## 12. Kết quả bước neo (chạy thật 2026-10-27, full 3000 query, 0 GPU)

Tái dựng từ `assemble_result/cot_5_things_fact_more_event_llama.json`.

### Khớp

| | Tái dựng | Bài báo |
|---|---|---|
| `repro_vgcap` mean length | **104.01** | 104 ✅ |
| `repro_unfiltered` mean length | **104.02** | 104 ✅ |
| Số caption khác nhau giữa 2 config | **2119/3000 (70.6%)** | — |

70.6% khớp chính xác với tỉ lệ draft ngắn hơn 104 từ (71.0%) — tức gate chỉ đổi caption ở đúng
những query mà enrichment chạy. Cơ chế đúng như mô tả ở `VGCAP_FLOW.md`.

### KHÔNG khớp: cột `#Ent`

Bài ghi 7.59 → 6.76 (22,759 → 20,276 mention). Thử 5 cách đếm, **không cách nào tái tạo được cặp số
này**:

| Cách đếm | unfiltered | VG-Cap | chênh |
|---|---|---|---|
| Mọi mention | 14.06 / 42,191 | 12.35 / 37,047 | −1.71 |
| Unique lowercase | 12.73 / 38,176 | 11.00 / 33,011 | −1.72 |
| Unique + substring-dedup | 10.99 / 32,966 | 9.41 / 28,230 | −1.58 |
| Unique lowercase, bỏ DATE/CARDINAL/… | 8.25 / 24,756 | **7.59 / 22,761** | −0.67 |
| Unique + dedup, bỏ DATE/CARDINAL/… | 6.83 / 20,489 | 6.21 / 18,633 | −0.62 |
| **Bài báo** | **7.59 / 22,759** | **6.76 / 20,276** | **−0.83** |

Hai điểm bất thường:

1. **`repro_vgcap` đếm kiểu "unique lowercase, bỏ DATE" ra 7.59 / 22,761 — trùng 4 chữ số có nghĩa
   với dòng *unfiltered* của bài (7.59 / 22,759).** Hai số độc lập cùng khớp thì khó là ngẫu nhiên.
2. Vân tay phân phối khớp một nửa: sd per-caption **2.41** (bài ghi 2.44) và tỉ lệ query *giảm*
   **53.8%** (bài 54.5%) — nhưng tỉ lệ *tăng* chỉ **5.1%** trong khi bài ghi **28.2%**, và gross
   thêm 3,649 so với 5,215.

### Việc phát sinh

| Việc | Vì sao |
|---|---|
| **Hỏi lại script đếm `#Ent` gốc** | Bài chỉ viết "recognized and deduplicated as in Section 3.4", không đủ để tái lập. §3.4 ngụ ý substring-dedup, nhưng cách đó cho 9.41, lệch xa |
| **Tính lại `#Ent` cho MỌI dòng bằng một script duy nhất** | Dù tìm được script cũ hay không, cả bảng phải dùng chung một cách đếm và **nói rõ cách đếm trong bài** — hiện bài không định nghĩa |
| **Kiểm xem run đã công bố có dùng đúng bộ draft này không** | Không set seed nên draft có thể đã được sinh lại; đó là giả thuyết đơn giản nhất giải thích chênh lệch |

### Kết luận tạm

Cơ chế tái dựng **đúng** (Len khớp tuyệt đối, tỉ lệ caption đổi khớp tỉ lệ enrichment chạy).
Chênh lệch nằm ở **cách đếm `#Ent`**, không ở pipeline.

Phép thử quyết định là submit `runs/repro_vgcap/submission.csv` và `runs/repro_unfiltered/submission.csv`
lên leaderboard: ra 0.6219 và 0.6166 thì bản tái dựng đúng hoàn toàn và `#Ent` chỉ là vấn đề đếm.
