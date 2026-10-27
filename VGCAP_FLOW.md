# VG-Cap — Flow của code hiện tại (trace từ source)

Cập nhật 2026-10-27. Mọi số dòng dưới đây trace từ code thật, không từ manuscript.
Đây là flow đã sinh ra con số **0.6219** đã công bố.

> Dòng 153–156 và 183–188 của `step_2_caption_normalizer.py` là phần mới thêm
> (`selection_map`); flow gốc không có, không ảnh hưởng mô tả bên dưới.

---

## Sơ đồ tổng

```
 [0] merge_function()                          step_2_caption_process.py:21-28
      5 nguồn  ->  final_merge_result.json
         │
         │  mỗi record: query_id, article, crawl_caption, generated_caption,
         │              question_answer, article_summary.fact_summary,
         │              name_entity_keyword   <-- keyword THÔ do Llama trích
         ▼
 [A] compose_input()                           step_2_caption_process.py:70-95
      render cot_5_things_fact_more_event.j2
         │
         │  prompt chứa 5 khối:
         │    1. summary.fact
         │    2. generated_caption   (InternVL mô tả ảnh query)
         │    3. crawl_caption       (editorial caption của ảnh database)
         │    4. name_entity_keyword <-- ĐƯỜNG 1: entity vào caption
         │    5. question_answer
         │  + chỉ thị độ dài: "must be between 100-120 words", "3-5 sentences"
         ▼
 [B] Llama.assemble()                          llama3.py:494-532
      min_new_tokens=100, max_new_tokens=1500
      do_sample=True, temperature=0.6, top_p=0.9, KHÔNG set seed
         │
         ▼
      assemble_result/cot_5_things_fact_more_event_llama.json   (draft thô)
         │
         ▼
 [C] CaptionNormalizer.process_single_caption()  step_2_caption_normalizer.py:244-277
      │
      ├─ C1  normalize_llm_output()                            :220
      │      bỏ preamble ("Here is the caption:") và dấu nháy
      │
      ├─ C2  truncate_caption_semantic(max_words=105, exp=104)  :97
      │      CHỈ chạy khi draft > 105 từ
      │      mask POS/NER bảo vệ proper noun + số, cắt token thường từ cuối
      │
      ├─ C3  extract_entities_and_phrases()                     :148
      │      ▸ use_entity_scoring=True  -> EntityHeuristicScorer.score_query()
      │                                     = E* (score>=3.0, top 10)
      │      ▸ use_entity_scoring=False -> TẤT CẢ spaCy entity của article
      │      ▸ cả hai nhánh đều kèm noun_chunks của toàn article (KHÔNG lọc)
      │      ^^^^ ĐÂY LÀ NƠI DUY NHẤT SCORER CHẠY
      │
      ├─ C4  enrich_caption(cutting_idx=104)                    :190
      │      nếu caption < 104 từ: nối entity vào đuôi, hết entity thì nối
      │      noun phrase, tới khi đủ 104 từ   <-- ĐƯỜNG 2: entity vào caption
      │
      └─ C5  chunking_words(104) + sửa dấu câu cuối             :271
         │
         ▼
 [D] create_submission()  ->  final_csv_result/submission_final.csv
```

---

## Hai đường entity đi vào caption

| | Đường 1 — PROMPT | Đường 2 — ENRICHMENT |
|---|---|---|
| Ở đâu | `compose_input` :77 -> `.j2` :13 | `enrich_caption` :190 |
| Nguồn entity | `name_entity_keyword` (Llama trích) | spaCy NER trên article (+ keyword) |
| Có qua scorer? | **KHÔNG** | **CÓ** (khi bật `--use_entity_scoring`) |
| Tác động | Llama tự viết entity vào thân caption | nối entity vào đuôi cho đủ 104 từ |
| Khi nào chạy | trước khi sinh | sau khi sinh |

⇒ Cờ `--use_entity_scoring` **chỉ** đổi Đường 2. Đường 1 luôn nhận danh sách thô.
Đó là định nghĩa của **enrichment-only gating**.

**Unified gating** = lọc cả hai đường bằng cùng một tập `E*` — chính là điều §3.6 của bài mô tả
(*"The same set E* is passed to both G_theta and N"*), nhưng code chưa làm.

---

## Two-stage length control nằm ở đâu

| Stage | Cơ chế | Vị trí | Ghi chú |
|---|---|---|---|
| **1. Bounded generation** | chỉ thị "100-120 words, 3-5 sentences" trong prompt | `.j2` :35-36 | ràng buộc mềm, Llama có thể không tuân |
| **2. Semantic normalization** | SGN: cắt (C2) + enrich (C4) + hard cut (C5) | normalizer | ràng buộc cứng về đúng 104 từ |

Trả lời thầy: **đúng**, two-stage length control bao gồm bước sinh caption; stage 1 chính là
bounded generation.

Nhưng **chưa đúng** ở chỗ: bounded generation **không** xài entity đã chọn lọc. Nó xài
`name_entity_keyword` thô. Entity đã lọc (`E*`) chỉ tồn tại từ C3 trở đi.

---

## Bằng chứng trong chính bài báo

§4.6 viết: *"the discarded entities concentrate in the passage that enrichment appends to reach the
target length"* — entity bị loại **dồn về đoạn do enrichment chắp thêm**. Đó đúng là dấu vân tay của
enrichment-only gating. Nếu prompt cũng bị lọc thì khác biệt phải trải đều khắp caption chứ không
dồn về đuôi.

Ví dụ §4.7 (ảnh Mark Webber xem tennis): caption unfiltered kết thúc bằng mảnh vỡ "Red Bull." —
đúng vị trí enrichment chắp thêm, không phải giữa thân câu.

---

## Hệ quả

1. **Hình overview phải vẽ lại cho đúng**: mũi tên từ Entity Heuristic Scorer hiện chỉ nên trỏ vào
   khối normalizer, KHÔNG trỏ vào khối caption generation. Hoặc sửa code cho khớp hình.
2. **§3.6 "Unified Entity Conditioning" đang mô tả thứ code chưa làm.**
3. **Panel (d) của Table 3 đang có 2 dòng trùng nhau**: "Enrichment-only gating" và
   "Unified gating (VG-Cap)" cùng là 0.6219.
4. **Về chi phí**: vì enrichment nằm sau bước sinh, đổi `E*` không cần sinh lại draft.
   Panel (b) và (c) chạy ở chế độ `enrichment` gần như miễn phí GPU
   (`run_ablation.py --reuse_drafts`). Unified thì đụng prompt -> mỗi dòng ~6h.

---

## Hai hướng xử lý

| | Sửa bài theo code | Sửa code theo bài |
|---|---|---|
| Việc | viết lại §3.6 thành enrichment-only, vẽ overview đúng | `run_ablation.py --gate unified` |
| Số cũ | giữ nguyên toàn bộ | **phải chạy lại hết**, kể cả Table 1 |
| Đóng góp "unified gating" | mất, phải hạ xuống thành enrichment gating | giữ được |
| GPU | 0 | >= 13 x 6h |

**Đề xuất:** chạy `--gate unified` cho đúng một dòng VG-Cap trước (~6h) rồi so với 0.6219.
Hơn -> chuyển hẳn sang unified và chạy lại. Không hơn -> sửa bài theo code, giữ số cũ, và nói thẳng
trong bài rằng gate ở enrichment là đủ (đây vẫn là một phát hiện có giá trị, không phải điểm yếu).

---

## Lệnh kiểm chứng nhanh

```bash
# xem prompt thật sự gửi cho Llama (đã render)
python step_2_caption_process.py --strategy cot_5_things_fact_more_event --template_test
cat assemble_caption_prompt_template/cot_5_things_fact_more_event.md

# đối chiếu: entity vào prompt là thô
sed -n 70,78p step_2_caption_process.py

# đối chiếu: scorer chỉ chạy trong normalizer
grep -n "score_query\|EntityHeuristicScorer" step_2_caption_normalizer.py step_2_caption_process.py
```

---

## Dữ liệu trung gian có sẵn trong repo

Trace thêm sau khi đối chiếu file (cập nhật 2026-10-27).

### Hai bộ draft, ứng với hai nhánh prompt

| File | Độ dài TB sau làm sạch | Prompt | Ứng với dòng nào trong Table 3(a) |
|---|---|---|---|
| `assemble_result/cot_5_things_fact_more_event_llama.json` | **97.4 từ** | CÓ length band | `+ unfiltered entities, prompt band` (0.6166) và `+ heuristic selector` (0.6219) |
| `assemble_result/cot_5_things_fact_more_event_llama_version1.json` | **135.2 từ** | KHÔNG length band | `ENRIC baseline` (bài ghi Len 136) và `+ SGN` (0.6124) |

Cả hai đều 3000 query, **không có query nào trùng caption** giữa hai file.

Hệ quả: **bốn dòng đã công bố của panel (a) đều tái dựng được mà không cần GPU**, vì phần ngẫu
nhiên duy nhất (LLM) đã đóng băng trong draft, còn normalizer và entity scorer đều deterministic.

### `submission_final.csv` KHÔNG phải output của VG-Cap

Đối chiếu từng dòng: `final_csv_result/submission_final.csv` trùng **100%** (3000/3000) với draft
thô `llama.json` sau khi chỉ làm sạch preamble và xuống dòng — chưa qua SGN.

```
độ dài TB 97.4 từ   |   25.6% caption dài hơn 104 từ
```

Nếu đã qua SGN thì mọi caption phải <= 104 từ. Nên đừng dùng file này làm mốc đối chiếu cho 0.6219.

### Caption của run 0.6219 không có trong repo

`normalized_captions.json` và `entity_heuristic_scores.json` (README nói hai file này được sinh ra)
không tồn tại, và `git log --all` cho thấy **chưa từng** được commit. Không khôi phục được từ
lịch sử git.

Không sao: tái dựng từ draft cho kết quả tương đương, và cho **4 điểm neo thay vì 1** — nếu cả bốn
lệch cùng một hệ số thì lỗi ở evaluator, nếu chỉ vài dòng lệch thì lỗi ở pipeline tái dựng.

### Đo tốc độ thật (máy local, CPU 8 nhân)

| Giai đoạn | Tốc độ |
|---|---|
| Chấm điểm entity (spaCy trên article) | ~11 query/giây |
| Normalize caption | ~7 query/giây |
| Tổng một run 3000 query | **~10–15 phút, 0 GPU** |

So với sinh caption bằng Llama-3-8B: **6h05 cho 3000 query** (đo từ
`step_2_caption_process.log.txt`, chạy tuần tự không batch).
