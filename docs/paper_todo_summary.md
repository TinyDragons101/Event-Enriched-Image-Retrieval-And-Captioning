# VG-Cap (ACIIDS 2027) – Danh sách việc còn lại

Tổng hợp từ các `\todo` trong [main.tex](../../VGCap_ACIIDS2027/main.tex), [VGCap_Revision_Log.pdf](../../VGCap_ACIIDS2027/VGCap_Revision_Log.pdf) và mã nguồn pipeline này.

## 0. Những chỗ code không khớp với paper (cần xác nhận trước khi chạy)

1. **Gate ở prompt không có trong code.** Prompt [cot_5_things_fact_more_event.j2](../assemble_caption_prompt_template/cot_5_things_fact_more_event.j2) đưa thẳng `name_entity_keyword` (toàn bộ keyword chưa lọc) vào LLM. Entity Scorer chỉ được gọi trong normalizer ([step_2_caption_normalizer.py:138](../step_2_caption_normalizer.py#L138)), tức là chỉ lọc ở bước enrichment. Paper lại viết "unified gating: prompt + enrichment" (Algorithm 1, Sec. 3.6) và nói bản 0.6166 với 0.6219 "chỉ khác entity set đưa vào generator". Cần xác nhận 0.6219 được tạo ra bằng code nào. Nếu đúng là code này thì phải sửa phần mô tả trong paper, hoặc bổ sung gate ở prompt rồi chạy lại.
2. **Submission CSV không phải output đã normalize.** [submission_final.csv](../final_csv_result/submission_final.csv) có độ dài trung bình 97.4 từ (min 67, max 144), không phải 104 như paper. *(Cập nhật: đã sửa ở commit `1ce3c7f`. `create_submission()` nay nhận tham số và đọc được cả output của normalizer; `run_ablation.py` tự ghi `runs/<tag>/submission.csv`.)* Trước bản vá, `create_submission()` trong [step_2_merge_all_elements.py](../step_2_merge_all_elements.py) không nhận tham số nhưng lại được gọi với `final_caption_json_path=...`, nên crash, và luôn đọc file LLM thô thay vì `normalized_captions.json`. Cần xác định file CSV nào thật sự cho ra 0.6219 / 0.6166 / 0.6124.
3. **Feature "article position" khác paper.** Code tính theo vị trí nhắc đến đầu tiên của entity. Paper nói có dùng cả vị trí của ảnh trong bài (URL/HTML marker), nhưng scorer không dùng `article_position`. Ngoài ra, keyword do LLM sinh ra (label `PARSED`) chỉ nhận type 0.5 và không có điểm position. Paper chưa nói rõ điểm này.
4. **Rerank khác paper.** [step_1_rerank.py](../step_1_rerank.py) không có detector entropy/top-1–top-2 gap, cũng không có fusion λ. Code lấy danh sách query "sai" có sẵn trong `temp_three_ways_wrong_samples_set.json`, rồi cộng `logit_scale·text_sim + image_sim`. Query được caption bằng InternVL2.5-**4B**, không phải 8B. Danh sách `reranked_ids` cũng không được sort lại. Cần lấy λ, temperature và threshold từ paper ENRIC, hoặc viết lại Sec. 3.2 cho đúng với code.
5. **Không có seed.** Code sinh caption bằng sampling (temperature 0.6, top-p 0.9) nhưng không set seed. *(Cập nhật: `run_ablation.py --seed` và `step_2_caption_process.py --seed` đã có. Run đã công bố vẫn không tái lập được; các run tái dùng draft cho caption giống hệt nhau giữa các seed, nên 3 seed thật phải sinh lại draft.)*
6. **Không có ground-truth caption ở local.** `query.csv` chỉ có query id. Mọi bảng (CIDEr, CLIP, entity P/R/F1) đều cần nộp lên evaluation server của EVENTA hoặc có reference captions của public test. Cần kiểm tra server còn mở không. Đây là điểm có thể chặn toàn bộ việc chạy thí nghiệm. *(Cập nhật 2026-10-27, nhóm xác nhận: không có GT public test và không có công cụ chấm; leaderboard còn khoảng 94 lượt nộp; có `gt_train.csv` của tập train trên drive, dùng cho Phụ lục A/B và P/R/F1 của Table 4 trên tập con train.)*

## 1. Những gì còn dùng được (đỡ phải chạy lại)

- Không cần chạy lại retrieval: [public_final_retrieval_merging_final_results.csv](../final_csv_result/public_final_retrieval_merging_final_results.csv) đã có.
- Còn đủ các file trung gian: InternVL caption, summary, QA, NER keyword, `result-tanphuc.json`, `final_merge_result.json`.
- Còn 2 file output LLM thô: `cot_..._llama.json` (~101 từ, prompt band) và `_version1.json` (~139 từ, gần với baseline ENRIC 136).
- Model đều là model public trên Hugging Face (Llama-3-8B-Instruct, InternVL), có thể tải lại.
- Bị mất: `normalized_captions.json`, `entity_heuristic_scores.json`. *(`configs/entity_scoring_config.json` đã được dựng lại ở commit `1ce3c7f`; hai file caption có thể tái dựng từ draft bằng `run_ablation.py --reuse_drafts`.)*
- Nếu gate thật sự chỉ nằm ở enrichment (mục 0.1), nhiều thí nghiệm chỉ cần normalize lại file LLM thô, không cần GPU.

## 2. Thí nghiệm cần chạy

7. **Table 3(a)**: thêm 2 dòng mới "+ prompt band only" và "+ unfiltered entities only", và điền cột #Ent cho baseline và SGN.
8. **Table 3(b)**: các selector thay thế với K=10: article-order, random-10 (trung bình 5 lần), caption-match-only, type-prior-only.
9. **Table 3(c)**: bỏ từng nhóm feature: presence, chỉ editorial caption, position, type, span penalty.
10. **Table 3(d)**: prompt-only, enrichment-only và unified gating. Phụ thuộc vào mục 0.1.
11. **Thống kê**: chạy 3 seeds cho ENRIC-repro, unfiltered và VG-Cap. Báo cáo std và paired bootstrap 95% CI, gồm cả CI cho phần tăng CIDEr/Overall so với ENRIC.
12. **Table 4**: entity P/R/F1 so với reference trên 3000 query, và leakage rate.
13. **Table 4**: gán nhãn khoảng 200 query (2 annotator, Cohen's κ) để tính unsupported rate. Human eval 3 rater, thang 1–5 cho factuality, fluency, informativeness. Viết annotation guideline. Nếu Unsup. không giảm thì phải giảm nhẹ claim ở abstract và conclusion.
14. **App. A – Table A1**: lưới τ ∈ {2,3,4,5} × K ∈ {5,10,15,no cap}.
15. **App. A – Table A2**: scale weight ×0.5/×1.5 cho từng nhóm, nhiễu ngẫu nhiên ±20% (10 lần), và sweep L* = 80/90/110/120. Nếu được thì vẽ thêm plot. Chỉ tune trên held-out train, không tune trên test.
16. **App. B**: selector logistic regression, chọn số query train held-out. So sánh hệ số học được với thứ tự weight trong Table 1.
17. **App. C**: đo latency và peak memory từng stage trên A100, và bàn về chi phí của budget 1500 token.
18. **Tùy chọn**: variant che entity bị loại trong summary/QA (R3.2), chạy lại với `en_core_web_trf` (SAR.2), baseline CLIP hoặc cross-encoder theo entity (SAR.7).

## 3. Việc tác giả cần kiểm tra hoặc viết

19. Ghi CIDEr 4 chữ số thập phân trong Table 2. ENRIC 0.257 tính lại ra 0.6089, không phải 0.6097.
20. Tìm nguồn của con số ReCap 0.5987 so với 0.61619 trong paper ReCap, trích dẫn cho đúng, và thêm dòng nếu cần.
21. Xác nhận với bản phát hành OpenEvents-V1 rằng reference caption không được tạo từ editorial caption (SAR.Q5).
22. Công bố đầy đủ prompt template (system/user, thứ tự các field) trong phần supplementary.
23. App. D: chuẩn bị 6–8 ví dụ (thành công, giữ sai, loại sai, enrichment lỗi kiểu "the Tiger Woods."), kèm breakdown điểm cho từng entity.
24. App. E: release code, prompt, script, seed, annotation, và thêm URL repo.
25. Kiểm tra metadata của 12 bib entry mới và venue của HMR/CIAN (SOICT 2025 hay arXiv).

## 4. Hình

26. Fig. 1: xem lại Overview2/Overview3, hình nào đúng, và vẽ lại. Nếu gate ở prompt không có thật thì hình hiện tại cũng sai.
27. Fig. 2a: sửa nhãn trục 0.6218 → 0.6219.
28. Fig. 3: đổi nhãn "0.6166"/"0.62" → "Unfiltered"/"VG-Cap".

## 5. Trước khi nộp

29. Đặt `\showrevisionsfalse` để chắc chắn không còn `\todo` nào, rồi build lại bằng llncs chính thức và kiểm tra giới hạn trang.

## Thứ tự nên làm

Làm mục 0.1, 0.2 và 0.6 trước, vì nó quyết định paper đang mô tả đúng hệ thống hay không. Sau đó chạy Table 3(b). Nếu article-order hoặc random tiến sát 0.6219 thì phải đổi cách kể chuyện sang "budget-aware gating".
