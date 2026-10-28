# VG-Cap — Runbook

Cách sinh toàn bộ file kết quả để submit lên leaderboard.
`VGCAP_FLOW.md` = code chạy thế nào. `VGCAP_TODO.md` = chạy cái gì và vì sao. File này = **chạy sao**.

Giả định: **không có ground truth public test**, điểm lấy từ leaderboard.

> **Nhóm đã xác nhận (2026-10-27):** leaderboard có **quota khoảng 100 lần, đã dùng 5–6**; không có
> công cụ chấm, không biết model CLIP chính thức; có `gt_train.csv` (trên drive). Luôn khử trùng
> lặp bằng `make_submissions.py` và theo bảng ngân sách ở `VGCAP_ACTION.md` B1. Phụ lục A chạy trên
> tập con train và chấm cục bộ, không nộp.

### Trạng thái kiểm chứng (2026-10-27)

| | |
|---|---|
| ✅ Đã chạy thật | smoke test 20 query trên macOS, 7.5 giây, ra đủ `submission.csv` + `stats.json` |
| ✅ Đã kiểm | chạy được **không cần torch**; Python 3.15 + spaCy 3.8.17 OK |
| ✅ Đã kiểm | template ablation render **byte-identical** với bản gốc khi bật cả hai khối |
| ✅ Đã kiểm | hai bộ draft khớp hai nhánh prompt (97.4 từ / 135.2 từ) |
| ✅ Đã kiểm | draft, `final_merge_result.json` và CSV retrieval: cùng 3000 query_id, cùng thứ tự; mọi preamble đều bị bóc |
| ❌ Chưa chứng minh | scorer mới == scorer đã công bố: `test_fixes.py` mới chạy trên NER giả lập, và file gốc `backup_pre_fix/` không có trong repo (xem `VGCAP_ACTION.md` B0) |
| ✅ Đã kiểm (git) | `version1.json` sinh từ template commit `c004e52`: **có** khối Keywords, **không** có band ("at least 5 sentences"). Dòng ENRIC/+SGN đã có entity trong prompt, xem `VGCAP_TODO.md` §2 panel (a) |
| ✅ Đã kiểm (code + log) | `generated_caption` do InternVL2.5-8B sinh (mặc định của `step_2_create_caption_query.py`, log nạp InternLM2) |
| 🔄 Đang chạy | `repro_vgcap` + `repro_unfiltered` full 3000 query |
| ❌ Chưa có | bất kỳ con số Overall/CIDEr nào — phải submit mới biết |

---

## 1. Chuẩn bị

```bash
cd phucpv/Eventa
conda deactivate
source ./.venv/bin/activate
nvidia-smi                      # chọn GPU trống; theo NOTE.md không dùng cuda:1
```

Thư viện còn thiếu trong `requirements.txt`:

```bash
pip install spacy jinja2 python-dotenv     # llama3.py import dotenv
python -m spacy download en_core_web_sm    # ghi lai version (spacy.load(...).meta["version"]): NER doi -> E* doi
# chỉ cần nếu chạy phụ lục B:
pip install scikit-learn
```

File cần có — **tất cả đều đã nằm trong repo**:

| File | Vai trò |
|---|---|
| `assemble_result/cot_5_things_fact_more_event_llama.json` | 3000 draft, prompt **có** length band (97.4 từ) |
| `assemble_result/cot_5_things_fact_more_event_llama_version1.json` | 3000 draft, prompt **không** length band (135.2 từ) |
| `public_test_final_elements_json/final_merge_result.json` | context đầu vào |
| `final_csv_result/public_final_retrieval_merging_final_results.csv` | khung submission |

Không cần ground truth, không cần ảnh query (trừ khi muốn tự tính CLIPScore trước khi submit).

---

## 2. Vì sao phần lớn không tốn GPU

Chỉ bước sinh caption là ngẫu nhiên, và 3000 draft đã lưu sẵn — **hai bộ**, ứng với hai nhánh prompt:

```
llama.json      97.4 từ  → prompt CÓ length band     → dòng 0.6166, 0.6219 và toàn bộ panel (b),(c)
version1.json  135.2 từ  → prompt KHÔNG length band, CÓ Keywords → dòng ENRIC baseline (bài ghi 136) và +SGN
```

Normalizer và entity scorer đều deterministic, nên mọi cấu hình chỉ đổi `E*` hoặc đổi `L*` đều tái
dùng được draft.

| Loại | Số run | GPU | Thời gian đo được |
|---|---|---|---|
| Tái dùng draft | 18 | 0 | ~10–15 phút/run trên CPU 8 nhân |
| Phải sinh mới | 4 | ~6h mỗi run | 6h05 đo từ log, chạy tuần tự |

Bốn run phải sinh mới: `ladder_band_only`, `ladder_entities_only`, `gate_prompt`, `gate_unified`
(hai cái sau đổi prompt nên không thể tái dùng).

---

## 3. Smoke test — làm trước tiên

```bash
python run_ablation.py --tag smoke --gate enrichment --selector heuristic --limit 20 \
    --reuse_drafts assemble_result/cot_5_things_fact_more_event_llama.json
```

~2–3 phút. Phải thấy `runs/smoke/submission.csv` và `stats.json`. Lỗi ở đây thì dừng, đừng chạy tiếp.

---

## 4. Bước neo — tái lập số đã công bố, 0 GPU

```bash
BAND=assemble_result/cot_5_things_fact_more_event_llama.json
NOBAND=assemble_result/cot_5_things_fact_more_event_llama_version1.json

python run_ablation.py --tag repro_enric      --gate none --no_prompt_entities --no_prompt_band \
    --no_normalizer --reuse_drafts "$NOBAND"
python run_ablation.py --tag repro_sgn        --gate none --no_prompt_entities --no_prompt_band \
    --reuse_drafts "$NOBAND"
python run_ablation.py --tag repro_unfiltered --gate none --reuse_drafts "$BAND"
python run_ablation.py --tag repro_vgcap      --gate enrichment --selector heuristic \
    --entity_config configs/entity_scoring_config.json --reuse_drafts "$BAND"
```

Mỗi lệnh ~10–30 phút CPU.

**Kiểm ngay, không cần submit:**

```bash
python scripts/collect_results.py
```

| Run | Len bài ghi | Len tái dựng (đã chạy 2026-10-27) |
|---|---|---|
| `repro_enric` | 136 | — |
| `repro_sgn` | 104 | — |
| `repro_unfiltered` | 104 | 104.02 |
| `repro_vgcap` | 104 | 104.01 |

Thêm một mốc: số caption khác nhau giữa `repro_unfiltered` và `repro_vgcap` phải là **2119/3000**.

Khớp ⇒ pipeline tái dựng đúng, submit tiếp yên tâm.
**Lệch Len hoặc số caption khác nhau ⇒ dừng lại báo**, vì 22 CSV phía sau sẽ sai theo.

> ⚠️ **Không** dùng `#Ent` (7.59 / 6.76) làm mốc dừng. Đã biết không cách đếm nào tái lập được
> cặp số này (`VGCAP_TODO.md` §12); dùng nó làm mốc thì pipeline sẽ dừng dù tái dựng đúng.
> `#Ent` là việc phải chốt định nghĩa và đếm lại, không phải phép kiểm.

Rồi submit 4 CSV này lên leaderboard. Điểm trả về phải ra 0.6097 / 0.6124 / 0.6166 / 0.6219.
Đây là phép kiểm mạnh nhất có thể làm mà không có ground truth.

---

## 5. Sinh toàn bộ CSV cho Table 3

```bash
bash scripts/run_panels.sh enrichment
```

Biến môi trường: `SEED` (0), `DEVICE` (cuda:5), `MODEL` (llama3), `LIMIT` (subsample),
`DRAFTS_BAND`, `DRAFTS_NOBAND`.

Sinh 22 run. Cuối script in ra danh sách `runs/*/submission.csv`.

Ba seed cho yêu cầu T8:
```bash
for S in 0 1 2; do SEED=$S bash scripts/run_panels.sh enrichment; done
```
⚠️ Seed chỉ đổi kết quả của 4 run sinh mới; 18 run tái dùng draft sẽ cho caption y hệt nhau giữa
các seed. Muốn 3 seed thật cho mọi dòng thì phải sinh lại draft 3 lần (3 × 6h) rồi truyền
`DRAFTS_BAND` tương ứng.

Phụ lục A:
```bash
bash scripts/run_appendix.sh enrichment      # 37 run, 0 GPU, ~10-15h CPU
```

---

## 6. Submit và ghi điểm ngược lại

Submit từng `runs/<tag>/submission.csv`, rồi ghi điểm về:

```bash
python scripts/record_score.py --run runs/repro_vgcap --overall 0.6219 --cider 0.2651 --clip 0.858
```

Script tự tính lại Overall từ các thành phần và **cảnh báo nếu lệch > 0.001** — đúng loại lỗi
đang có ở dòng ENRIC (0.257 → 0.6089, bài ghi 0.6097). Khi thấy cảnh báo thì xin thêm chữ số thập
phân của CIDEr trước khi đưa vào bảng.

Gom bảng:
```bash
python scripts/collect_results.py --latex
```

Khoảng tin cậy (T8/T9) cần điểm từng query, mà leaderboard chỉ trả điểm tổng → **chỉ làm được
bootstrap cho CLIPScore** (tự tính cục bộ nếu có ảnh query):

```bash
python scripts/evaluate.py --run runs/repro_vgcap --query_images data/track1_public/query --device cuda:0
python scripts/bootstrap.py --a runs/gate_none_s0 --b runs/repro_vgcap --metric clip
```

---

## 7. Table 4 — cái làm được và cái không

| Cột | Không có GT public test thì sao |
|---|---|
| Leakage rate | ⚠️ `stats.json` đã có số, nhưng đo bằng chuỗi con thô và mẫu số có blob keyword → phải sửa cách đo trước khi đưa vào bài |
| Entity P/R/F1 | ❌ **không làm được** — cần reference caption |
| Unsupported-entity rate | ✅ người chấm, không cần GT |
| Human 1–5 | ✅ người chấm |

```bash
python scripts/eval_entities.py --run runs/repro_vgcap --sample 200 --seed 0   # cần --gt cho P/R/F1
```

Cột P/R/F1: **đã chốt** chạy trên một tập con của **train split** bằng `gt_train.csv` (nhóm xác
nhận không có GT public test). Đây cũng là điều phụ lục A của bài đã hứa ("use a held-out split of
the OpenEvents training queries"). Sửa câu "over all 3,000 queries" ở dòng 380 của bài.

⚠️ κ cho 3 rater phải là **Fleiss**, không phải Cohen như bài đang viết.

---

## 8. Phụ lục B, D, E

```bash
# D — ví dụ định tính (phụ lục D)
python scripts/make_qualitative.py --run runs/repro_vgcap --baseline runs/repro_unfiltered --per_bucket 3

# E — prompt template (T4)
python scripts/dump_prompts.py --n 3 --out supplementary/prompts \
    --selected_entities runs/repro_vgcap/selected_entities.json

# B — learned selector: cần GT của TRAIN split
python scripts/train_learned_selector.py --merge <train_merge.json> --gt <train_gt.json> \
    --n_queries 500 --out configs/learned_selector.json
```

> ⚠️ `<train_merge.json>` **chưa tồn tại**. Không được thay bằng
> `public_test_final_elements_json/final_merge_result.json` (đó là tập test, không có query chung
> với GT train). Phải dựng lại context Stage 2 cho ~500 query train trên server (cần GPU).
>
> ⚠️ Phụ lục D: sửa `_pack()` trong `step_2_entity_selectors.py` để ghi `features` vào
> `entity_debug.json` trước, nếu không `make_qualitative.py` phân nhóm sai và không có bảng điểm.

Phụ lục C (efficiency) **cần lệnh riêng**: khối `timings` trong `stats.json` chỉ là tổng thời gian
cả run, không có thời gian mỗi query, peak GPU memory, hay giai đoạn retrieval/re-ranking mà bảng
của bài yêu cầu.

---

## 9. Chạy nền cho đúng

`nohup ... &` **báo xong ngay lập tức** dù việc thật vẫn đang chạy — tôi đã dính lỗi này. Dùng một
trong hai cách:

```bash
# cách 1: chạy tuần tự trong một phiên, xem tiến độ trực tiếp
bash scripts/run_panels.sh enrichment 2>&1 | tee run_panels.log

# cách 2: chạy nền nhưng theo dõi được
nohup bash scripts/run_panels.sh enrichment > run_panels.log 2>&1 &
echo $!                       # nhớ PID
tail -f run_panels.log        # xem tiến độ
pgrep -f run_ablation.py      # còn sống không
```

Mỗi run ghi `runs/<tag>/stats.json` **ở bước cuối cùng**, nên file đó xuất hiện = run đó đã xong.
Kiểm tiến độ tổng:

```bash
ls runs/*/stats.json | wc -l
```

---

## 10. Xử lý sự cố

| Lỗi | Xử lý |
|---|---|
| `--reuse_drafts is invalid with --gate prompt/unified` | hai gate đó đổi prompt, phải sinh lại |
| `--budget_match per_query requires --reference_selection` | chạy run VG-Cap trước, trỏ vào `selected_entities.json` của nó |
| `--gate prompt/unified needs the Keywords block` | bỏ `--no_prompt_entities` |
| Generation đứt giữa chừng | chạy lại cùng `--tag` kèm `--resume`. Lưu ý: (1) file checkpoint ghi đè trực tiếp, bị kill đúng lúc ghi thì JSON hỏng và `--resume` lỗi; (2) sau resume, luồng ngẫu nhiên khác run liền mạch, nên run đó không tái lập được theo seed |
| Lỡ chạy lại một `--tag` đã xong | `run_ablation.py` **không chặn**: mọi file trong `runs/<tag>/` bị ghi đè, kể cả `llm_raw.json` (6h GPU). Sao lưu `runs/` sau mỗi run GPU; `runs/` bị gitignore |
| OOM | đổi `--device`, xem `nvidia-smi` |
| spaCy chậm | bình thường, ~10–30 phút cho 3000 article |

Mỗi run lưu `config.json` và `effective_entity_config.json` nên luôn truy lại được tham số.

---

## 11. Checklist

```
[ ] 1  pip install spacy jinja2 python-dotenv + en_core_web_sm (ghi lại version)
[ ] 1  khôi phục backup_pre_fix từ git; test_fixes.py dòng [1] PASS với spaCy thật
[ ] 3  smoke test --limit 20 chạy trót lọt
[ ] 4  4 run neo: Len = 136/104/104/104; 2119 caption khác nhau giữa unfiltered và vgcap
[ ] 4  CLIP cục bộ của repro_vgcap ≈ 0.858 (kiểm evaluator trước khi chấm hàng loạt)
[ ] 4  submit 4 CSV neo -> ra 0.6097 / 0.6124 / 0.6166 / 0.6219
[ ] 5  run_panels.sh xong -> 22 CSV
[ ] 5  run_appendix.sh xong -> 37 CSV
[ ] 6  submit hết, record_score.py từng cái
[ ] 6  collect_results.py --latex ra đủ bảng
[ ] 7  eval_entities.py + phát annotation sheet
[ ] 8  qualitative + prompt dump
```
