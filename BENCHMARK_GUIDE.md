# 🧪 HƯỚNG DẪN CHẠY BENCHMARK TÁCH RỜI (NHIỀU NOTEBOOK)

Mỗi biến thể = **1 engine TTS × 1 chế độ dịch × 1 model**. Mỗi lệnh chạy độc lập trên 1 notebook Molab,
chạy xong **tự đẩy kết quả lên** `gdrive:runs/benchmark/<tag>/`. Khi đủ, chạy `analyze` ở bất kỳ notebook nào.

## 0. Chuẩn bị mỗi notebook
1. Dán khối lệnh trong `MOLAB_1CLICK_SETUP.md` (rclone + repo + thư viện).
2. Kiểm tra: `rclone lsd gdrive:` và `git log --oneline -1` (phải là commit mới nhất).

## 1. 13 lệnh độc lập (10 mẫu/lệnh)
In lại bất cứ lúc nào: `bash run.sh bench-list 10`

| # | Lệnh | Ý nghĩa |
| :-: | :--- | :--- |
| 0 | `bash run.sh bench seamless 10` | **SeamlessM4T-v2-large** · Direct S2ST SOTA (1 model, không voice cloning) |
| 1 | `bash run.sh bench xtts 2step openai/whisper-small 10` | XTTS · dịch thẳng · Whisper nhỏ |
| 2 | `bash run.sh bench xtts 2step openai/whisper-large-v3 10` | XTTS · dịch thẳng · Whisper lớn |
| 3 | `bash run.sh bench xtts 2step vinai/PhoWhisper-small 10` | XTTS · kiểm chứng PhoWhisper có dịch được không |
| 4 | `bash run.sh bench xtts 2step vinai/PhoWhisper-large 10` | như trên, bản large |
| 5 | `bash run.sh bench xtts 3step vinai/PhoWhisper-small 10` | XTTS · PhoWhisper-small + VinAI MT |
| 6 | `bash run.sh bench xtts 3step vinai/PhoWhisper-large 10` | XTTS · PhoWhisper-large + VinAI MT |
| 7 | `bash run.sh bench f5 2step openai/whisper-small 10` | F5-TTS · ... |
| 8 | `bash run.sh bench f5 2step openai/whisper-large-v3 10` | |
| 9 | `bash run.sh bench f5 2step vinai/PhoWhisper-small 10` | |
| 10 | `bash run.sh bench f5 2step vinai/PhoWhisper-large 10` | |
| 11 | `bash run.sh bench f5 3step vinai/PhoWhisper-small 10` | |
| 12 | `bash run.sh bench f5 3step vinai/PhoWhisper-large 10` | |

- Log riêng: `logs/bench_<tag>.log` (cũng được đẩy lên Drive cùng kết quả).
- Lệnh lỗi → **không** đẩy lên Drive (in ❌). Chạy lại đúng lệnh đó là đủ, các lệnh khác không ảnh hưởng.
- Mọi biến thể dùng cùng các câu FLEURS test theo đúng thứ tự (không shuffle) → so sánh ghép cặp hợp lệ;
  `analyze` tự kiểm tra lại điều này.

## 2. Gom & phân tích
```bash
bash run.sh analyze
```
Xuất ra `outputs/benchmark_all/`:
- `analysis_report.md` — bảng tổng hợp + so sánh ghép cặp + nhận xét tự động
- `analysis_summary.csv` — đưa vào Excel / LaTeX
- `analysis.json` — dữ liệu thô

Chạy `analyze` nhiều lần cũng được (VD khi mới xong 6/12 lệnh) — nó phân tích những gì đang có.

## 3. Đọc báo cáo
| Chỉ số | Ý nghĩa |
| :--- | :--- |
| **S2TT-BLEU** | Văn bản dịch (trước TTS) vs Target Ref → chất lượng **dịch** |
| **ASR-BLEU** | Whisper-En nghe lại audio sinh ra vs Target Ref → chất lượng **S2ST cuối cùng** (số chính thức) |
| **Mất do TTS** | S2TT-BLEU − ASR-BLEU → TTS đọc sai / cắt cụt bao nhiêu |
| **SECS** | Độ giống giọng nguồn (WavLM-SV), cao = giữ giọng tốt |
| **RTF** | Thời gian xử lý / thời lượng audio, < 1 = nhanh hơn thời gian thực. Gồm cả bước dịch + TTS. Dòng `🖥️ Thiết bị` đầu log cho biết chạy GPU hay CPU |
| **Lỗi TTS** | Số mẫu TTS bị crash (BLEU tính 0, SECS bỏ qua) — khác 0 thì nên sửa & chạy lại |
| **Câu không dịch** | Số câu đầu ra vẫn là tiếng Việt (dự kiến cao với PhoWhisper 2-step) |
| **[CI 95%]** | Khoảng tin cậy bootstrap. Hai biến thể chỉ khác nhau **có ý nghĩa** khi CI của Δ (mục 2) không chứa 0 |

⚠️ 10 mẫu chỉ là thử nghiệm sơ bộ — CI rất rộng. Số liệu chính thức cho khóa luận nên chạy ≥ 100 mẫu
(đổi `10` thành `100` ở cuối mỗi lệnh).
