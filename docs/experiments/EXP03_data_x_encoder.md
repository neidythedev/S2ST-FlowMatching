# EXP03 — Lưới dữ liệu (FLEURS | FLEURS + PhoST) × encoder đích (PhoWhisper | whisper-small)

*Ngày chạy: 24/11/2026 · Commit: `2f07fd5` · Lệnh: `bash run.sh exp03 3000`*

## 1. Mục đích
EXP02 cho thấy encoder đích không phải điểm nghẽn và gợi ý **thiếu dữ liệu** (mô hình đọc hết FLEURS `en_us` ~10 lượt
trong 3000 bước, val > train). EXP03 kiểm tra: thêm ~20 giờ audio tiếng Anh PhoST (TED, không cần text) có làm CFM phát âm
rõ hơn không, và khi có thêm dữ liệu thì hai encoder đích có tách nhau không.

## 2. Thiết lập
- Decoder `v2` (20.49M), `t` uniform, 3000 bước, batch 8, seed 0 — như EXP01/02.
- Dữ liệu train:
  - `fleurs`: FLEURS `en_us` train (vài giờ).
  - `fleurs+phost`: mỗi câu lấy từ FLEURS với xác suất 0.3, từ PhoST với xác suất 0.7. PhoST: 14 887 câu, **20.03 giờ**
    (60+100 bài TED, cắt theo yaml MuST-C, lọc 1–12 s; `tools/prepare_phost.py`).
- Đánh giá: như EXP01/02 — 32 câu **FLEURS validation** cố định; WER tái tạo trên 16 câu (audio gốc: 9.2%).

## 3. Kết quả

| Dữ liệu | Encoder đích | Val loss ↓ | Mel L1 ↓ | WER tái tạo ↓ | Train loss cuối | s/bước |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| fleurs | PhoWhisper-small | 1.3281 | 1.7591 | 69.2% | 1.059 | 0.20 |
| fleurs | whisper-small | 1.2617 | 1.3931 | **68.9%** | 1.042 | 0.18 |
| fleurs+phost | PhoWhisper-small | 1.4082 | 2.2459 | 86.2% | 1.135 | 0.14 |
| fleurs+phost | whisper-small | 1.3613 | 1.7152 | 96.4% | 1.137 | 0.13 |

## 4. Nhận xét

**4.1. Encoder đích: vẫn ngang nhau** trên FLEURS (69.2% vs 68.9%) — lặp lại kết quả EXP02. Kết luận không đổi.

**4.2. Thêm PhoST làm WER trên FLEURS validation tệ hơn 17–27 điểm — nhưng thí nghiệm chưa trả lời đúng câu hỏi "nhiều dữ
liệu có giúp không":**
- **Ngân sách cố định bất lợi cho phía nhiều dữ liệu:** với 70% PhoST, mô hình chỉ thấy ~7 nghìn câu FLEURS (thay vì ~24
  nghìn) trong khi đánh giá trên FLEURS; 20 giờ PhoST mới được đọc ~1 lượt. Train loss cao hơn (1.14 vs 1.05) ⇒ mô hình
  **chưa hội tụ** trên dữ liệu đa dạng hơn, chứ chưa chứng minh dữ liệu kém.
- **Lệch miền:** TED là nói tự nhiên, phòng lớn có vang, vỗ tay, tiếng cười; FLEURS là giọng đọc sạch. Ở ngân sách nhỏ,
  trộn dữ liệu khác miền làm hại tập đánh giá trong miền.
- Chênh 10 điểm giữa hai encoder ở hàng PhoST không có ý nghĩa (cả hai ở vùng gần như không nghe ra chữ).

**4.3. Bài học thiết kế thí nghiệm:** so sánh lượng/nguồn dữ liệu ở ngân sách bước cố định và nhỏ là **không công bằng**
với phía nhiều dữ liệu; cần so ở điều kiện hội tụ (hoặc ít nhất cùng số lượt đọc dữ liệu trong miền).

**4.4. Câu hỏi lớn hơn vẫn mở:** mọi cấu hình đều kẹt ~69% WER sau 3000 bước; chưa từng train đủ lâu để biết WER giảm
được tới đâu.

## 5. Quyết định
- Giữ `data.sources: fleurs`, `target_encoder: vinai/PhoWhisper-small`, `decoder_preset: v2`.
- **Train Stage 1 đầy đủ** (`bash run.sh start cfm_en`: 60 epoch × 300 bước = 18 000 bước, ~1 giờ) và theo dõi đường
  WER tái tạo mỗi 5 epoch:
  - WER giảm dần về ~20–30% ⇒ Stage 1 ổn, sang Stage 2.
  - WER kẹt ~60–70% ⇒ chạy thêm 1 lần train đầy đủ với `--data fleurs+phost` để so ở điều kiện hội tụ, hoặc xem lại cách
    tạo điều kiện `c` (VD dùng tầng giữa của Whisper thay vì tầng cuối).
- Nghe kiểm tra chất lượng cắt câu PhoST (loại trừ khả năng lệch thời gian cắt).

## 6. Giới hạn
- 1 seed, 3000 bước; WER trên 16 câu; biến thiên giữa các lần chạy ≈ 3 điểm WER.
- Đánh giá chỉ trên FLEURS (trong miền của FLEURS); chưa đo trên tập TED/PhoST dev.
