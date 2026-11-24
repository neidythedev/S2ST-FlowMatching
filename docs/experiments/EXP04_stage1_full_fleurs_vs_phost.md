# EXP04 — Train Stage 1 đầy đủ: chỉ FLEURS vs FLEURS + PhoST (đến hội tụ)

*Ngày chạy: 24/11/2026 · Commit: `1ddb16b` · Chạy song song trên 2 notebook Molab*
- `cfm_en`: `bash run.sh start cfm_en` (data.sources = fleurs)
- `cfm_en_phost`: `python train_cfm.py --exp-name cfm_en_phost --data fleurs+phost`

## 1. Mục đích
EXP03 (3000 bước) không công bằng với phía nhiều dữ liệu; bản kiểm tra 900 bước cho thấy PhoST **giúp** khi FLEURS khan hiếm.
EXP04 so hai nguồn dữ liệu **ở điều kiện hội tụ** để quyết định checkpoint Stage 1 dùng cho Stage 2.

## 2. Thiết lập (giống nhau trừ dữ liệu)
- Decoder `v2` (20.49M), `t` uniform, encoder đích `vinai/PhoWhisper-small`.
- 60 epoch × 300 bước = **18 000 bước**, batch 8, LR cosine 2e-4 → 1e-5, seed/thứ tự dữ liệu cố định.
- `fleurs+phost`: mỗi câu lấy từ PhoST với xác suất 0.7 (20.03 giờ TED, 14 887 câu), FLEURS 0.3.
- Đánh giá mỗi 5 epoch trên 32 câu **FLEURS validation** cố định; WER tái tạo trên 16 câu (audio gốc: **9.2%**).

## 3. Kết quả — WER tái tạo theo epoch (%)

| Epoch | 4 | 9 | 14 | 19 | 24 | 29 | 34 | 39 | 44 | 49 | 54 | 59 |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **chỉ FLEURS** (`cfm_en`) | 74.1 | 53.4 | 46.9 | 40.0 | 39.3 | 29.5 | 28.5 | 31.5 | 29.2 | 29.5 | 30.2 | **29.2** |
| FLEURS + PhoST (`cfm_en_phost`) | 102.0 | 83.3 | 75.7 | 57.0 | 53.1 | 47.5 | 55.4 | 46.2 | 48.5 | 49.2 | 47.9 | **45.6** |
| Chênh lệch (PhoST − FLEURS) | +27.9 | +29.9 | +28.8 | +17.0 | +13.8 | +18.0 | +26.9 | +14.7 | +19.3 | +19.7 | +17.7 | +16.4 |

| Cuối train | chỉ FLEURS | FLEURS + PhoST |
| :--- | :---: | :---: |
| WER tái tạo (epoch 59) | **29.2%** | 45.6% |
| Mel L1 (epoch 59) | 1.478 | 1.780 |
| Train loss epoch cuối | 0.66 | 0.72 |
| Val loss tốt nhất | 0.8184 | — |
| Thời gian/epoch | ~52 s | ~36 s (đọc PhoST từ đĩa nhanh hơn stream FLEURS) |

## 4. Nhận xét

**4.1. Train đủ lâu là yếu tố quyết định.** Chỉ-FLEURS giảm đều từ 74% xuống ~29% ở epoch 29 (≈ 8 700 bước), sau đó đi
ngang suốt nửa sau ⇒ đã chạm trần với lượng dữ liệu FLEURS. So với ~69% ở 3000 bước (EXP01–03), mọi kết luận rút ra ở
ngân sách 3000 bước chỉ phản ánh giai đoạn đầu.

**4.2. Ở điều kiện hội tụ, thêm PhoST (70%) làm kết quả trên FLEURS tệ hơn ~16–20 điểm WER.** Đường PhoST chậm hơn ngay
từ đầu và chạm trần cao hơn (~46–49% từ epoch 29). Không thể quy cho thiếu FLEURS: bản trộn vẫn thấy ~43 nghìn câu FLEURS
(~18 lượt), trong khi bản chỉ-FLEURS đã chạm trần sau ~70 nghìn câu. ⇒ **ảnh hưởng thật của lệch miền**: 70% số bước học
âm thanh TED (phòng lớn có vang, nói tự nhiên, vỗ tay/cười) khiến decoder 20M tham số tái tạo giọng đọc sạch của FLEURS kém đi.

**4.3. Kết hợp với kiểm tra 900 bước (EXP03 mục 5b):** PhoST **giúp** khi dữ liệu trong miền khan hiếm, nhưng **làm loãng**
khi dữ liệu trong miền đã đủ. Hai kết quả nhất quán với nhau.

## 5. Quyết định
- **Dùng `cfm_en` (chỉ FLEURS, WER 29.2%) cho Stage 2** — `checkpoints/cfm_en-best.pth` (Drive: `gdrive:runs/cfm_en/checkpoints/`).
- PhoST **vẫn dùng ở Stage 2** (cặp giọng Việt tổng hợp → audio Anh thật): khi đó CFM đã đóng băng nên dữ liệu TED không làm
  hỏng decoder.
- Hướng có thể thử sau (không chặn tiến độ): tỉ lệ PhoST 20–30%, hoặc pretrain trên dữ liệu trộn rồi fine-tune lại trên FLEURS.

## 6. Giới hạn
- 1 seed mỗi cấu hình; WER trên 16 câu dao động ±3 điểm giữa các lần đo (thấy rõ ở các epoch cuối).
- Chỉ đánh giá trên FLEURS (trong miền của FLEURS); chưa đo trên tập TED/PhoST dev, nơi bản trộn có thể tốt hơn.
- Checkpoint `cfm_en_phost` chọn theo val loss nhưng val loss của bản này chưa được ghi vào báo cáo (lấy từ log nếu cần).
