# EXP02 — So sánh encoder PHÍA ĐÍCH cho Stage 1 (PhoWhisper-small vs whisper-small)

*Ngày chạy: 24/11/2026 · Commit: `4643f85` · Lệnh: `bash run.sh compare-encoder 3000`*

## 1. Mục đích
Kiểm tra giả thuyết rút ra từ EXP01: *CFM phát âm mờ (WER tái tạo ~70%) vì điều kiện `c` được tạo từ PhoWhisper — encoder đã
fine-tune cho tiếng Việt — khi nghe audio **tiếng Anh**, nên đặc trưng nghèo thông tin âm vị tiếng Anh.*

Encoder đích chỉ dùng cho audio tiếng Anh (điều kiện `c` ở Stage 1, nhãn `c*` và speaker ở Stage 2); encoder nguồn
(PhoWhisper, nghe audio Việt → Bridge) **không đổi**.

## 2. Thiết lập
- Decoder cố định `v2` (20.49M), `t` uniform — lựa chọn từ EXP01.
- Hai biến thể encoder đích: `vinai/PhoWhisper-small` và `openai/whisper-small` (cùng 768 chiều).
- Điều kiện công bằng như EXP01: FLEURS `en_us` train, cùng thứ tự dữ liệu, seed 0, **3000 bước**, batch 8.
- Đánh giá: 32 câu validation cố định (val loss, Mel L1), WER tái tạo trên 16 câu đầu (Whisper-small En nghe lại; audio gốc: **9.2%**).
- Đánh giá nghe: người làm khóa luận nghe so sánh `val_XX_pred.wav` của 2 biến thể với `val_XX_gt_mel_vocos.wav`.

## 3. Kết quả

| Encoder đích | Val loss ↓ | Mel L1 ↓ | WER tái tạo ↓ | s/bước | Nghe thử |
| :--- | :---: | :---: | :---: | :---: | :--- |
| PhoWhisper-small | 1.3125 | 1.7863 | 72.1% | 0.18 | Không phân biệt được |
| whisper-small | 1.2617 | **1.4071** | 72.5% | 0.17 | Không phân biệt được |

## 4. Nhận xét

**4.1. Giả thuyết bị bác bỏ ở quy mô dữ liệu hiện tại.** WER chênh 0.4 điểm. Cấu hình PhoWhisper + v2 chạy lại cho 72.1%
so với 69.5% ở EXP01 ⇒ biến thiên giữa hai lần chạy ≈ **3 điểm**, nên 0.4 điểm nằm hoàn toàn trong nhiễu. Nghe cũng không
phân biệt được. ⇒ Encoder đích **không phải điểm nghẽn** gây phát âm mờ lúc này.

**4.2. Mel L1 của whisper-small thấp hơn rõ (1.41 vs 1.79)** — khác EXP01, ở đây decoder giống hệt, chỉ khác điều kiện đầu
vào, nên đây có thể là tín hiệu thật về độ chi tiết âm học của đặc trưng. Tuy nhiên chênh lệch **không nghe ra được** và không
cải thiện độ rõ chữ, nên chưa đủ để quyết định.

**4.3. Manh mối về điểm nghẽn thật: thiếu dữ liệu.** Trong 3000 bước, mô hình đọc hết FLEURS `en_us` train **~10 lượt**;
train loss đi ngang ~1.05 trong khi val loss 1.26–1.31 ⇒ bắt đầu học thuộc. Câu tái tạo thường đúng vài từ đầu
("When you call someone who…") rồi trôi sang từ khác: mô hình đã nắm nhịp/kiểu tiếng Anh nhưng chưa đủ dữ liệu để học chính
xác từng âm vị. Hai encoder cùng bị chặn bởi giới hạn này nên khác biệt (nếu có) bị che khuất.

## 5. Kết luận & quyết định
- **Chưa chốt encoder đích.** Giữ mặc định `vinai/PhoWhisper-small` (đơn giản hơn: một encoder dùng chung, Stage 2 nhẹ hơn).
- So sánh lại hai encoder **khi có thêm dữ liệu** (EXP03) — nếu khác biệt có thật, nó sẽ lộ ra khi mô hình không còn bị giới hạn bởi dữ liệu.

## 6. Giới hạn
- 1 seed, 3000 bước, WER trên 16 câu; biến thiên giữa các lần chạy ≈ 3 điểm WER.
- Đánh giá nghe do một người, không mù (biết biến thể nào là gì).

## 7. Việc tiếp theo — EXP03 (thiết kế 2×2)
Dữ liệu × encoder đích, cùng 3000 bước, cùng seed:

| | FLEURS en (vài giờ) | FLEURS + audio Anh PhoST (~50 giờ, không cần text) |
| :--- | :---: | :---: |
| PhoWhisper-small | (≈ EXP02) | ? |
| whisper-small | (≈ EXP02) | ? |

- Nếu cột PhoST giảm WER mạnh ⇒ xác nhận điểm nghẽn là dữ liệu → train Stage 1 trên dữ liệu gộp.
- Nếu ở cột PhoST hai encoder tách nhau ⇒ chọn encoder theo đó.
- Cần: `bash run.sh prepare-phost train 50` xong trước, và bộ đọc manifest PhoST cho `train_cfm.py`.
