# EXP01 — So sánh kiến trúc decoder CFM (Stage 1)

*Ngày chạy: 23–24/11/2026 · Commit: `d6812dc` · Lệnh: `bash run.sh compare-decoder 3000`*

## 1. Mục đích
Chọn kiến trúc decoder OT-CFM cho Stage 1 (sinh Mel tiếng Anh từ đặc trưng PhoWhisper) **bằng số liệu**, trước khi train dài.
Câu hỏi cụ thể:
1. Decoder v2 (FiLM cho `t` và speaker, conv giãn) có tốt hơn v1 (bản gốc) không?
2. Nếu có, cải thiện đến từ **kiến trúc** hay chỉ từ **kích thước**?
3. Lấy mẫu `t` kiểu logit-normal có giúp gì không?

## 2. Thiết lập

| Biến thể | Mô tả | Tham số |
| :--- | :--- | :---: |
| `v1` | Bản gốc: `t` là 1 kênh hằng số ở lớp đầu, conv k=3 không giãn (tầm nhìn ≈ ±0.1 s), 4 khối × 384 | 4.59M |
| `v2-small` | Kiến trúc v2 nhưng nhỏ (4 khối × 384) — **cùng cỡ v1** để tách hiệu ứng kiến trúc | 6.66M |
| `v2` | `t` sinusoidal + MLP và speaker đưa vào **mọi khối** qua FiLM; điều kiện `c` vào từng khối; conv giãn 1/3/9/27 (tầm nhìn ≈ ±1 s); 8 khối × 512 | 20.49M |
| `v2+ln` | Như `v2`, nhưng lấy mẫu `t` theo logit-normal (dồn về giữa) | 20.49M |

**Điều kiện công bằng:** cùng dữ liệu FLEURS `en_us` train, cùng thứ tự dữ liệu (shuffle seed 42), cùng khởi tạo (seed 0), cùng
**3000 bước**, batch 8, LR cosine 2e-4 → 1e-5.

**Đánh giá** trên 32 câu FLEURS validation **cố định** (chưa thấy khi train):
- **Val loss**: loss flow-matching với `t`, `x0` cố định.
- **Mel L1**: Mel sinh ra (ODE 32 bước midpoint, noise cố định) so với Mel thật, chỉ trên frame thật.
- **WER tái tạo** (16 câu đầu): Mel sinh ra → Vocos → Whisper-small (En) → WER so với transcript. Mốc tham chiếu: audio gốc qua cùng ASR đạt **9.2%**.

## 3. Kết quả

| Biến thể | Tham số | Val loss ↓ | Mel L1 ↓ | WER tái tạo ↓ | s/bước |
| :--- | :---: | :---: | :---: | :---: | :---: |
| v1 | 4.59M | **1.1152** | **1.7169** | 95.1% | 0.15 |
| v2-small | 6.66M | 1.5508 | 1.8448 | 77.0% | 0.15 |
| v2 | 20.49M | 1.3027 | 1.8277 | **69.5%** | 0.15 |
| v2+ln | 20.49M | 1.4668 | 2.2693 | 94.4% | 0.15 |
| *Audio gốc (tham chiếu)* | — | — | — | *9.2%* | — |

## 4. Nhận xét

**4.1. v2 rõ chữ hơn v1 rất nhiều, và phần lớn nhờ kiến trúc.**
WER của `v2` thấp hơn `v1` **25.6 điểm** (69.5% so với 95.1%); khoảng chênh này lớn hơn hẳn mức nhiễu của 16 câu (~300 từ).
`v2-small` — cùng cỡ với v1 — cũng thấp hơn 18 điểm (77.0%), nên **phần lớn cải thiện đến từ kiến trúc** (đưa `t`/speaker
vào mọi khối qua FiLM + tầm nhìn rộng), phần còn lại (~7.5 điểm) đến từ việc mở rộng mạng.

**4.2. Mel L1 và val loss "thích" v1 — nhưng đó là dấu hiệu Mel bị làm mượt, không phải chất lượng tốt.**
v1 có val loss và Mel L1 thấp nhất nhưng gần như **không nghe ra chữ** (WER 95%). Lý do: loss MSE/L1 thưởng cho dự đoán
"trung bình" — một phổ mờ, mất chi tiết vẫn có sai số trung bình nhỏ nhưng không còn thông tin âm vị. WER mới phản ánh độ
rõ chữ. ⇒ **Không dùng Mel L1 / val loss để xếp hạng giữa các kiến trúc khác nhau**; chỉ dùng chúng để theo dõi *cùng một*
mô hình qua các epoch. (Ghi chú trong `compare_report.md` tự sinh bảo "ưu tiên Mel L1" là **sai** — đã sửa trong code.)

**4.3. Logit-normal làm kết quả tệ hơn ở ngân sách này.**
`v2+ln` kém `v2` ở cả ba chỉ số (WER 94.4% so với 69.5%). Logit-normal dồn mẫu `t` về giữa, nên mạng ít được học ở hai đầu
quỹ đạo (gần nhiễu và gần dữ liệu) — đúng những bước quyết định chi tiết cuối cùng khi giải ODE. ⇒ Giữ `t` **đều (uniform)**.

**4.4. Chi phí train như nhau.**
Cả 4 biến thể đều 0.15 s/bước: thời gian bị chi phối bởi việc nạp dữ liệu và PhoWhisper encoder (đệm 30 s), không phải
decoder. Mạng to hơn 4.5 lần **không làm train chậm hơn**. Chi phí lúc suy luận (RTF) của v2 cần đo riêng ở benchmark Case 4.

**4.5. ⚠️ Vấn đề lớn hơn: chưa biến thể nào nói rõ.**
WER tốt nhất 69.5% vẫn cách xa mốc 9.2% của audio gốc. 3000 bước (≈ 24k câu, ~10 lượt dữ liệu) là quá ngắn so với
lần train thật (60 epoch × 300 bước = 18k bước), nhưng cũng là **cảnh báo**: cần theo dõi đường WER khi train thật.
Nếu sau ~15 epoch WER không giảm mạnh, nghi vấn hàng đầu là **điều kiện `c`**: PhoWhisper đã fine-tune cho tiếng Việt,
nên đặc trưng của nó trên audio tiếng Anh có thể nghèo thông tin âm vị tiếng Anh.

## 5. Kết luận & quyết định
- **Chọn `decoder_preset: v2`, `t_sampling: uniform`** cho Stage 1 (đây cũng là mặc định hiện tại trong `configs/stage1_cfm_en.yaml`).
- Train Stage 1 đầy đủ: `bash run.sh start cfm_en`, theo dõi dòng **"WER tái tạo"** mỗi 5 epoch.
- Trước đó, **nghe** `outputs/sanity/decoder_compare/<biến thể>/val_XX_pred.wav` cạnh `val_XX_gt_mel_vocos.wav` để xác nhận bằng tai.

## 6. Giới hạn của thí nghiệm
- Chỉ **1 seed** và **3000 bước** — thứ hạng ở giai đoạn đầu có thể thay đổi khi train lâu.
- WER chỉ trên **16 câu** (nhiễu lớn với chênh lệch nhỏ; các chênh lệch ở đây đều > 15 điểm nên vẫn đáng tin về chiều hướng).
- Không có đánh giá nghe chủ quan (MOS); Mel L1 không phù hợp để so sánh mô hình sinh (xem 4.2).

## 7. Việc tiếp theo nếu WER Stage 1 vẫn cao
1. Thử **điều kiện tiếng Anh** cho phía đích: dùng `openai/whisper-small` (encoder đa ngữ, không fine-tune Việt) để tính `c`
   ở Stage 1 và `c*` ở Stage 2; Bridge vẫn nhận PhoWhisper(Vi) ở đầu vào.
2. Train lâu hơn / thêm dữ liệu tiếng Anh (audio PhoST — không cần text).
3. Thử classifier-free guidance cho điều kiện `c` khi suy luận.
