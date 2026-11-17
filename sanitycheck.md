**Đúng hoàn toàn!** Nhiều người bỏ qua bước này, lao vào train ngay hàng trăm giờ audio rồi mất cả tuần mới phát hiện code tính sai chuẩn Mel hoặc dữ liệu bị lệch — rất tốn thời gian và GPU.

Thời điểm chuẩn nhất để làm **EDA & Sanity Check** là **NGAY TRƯỚC KHI bấm nút train thật (tức là sau khi tải data + viết xong code mô hình, ngay trước Việc 3)**. Bạn chỉ cần dành khoảng **1 ngày** làm gọn đúng **4 bước kiểm tra sinh tử** dưới đây (không cần vẽ biểu đồ rườm rà):

---

### 1. EDA Dữ liệu (Làm trong 2–3 tiếng)
Mục tiêu không phải để viết báo cáo đẹp, mà là **lọc rác để mô hình không bị sập loss**:
*   **Kiểm tra phân phối độ dài (giây):** Vẽ nhanh histogram độ dài các tập (LibriTTS, VIVOS, Common Voice, FLEURS).
    *   *Hành động:* Cắt/lọc bỏ các câu quá ngắn (< 0.5s, thường là tiếng thở/lỗi) và quá dài (> 15s, dễ gây OOM tràn VRAM và khó căn chỉnh Attention).
*   **Kiểm tra Sample Rate & Khoảng lặng:** Đảm bảo 100% file đã đưa về đúng **16kHz** (cho Whisper & Speaker Encoder) và **24kHz** (cho Mel đích của Vocos). Kiểm tra xem có file nào bị rỗng (toàn số 0) hay mất tiếng không.
*   **Kiểm tra tỷ lệ độ dài `Đích / Nguồn` (cho dữ liệu song ngữ):** Tính tỷ lệ `len(En_audio) / len(Vi_audio)`. Lọc bỏ các cặp ngoại lai (ví dụ: câu Việt 2 giây nhưng câu Anh tận 15 giây do lỗi dịch/tổng hợp F5-TTS bị lặp từ!).
*   **Nghe kiểm tra bằng tai (Spot Check):** Mở ngẫu nhiên 10–15 file audio tiếng Anh do `F5-TTS` vừa tạo ở bước data augmentation để chắc chắn nó không bị đọc ngọng hay sai chữ.

---

### 2. Sanity Check #1: "Vocos Round-Trip Test" (15 phút — Bắt buộc!)
Đây là lỗi kinh điển nhất khi làm Audio Flow Matching:
*   **Cách làm:** Lấy 5 file audio tiếng Anh chuẩn 24kHz $\rightarrow$ chạy qua **hàm trích xuất Mel-Spectrogram 100 kênh** của bạn $\rightarrow$ đưa thẳng vào **Vocos** giải mã ngược lại thành `.wav` (chưa đụng gì tới OT-CFM).
*   **Tiêu chuẩn đạt (Pass):** Nghe lại 5 file `.wav` đó phải **trong vắt, giống hệt file gốc**. Nếu bị rè, méo giọng hoặc sai tốc độ $\rightarrow$ hàm tính Mel của bạn (`n_fft`, `hop_length`, `fmin/fmax`, `log` cơ số $e$ hay $10$, hoặc chuẩn hóa mean/std) đang lệch với Vocos. Phải sửa cho khớp 100% rồi mới được train!

---

### 3. Sanity Check #2: Kiểm tra "Độ khỏe" của Whisper Encoder trên tiếng Việt (30 phút)
*   **Cách làm:** Vì bạn đóng băng `Whisper-Small` làm đầu vào tiếng Việt, hãy cho `Whisper-Small` chạy thử nhận dạng (ASR) trên khoảng 100 câu ngẫu nhiên của VIVOS và Common Voice (nhất là giọng miền Trung/Nam và file hơi nhiễu).
*   **Tiêu chuẩn đạt:** Nếu `Whisper-Small` nghe tiếng Việt quá tệ (WER cao ngất, rớt hết chữ), thì đặc trưng $H_{src}$ đưa vào Bridge sẽ bị mất nghĩa ngay từ cửa vào. Lúc đó bạn biết ngay để nâng cấp từ `Whisper-Small` lên `Whisper-Medium` hoặc `Whisper-Large-v3` (vẫn đóng băng) **trước khi train**, tránh tốn công train xong mới đổi!

---

### 4. Sanity Check #3: Overfit 1 Batch nhỏ (1–2 tiếng — Kiểm tra khả năng học!)
Đây chính là bước **"kiểm tra khả năng học"** mà bạn nhắc tới. Tuyệt đối **không** mang 100 giờ ra train ngay nếu bước này chưa qua:
*   **Cách làm:**
    *   *Cho Giai đoạn 1 (OT-CFM):* Lấy đúng **16 câu tiếng Anh** $\rightarrow$ cho OT-CFM train lặp đi lặp lại trên đúng 16 câu này (khoảng 500–1.000 bước).
    *   *Cho Giai đoạn 2 (Full `Bridge + OT-CFM`):* Lấy đúng **16–32 cặp câu Việt–Anh** từ FLEURS $\rightarrow$ cho mô hình overfit trên đúng 16–32 câu này.
*   **Tiêu chuẩn đạt (Pass):**
    *   Đường Loss phải cắm đầu giảm đều và mượt.
    *   Khi chạy infer lại trên **chính 16 câu đó**, âm thanh sinh ra phải **đọc đúng nội dung câu dịch tiếng Anh** và giữ đúng màu giọng!
*   **Nếu Fail (loss không giảm hoặc đọc ra tiếng rè/vô nghĩa trên 16 câu):** Chắc chắn code đang có bug (lỗi Attention Mask, lỗi tính chiều thời gian $t$ trong Flow Matching, sai chiều tensor Mel, hoặc Bridge làm mất thông tin). Sửa bug ngay trên 16 câu này cực kỳ nhanh (chỉ mất vài phút mỗi lần chạy lại!).

---

### Tóm lại thứ tự chuẩn từ A $\rightarrow$ Z cho bạn:
1. Dựng xong script chấm điểm + chạy Baseline (`SeamlessM4T-v2`, `Whisper + F5-TTS`) trên FLEURS Test.
2. Chuẩn bị xong dữ liệu + sửa code kiến trúc `Vi -> En`.
3. **Làm 4 bước EDA & Sanity Check ở trên (mất ~1 ngày)** $\rightarrow$ Pass hết (đặc biệt là Overfit 16 câu ra tiếng chuẩn).
4. Bấm nút **Train chính thức** (Stage 1 trên 100–200h Anh $\rightarrow$ Stage 2 trên VIVOS/CV/FLEURS).