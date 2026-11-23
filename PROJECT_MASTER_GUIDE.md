# 📘 TỔNG HỢP TOÀN DIỆN DỰ ÁN S2ST FLOW-MATCHING (VIỆT ➔ ANH) & TÀI LIỆU CHUYỂN GIAO (CLAUDE MASTER GUIDE)

> **Mục đích tài liệu:** Lưu trữ toàn bộ tri thức, kiến trúc hệ thống, quy trình hạ tầng, kết quả thực nghiệm và lộ trình phát triển của dự án Dịch nói trực tiếp Tiếng Việt sang Tiếng Anh (**Vietnamese-to-English Direct Speech-to-Speech Translation - S2ST**). Tài liệu này được thiết kế để bất kỳ AI Assistant (đặc biệt là Claude) hoặc kỹ sư nào đọc vào cũng có thể nắm bắt 100% ngữ cảnh và tiếp tục phát triển mà không bị đứt đoạn.

---

## 📑 MỤC LỤC
1. [Tổng Quan Bài Toán & Điểm Mới Khoa Học](#1-tổng-quan-bài-toán--điểm-mới-khoa-học)
2. [Hạ Tầng, Kho Lưu Trữ Git & Môi Trường Molab](#2-hạ-tầng-kho-lưu-trữ-git--môi-trường-molab)
3. [Kiến Trúc Mô Hình Chi Tiết (Architecture)](#3-kiến-trúc-mô-hình-chi-tiết-architecture)
4. [Chiến Lược Huấn Luyện 2 Giai Đoạn & Hàm Loss](#4-chiến-lược-huấn-luyện-2-giai-đoạn--hàm-loss)
5. [Quy Trình EDA & Sanity Check Sống Còn](#5-quy-trình-eda--sanity-check-sống-còn)
6. [Hệ Thống Benchmark Đánh Giá & Các Baseline SOTA](#6-hệ-thống-benchmark-đánh-giá--các-baseline-sota)
7. [Dữ Liệu & Bộ Nạp Streaming Đa Ngữ (FLEURS, VietMed, VIVOS)](#7-dữ-liệu--bộ-nạp-streaming-đa-ngữ)
8. [Các Lỗi Hạ Tầng Đã Giải Quyết (Troubleshooting Log)](#8-các-lỗi-hạ-tầng-đã-giải-quyết-troubleshooting-log)
9. [Cẩm Nang Lệnh Thực Thi 1-Chạm (`run.sh`)](#9-cẩm-nang-lệnh-thực-thi-1-chạm-runsh)
10. [Lộ Trình Hành Động Tiếp Theo (Next Steps for Claude / Team)](#10-lộ-trình-hành-động-tiếp-theo-next-steps-for-claude--team)

---

## 1. TỔNG QUAN BÀI TOÁN & ĐIỂM MỚI KHOA HỌC

### 1.1. Bài toán nghiên cứu
Xây dựng hệ thống dịch tiếng nói trực tiếp từ **Tiếng Việt sang Tiếng Anh (Vi $\rightarrow$ En S2ST)**:
$$\text{Audio Tiếng Việt (16kHz)} \xrightarrow{\text{Direct S2ST}} \text{Audio Tiếng Anh (24kHz chuẩn phòng thu)}$$
Yêu cầu cốt lõi:
1. **Dịch đúng ngữ nghĩa** từ Tiếng Việt sang Tiếng Anh.
2. **Bảo tồn âm sắc người nói (Zero-shot Cross-lingual Voice Cloning):** Giữ nguyên tông giọng, cao độ, giới tính của người nói tiếng Việt khi chuyển sang nói tiếng Anh.
3. **Tốc độ suy luận siêu thời gian thực (Ultra Fast RTF):** Thích hợp cho ứng dụng đàm thoại trực tiếp thời gian thực.

### 1.2. Tính mới & Đột phá công nghệ so với thế giới
* **Từ bỏ mô hình rời rạc (Discrete Codec / Speech LLM):** Các hệ thống như AudioPaLM, VALL-E thường rời rạc hóa âm thanh thành discrete acoustic tokens $\rightarrow$ chạy autoregressive LLM $\rightarrow$ tốc độ chậm, dễ hallucination, vỡ âm sắc và sụp đổ codebook.
* **Đưa vào Continuous Optimal Transport Flow Matching (OT-CFM):** Mô hình hóa quá trình sinh âm học thành đường thẳng tối ưu từ nhiễu Gaussian $x_0 \sim \mathcal{N}(0, I)$ sang phổ Mel 100 kênh $x_1$ thông qua phương trình vi phân thường (ODE).
* **Cross-Attention Bridge Non-Autoregressive:** Thay vì dùng Length Regulator lặp khung (kiểu FastSpeech2 vốn gặp lỗi khi hai ngôn ngữ đảo trật tự từ), dự án sử dụng Transformer Decoder với Cross-Attention để ánh xạ linh hoạt cấu trúc ngữ pháp Việt $\rightarrow$ Anh.
* **Siêu nhẹ (Lightweight):** Toàn bộ phần cần huấn luyện chỉ khoảng **~40M tham số**, ăn **< 6GB VRAM**, chạy mượt trên mọi GPU tầm trung.

---

## 2. HẠ TẦNG, KHO LƯU TRỮ GIT & MÔI TRƯỜNG MOLAB

### 2.1. Quản lý mã nguồn (Git Repository)
* **Thư mục local:** `d:\Github\ai biet\S2ST`
* **Remote GitHub:** `https://github.com/neidythedev/S2ST-FlowMatching.git`
* **Nhánh hoạt động chính (Active Branch):** `s2st-vi-en`
* **Quy trình đồng bộ mã nguồn:**
  * Chỉnh sửa và kiểm tra cú pháp trên máy cục bộ $\rightarrow$ `git commit` $\rightarrow$ `git push origin s2st-vi-en`.
  * Trên server Molab: Chạy lệnh cập nhật:
    ```bash
    git fetch origin s2st-vi-en && git reset --hard origin/s2st-vi-en
    ```

### 2.2. Môi trường tính toán GPU Molab
* **Cấu hình server:** Linux, GPU NVIDIA (T4 / V100 / A10), Python 3.13+, PyTorch 2.9+.
* **Đường dẫn thư mục làm việc trên Molab:** `/marimo/S2ST-FlowMatching` (được liên kết symlink với `~/S2ST-FlowMatching` để Marimo file explorer luôn nhận diện đúng).
* **Xử lý lỗi ABI Torchaudio trên Python 3.13:**
  * Phiên bản PyTorch 2.9+ / Python 3.13 trên Molab gặp lỗi nghiêm trọng do torchaudio C++ binary bị thiếu symbol ABI (`undefined symbol`).
  * **Giải pháp dứt điểm:** Gỡ bỏ hoàn toàn bản torchaudio C++ (`pip uninstall torchaudio -y`) và tích hợp **In-Memory Mock Torchaudio Shim** ngay đầu mỗi script (`eval_benchmark.py`, `train_bridge.py`, `train_cfm.py`, `sanity_and_eda_suite.py`). Bộ shim này dùng `soundfile` cho I/O và `torch.stft` / `torch.nn.functional.interpolate` cho tính toán Mel/Resample.

### 2.3. Tích hợp Rclone & Google Drive tự động
* Lưu trữ checkpoint và log huấn luyện an toàn lên Google Drive qua remote `gdrive:runs/<exp_name>/`.
* Cấu hình Rclone tự động bằng chuỗi Base64 chứa OAuth2 credentials (`client_id`, `client_secret`, `refresh_token` dài hạn) nhúng trực tiếp trong `MOLAB_1CLICK_SETUP.md`.
* Lệnh 1-chạm thiết lập server mới từ con số 0 được lưu tại: [`MOLAB_1CLICK_SETUP.md`](file:///d:/Github/ai%20biet/S2ST/MOLAB_1CLICK_SETUP.md).

---

## 3. KIẾN TRÚC MÔ HÌNH CHI TIẾT (ARCHITECTURE)

Hệ thống S2ST hoàn chỉnh (`ViEnS2STSystem`) gồm 4 khối module liên hoàn:

```
[Audio Tiếng Việt 16kHz]
   ├──> PhoWhisper Encoder (Frozen) ─────────────> H_src [B, T_src, 768]
   │                                                    │
   ├──> AcousticSpeakerEncoder ────> spk_emb [B, 192]   │
   │                                       │            │
   │                                       ▼            ▼
   │                            CrossAttentionBridge (Trainable)
   │                                       │
   │                                       ▼
   │                                cond [B, 512, T_tgt]
   │                                       │
   ▼                                       ▼
[Nhiễu x_0] ─────────────> FlowMatchingTargetDecoder (OT-CFM)
                                           │
                                           ▼
                               Mel-Spectrogram 100-band [B, 100, T_tgt]
                                           │
                                           ▼
                                 Vocos Vocoder (24kHz)
                                           │
                                           ▼
                             [Audio Tiếng Anh 24kHz]
```

### 3.1. Chi tiết các thành phần:

1. **Speech Encoder (`vinai/PhoWhisper-small`):**
   * Tham số: ~100M (Trạng thái: **ĐÓNG BĂNG 100% - Frozen**).
   * WER nhận dạng tiếng Việt xuất sắc (9.0%), trích xuất đặc trưng ngữ nghĩa âm học $H_{src} \in \mathbb{R}^{B \times T_{src} \times 768}$.
2. **Acoustic Speaker Encoder (`AcousticSpeakerEncoder`):**
   * Tham số: **~0.33M** (Trạng thái: **Trainable**).
   * Trích xuất thống kê Mean & Std trên các khung thời gian để tạo ra d-vector âm sắc 192 chiều chuẩn hóa L2 ($s \in \mathbb{R}^{B \times 192}$).
3. **Cross-Attention Bridge (`CrossAttentionBridge`):**
   * Tham số: **~35M** (Trạng thái: **Trainable - Trọng tâm Stage 2**).
   * Gồm: 1 tầng chiếu `src_proj`, 3 tầng `TransformerEncoder` tinh chỉnh ngữ cảnh tiếng Việt, và 6 tầng `TransformerDecoder` sử dụng Cross-Attention để sinh chuỗi điều kiện tiếng Anh $c \in \mathbb{R}^{B \times 512 \times T_{tgt}}$.
   * Kèm khối `DurationPredictor` dự đoán tỷ lệ co giãn thời gian giữa câu Việt và câu Anh.
4. **Target Acoustic Decoder (`FlowMatchingTargetDecoder`):**
   * Tham số: **~4.6M** (Trạng thái: **Huấn luyện ở Stage 1, Đóng băng ở Stage 2**).
   * Mạng 1D Convolutional ResNet (4 khối residual, hidden_dim = 384, tích hợp điều kiện thời gian $t$, ngữ nghĩa $c$ và âm sắc $s$).
5. **Neural Vocoder (`charactr/vocos-mel-24khz`):**
   * Tham số: ~13M (Trạng thái: **ĐÓNG BĂNG 100% - Frozen**).
   * Dựa trên ConvNeXt và iSTFT, chuyển đổi tức thì phổ Mel 100 kênh thành sóng âm 24kHz trong trẻo, không có tạp âm rè kim loại.

### 3.2. Bảng tổng kết số lượng tham số
* **Tổng số tham số toàn hệ thống:** ~153M
* **Số tham số cần huấn luyện (Stage 2):** **~35M – 40M** (Chiếm ~25% hệ thống).
* **Bộ nhớ GPU yêu cầu:** 4GB – 6GB VRAM (Hoàn toàn miễn nhiễm với lỗi OOM).

---

## 4. CHIẾN LƯỢC HUẤN LUYỆN 2 GIAI ĐOẠN & HÀM LOSS

Mô hình được huấn luyện theo chiến lược chia để trị (Decoupled 2-Stage Training) để tránh hiện tượng "Moving Target":

### 4.1. Giai đoạn 1: Huấn luyện CFM Tiếng Anh (`train_cfm.py`)
* **Mục tiêu:** Dạy cho CFM Decoder kỹ năng tái tạo âm thanh tiếng Anh chuẩn xác từ phổ Mel 100 kênh.
* **Dữ liệu:** Tiếng Anh đơn ngữ (FLEURS `en_us` hoặc LibriTTS).
* **Hàm mất mát:** **OT-CFM Velocity Field Loss ($\mathcal{L}_{\text{CFM}}$)**:
  $$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t, x_0, x_1} \left[ \| v_\theta(x_t, t, c, s) - (x_1 - x_0) \|_2^2 \right]$$
  Trong đó đường đi tối ưu là đường thẳng: $x_t = (1 - t)x_0 + t x_1$ với vận tốc lý tưởng $u_t = x_1 - x_0$.
* **Lệnh chạy:** `bash run.sh start cfm_en`

### 4.2. Giai đoạn 2: Huấn luyện Cross-Attention Bridge (`train_bridge.py`)
* **Mục tiêu:** Dạy Bridge học cách ánh xạ ngữ nghĩa từ sóng âm tiếng Việt sang không gian điều kiện mà CFM Decoder tiếng Anh có thể hiểu.
* **Cơ chế đóng băng:** **PhoWhisper đóng băng 100%**, **CFM Decoder đóng băng 100%**. Toàn bộ áp lực học tập trung vào Bridge và Speaker Encoder.
* **Hàm mất mát tổng hợp:**
  $$\mathcal{L}_{\text{total}} = 1.0 \times \mathcal{L}_{\text{CFM}} + 0.1 \times \mathcal{L}_{\text{dur}}$$
  * $\mathcal{L}_{\text{CFM}}$: Sai số khử nhiễu Mel tiếng Anh. Gradient chạy xuyên qua CFM Decoder đóng băng (đóng vai trò **Differentiable Acoustic Critic**) truyền trực tiếp vào Bridge!
  * $\mathcal{L}_{\text{dur}} = \text{SmoothL1}(\hat{r}_{dur}, r_{dur})$: Sai số dự đoán tỷ lệ độ dài âm thanh giữa 2 ngôn ngữ.
* **Lệnh chạy:** `bash run.sh start bridge_vien`

---

## 5. QUY TRÌNH EDA & SANITY CHECK SỐNG CÒN

Không bao giờ được mang dữ liệu lớn vào huấn luyện nếu chưa vượt qua bộ kiểm tra tại [`sanity_and_eda_suite.py`](file:///d:/Github/ai%20biet/S2ST/sanity_and_eda_suite.py):

| Kiểm tra | Mục đích | Tiêu chuẩn ĐẠT (PASS) | Lệnh chạy |
| :--- | :--- | :--- | :--- |
| **EDA Dữ liệu** | Kiểm tra độ dài, sample rate (16k vs 24k), lọc tỷ lệ $T_{en}/T_{vi}$ dị thường | Tỷ lệ En/Vi trung bình $0.7\times \sim 1.5\times$, không có audio rỗng | `bash run.sh sanity` |
| **Sanity #1: Vocos Round-Trip** | Kiểm tra hàm trích xuất Mel 100 kênh ($n\_fft=1024, hop=256$) có khớp chuẩn Vocos không | Mel L1 Reconstruction Error $< 0.40$, audio tái tạo nghe trong vắt | `bash run.sh sanity` |
| **Sanity #2: Whisper Health** | Kiểm tra độ khỏe của Whisper trên tiếng Việt thực tế | $\text{WER} < 20\%$ trên tập câu test | `bash run.sh sanity` |
| **Sanity #3: Overfit 16 câu** | Kiểm tra mô hình có khả năng học và lan truyền gradient không | Loss giảm $> 95\%$ sau 500 bước trên 16 câu cố định | `bash run.sh overfit [cfm\|bridge]` |

---

## 6. HỆ THỐNG BENCHMARK ĐÁNH GIÁ & CÁC BASELINE SOTA

Hệ thống đánh giá toàn diện được xây dựng tại [`eval_benchmark.py`](file:///d:/Github/ai%20biet/S2ST/eval_benchmark.py), hỗ trợ so sánh đối đầu Case 4 đề xuất với toàn bộ các giải pháp SOTA hiện nay.

### 6.1. Danh mục các Case thử nghiệm:
* **Case 0 (Pretrained Cascaded Baseline):** PhoWhisper-small (ASR) $\rightarrow$ VinAI-Translate/NLLB-200 (MT) $\rightarrow$ MMS-TTS-Eng (TTS).
* **Case 5 (SOTA Direct S2ST):** Meta SeamlessM4T-v2 (`facebook/seamless-m4t-v2-large`).
* **Case 6 (Cascaded Voice Cloning XTTS-v2):** Chạy cả 2 chế độ dịch (2-step Whisper translate & 3-step PhoWhisper+MT) kết hợp XTTS-v2 Zero-shot voice cloning.
* **Case 7 (Cascaded Voice Cloning F5-TTS):** Chạy cả 2 chế độ dịch kết hợp F5-TTS Non-autoregressive Flow-Matching voice cloning.
* **Case 1 $\rightarrow$ 3 (Ablation Studies):**
  * Case 1: Zero-baseline (Bridge ngẫu nhiên + CFM ngẫu nhiên).
  * Case 2: Acoustic Only (Bridge ngẫu nhiên + CFM đã train).
  * Case 3: Semantic Only (Bridge đã train + CFM ngẫu nhiên).
* **Case 4 (Full Proposed System):** PhoWhisper + CrossAttentionBridge đã train + OT-CFM đã train + Vocos.

### 6.2. Bốn chỉ số đo lường học thuật (Metrics):
1. **RTF (Real-Time Factor) $\downarrow$:** Thời gian suy luận / Thời lượng âm thanh sinh ra.
2. **Mel L1 Error $\downarrow$:** Độ sai lệch phổ âm học giữa Mel sinh ra và Mel tiếng Anh người bản xứ.
3. **SECS (Speaker Embedding Cosine Similarity) $\uparrow$:** Độ tương đồng cosine giữa vector âm sắc giọng Việt nguồn và giọng Anh đích (đo mức độ giữ giọng).
4. **ASR-BLEU $\uparrow$:** Dùng `openai/whisper-small` (task `transcribe`) phiên âm file audio tiếng Anh sinh ra $\rightarrow$ tính điểm SacreBLEU (0 - 100) so với câu văn bản tiếng Anh chuẩn (`Target Ref`).

### 6.3. Báo cáo & Thư viện âm thanh tương tác:
* Tự động sinh bảng tổng kết Markdown: `outputs/benchmark/ablation_summary.md`.
* Tự động tạo giao diện Web trực quan: `outputs/benchmark/index.html` gồm 6 cột phát âm thanh song song để nghe so sánh trực tiếp.

---

## 7. DỮ LIỆU & BỘ NẠP STREAMING ĐA NGỮ

* **Tập dữ liệu chính (FLEURS):** Sử dụng `google/fleurs` (cặp `vi_vn` và `en_us`).
* **Cơ chế căn chỉnh song song qua Streaming:** 
  * Do HuggingFace nạp streaming riêng rẽ từng ngôn ngữ, pipeline sử dụng trường `id` (FLoRes Sentence ID) để căn chỉnh 1-1 chính xác tuyệt đối giữa audio tiếng Việt và audio tiếng Anh bản xứ.
  * Bộ đệm song song (buffer) hỗ trợ lên đến **400–600 cặp câu** cho mỗi split (`train`, `test`).
* **Bộ dữ liệu kiểm định ASR bổ sung:**
  * **VietMed:** Tập âm thanh hội thoại y tế 3 miền thực tế (chưa từng xuất hiện trong tập train của Whisper lẫn PhoWhisper).
  * **VIVOS & Mozilla Common Voice:** Đánh giá tính tổng quát hóa của Speech Encoder.

---

## 8. CÁC LỖI HẠ TẦNG ĐÃ GIẢI QUYẾT (TROUBLESHOOTING LOG)

Trong quá trình triển khai thực tế trên Molab, hệ thống đã vượt qua các lỗi kỹ thuật quan trọng sau:

1. **Lỗi Torchaudio C++ ABI (Python 3.13):**
   * *Hiện tượng:* `undefined symbol: ...` khi import torchaudio.
   * *Khắc phục:* Viết Mock Shim hoàn chỉnh trong bộ nhớ (`apply_torchaudio_mock_if_needed`), tích hợp hàm `resample` qua `torch.nn.functional.interpolate`.
2. **Lỗi XTTS-v2 Shape Mismatch (`MockMelSpectrogram`):**
   * *Hiện tượng:* XTTS yêu cầu buffer window `[400]` và mel scale `[257, 64]`, trong khi mock cũ fix cứng `[1024]` và `[513, 100]`.
   * *Khắc phục:* Nâng cấp `MockSpectrogram` và `MockMelSpectrogram` tự động tính kích thước buffer động theo tham số khởi tạo (`n_fft`, `win_length`, `n_mels`).
3. **Lỗi F5-TTS Flash-Attention:**
   * *Hiện tượng:* `cannot import name 'flash_attn_func' from 'flash_attn'`.
   * *Khắc phục:* Tự động phát hiện và gỡ bỏ `flash-attn` bị hỏng, ép F5-TTS sử dụng cơ chế PyTorch SDPA (Scaled Dot-Product Attention) chuẩn có sẵn trong PyTorch.
4. **Lỗi Parse Sentencepiece của VinAI-Translate:**
   * *Hiện tượng:* `Error parsing line b'\x0e' in sentencepiece.bpe.model`.
   * *Khắc phục:* Nâng cấp `sentencepiece` và `protobuf`, thêm cơ chế kiểm tra tính toàn vẹn của cache; nếu lỗi thì tự động xóa và tải lại, đồng thời tích hợp fallback mượt mà sang `facebook/nllb-200-distilled-600M`.
5. **Cửa sổ xác nhận License CPML của XTTS-v2:**
   * *Khắc phục:* Đặt biến môi trường `COQUI_TOS_AGREED=1` trong cả Bash lẫn Python để bypass prompt interactive.

---

## 9. CẨM NANG LỆNH THỰC THI 1-CHẠM (`run.sh`)

Mọi thao tác trong dự án đều được gói gọn qua tệp điều khiển trung tâm [`run.sh`](file:///d:/Github/ai%20biet/S2ST/run.sh):

```bash
# 1. Kiểm tra môi trường & kiểm định sức khỏe hệ thống
bash run.sh sanity

# 2. Overfit kiểm tra học trên 16 câu (Sanity Check #3)
bash run.sh overfit cfm          # Overfit Stage 1 CFM tiếng Anh
bash run.sh overfit bridge       # Overfit Stage 2 Bridge Việt -> Anh

# 3. Huấn luyện chính thức
bash run.sh start cfm_en         # Train Stage 1: CFM Tiếng Anh
bash run.sh start bridge_vien    # Train Stage 2: CrossAttentionBridge Vi -> En
bash run.sh logs [tên_exp]        # Xem log huấn luyện thời gian thực
bash run.sh status [tên_exp]      # Kiểm tra trạng thái tiến trình
bash run.sh stop [tên_exp]        # Dừng tiến trình huấn luyện

# 4. Sao lưu & Đồng bộ Google Drive
bash run.sh sync [tên_exp]       # Đẩy checkpoint và logs lên Google Drive
bash run.sh restore [tên_exp]    # Khôi phục checkpoint từ Google Drive về máy

# 5. Benchmark thực nghiệm & Đánh giá SOTA
bash run.sh eval sota both 10    # Benchmark cả XTTS-v2 & F5-TTS ở 2 chế độ (2-step & 3-step)
bash run.sh eval 0 both 10       # Benchmark Case 0 (Cascaded Baseline)
bash run.sh eval all both 20     # Chạy toàn bộ 7 Cases (0, 6, 7, 1, 2, 3, 4)

# 6. Đẩy outputs (Audio, HTML Gallery, Metrics) lên GitHub để tải về máy
bash run.sh upload
```

---

## 10. LỘ TRÌNH HÀNH ĐỘNG TIẾP THEO (NEXT STEPS FOR CLAUDE / TEAM)

Khi Claude hoặc kỹ sư mới tiếp nhận dự án, hãy thực hiện tuần tự theo các bước ưu tiên sau:

### 🎯 Bước 1: Hoàn tất Benchmark các Baseline Voice Cloning (Case 6 & Case 7)
* Đảm bảo `bash run.sh eval sota both 10` chạy trơn tru trên Molab sau khi đã fix `MockMelSpectrogram.forward()` và `torchaudio.functional.resample`.
* Kiểm tra file `outputs/benchmark/index.html` và chạy `bash run.sh upload` để kiểm tra chất lượng file `.wav` sinh ra từ XTTS-v2 và F5-TTS.

### 🎯 Bước 2: Huấn luyện Stage 1 OT-CFM Tiếng Anh (`cfm_en`)
* Chạy: `bash run.sh start cfm_en` trên Molab.
* Mục tiêu: Huấn luyện mạng Flow-Matching đạt validation loss thấp ($< 1.0$), lưu checkpoint tốt nhất tại `checkpoints/cfm_en-best.pth`.
* Sau khi train xong, copy checkpoint sang `checkpoints/cfm-best.pth`.

### 🎯 Bước 3: Huấn luyện Stage 2 Cross-Attention Bridge (`bridge_vien`)
* Chạy: `bash run.sh start bridge_vien` trên Molab.
* Hệ thống sẽ tự động nạp `checkpoints/cfm-best.pth`, đóng băng CFM và chỉ huấn luyện Bridge + Speaker Encoder trên cặp câu song ngữ FLEURS (Vi $\rightarrow$ En).
* Theo dõi đồ thị loss bằng: `bash run.sh plot bridge_vien`.

### 🎯 Bước 4: Chạy Full Benchmark Hoàn Chỉnh (Case 4 vs All Baselines)
* Chạy: `bash run.sh eval all both 30`.
* Thu thập bảng chỉ số chính thức:
  * Điểm **ASR-BLEU** của Case 4 đề xuất.
  * Tốc độ suy luận **RTF** (chứng minh tính chất Non-autoregressive nhanh vượt trội so với Cascaded).
  * Điểm giữ giọng **SECS** (chứng minh khả năng giữ âm sắc xuyên ngôn ngữ).
* Xuất báo cáo tổng kết hoàn chỉnh phục vụ viết Khóa luận Tốt nghiệp hoặc bài báo khoa học quốc tế.

---
*Tài liệu được cập nhật tự động và đồng bộ với nhánh `s2st-vi-en` của dự án.*
