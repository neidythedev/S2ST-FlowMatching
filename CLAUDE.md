# 🤖 CLAUDE.md - MASTER INSTRUCTION & CONTEXT GUIDE

> **Dành cho Claude / AI Assistants:** Đây là tài liệu chỉ dẫn kỹ thuật, quy tắc mã nguồn và bức tranh toàn cảnh dự án Dịch nói trực tiếp Tiếng Việt sang Tiếng Anh (**Vietnamese-to-English Direct Speech-to-Speech Translation - S2ST**) sử dụng **Continuous Optimal Transport Flow Matching (OT-CFM)** và **Cross-Attention Bridge**.

---

## 📌 1. BẢN ĐỒ DỰ ÁN & QUY TẮC MÃ NGUỒN (CORE INVARIANTS)

1. **Kho lưu trữ Git:**
   * Remote: `https://github.com/neidythedev/S2ST-FlowMatching.git`
   * Nhánh chính: `s2st-vi-en` (luôn làm việc và push lên nhánh này).
   * Triển khai lên server GPU Molab:
     ```bash
     git fetch origin s2st-vi-en && git reset --hard origin/s2st-vi-en
     ```
2. **Quy ước ngôn ngữ (Language Convention):**
   * **Bắt buộc:** Toàn bộ ghi chú mã nguồn (code comments), log in ra terminal (`print`), thông báo trạng thái và tài liệu hướng dẫn **phải viết bằng Tiếng Việt**.
3. **Quy tắc Torchaudio Mock Shim (TUYỆT ĐỐI KHÔNG XÓA):**
   * Môi trường Molab chạy Python 3.13 / PyTorch 2.9+ gặp lỗi vỡ ABI C++ của `torchaudio` (`undefined symbol`).
   * Tất cả các file chạy (`eval_benchmark.py`, `train_bridge.py`, `train_cfm.py`, `sanity_and_eda_suite.py`) đều bắt buộc phải có đoạn **In-Memory Mock Torchaudio Shim** (`apply_torchaudio_mock_if_needed`).
   * Shim này cung cấp: `torchaudio.load`, `torchaudio.save`, `torchaudio.info` (qua `soundfile`), `MockSpectrogram`, `MockMelScale`, `MockMelSpectrogram` (tính qua `torch.stft`), và `torchaudio.functional.resample` (tính qua `torch.nn.functional.interpolate`).
4. **Decoder CFM dùng chung:** `FlowMatchingTargetDecoder` / `ResidualBlock1D` / `MaskedGroupNorm` chỉ được định nghĩa trong `train_utils.py` (Stage 1 và Stage 2 cùng import). Trước 23/11/2026 hai file định nghĩa lệch tên tham số nên Stage 2 không nạp được CFM — tuyệt đối không tách lại.
5. **Checkpoint có meta:** Stage 1 lưu `stage1_meta` + `full_system_state_dict` (gồm `cond_proj`, `spk_encoder`); Stage 2 lưu `stage2_meta`. Checkpoint không có meta (bản En→Vi cũ 13/11) bị từ chối.
6. **Các thành phần đóng băng (Frozen Models):**
   * `vinai/PhoWhisper-small` Encoder: **Đóng băng 100% (`requires_grad = False`)**.
   * `charactr/vocos-mel-24khz` Vocoder: **Đóng băng 100% (`eval()`)**.
   * `FlowMatchingTargetDecoder`: Được train ở Stage 1, sang Stage 2 **đóng băng 100%** làm Giám khảo âm học (Acoustic Critic).

---

## 🏗️ 2. KIẾN TRÚC MÔ HÌNH (`ViEnS2STSystem`)

Hệ thống S2ST hoàn chỉnh hoạt động trực tiếp không qua văn bản trung gian (Direct Speech-to-Speech):

$$\text{Vi Audio (16kHz)} \xrightarrow{\text{PhoWhisper Encoder}} H_{src} \xrightarrow{\text{CrossAttentionBridge}} c \xrightarrow{\text{OT-CFM Decoder (ODE)}} \text{100-Mel} \xrightarrow{\text{Vocos}} \text{En Audio (24kHz)}$$

| Khối module | Class / Pretrained | Số tham số | Trạng thái train | Vai trò kỹ thuật |
| :--- | :--- | :---: | :---: | :--- |
| **Speech Encoder** | `vinai/PhoWhisper-small` | ~100M | 🔒 Đóng băng | Trích xuất ngữ nghĩa âm học tiếng Việt ($H_{src} \in \mathbb{R}^{B \times T_{src} \times 768}$). WER 9.0%. |
| **Speaker Encoder** | `AcousticSpeakerEncoder` | ~0.33M | 🔥 Trainable | Rút trích d-vector âm sắc 192d từ Mean/Std của $H_{src}$ để giữ giọng nói gốc. |
| **Translation Bridge** | `CrossAttentionBridge` | ~35M | 🔥 Trainable | 6 tầng Transformer Decoder + Cross-Attention ánh xạ ngữ pháp Việt sang điều kiện tiếng Anh $c \in \mathbb{R}^{B \times 512 \times T_{tgt}}$. Kèm `DurationPredictor`. |
| **Acoustic Decoder** | `FlowMatchingTargetDecoder` | ~4.6M | 🔥 Stage 1 / 🔒 Stage 2 | Mạng 1D Conv ResNet (4 blocks, $d=384$) giải mã hạt nhiễu $x_0 \sim \mathcal{N}(0, I)$ thành Mel 100 kênh. |
| **Neural Vocoder** | `charactr/vocos-mel-24khz` | ~13M | 🔒 Đóng băng | ConvNeXt + iSTFT chuyển đổi Mel 100 kênh sang sóng âm 24kHz trong trẻo trong vài mili-giây. |

* **Tổng tham số cần train (Stage 2):** **~35M – 40M** (Cực nhẹ, ăn **< 6GB VRAM**, không bao giờ bị OOM).

---

## ⚡ 3. CHIẾN LƯỢC HUẤN LUYỆN 2 GIAI ĐOẠN & HÀM LOSS

### Stage 1: Huấn luyện CFM Decoder tiếng Anh (`train_cfm.py`)
* **Mục tiêu:** Dạy mạng CFM học quy luật khử nhiễu để sinh ra phổ Mel tiếng Anh chuẩn.
* **Hàm loss:** $\mathcal{L}_{\text{CFM}} = \text{MSE}\big(v_\theta(x_t, t, c, s), \; (x_1 - x_0)\big)$ trên đường thẳng Optimal Transport.
* **Lệnh:** `bash run.sh start cfm_en` $\rightarrow$ Checkpoint lưu tại `checkpoints/cfm_en-best.pth`.

### Stage 2: Huấn luyện Cross-Attention Bridge Việt $\rightarrow$ Anh (`train_bridge.py`)
* **Mục tiêu:** Dạy Bridge học cách dịch ngữ nghĩa sóng âm tiếng Việt sang biểu diễn điều kiện mà CFM tiếng Anh có thể phát âm được.
* **Cơ chế gradient:** Đóng băng PhoWhisper và CFM Decoder. Gradient từ sai số âm thanh tiếng Anh $\mathcal{L}_{\text{CFM}}$ chạy xuyên qua CFM Decoder đóng băng (đóng vai trò **Differentiable Acoustic Critic**) truyền thẳng về cập nhật trọng số của Bridge!
* **Hàm loss tổng:** 
  $$\mathcal{L}_{\text{total}} = 1.0 \times \mathcal{L}_{\text{CFM}} + 0.1 \times \mathcal{L}_{\text{dur}} + 1.0 \times \mathcal{L}_{\text{cond}}$$
  * $\mathcal{L}_{\text{dur}} = \text{SmoothL1}(\text{dur\_scale}, \; \text{target\_dur\_ratio})$: Dự đoán độ dài câu đích (tỉ lệ tính **theo từng câu**, không theo batch đệm).
  * $\mathcal{L}_{\text{cond}} = \text{L1}(\text{Bridge}(H_{vi}), \; c^*)$ — **chưng cất**: $c^*$ = `cond_proj` của Stage 1 áp lên PhoWhisper(audio **tiếng Anh đích**), tức đúng điều kiện CFM Stage 1 đã quen. FLEURS có audio En song song nên Bridge có nhãn trực tiếp.
  * **Speaker:** lúc train dùng embedding của audio **đích** (`spk_encoder` Stage 1, đóng băng); lúc suy luận dùng giọng **nguồn** → giữ giọng.
  * Mọi loss có **mask phần đệm**; decoder dùng `MaskedGroupNorm` nên train (batch có đệm) và suy luận (1 câu) nhất quán.
* **Lệnh:** `bash run.sh start bridge_vien` $\rightarrow$ Checkpoint lưu tại `checkpoints/bridge_vien-best.pth`.

---

## 🧪 4. BỘ SANITY CHECKS (KIỂM ĐỊNH TRƯỚC KHI TRAIN)

Tất cả nằm trong [`sanity_and_eda_suite.py`](file:///d:/Github/ai%20biet/S2ST/sanity_and_eda_suite.py):
1. **EDA Dữ liệu:** Kiểm tra phân phối độ dài câu, sample rate (16k vs 24k), tỷ lệ thời lượng $En/Vi \in [0.7\times, 1.5\times]$.
2. **Sanity Check #1 (Vocos Round-trip):** Lấy audio tiếng Anh 24k $\rightarrow$ trích xuất Mel 100 kênh $\rightarrow$ đưa vào Vocos tái tạo. Phải đạt Mel L1 error $< 0.40$ và nghe trong vắt.
3. **Sanity Check #2 (Whisper Health):** Đo WER của PhoWhisper trên tập câu tiếng Việt test, phải đạt $\text{WER} < 20\%$.
4. **Sanity Check #3 (Overfit 16 câu):** Cho mô hình overfit lặp lại trên đúng 16 câu trong 500 bước. Loss phải giảm $> 95\%$ để chứng minh gradient flow thông suốt.
   * Lệnh: `bash run.sh overfit cfm` hoặc `bash run.sh overfit bridge`.

---

## 📊 5. HỆ THỐNG BENCHMARK & METRICS (`eval_benchmark.py`)

### Danh mục 7 Kịch bản Thử nghiệm (Cases):
* **Case 0:** Pretrained Cascaded Baseline (PhoWhisper-small + VinAI-Translate/NLLB-200 + MMS-TTS-Eng).
* **Case 5:** Meta SeamlessM4T-v2 SOTA Direct S2ST (`facebook/seamless-m4t-v2-large`).
* **Case 6:** Cascaded Voice Cloning XTTS-v2 (hỗ trợ cả 2-step Whisper translate và 3-step PhoWhisper+MT).
* **Case 7:** Cascaded Voice Cloning F5-TTS Flow-Matching (hỗ trợ cả 2-step và 3-step).
* **Case 1 $\rightarrow$ 3:** Ablation Studies (Zero baseline, Acoustic only, Semantic only).
* **Case 4:** Full Proposed System (PhoWhisper + CrossAttentionBridge + OT-CFM + Vocos).

### Bốn chỉ số đo lường học thuật:
1. **RTF (Real-Time Factor) $\downarrow$:** Thời gian suy luận / Thời lượng audio sinh ra (Proposed: $\approx 0.05$, nhanh gấp 5-10 lần Cascaded).
2. **Mel L1 Error $\downarrow$:** Sai số phổ âm học so với English Ground Truth.
3. **SECS (Speaker Embedding Cosine Similarity) $\uparrow$:** Độ tương đồng cosine giữa vector âm sắc giọng nguồn và giọng đích (đo khả năng giữ giọng).
4. **ASR-BLEU (0 - 100) $\uparrow$:** Dùng `openai/whisper-small` transcribe audio tiếng Anh sinh ra $\rightarrow$ tính SacreBLEU so với văn bản gốc (`Target Ref`).

### Giao diện kết quả trực quan:
* Báo cáo Markdown: `outputs/benchmark/ablation_summary.md`.
* Giao diện HTML Gallery: `outputs/benchmark/index.html` (trình phát audio 6 cột so sánh song song).

---

## 🛠️ 6. CẨM NANG LỆNH 1-CHẠM TRÊN MOLAB (`run.sh`)

| Nhu cầu thao tác | Lệnh thực thi | Ghi chú |
| :--- | :--- | :--- |
| **EDA & Sanity Check** | `bash run.sh sanity` | Kiểm tra Vocos round-trip + PhoWhisper health |
| **Overfit 16 câu** | `bash run.sh overfit [cfm\|bridge]` | Chạy 500 bước kiểm tra học trên batch nhỏ |
| **Train Stage 1 CFM** | `bash run.sh start cfm_en` | Huấn luyện bộ giải mã Flow-Matching tiếng Anh |
| **Train Stage 2 Bridge** | `bash run.sh start bridge_vien` | Huấn luyện Cross-Attention Bridge Vi $\rightarrow$ En |
| **Theo dõi tiến trình** | `bash run.sh logs [tên_exp]` | Xem log thời gian thực |
| **Kiểm tra trạng thái** | `bash run.sh status [tên_exp]` | Kiểm tra PID tiến trình đang chạy |
| **Vẽ đồ thị Loss** | `bash run.sh plot [tên_exp]` | Xuất biểu đồ ASCII và file ảnh SVG |
| **Đồng bộ Google Drive**| `bash run.sh sync [tên_exp]` | Đẩy checkpoints và logs lên Google Drive |
| **Khôi phục Checkpoint**| `bash run.sh restore [tên_exp]`| Tải checkpoints từ Drive về Molab |
| **Benchmark SOTA TTS** | `bash run.sh eval sota both 10` | Chạy 10 mẫu test cho cả XTTS-v2 & F5-TTS |
| **Benchmark Baseline 0**| `bash run.sh eval 0 both 10` | Chạy Cascaded Baseline (PhoWhisper+MT+MMS) |
| **Benchmark Toàn bộ** | `bash run.sh eval all both 20` | Chạy đầy đủ 7 Cases |
| **Upload Outputs** | `bash run.sh upload` | Đẩy toàn bộ audio .wav, HTML, JSON lên GitHub |

---

## ⚠️ 7. CÁC BẪY KỸ THUẬT & CÁCH XỬ LÝ (GOTCHAS)

1. **XTTS-v2 Shape Mismatch:** Checkpoint của XTTS-v2 chứa các trọng số cố định cho `hifigan_decoder.speaker_encoder` với window `[400]` và mel scale `[257, 64]`. Lớp `MockMelSpectrogram` và `MockSpectrogram` trong `eval_benchmark.py` và `train_bridge.py` bắt buộc phải tạo kích thước buffer động theo tham số truyền vào constructor (`n_fft`, `win_length`, `n_mels`).
2. **F5-TTS Flash-Attention:** Môi trường Molab không tương thích với binary precompiled của `flash-attn`. Trong `run.sh` và `eval_benchmark.py`, luôn tự động gỡ bỏ `flash-attn` (`pip uninstall flash-attn -y`) để F5-TTS tự động chuyển sang PyTorch SDPA fallback.
3. **VinAI-Translate Sentencepiece:** Khi file `sentencepiece.bpe.model` trong cache Hugging Face bị lỗi, script tự động xóa cache thư mục `~/.cache/huggingface/hub/models--vinai--vinai-translate-vi2en-v2` và fallback sang Meta `facebook/nllb-200-distilled-600M`.
4. **Resample không có torchaudio C++:** Dùng hàm `_resample` dựa trên `torch.nn.functional.interpolate` để thay đổi tần số lấy mẫu an toàn.

---

## 🎯 8. KẾ HOẠCH HÀNH ĐỘNG TIẾP THEO (NEXT PRIORITIES)

Khi tiếp tục phiên làm việc, hãy ưu tiên các nhiệm vụ theo thứ tự sau:

1. **Kiểm tra kết quả Benchmark SOTA Baselines:**
   * Chạy: `bash run.sh eval sota both 10`.
   * Chạy: `bash run.sh upload` để đẩy audio sinh ra lên GitHub kiểm tra chất lượng nghe thực tế của XTTS-v2 và F5-TTS.
2. **Tiến hành Huấn luyện Stage 1 OT-CFM Tiếng Anh:**
   * Chạy: `bash run.sh start cfm_en`.
   * Theo dõi **WER tái tạo** (in mỗi 5 epoch, so với WER của audio gốc) thay vì ngưỡng val loss. Checkpoint `checkpoints/cfm_en-best.pth` — Stage 2 và eval tự tìm đúng tên này, KHÔNG cần copy.
3. **Tiến hành Huấn luyện Stage 2 Cross-Attention Bridge:**
   * Chạy: `bash run.sh start bridge_vien`.
   * Mô hình sẽ tự nạp CFM checkpoint từ Stage 1, đóng băng CFM và chỉ train Bridge trên cặp câu FLEURS Vi $\rightarrow$ En.
4. **Đánh giá Toàn Diện Case 4 Đề Xuất so với Toàn Bộ Baseline:**
   * Chạy: `bash run.sh eval all both 30`.
   * Thu thập bảng số liệu chính thức: ASR-BLEU, RTF, SECS, Mel L1 để hoàn tất Khóa luận Tốt nghiệp hoặc nộp bài báo khoa học.
