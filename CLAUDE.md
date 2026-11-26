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
4. **Decoder CFM dùng chung:** `FlowMatchingTargetDecoder` (hàm tạo theo preset `v1` | `v2` | `v2-small`) cùng `CFMDecoderV1/V2`, `MaskedGroupNorm` chỉ được định nghĩa trong `train_utils.py` (Stage 1, Stage 2, eval cùng import). Kiến trúc được lưu trong `stage1_meta.decoder_config` / `stage2_meta.decoder_config` và luôn dựng lại từ đó khi nạp (strict). Trước 23/11/2026 hai file định nghĩa lệch tên tham số nên Stage 2 không nạp được CFM — tuyệt đối không tách lại.
   * Chọn kiến trúc bằng số liệu: `bash run.sh compare-decoder [bước] [biến_thể]` (cùng dữ liệu/seed → bảng val loss, Mel L1, WER tái tạo, s/bước + audio nghe thử ở `outputs/sanity/decoder_compare/`), rồi đặt `decoder_preset` / `t_sampling` trong `configs/stage1_cfm_en.yaml`.
5. **Hai encoder tách biệt:** encoder **nguồn** (`model.whisper_model`, PhoWhisper) nghe audio Việt → đầu vào Bridge; encoder **đích** (`model.target_encoder` trong `stage1_cfm_en.yaml`) nghe audio Anh → điều kiện `c` (Stage 1), nhãn `c*` và speaker (Stage 2), lưu trong `stage1_meta.target_encoder` / `stage2_meta.target_encoder`. Speaker lúc suy luận = `spk_encoder(encoder_đích(audio nguồn))`. So sánh: `bash run.sh compare-encoder` (EXP02).
6. **Checkpoint có meta:** Stage 1 lưu `stage1_meta` + `full_system_state_dict` (gồm `cond_proj`, `spk_encoder`); Stage 2 lưu `stage2_meta`. Checkpoint không có meta (bản En→Vi cũ 13/11) bị từ chối.
7. **Các thành phần đóng băng (Frozen Models):**
   * `vinai/PhoWhisper-small` Encoder: **Đóng băng 100% (`requires_grad = False`)**.
   * `charactr/vocos-mel-24khz` Vocoder: **Đóng băng 100% (`eval()`)**.
   * `FlowMatchingTargetDecoder`: Được train ở Stage 1, sang Stage 2 **đóng băng 100%** làm Giám khảo âm học (Acoustic Critic).

---

## 🏗️ 2. KIẾN TRÚC MÔ HÌNH (`ViEnS2STSystem`) — HIỆN TẠI: HAI LƯỢT NGỮ NGHĨA (từ 26/11/2026)

> Bản trực tiếp (PhoWhisper(audio Việt) → `CrossAttentionBridge` → c) **đã thất bại** (val L1 kẹt 0.44, ASR-BLEU 0, cả khi thêm CTC) và chỉ còn là baseline/ablation (`configs/stage2_bridge.yaml`). Lịch sử đầy đủ: `docs/worklog/2026-11-26.md`.

Suy luận: Vi Audio 16 kHz → **PhoWhisper-small** (ASR Việt) → **NLLB-200-distilled-600M** (dịch Vi→En, rồi chạy lại decoder trên câu Anh để lấy trạng thái ẩn 1024 chiều/token) → tách thành **đơn vị KÝ TỰ** (trạng thái NLLB của token chứa ký tự + id ký tự; thêm đơn vị **LẶNG** id 0 ở đầu/cuối) → **`MASBridge`** (Transformer pre-norm trên ký tự → μ → **bộ dự đoán thời lượng nhận vector giọng** → kéo dài μ theo frame → refiner conv) → c [B, 512, T] → **CFM Stage 1** (đóng băng, ODE 32 bước) → Mel 100 → **Vocos** → En Audio 24 kHz. Giọng: `spk_encoder(encoder_đích(audio Việt nguồn))`.

| Khối | Class / Pretrained | Tham số | Trạng thái | Vai trò |
| :--- | :--- | :---: | :---: | :--- |
| ASR Việt | `vinai/PhoWhisper-small` | ~240M | 🔒 | Chép lời tiếng Việt; encoder cũng tạo c\* và vector giọng |
| Dịch máy | `facebook/nllb-200-distilled-600M` | 600M | 🔒 | Kiến thức dịch; ngữ nghĩa câu Anh theo token |
| Bridge | `MASBridge` (`bridge_type: mas`, `semantic.unit: char`) | ~20.4M | 🔥 Stage 2 | Ký tự → điều kiện âm học theo thời gian (nói gì, lúc nào, bao lâu) |
| Căn chỉnh (chỉ train) | `CTCCharAligner` = `facebook/wav2vec2-base-960h` | ~95M | 🔒 | Forced alignment ký tự trên audio Anh đích (Viterbi CTC tự viết) → thời lượng từng ký tự |
| Decoder âm học | `FlowMatchingTargetDecoder` preset v2 | ~20.5M | 🔥 Stage 1 / 🔒 Stage 2 | OT-CFM sinh Mel; ở Stage 2 làm giám khảo âm học khả vi |
| Teacher / giọng | `cond_proj`, `spk_encoder` (Stage 1) | ~0.4M / 0.33M | 🔒 | Nhãn c\*, vector giọng 192 chiều |
| Vocoder | `charactr/vocos-mel-24khz` | ~13M | 🔒 | Mel → sóng âm |

---

## ⚡ 3. CHIẾN LƯỢC HUẤN LUYỆN 2 GIAI ĐOẠN & HÀM LOSS

### Stage 1: Huấn luyện CFM Decoder tiếng Anh (`train_cfm.py`) — ✅ XONG
* $\mathcal{L}_{\text{CFM}} = \text{MSE}\big(v_\theta(x_t, t, c, s), \; (x_1 - x_0)\big)$, $c$ = interp(`cond_proj`(PhoWhisper_enc(audio Anh))).
* `bash run.sh start cfm_en` → `checkpoints/cfm_en-best.pth` (chỉ FLEURS; WER tái tạo **29.2%**, audio gốc 9.2%). Drive: `gdrive:runs/cfm_en/checkpoints/`.

### Stage 2: Huấn luyện MASBridge (`train_bridge.py --config configs/stage2_bridge_sem.yaml`)
* Đóng băng mọi thứ trừ Bridge. Văn bản Anh lúc train = tham chiếu (`train_text: ref`); lúc đo ASR-BLEU = bản dịch NLLB (`eval_text: mt`).
* $\mathcal{L}_{\text{total}} = 1.0\,\mathcal{L}_{\text{CFM}} + 1.0\,\mathcal{L}_{\text{cond}} + 1.0\,\mathcal{L}_{\text{prior}} + 1.0\,\mathcal{L}_{\text{prior\_mel}} + 0.1\,\mathcal{L}_{\text{dur}}$ (CTC = 0)
  * $\mathcal{L}_{\text{CFM}}$ qua decoder đóng băng; $\mathcal{L}_{\text{cond}}$ = L1(c, c\*), c\* = `teacher_condition`(PhoWhisper_enc(audio **Anh đích**)).
  * $\mathcal{L}_{\text{prior}}$ = L1(μ kéo dài, c\*); $\mathcal{L}_{\text{prior\_mel}}$ = L1(`mel_head`(μ) kéo dài, Mel chuẩn hóa).
  * $\mathcal{L}_{\text{dur}}$ = MSE(log d̂, log d), d = thời lượng **forced alignment wav2vec2** (`semantic.alignment: w2v`; `mas` = MAS tự học, đã thất bại trên dữ liệu đầy đủ).
  * Speaker lúc train = audio **đích**; suy luận = giọng **nguồn**. Mọi loss mask phần đệm.
* **Chỉ số quyết định: ASR-BLEU / WER** (mỗi `eval_asr_every` epoch) và CFM so với 2 mốc (c\* ≈ 0.65, c = 0 ≈ 1.10). **Val Cond có sàn cao** (c\* chứa thông tin riêng của bản ghi) nên không dùng để kết luận; đo sàn bằng `bash run.sh floor 40`.
* Best checkpoint theo **ASR-BLEU** (`training.best_metric: asr_bleu`).
* Kết quả 26/11 (20 epoch, `bridge_vien_mas`): ASR-BLEU 4.8 → 9.6 → 13.8 → **21.1** (WER 68%), chưa bão hòa → config 60 epoch.
* Lệnh: overfit `bash run.sh overfit bridge --config configs/stage2_bridge_sem.yaml --overfit-steps 2000`; train `bash run.sh start bridge_vien_mas --fresh` (tên exp có `mas` → run.sh tự dùng config sem).

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
| **Train Stage 2 Bridge** | `bash run.sh start bridge_vien_mas --fresh` | MASBridge ngữ nghĩa (config sem). `bridge_vien` = bản trực tiếp cũ (baseline) |
| **Đo sàn Cond** | `bash run.sh floor 40` | L1 giữa c\* của 2 người đọc cùng câu (căn theo ký tự) |
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

Trạng thái và việc dở dang mới nhất luôn ở **mục cuối của file mới nhất trong `docs/worklog/`** — đọc trước khi làm tiếp. Tóm tắt (26/11 chiều):

1. Đo sàn Cond: `bash run.sh floor 40`.
2. Train Stage 2 lên 60 epoch: `bash run.sh start bridge_vien_mas --resume` (hoặc `--fresh`), theo dõi ASR-BLEU.
3. Đo S2TT-BLEU của PhoWhisper + NLLB (cận trên của ASR-BLEU).
4. Benchmark Case 4 (checkpoint ngữ nghĩa) so với baseline: `bash run.sh eval all both 30`.
5. Nếu BLEU bão hòa thấp: thêm dữ liệu phía Anh cho Stage 2 (PhoST, LibriTTS).
6. Gia hạn token rclone ~30/11, thu hồi token đã lộ.
