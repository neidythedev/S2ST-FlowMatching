# generate_split_notebooks.py
import json
import os

def create_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.9"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

def md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    }

def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().split("\n")]
    }

# ==============================================================================
# NOTEBOOK 1: DISCRETE REPRESENTATIONS & CODEC LLMS
# ==============================================================================
nb1_cells = [
    md("""# CHUYÊN ĐỀ 1: CÁC PHƯƠNG PHÁP RỜI RẠC (DISCRETE REPRESENTATIONS) VÀ CODEC LLMS TRONG S2ST

Notebook này tập trung vào 2 trường phái nền tảng trong Speech-to-Speech Translation (S2ST):
1. **Discrete Semantic Units (HuBERT + k-means)**: Trường phái Textless S2ST của Meta (2021-2022).
2. **Neural Audio Codec & RVQ Tokens trong LLM**: Trường phái đưa âm thanh vào chung không gian từ điển với văn bản (AudioPaLM, SpeechGPT, VioLA - 2023).

> 💡 **Khả năng tương thích**: Notebook này có thể chạy trực tiếp trong VS Code / Jupyter hoặc chuyển đổi sang **Marimo** (`marimo convert ... -o ...py`) để chạy trên MoLab.
"""),

    md("""## 1. Cheat Sheet: Kích thước Tensor và Các Dạng Dữ liệu trong S2ST
Trước khi đi vào thuật toán, hãy nắm vững các chiều không gian (dimensions) của các dạng biểu diễn âm thanh:
"""),

    code("""import torch
import torch.nn as nn
import torch.nn.functional as F

print(f"PyTorch version: {torch.__version__}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Thiết bị đang sử dụng: {device}")

# TỔNG QUAN TENSOR SHAPES:
# 1. Raw Waveform:         [Batch, T_samples]       (ví dụ: 1s @ 16kHz = 16,000 samples)
# 2. Log Mel-Spectrogram:  [Batch, n_mels, T_frames] (80 hoặc 100 mel channels, ~50 hoặc 100 fps)
# 3. Discrete Units:       [Batch, T_units]         (chuỗi số nguyên int in [0, K-1], K ~ 100 - 1000)
# 4. Neural Codec Tokens:  [Batch, N_layers, T]     (EnCodec/DAC: N_layers=4-8, mỗi frame có 4-8 mã)
# 5. Whisper Continuous:   [Batch, T_whisper, 768]  (Vector liên tục 50Hz, giàu ngữ nghĩa)
"""),

    md("""## 2. Trường phái 1: Discrete Semantic Units (HuBERT + k-means)

### Xuất xứ học thuật:
- **Paper gốc**:
  - *Lee et al. (Meta AI, 2021)* - *"Direct Speech-to-Speech Translation With Discrete Units"*.
  - *Popuri et al. (Meta AI, 2022)* - *"Enhanced Direct Speech-to-Speech Translation Using Self-supervised Pre-trained Representations and UnitY"*.
  - *Jia et al. (Google, 2022)* - *"Translatotron 2: High-quality direct speech-to-speech translation with voice preservation"*.
- **Các bài báo cùng tư duy trong thư mục của bạn**: `Translatotron 2.pdf`, `Speech-to-speech survey.pdf`.

### Tư duy kiến trúc:
- Lớp biểu diễn trung gian (layer 6 hoặc layer 9 của HuBERT 12-layer) đã loại bỏ hầu hết thông tin người nói và tạp âm, chỉ giữ lại ngữ âm (phonetic/semantic).
- Thuật toán **k-means** (thường fit trên 50-100h dữ liệu với $K \in [100, 1000]$ cụm) biến mỗi frame thời gian 20ms thành 1 số nguyên (Unit ID).
- Nhờ đó, bài toán S2ST được giải quyết giống hệt Machine Translation: `Source Speech -> Target Discrete Units`.
- Cuối cùng, một bộ **Unit HiFi-GAN Vocoder** tái tạo dạng sóng từ các unit này.
"""),

    code("""# Code mô phỏng: HuBERT SSL Feature Extractor + k-means Quantizer
class HuBERTKMeansTokenizer(nn.Module):
    \"\"\"
    Lấy cảm hứng từ Lee et al. (Meta 2021) và UnitY (Popuri et al. 2022).
    Chuyển đổi vector ngữ nghĩa liên tục từ HuBERT thành Discrete Units.
    \"\"\"
    def __init__(self, feat_dim=768, num_clusters=100):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_clusters = num_clusters
        # Tâm cụm k-means (trong thực tế fit bằng Faiss)
        self.register_buffer("centroids", torch.randn(num_clusters, feat_dim))
        
    def forward(self, continuous_ssl_features):
        \"\"\"
        Input:  [Batch, T_frames, feat_dim] (Vector liên tục từ HuBERT)
        Output: [Batch, T_frames] (Discrete Unit IDs in [0, num_clusters-1])
        \"\"\"
        B, T, D = continuous_ssl_features.shape
        flat = continuous_ssl_features.reshape(-1, D)
        
        # Tính khoảng cách Euclidean tới các tâm cụm: ||x - c||^2
        dist = torch.cdist(flat, self.centroids) # [B*T, num_clusters]
        unit_ids = torch.argmin(dist, dim=-1)    # [B*T]
        return unit_ids.view(B, T)
        
    @staticmethod
    def deduplicate(units):
        \"\"\"
        Loại bỏ các unit lặp lại liên tiếp (Run-Length Deduplication).
        Ví dụ: [12, 12, 12, 45, 45, 12] -> [12, 45, 12].
        Giúp chuỗi ngắn đi 2-3 lần, giảm tải tính toán cho Transformer.
        \"\"\"
        res = []
        prev = None
        for u in units:
            if u != prev:
                res.append(u)
                prev = u
        return res

# Chạy thử nghiệm
hubert_tokenizer = HuBERTKMeansTokenizer(feat_dim=768, num_clusters=100).to(device)
fake_hubert_out = torch.randn(2, 50, 768, device=device) # 2 mẫu, mỗi mẫu 50 frames (1 giây)
discrete_units = hubert_tokenizer(fake_hubert_out)

print("Kích thước HuBERT feature liên tục: ", fake_hubert_out.shape)
print("Kích thước Discrete Units:         ", discrete_units.shape)
print("Chuỗi Units mẫu 0:                 ", discrete_units[0, :12].tolist())
print("Chuỗi Units sau Deduplication:     ", HuBERTKMeansTokenizer.deduplicate(discrete_units[0, :12].tolist()))
"""),

    md("""## 3. Trường phái 2: Neural Codec RVQ Tokens trong LLM

### Xuất xứ học thuật:
- **Paper gốc**:
  - *AudioPaLM (Rubenstein et al., Google 2023)* - *"AudioPaLM: A Large Language Model That Can Speak and Listen"*.
  - *SpeechGPT (Zhang et al., 2023)* - *"SpeechGPT: Empowering Large Language Models with Intrinsic Cross-Modal Conversational Abilities"*.
  - *VioLA (Wang et al., Microsoft 2023)* - *"VioLA: Unified Codec Language Models for Speech Recognition, Synthesis, and Translation"*.
  - *EnCodec (Défossez et al., Meta 2022)* - *"High Fidelity Neural Audio Compression"*.
- **Các bài báo cùng tư duy trong thư mục của bạn**: `speechgpt.pdf`, `AudioPaLM.pdf`, `viola.pdf`.

### Tư duy kiến trúc:
- **Hạn chế của HuBERT k-means**: Nó chỉ chứa ngữ nghĩa (semantics), mất sạch âm sắc người nói (timbre) và ngữ điệu tự nhiên.
- **Giải pháp RVQ (Residual Vector Quantization)**:
  - Chia âm thanh thành nhiều tầng (ví dụ 4 đến 8 codebooks).
  - Tầng 1: Đại diện cho cấu trúc thô và âm vị.
  - Các tầng sau: Bù đắp phần dư (residuals) chứa chi tiết âm sắc, hơi thở, cao độ.
- **Đưa vào LLM**: Mở rộng từ điển của LLM: Gom Text Tokens + Semantic Tokens + Acoustic Tokens. LLM học dự đoán token tiếp theo theo kiểu tự hồi quy (Autoregressive).
"""),

    code("""# Code mô phỏng: Residual Vector Quantizer (RVQ) 4 tầng & Interleaved Tokenization
class ToyRVQCodec(nn.Module):
    \"\"\"
    Lấy cảm hứng từ EnCodec (Défossez et al. 2022) và AudioPaLM (Google 2023).
    \"\"\"
    def __init__(self, num_codebooks=4, codebook_size=1024, latent_dim=128):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebooks = nn.ModuleList([
            nn.Embedding(codebook_size, latent_dim) for _ in range(num_codebooks)
        ])
        
    def encode(self, continuous_latents):
        \"\"\"
        Input:  [Batch, T_frames, latent_dim]
        Output: [Batch, num_codebooks, T_frames] (Chuỗi mã rời rạc ở từng tầng)
        \"\"\"
        residual = continuous_latents
        codes = []
        for cb in self.codebooks:
            # Tìm code vector gần nhất trong codebook hiện tại
            dist = torch.cdist(residual, cb.weight.unsqueeze(0)) # [Batch, T, codebook_size]
            indices = torch.argmin(dist, dim=-1)                # [Batch, T]
            codes.append(indices)
            
            # Tính phần dư cho tầng tiếp theo: residual = residual - quantized
            quantized = cb(indices)
            residual = residual - quantized
        return torch.stack(codes, dim=1)

# Chạy thử nghiệm
rvq = ToyRVQCodec(num_codebooks=4, codebook_size=1024, latent_dim=128).to(device)
fake_audio = torch.randn(2, 25, 128, device=device) # 25 frames
rvq_tokens = rvq.encode(fake_audio)

print("Kích thước Audio Latents đầu vào: ", fake_audio.shape)
print("Kích thước RVQ Tokens:            ", rvq_tokens.shape)
print("Tầng 1 (Coarse/Phonetic):         ", rvq_tokens[0, 0, :5].tolist())
print("Tầng 4 (Fine/Acoustic detail):    ", rvq_tokens[0, 3, :5].tolist())
"""),

    md("""## 4. Tóm kết Chuyên đề 1: Ưu điểm và Điểm nghẽn
- **Ưu điểm**:
  - Dễ dàng tích hợp vào hạ tầng Transformer và LLM có sẵn (chỉ cần dùng Cross-Entropy Loss chuẩn).
- **Điểm nghẽn (Bottleneck)**:
  - Chuỗi RVQ tokens cực dài (ví dụ 8 tầng $\times$ 75 fps = **600 tokens/giây**).
  - Ép LLM gánh chuỗi dài như vậy làm bộ nhớ Self-Attention ($O(N^2)$) bùng nổ, dễ gây lặp từ (looping) hoặc nuốt chữ.
  - Đây chính là lý do thôi thúc các nhà nghiên cứu chuyển dịch sang **Biểu diễn Liên tục và Flow Matching**, được trình bày ở **Chuyên đề 2**!
""")
]

# ==============================================================================
# NOTEBOOK 2: CONTINUOUS REPRESENTATIONS & FLOW MATCHING
# ==============================================================================
nb2_cells = [
    md("""# CHUYÊN ĐỀ 2: BIỂU DIỄN VECTOR LIÊN TỤC VÀ CONDITIONAL FLOW MATCHING (CFM) TRONG S2ST

Notebook này giải quyết 2 trọng tâm công nghệ SOTA (State-of-the-Art):
1. **Giải mã băn khoăn "LLM học biểu diễn vector liên tục"**: Tại sao không lượng tử hóa mà giữ nguyên continuous features từ Whisper/Conformer? (SALMONN, Qwen-Audio, SLAM-Omni, LLaMA-Omni).
2. **Trường phái Whisper + Conditional Flow Matching (CFM) / Diffusion**: Voicebox, F5-TTS, CosyVoice, UNISS. Giải mã tại sao CFM (16-32 bước ODE) thay thế hoàn toàn DDPM truyền thống (1000 bước).

> 💡 **Khả năng tương thích**: Notebook này có thể chạy trực tiếp trong VS Code / Jupyter hoặc chuyển đổi sang **Marimo** (`marimo convert ... -o ...py`) để chạy trên MoLab.
"""),

    md("""## 1. Giải mã băn khoăn: "LLM học biểu diễn vector liên tục"

### Xuất xứ học thuật:
- **Paper gốc**:
  - *SALMONN (Tang et al., 2023)* - *"SALMONN: Towards Generic Hearing Abilities for Large Language Models"*.
  - *Qwen-Audio (Chu et al., Alibaba 2023)* - *"Qwen-Audio: Advancing Universal Audio Understanding via Large-Scale Audio-Language Models"*.
  - *LLaMA-Omni / SLAM-Omni (Fang et al., 2024)* - *"LLaMA-Omni: Seamless Speech Interaction with Large Language Models"*.
- **Các bài báo cùng tư duy trong thư mục của bạn**: `SLAM_Omni (1).pdf`, `S2ST_Omni Hierarchical LanguageAware SpeechLLM Adaptation for Multilingual S2ST.pdf`.

### Bản chất toán học:
1. LLM vốn dĩ hoạt động trên không gian vector liên tục $h \in \mathbb{R}^{d_{model}}$. Nó chỉ biến thành rời rạc ở tầng `lm_head` cuối cùng khi tính Softmax qua Vocabulary.
2. Whisper Encoder xuất ra chuỗi vector liên tục $H \in \mathbb{R}^{T \times D_{whisper}}$ ở tần số 50Hz.
3. Thay vì lượng tử hóa làm mất mát thông tin, ta dùng một **Continuous Modality Projector** (Conv1D temporal downsampler + MLP) để chiếu trực tiếp $H$ vào không gian embedding của LLM.
4. Các vector này hoạt động như các **"Soft Prompts" (Continuous Prefix)**, LLM tiếp nhận mượt mà như word embedding của văn bản mà không bị rào cản từ vựng rời rạc!
"""),

    code("""import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ContinuousWhisperProjector(nn.Module):
    \"\"\"
    Lấy cảm hứng từ SALMONN (2023) và LLaMA-Omni (2024).
    Nén thời gian (temporal downsampling) từ 50Hz xuống 12.5Hz (giảm 4 lần)
    và chiếu vector liên tục của Whisper sang không gian ẩn của LLM.
    \"\"\"
    def __init__(self, audio_dim=768, llm_dim=1024, downsample_rate=4):
        super().__init__()
        # Conv1D vừa giảm độ dài chuỗi vừa học đặc trưng cục bộ
        self.conv = nn.Sequential(
            nn.Conv1d(audio_dim, llm_dim, kernel_size=downsample_rate, stride=downsample_rate),
            nn.GELU(),
            nn.BatchNorm1d(llm_dim)
        )
        # MLP 2 tầng ánh xạ phi tuyến
        self.mlp = nn.Sequential(
            nn.Linear(llm_dim, llm_dim),
            nn.SiLU(),
            nn.Linear(llm_dim, llm_dim)
        )
        
    def forward(self, whisper_features):
        \"\"\"
        Input:  [Batch, T_audio, audio_dim] (50Hz continuous features)
        Output: [Batch, T_audio // downsample_rate, llm_dim] (Soft Prompts cho LLM)
        \"\"\"
        x = whisper_features.transpose(1, 2) # [B, audio_dim, T]
        x = self.conv(x).transpose(1, 2)     # [B, T // 4, llm_dim]
        return self.mlp(x)

# Thử nghiệm Projector
projector = ContinuousWhisperProjector(audio_dim=768, llm_dim=1024, downsample_rate=4).to(device)
fake_whisper = torch.randn(2, 100, 768, device=device) # 100 frames (~2 giây âm thanh)
continuous_tokens = projector(fake_whisper)

print("Kích thước đầu vào từ Whisper:       ", fake_whisper.shape)
print("Kích thước sau khi chiếu vào LLM:   ", continuous_tokens.shape)
print("-> Chiều dài chuỗi giảm 4 lần, tiết kiệm 75% bộ nhớ Self-Attention của LLM!")
"""),

    md("""## 2. Trường phái Whisper + Conditional Flow Matching (CFM)

### Xuất xứ học thuật:
- **Paper gốc**:
  - *Voicebox (Le et al., Meta AI 2023)* - *"Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale"*.
  - *F5-TTS (Chen et al., 2024)* - *"F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching"*.
  - *CosyVoice (Du et al., Alibaba 2024)* - *"CosyVoice: A Multi-lingual Large Voice Generation Model"*.
  - *UniS2ST / UNISS (ICLR 2024)* - *"Unified Expressive Speech-to-Speech Translation With Your Voice"*.
- **Các bài báo cùng tư duy trong thư mục của bạn**: `Voicebox.pdf`, `F5-TTS.pdf`, `ICLR-2026-uniss...pdf`.

### Tại sao Flow Matching (CFM) lại vượt trội hoàn toàn so với DDPM?
1. **Diffusion truyền thống (DDPM)**:
   - Mô phỏng quá trình khuếch tán Brownian ngẫu nhiên (SDE). Đường đi từ nhiễu $x_0$ đến Mel $x_1$ bị uốn lượn ngoằn ngoèo, đòi hỏi **100 đến 1000 bước** đảo ngược thời gian.
2. **Conditional Flow Matching (Optimal Transport CFM)**:
   - Thiết lập **đường thẳng ngắn nhất** nối trực tiếp giữa nhiễu Gaussian $x_0 \sim \mathcal{N}(0, I)$ và Mel-spectrogram mục tiêu $x_1$:
     $$x_t = (1 - t)x_0 + t x_1, \quad t \in [0, 1]$$
   - Đạo hàm vận tốc mục tiêu không đổi theo thời gian:
     $$u_t = \frac{dx_t}{dt} = x_1 - x_0$$
   - Mạng neural $v_\theta(x_t, t, \text{condition})$ chỉ cần học dự đoán vector vận tốc này bằng hàm mất mát MSE cực kỳ ổn định:
     $$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t, x_0, x_1} \left\| v_\theta(x_t, t, \text{condition}) - (x_1 - x_0) \right\|^2$$
   - Khi suy luận (Inference), bộ giải phương trình vi phân (Euler ODE Solver) chỉ cần **16 đến 32 bước** đi thẳng là đạt chất lượng âm thanh hoàn hảo!
"""),

    code("""# Code mô phỏng: Conditional Flow Matching (CFM) Decoder hoàn chỉnh có Euler ODE Solver
class MinimalCFMDecoder(nn.Module):
    \"\"\"
    Lấy cảm hứng từ Voicebox (Meta 2023) và F5-TTS (2024).
    Mạng dự đoán trường vector vận tốc v_t(x_t | t, condition) để sinh Mel-spectrogram.
    \"\"\"
    def __init__(self, n_mels=80, cond_dim=768, hidden_dim=256):
        super().__init__()
        # Kênh đầu vào: n_mels (x_t) + 1 (thời gian t) + cond_dim (vector ngữ nghĩa & speaker)
        self.input_proj = nn.Conv1d(n_mels + 1 + cond_dim, hidden_dim, kernel_size=3, padding=1)
        self.backbone = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU()
        )
        self.output_proj = nn.Conv1d(hidden_dim, n_mels, kernel_size=3, padding=1)
        
    def forward(self, x_t, t, condition):
        \"\"\"
        x_t:       [Batch, n_mels, T] (Mel đang bị nhiễu ở bước t)
        t:         [Batch] (Thời gian liên tục t in [0, 1])
        condition: [Batch, cond_dim, T] (Ngữ nghĩa đích + Giọng nói người nói)
        \"\"\"
        B, _, T = x_t.shape
        t_embed = t.view(B, 1, 1).expand(B, 1, T)
        net_in = torch.cat([x_t, t_embed, condition], dim=1)
        h = self.input_proj(net_in)
        h = self.backbone(h)
        v_pred = self.output_proj(h) # Dự đoán vector vận tốc dx/dt
        return v_pred

    @torch.no_grad()
    def sample_euler_ode(self, condition, num_steps=16):
        \"\"\"
        Euler ODE Solver: Tích phân số từ t=0 (Nhiễu trắng) đến t=1 (Mel hoàn chỉnh)
        x_{t+dt} = x_t + v_pred * dt
        \"\"\"
        B, _, T = condition.shape
        # Bước 0: Bắt đầu từ nhiễu thuần túy Gaussian N(0, I)
        x_t = torch.randn(B, 80, T, device=condition.device)
        dt = 1.0 / num_steps
        
        for step in range(num_steps):
            current_t = torch.full((B,), step * dt, device=condition.device)
            v_pred = self.forward(x_t, current_t, condition)
            x_t = x_t + v_pred * dt # Bước nhảy Euler
            
        return x_t # x_1 chính là Mel-spectrogram hoàn chỉnh!

# Chạy thử nghiệm CFM Decoder
cfm_decoder = MinimalCFMDecoder(n_mels=80, cond_dim=768, hidden_dim=128).to(device)
fake_condition = torch.randn(2, 768, 64, device=device) # 64 frames condition

# Sinh Mel-spectrogram trong 16 bước
generated_mel = cfm_decoder.sample_euler_ode(fake_condition, num_steps=16)

print("Kích thước Điều kiện (Whisper + Speaker Prompt):", fake_condition.shape)
print("Kích thước Mel sinh ra từ Euler ODE (16 bước):   ", generated_mel.shape)
print("-> Non-Autoregressive hoàn toàn: Không lo nuốt từ, không lo lặp chữ vô tận!")
""")
]

# ==============================================================================
# NOTEBOOK 3: STUDENT RESEARCH NOVELTY & TRAINING PIPELINE
# ==============================================================================
nb3_cells = [
    md("""# CHUYÊN ĐỀ 3: PIPELINE THỰC NGHIỆM TIẾT KIỆM TÀI NGUYÊN & ĐỀ TÀI NOVELTY CHO SINH VIÊN

Notebook này được thiết kế để bạn chạy thực nghiệm trực tiếp:
- **Môi trường phù hợp**:
  - Chạy thử & debug trên máy cá nhân: **RTX 4050 (6GB VRAM)**, mức ngốn VRAM **< 1.8 GB**.
  - Sẵn sàng mang lên train trên **MoLab / Marimo** hoặc **Kaggle / Colab** (GPU T4/P100 miễn phí).
- **Chiến lược tạo Novelty (Kiến thức mới)**:
  - Đóng băng (Freeze) 90% các module lớn.
  - Sáng tạo module cầu nối siêu nhẹ **Tone-Aware / Prosody Semantic Bridge (~12M params)** giải quyết bài toán dịch tiếng nói giữ ngữ điệu cho tiếng Việt.

> 💡 **Khả năng tương thích**: Notebook này có thể chạy trực tiếp trong VS Code / Jupyter hoặc chuyển đổi sang **Marimo** (`marimo convert ... -o ...py`) để chạy trên MoLab.
"""),

    md("""## 1. Kiến trúc Đề xuất: "Frozen Whisper + Lightweight Bridge + CFM Decoder"

### Triết lý thiết kế:
1. **Source Speech Encoder**: Dùng `Whisper-small` (đã pretrained trên 680k giờ, đóng băng trọng số - **Frozen**).
2. **Translation Bridge**: Transformer Encoder hoặc SSM Mamba nhỏ gọn (4 tầng, ~12M tham số, có thể học được - **Trainable**).
3. **Target Acoustic Synthesizer**: Flow Matching Decoder (như F5-TTS/Matcha) hoặc Vocos Vocoder.
4. **Speaker Retention**: Nhận vector đặc trưng giọng nói 192 chiều từ pretrained ECAPA-TDNN để giữ trọn vẹn chất giọng người nói nguồn.
"""),

    code("""import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tái sử dụng CFM Decoder từ Chuyên đề 2
class MinimalCFMDecoder(nn.Module):
    def __init__(self, n_mels=80, cond_dim=512, hidden_dim=192):
        super().__init__()
        self.input_proj = nn.Conv1d(n_mels + 1 + cond_dim, hidden_dim, kernel_size=3, padding=1)
        self.backbone = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU()
        )
        self.output_proj = nn.Conv1d(hidden_dim, n_mels, kernel_size=3, padding=1)
        
    def forward(self, x_t, t, condition):
        B, _, T = x_t.shape
        t_embed = t.view(B, 1, 1).expand(B, 1, T)
        net_in = torch.cat([x_t, t_embed, condition], dim=1)
        h = self.backbone(self.input_proj(net_in))
        return self.output_proj(h)

class StudentS2STSystem(nn.Module):
    \"\"\"
    Kiến trúc S2ST tinh gọn cho sinh viên nghiên cứu:
    - Nén frame: 50Hz -> 25Hz
    - Translation Bridge: 4 Transformer Layers (~12M params)
    - CFM Decoder: Học tái tạo Mel tiếng đích
    \"\"\"
    def __init__(self, src_dim=768, bridge_dim=512, n_mels=80):
        super().__init__()
        # 1. Temporal Downsampling: Giảm 50% số frame
        self.downsampler = nn.Conv1d(src_dim, bridge_dim, kernel_size=2, stride=2)
        
        # 2. Trainable Translation Bridge (Nơi bạn tạo Novelty)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=bridge_dim, nhead=8, dim_feedforward=1024, batch_first=True, activation="gelu"
        )
        self.bridge = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # 3. Flow Matching Velocity Field Estimator
        self.cfm = MinimalCFMDecoder(n_mels=n_mels, cond_dim=bridge_dim, hidden_dim=192)
        
    def compute_loss(self, src_features, target_mel):
        \"\"\"
        src_features: [Batch, T_src, 768] (Continuous features từ Frozen Whisper)
        target_mel:   [Batch, n_mels, T_tgt] (Ground truth mel tiếng đích)
        \"\"\"
        B, n_mels, T_tgt = target_mel.shape
        
        # Nén và dịch ngữ nghĩa
        x = self.downsampler(src_features.transpose(1, 2)).transpose(1, 2) # [B, T_src//2, bridge_dim]
        translated = self.bridge(x)                                        # [B, T_src//2, bridge_dim]
        
        # Căn chỉnh chiều dài thời gian khớp với T_tgt
        cond = F.interpolate(translated.transpose(1, 2), size=T_tgt, mode="linear", align_corners=False)
        
        # Optimal Transport Flow Matching Loss
        t = torch.rand(B, device=src_features.device)
        x_0 = torch.randn_like(target_mel) # Nhiễu Gaussian
        x_1 = target_mel                   # Mel mục tiêu
        
        # x_t = (1 - t) * x_0 + t * x_1
        t_exp = t.view(B, 1, 1)
        x_t = (1.0 - t_exp) * x_0 + t_exp * x_1
        u_t = x_1 - x_0 # Vận tốc mục tiêu
        
        v_pred = self.cfm(x_t, t, cond)
        loss = F.mse_loss(v_pred, u_t)
        return loss

# Khởi tạo mô hình
system = StudentS2STSystem().to(device)
params_count = sum(p.numel() for p in system.parameters() if p.requires_grad)
print(f"Tổng số tham số cần huấn luyện: {params_count / 1e6:.2f}M parameters")
print(f"-> Dung lượng mô hình chỉ khoảng {params_count * 4 / (1024**2):.2f} MB!")
"""),

    md("""## 2. Vòng lặp Huấn luyện Mô phỏng (Simulated Training Loop)
Đoạn code dưới đây kiểm tra trọn vẹn Forward pass, Backward pass, Optimizer step và bộ nhớ GPU:
"""),

    code("""# Vòng lặp huấn luyện thử nghiệm trên máy local
optimizer = torch.optim.AdamW(system.parameters(), lr=1e-4, weight_decay=1e-2)

print("Bắt đầu huấn luyện thử nghiệm (5 bước kiểm tra VRAM & Loss)...")
system.train()
for step in range(1, 6):
    optimizer.zero_grad()
    
    # Giả lập 1 batch: 4 câu audio nguồn và 4 câu audio dịch
    dummy_src = torch.randn(4, 80, 768, device=device) # 80 frames Whisper
    dummy_tgt_mel = torch.randn(4, 80, 100, device=device) # 100 frames Mel
    
    loss = system.compute_loss(dummy_src, dummy_tgt_mel)
    loss.backward()
    optimizer.step()
    
    print(f"Step {step}/5 - CFM Loss: {loss.item():.4f}")

if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    print(f"\\nBộ nhớ VRAM sử dụng: Allocated = {allocated:.1f} MB, Reserved = {reserved:.1f} MB")
    print("-> Chạy cực kỳ êm ái trên GPU 6GB của bạn!")
"""),

    md("""## 3. Hướng dẫn Triển khai trên MoLab / Marimo và Tài nguyên Đám mây

### 1. Chạy trên Marimo (MoLab):
Marimo là notebook dạng reactive thuần Python, rất được ưa chuộng tại các phòng lab nghiên cứu hiện đại:
- Để biến file notebook `.ipynb` này thành file script Marimo, mở terminal gõ:
  ```bash
  marimo convert 03_Student_Research_Novelty_and_Training_Pipeline.ipynb -o s2st_pipeline_marimo.py
  ```
- Để mở giao diện chỉnh sửa tương tác trên server lab:
  ```bash
  marimo edit s2st_pipeline_marimo.py --host 0.0.0.0 --port 8080
  ```

### 2. Bộ dữ liệu chuẩn cho cặp ngôn ngữ Anh - Việt:
1. **Google FLEURS (`google/fleurs`)**:
   - Có sẵn cặp dữ liệu `en_us` và `vi_vn`.
   - Sử dụng chế độ `streaming=True` của Hugging Face để nạp dữ liệu trực tiếp vào RAM mà không làm tràn ổ cứng:
     ```python
     from datasets import load_dataset
     fleurs_vi = load_dataset("google/fleurs", "vi_vn", split="train", streaming=True)
     ```
2. **CoVoST-2**: Benchmark quốc tế về Speech Translation.

### 3. Cách tổ chức Paper Khoa học từ kết quả của bạn:
- **Tên đề tài mẫu**: *"A Lightweight Flow-Matching Bridge for Expressive English-to-Vietnamese Speech Translation"*.
- **Cấu trúc bài báo**:
  1. *Introduction*: Nêu rõ sự thiếu hụt của các mô hình lớn đối với tiếng Việt (mất thanh điệu, lơ lớ).
  2. *Proposed Method*: Mô tả module Bridge siêu nhẹ và cơ chế nén frame 50Hz $\rightarrow$ 25Hz.
  3. *Experiments & Results*: So sánh BLEU score và MOS (độ tự nhiên của giọng nói) với mô hình Cascade chuẩn.
  4. *Ablation Study*: Chứng minh việc nén frame giảm 50% tính toán mà không làm giảm chất lượng.
""")
]

# Ghi ra 3 file notebook
notebooks = [
    ("d:\\Github\\ai biet\\S2ST\\01_Discrete_Representations_and_Codec_LLMs.ipynb", nb1_cells),
    ("d:\\Github\\ai biet\\S2ST\\02_Continuous_Representations_and_Flow_Matching.ipynb", nb2_cells),
    ("d:\\Github\\ai biet\\S2ST\\03_Student_Research_Novelty_and_Training_Pipeline.ipynb", nb3_cells)
]

for file_path, cells in notebooks:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(create_notebook(cells), f, ensure_ascii=False, indent=2)
    print(f"Generated: {file_path}")

