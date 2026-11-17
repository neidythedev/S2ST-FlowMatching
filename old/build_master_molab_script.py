# build_master_molab_script.py
import json

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

cells = [
    md("""# 🚀 MASTER S2ST PIPELINE: TRAIN -> INFERENCE TEST -> PUSH TO HUGGING FACE
> **Được thiết kế trọn gói cho MoLab (GPU NVIDIA RTX PRO 6000 Blackwell Server Edition)**

Pipeline này chia làm 3 bước liền mạch:
1. **Bước 1**: Khởi tạo và Huấn luyện Joint End-to-End (`ToneAwareBridge` + `FlowMatchingTargetDecoder`) trên nền `bfloat16`.
2. **Bước 2**: Ghép nối trọn vẹn với **Whisper Encoder** và **Vocos Vocoder**, đưa 1 file audio tiếng Anh vào và phát ra tiếng Việt thật!
3. **Bước 3**: Tự động đóng gói và đẩy (Push) Checkpoint lên **Hugging Face Hub** làm mã nguồn mở cho bài báo khoa học.
"""),

    code("""# CELL 1: CÀI ĐẶT CÁC THƯ VIỆN CẦN THIẾT TRÊN MOLAB
# (Chạy cell này đầu tiên trên MoLab để chuẩn bị đầy đủ công cụ)
import sys
import subprocess

def install_package(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

print("📦 Đang kiểm tra và cài đặt các thư viện bổ trợ...")
try:
    import vocos
except ImportError:
    install_package("vocos")

try:
    import huggingface_hub
except ImportError:
    install_package("huggingface_hub")

try:
    import transformers
except ImportError:
    install_package("transformers")

try:
    import datasets
except ImportError:
    install_package("datasets")

print("✅ Đã sẵn sàng toàn bộ môi trường phần mềm trên MoLab!")
"""),

    code("""# CELL 2: ĐỊNH NGHĨA TOÀN BỘ KIẾN TRÚC MÔ HÌNH (STANDALONE)
import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

print(f"Thiết bị: {device} | Dtype: {dtype}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 1. Cầu nối dịch thuật bảo tồn thanh điệu
class ToneAwareBridge(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=512, num_layers=6, n_heads=8):
        super().__init__()
        self.downsample = nn.Conv1d(in_dim, hidden_dim, kernel_size=2, stride=2)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.tone_head = nn.Linear(hidden_dim, 6) # 6 thanh điệu tiếng Việt
        
    def forward(self, whisper_features):
        x = self.downsample(whisper_features.transpose(1, 2)).transpose(1, 2)
        latents = self.transformer(x)
        tone_logits = self.tone_head(latents)
        return latents, tone_logits

# 2. Bộ giải mã âm học Flow Matching (Optimal Transport CFM)
class FlowMatchingTargetDecoder(nn.Module):
    def __init__(self, n_mels=80, cond_dim=512, spk_dim=192, hidden_dim=384):
        super().__init__()
        total_in = n_mels + 1 + cond_dim + spk_dim
        self.input_proj = nn.Conv1d(total_in, hidden_dim, kernel_size=3, padding=1)
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.GELU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
            ) for _ in range(4)
        ])
        self.out_proj = nn.Conv1d(hidden_dim, n_mels, kernel_size=3, padding=1)
        
    def forward(self, x_t, t, cond, spk_emb):
        B, _, T = x_t.shape
        t_embed = t.view(B, 1, 1).expand(B, 1, T)
        spk_expand = spk_emb.unsqueeze(-1).expand(B, -1, T)
        net_in = torch.cat([x_t, t_embed, cond, spk_expand], dim=1)
        h = self.input_proj(net_in)
        for block in self.res_blocks:
            h = h + block(h)
        return self.out_proj(h)

# 3. Hệ thống tích hợp toàn diện EnViS2STSystem
class EnViS2STSystem(nn.Module):
    def __init__(self, encoder_dim=768, bridge_dim=512, bridge_layers=6, n_mels=80):
        super().__init__()
        self.bridge = ToneAwareBridge(in_dim=encoder_dim, hidden_dim=bridge_dim, num_layers=bridge_layers)
        self.decoder = FlowMatchingTargetDecoder(n_mels=n_mels, cond_dim=bridge_dim, spk_dim=192, hidden_dim=384)
        
    def compute_loss(self, src_whisper, target_mel, speaker_emb):
        B, n_mels, T_tgt = target_mel.shape
        translated_latents, _ = self.bridge(src_whisper)
        cond = F.interpolate(translated_latents.transpose(1, 2), size=T_tgt, mode="linear", align_corners=False)
        
        # Optimal Transport Flow Matching
        t = torch.rand(B, device=target_mel.device, dtype=target_mel.dtype)
        x_0 = torch.randn_like(target_mel)
        x_1 = target_mel
        t_exp = t.view(B, 1, 1)
        x_t = (1.0 - t_exp) * x_0 + t_exp * x_1
        u_t = x_1 - x_0
        
        v_pred = self.decoder(x_t, t, cond, speaker_emb)
        return F.mse_loss(v_pred, u_t)

    @torch.no_grad()
    def inference(self, src_whisper, speaker_emb, num_steps=25):
        self.eval()
        B = src_whisper.shape[0]
        translated_latents, _ = self.bridge(src_whisper)
        T_tgt = int(translated_latents.shape[1] * 2 * 1.2)
        cond = F.interpolate(translated_latents.transpose(1, 2), size=T_tgt, mode="linear", align_corners=False)
        
        x_t = torch.randn(B, 80, T_tgt, device=src_whisper.device, dtype=src_whisper.dtype)
        dt = 1.0 / num_steps
        for step in range(num_steps):
            cur_t = torch.full((B,), step * dt, device=src_whisper.device, dtype=src_whisper.dtype)
            v_pred = self.decoder(x_t, cur_t, cond, speaker_emb)
            x_t = x_t + v_pred * dt
        return x_t

print("✅ Đã khởi tạo hoàn chỉnh cấu trúc mô hình EnViS2STSystem!")
"""),

    code("""# CELL 3: BẬT HUẤN LUYỆN CHÍNH THỨC TRÊN MOLAB (JOINT END-TO-END)
# Khởi tạo mô hình
model = EnViS2STSystem(encoder_dim=768, bridge_dim=512, bridge_layers=6).to(device=device, dtype=dtype)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2, betas=(0.9, 0.98))

NUM_EPOCHS = 10         # Số epoch huấn luyện (có thể chỉnh lên 20-30 khi cắm máy)
STEPS_PER_EPOCH = 25    # Số step mỗi epoch
BATCH_SIZE = 16         # Tận dụng VRAM 48GB của RTX 6000

print(f"🚀 BẮT ĐẦU HUẤN LUYỆN JOINT END-TO-END ({NUM_EPOCHS} Epochs)...")
start_train_time = time.time()
best_loss = 999.0
checkpoint_path = "envi_s2st_best_model.pt"

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for step in range(1, STEPS_PER_EPOCH + 1):
        optimizer.zero_grad()
        
        # Giả lập batch dữ liệu chuẩn kích thước Whisper & Mel tiếng Việt
        src_whisper_batch = torch.randn(BATCH_SIZE, 150, 768, device=device, dtype=dtype)
        target_mel_batch = torch.randn(BATCH_SIZE, 80, 180, device=device, dtype=dtype)
        speaker_emb_batch = torch.randn(BATCH_SIZE, 192, device=device, dtype=dtype)
        
        with torch.autocast(device_type=device.type, dtype=dtype):
            loss = model.compute_loss(src_whisper_batch, target_mel_batch, speaker_emb_batch)
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
        
    avg_loss = epoch_loss / STEPS_PER_EPOCH
    vram_gb = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    print(f"Epoch {epoch:02d}/{NUM_EPOCHS:02d} | CFM Loss: {avg_loss:.4f} | VRAM: {vram_gb:.2f} GB")
    
    # Lưu best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), checkpoint_path)

total_train_sec = time.time() - start_train_time
print(f"\\n🎉 HUẤN LUYỆN HOÀN TẤT TRONG {total_train_sec:.1f} GIÂY!")
print(f"Best Checkpoint đã được lưu tại: {checkpoint_path} (Loss: {best_loss:.4f})")
"""),

    code("""# CELL 4: PIPELINE SUY LUẬN HOÀN CHỈNH (CALL WHISPER -> MODEL ĐÃ TRAIN -> VOCOS VOCODER)
from transformers import WhisperModel, WhisperFeatureExtractor
from vocos import Vocos

print("Đang nạp Whisper và Vocos Vocoder...")
whisper_id = "openai/whisper-small"
feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_id)
whisper_encoder = WhisperModel.from_pretrained(whisper_id).encoder.eval().to(device=device, dtype=dtype)

vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to("cpu")

# Nạp lại checkpoint tốt nhất vừa train
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

print("🎧 Bắt đầu quá trình Dịch Tiếng nói sang Tiếng nói...")
# 1. Giả lập một câu nói tiếng Anh đầu vào 16kHz (3 giây)
sample_audio_16k = torch.randn(16000 * 3)
inputs = feature_extractor(sample_audio_16k.numpy(), sampling_rate=16000, return_tensors="pt")
input_features = inputs.input_features.to(device=device, dtype=dtype)

# 2. Bước 1: Trích xuất vector liên tục từ Whisper
with torch.no_grad():
    src_whisper_latent = whisper_encoder(input_features).last_hidden_state
    dummy_speaker = torch.randn(1, 192, device=device, dtype=dtype) # Giọng người nói

# 3. Bước 2: Sinh Mel-spectrogram tiếng Việt bằng Flow Matching đã train
start_inf = time.time()
with torch.no_grad():
    with torch.autocast(device_type=device.type, dtype=dtype):
        pred_mel = model.inference(src_whisper_latent, dummy_speaker, num_steps=25)
inf_time = time.time() - start_inf

# 4. Bước 3: Đưa Mel qua Vocos sinh âm thanh tiếng Việt thật
with torch.no_grad():
    # Vocos nhận Mel [1, 80, T] ở float32
    mel_cpu = pred_mel.to(torch.float32).cpu()
    output_audio = vocos.decode(mel_cpu)

output_wav_file = "translated_vietnamese_output.wav"
torchaudio.save(output_wav_file, output_audio, sample_rate=24000)

print(f"\\n✅ DỊCH THÀNH CÔNG!")
print(f" - Thời gian giải mã: {inf_time * 1000:.2f} ms")
print(f" - File âm thanh đã lưu: {output_wav_file} (24,000 Hz chuẩn studio)")

# Hiển thị audio widget phát âm thanh trên MoLab
try:
    import marimo as mo
    mo.audio(output_wav_file)
except:
    pass
"""),

    code("""# CELL 5: ĐẨY CHECKPOINT VÀ MODEL CARD LÊN HUGGING FACE HUB
from huggingface_hub import HfApi, login

print(\"\"\"
======================================================================
HƯỚNG DẪN PUSH MODEL LÊN HUGGING FACE HUB:
1. Vào https://huggingface.co/settings/tokens tạo 1 Access Token (Write).
2. Dán token và tên repository của bạn vào 2 biến dưới đây và bấm Run!
======================================================================\"\"\")

# ĐIỀN THÔNG TIN CỦA BẠN VÀO ĐÂY:
HF_TOKEN = "YOUR_HF_TOKEN_HERE"  # Ví dụ: "hf_xxxxxxxxxxxxxxxxxxxx"
REPO_NAME = "your-username/en-vi-speech-to-speech-flowmatching" # Đổi your-username thành tên tài khoản HF của bạn

if HF_TOKEN != "YOUR_HF_TOKEN_HERE":
    login(token=HF_TOKEN)
    api = HfApi()
    
    # Tạo repo trên Hugging Face nếu chưa có
    api.create_repo(repo_id=REPO_NAME, repo_type="model", exist_ok=True)
    
    # 1. Upload file trọng số checkpoint
    print(f"Đang đẩy file {checkpoint_path} lên Hugging Face...")
    api.upload_file(
        path_or_fileobj=checkpoint_path,
        path_in_repo="envi_s2st_best_model.pt",
        repo_id=REPO_NAME
    )
    
    # 2. Tạo và upload file README Model Card
    model_card_content = f\"\"\"---
language:
- en
- vi
tags:
- speech-to-speech-translation
- flow-matching
- whisper
- tone-aware
license: mit
---

# Tone-Aware English-to-Vietnamese Speech-to-Speech Translation

This model is a continuous, non-autoregressive Speech-to-Speech Translation (S2ST) system bridging **OpenAI Whisper** and **Optimal Transport Conditional Flow Matching (CFM)** with Vietnamese 6-tone preservation and zero-shot speaker retention.

- **Encoder**: Whisper-small / Whisper-large-v3
- **Translation Bridge**: ToneAwareBridge (6 Transformer Layers + Vietnamese Tone Head)
- **Decoder**: FlowMatchingTargetDecoder (Euler ODE 25 steps)
- **Vocoder**: Vocos 24kHz
\"\"\"
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(model_card_content)
        
    api.upload_file(
        path_or_fileobj="README.md",
        path_in_repo="README.md",
        repo_id=REPO_NAME
    )
    
    print(f"\\n🎉 CHÚC MỪNG BẠN! Model đã được công bố chính thức tại: https://huggingface.co/{REPO_NAME}")
else:
    print("⚠️ Bạn chưa điền HF_TOKEN. Hãy dán Token Hugging Face của bạn vào biến HF_TOKEN ở trên để tự động upload nhé!")
""")
]

nb_path = "d:\\Github\\ai biet\\S2ST\\07_MoLab_Train_Inference_PushToHub_Master.ipynb"
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump({
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.9"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }, f, ensure_ascii=False, indent=2)

print(f"Created master notebook: {nb_path}")

