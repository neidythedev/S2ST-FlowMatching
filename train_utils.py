#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# TIỆN ÍCH DÙNG CHUNG CHO HUẤN LUYỆN STAGE 1 (train_cfm.py) VÀ STAGE 2 (train_bridge.py)
#   - Độ dài thật của từng câu trong batch đã đệm (latent Whisper, frame Mel)
#   - Mask + loss bỏ qua phần đệm (padding)
#   - Trích đặc trưng Whisper encoder cho batch
#   - ASR tiếng Anh + WER để đo chất lượng tái tạo
#   - Đồng bộ checkpoint lên Google Drive (không chặn tiến trình train)
# File này KHÔNG import torchaudio nên không cần Mock Shim.
# ==============================================================================
import os
import re
import shutil
import subprocess

import torch
import torch.nn.functional as F

WHISPER_SAMPLES_PER_FRAME = 320   # Whisper encoder: 16kHz, 50 frame/giây
MEL_HOP = 256                      # Mel 24kHz của Vocos: hop 256, center=True


# ------------------------------------------------------------------------------
# Độ dài & mask
# ------------------------------------------------------------------------------
def latent_lengths(n_samples_16k):
    """Số frame latent Whisper ứng với từng câu (n mẫu ở 16kHz)."""
    return [max(1, (int(n) + WHISPER_SAMPLES_PER_FRAME - 1) // WHISPER_SAMPLES_PER_FRAME) for n in n_samples_16k]


def mel_lengths(n_samples_16k):
    """Số frame Mel 24kHz (hop 256, center=True) ứng với từng câu (n mẫu ở 16kHz)."""
    out = []
    for n in n_samples_16k:
        n24 = int(round(int(n) * 24000.0 / 16000.0))
        out.append(n24 // MEL_HOP + 1)
    return out


def length_mask(lengths, max_len, device):
    """Mask [B, max_len]: 1 ở frame thật, 0 ở phần đệm."""
    lens = torch.as_tensor(lengths, device=device).view(-1, 1)
    ar = torch.arange(max_len, device=device).view(1, -1)
    return (ar < lens).float()


def masked_mse(pred, target, mask):
    """MSE chỉ tính trên frame thật. pred/target: [B, C, T]; mask: [B, T]."""
    if mask is None:
        return F.mse_loss(pred, target)
    m = mask.to(pred.dtype).unsqueeze(1)
    se = (pred - target).pow(2) * m
    return se.sum() / (m.sum() * pred.shape[1]).clamp(min=1.0)


def masked_l1(pred, target, mask):
    """L1 chỉ tính trên frame thật. pred/target: [B, C, T]; mask: [B, T]."""
    if mask is None:
        return F.l1_loss(pred, target)
    m = mask.to(pred.dtype).unsqueeze(1)
    ae = (pred - target).abs() * m
    return ae.sum() / (m.sum() * pred.shape[1]).clamp(min=1.0)


# ------------------------------------------------------------------------------
# CFM DECODER DÙNG CHUNG CHO STAGE 1 + STAGE 2 (một định nghĩa duy nhất -> không thể lệch kiến trúc)
# ------------------------------------------------------------------------------
import torch.nn as nn


class MaskedGroupNorm(nn.GroupNorm):
    """GroupNorm chỉ tính mean/var trên frame thật (bỏ phần đệm) -> train (batch có đệm) và
    suy luận (1 câu, không đệm) cùng một phân phối. mask=None -> GroupNorm thường."""
    def forward(self, x, mask=None):
        if mask is None:
            return super().forward(x)
        B, C, T = x.shape
        g = self.num_groups
        xg = x.view(B, g, C // g, T)
        m = mask.to(x.dtype).view(B, 1, 1, T)
        n = (m.sum(dim=(2, 3), keepdim=True) * (C // g)).clamp(min=1.0)
        mean = (xg * m).sum(dim=(2, 3), keepdim=True) / n
        var = (((xg - mean) ** 2) * m).sum(dim=(2, 3), keepdim=True) / n
        xn = ((xg - mean) / torch.sqrt(var + self.eps)).view(B, C, T)
        if self.affine:
            xn = xn * self.weight.view(1, C, 1) + self.bias.view(1, C, 1)
        return xn


class ResidualBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm = MaskedGroupNorm(8, channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x, mask=None):
        res = x
        x = self.conv1(x)
        x = self.norm(x, mask)
        x = self.act(x)
        if mask is not None:
            # Đưa phần đệm về 0 trước conv2 -> frame biên không "nhìn thấy" phần đệm (giống hệt chạy 1 câu)
            x = x * mask.unsqueeze(1).to(x.dtype)
        x = self.conv2(x)
        out = x + res
        return out * mask.unsqueeze(1).to(out.dtype) if mask is not None else out


class FlowMatchingTargetDecoder(nn.Module):
    """OT-CFM decoder: v_theta(x_t, t, c, s) -> vận tốc trên 100-Mel.
    Tên tham số: in_proj / res_blocks.*.conv1|norm|conv2 / out_proj (Stage 1 và Stage 2 dùng chung)."""
    def __init__(self, n_mels=100, cond_dim=512, spk_dim=192, hidden_dim=384, num_res_blocks=4):
        super().__init__()
        self.n_mels = n_mels
        self.cond_dim = cond_dim
        self.spk_dim = spk_dim
        self.hidden_dim = hidden_dim
        total_in_channels = n_mels + 1 + cond_dim + spk_dim
        self.in_proj = nn.Conv1d(total_in_channels, hidden_dim, kernel_size=3, padding=1)
        self.res_blocks = nn.ModuleList([ResidualBlock1D(hidden_dim) for _ in range(num_res_blocks)])
        self.out_proj = nn.Conv1d(hidden_dim, n_mels, kernel_size=3, padding=1)

    def forward(self, x_t, t, cond, spk_emb, mask=None):
        """mask [B, T] (1 = frame thật): phần đệm bị đưa về 0 sau mỗi lớp và không vào thống kê GroupNorm."""
        B, C, T = x_t.shape
        t_embed = t.view(B, 1, 1).expand(B, 1, T).to(x_t.dtype)
        spk_expand = spk_emb.unsqueeze(-1).expand(B, -1, T)
        net_in = torch.cat([x_t, t_embed, cond, spk_expand], dim=1)
        if mask is not None:
            net_in = net_in * mask.unsqueeze(1).to(net_in.dtype)
        h = self.in_proj(net_in)
        if mask is not None:
            h = h * mask.unsqueeze(1).to(h.dtype)
        for block in self.res_blocks:
            h = block(h, mask)
        return self.out_proj(h)


# ------------------------------------------------------------------------------
# Whisper encoder
# ------------------------------------------------------------------------------
def encode_whisper(audio_16k_padded, feat_extractor, whisper_enc, device, dtype):
    """Trích latent Whisper cho batch audio 16kHz đã đệm [B, N]. Trả về [B, T_lat, 768],
    T_lat = độ dài theo câu dài nhất trong batch (giống cách cắt cũ của dự án)."""
    in_p = feat_extractor(list(audio_16k_padded.cpu().numpy()), sampling_rate=16000,
                          return_tensors="pt", padding="max_length")
    f = in_p.input_features.to(device=device, dtype=dtype)
    if f.shape[-1] < 3000:
        f = F.pad(f, (0, 3000 - f.shape[-1]))
    elif f.shape[-1] > 3000:
        f = f[:, :, :3000]
    with torch.no_grad():
        fl = whisper_enc(f).last_hidden_state
    real_len = max(10, min(1500, (audio_16k_padded.shape[-1] + WHISPER_SAMPLES_PER_FRAME - 1) // WHISPER_SAMPLES_PER_FRAME))
    return fl[:, :real_len, :]


# ------------------------------------------------------------------------------
# ASR tiếng Anh + WER (đo chất lượng tái tạo / dịch)
# ------------------------------------------------------------------------------
def normalize_en(text):
    t = str(text or "").lower().replace("’", "'")
    t = re.sub(r"[^\w\s']", " ", t)
    t = re.sub(r"(?<!\w)'|'(?!\w)", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def word_errors(hyp, ref):
    """(số lỗi edit distance theo từ, số từ tham chiếu)."""
    h, r = normalize_en(hyp).split(), normalize_en(ref).split()
    prev = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        cur = [i] + [0] * len(h)
        for j in range(1, len(h) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
        prev = cur
    return prev[-1], len(r)


def corpus_wer(hyps, refs):
    err = tot = 0
    for h, r in zip(hyps, refs):
        e, n = word_errors(h, r)
        err += e
        tot += n
    return 100.0 * err / max(1, tot)


class EnglishASR:
    """Whisper tiếng Anh (mặc định openai/whisper-small) — nạp lười, dùng để đo WER."""
    def __init__(self, device, model_id="openai/whisper-small"):
        self.device, self.model_id = device, model_id
        self.feat = self.tok = self.mod = None

    def _load(self):
        if self.mod is None:
            from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperForConditionalGeneration
            self.feat = WhisperFeatureExtractor.from_pretrained(self.model_id)
            self.tok = WhisperTokenizer.from_pretrained(self.model_id, language="en", task="transcribe")
            self.mod = WhisperForConditionalGeneration.from_pretrained(self.model_id).to(self.device).eval()
            if getattr(self.mod, "generation_config", None) is not None:
                self.mod.generation_config.max_length = None

    @torch.no_grad()
    def transcribe(self, wav_16k):
        """wav_16k: tensor 1D (16kHz). Trả về chuỗi tiếng Anh."""
        self._load()
        w = torch.as_tensor(wav_16k).float().reshape(-1).cpu().numpy()
        inp = self.feat(w, sampling_rate=16000, return_tensors="pt")
        ids = self.mod.generate(inp.input_features.to(self.device), language="en", task="transcribe",
                                max_new_tokens=128)
        return self.tok.batch_decode(ids, skip_special_tokens=True)[0].strip()


def resample_24k_to_16k_fft(w24):
    """Resample FFT 24k -> 16k cho 1 waveform 1D (dùng trước khi đưa vào Whisper)."""
    w = torch.as_tensor(w24).float().reshape(-1)
    n = w.shape[-1]
    m = int(round(n * 16000 / 24000))
    X = torch.fft.rfft(w)[: m // 2 + 1]
    return torch.fft.irfft(X, n=m) * (m / n)


# ------------------------------------------------------------------------------
# Đồng bộ Google Drive (không chặn)
# ------------------------------------------------------------------------------
def sync_to_drive(files, exp_name, enabled=True):
    """Đẩy các file checkpoint lên gdrive:runs/<exp>/checkpoints/ ở tiến trình nền.
    Bỏ qua im lặng nếu máy không có rclone; in cảnh báo nếu không khởi chạy được."""
    if not enabled or shutil.which("rclone") is None:
        return
    dest = f"gdrive:runs/{exp_name}/checkpoints/"
    for fpath in files:
        if fpath and os.path.exists(fpath):
            try:
                subprocess.Popen(["rclone", "copyto", fpath, dest + os.path.basename(fpath)],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                print(f"[Drive] Không đồng bộ được {fpath}: {e}")
