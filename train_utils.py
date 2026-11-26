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
import math
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
    pred, target = pred.float(), target.float()          # loss luôn tính ở fp32 (ổn định khi autocast bf16)
    if mask is None:
        return F.mse_loss(pred, target)
    m = mask.to(pred.dtype).unsqueeze(1)
    se = (pred - target).pow(2) * m
    return se.sum() / (m.sum() * pred.shape[1]).clamp(min=1.0)


def masked_l1(pred, target, mask):
    """L1 chỉ tính trên frame thật. pred/target: [B, C, T]; mask: [B, T]."""
    pred, target = pred.float(), target.float()
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


class CFMDecoderV1(nn.Module):
    """Decoder v1 (cũ, giữ lại để SO SÁNH): t chỉ là 1 kênh hằng số ở lớp đầu, conv k=3 không dilation
    (tầm nhìn ~±10 frame ≈ 0.1s), 4 khối, ~4.6M tham số."""
    def __init__(self, n_mels=100, cond_dim=512, spk_dim=192, hidden_dim=384, num_res_blocks=4, **_):
        super().__init__()
        self.n_mels = n_mels
        self.cond_dim = cond_dim
        self.spk_dim = spk_dim
        self.hidden_dim = hidden_dim
        self.config = {"arch": "v1", "n_mels": n_mels, "cond_dim": cond_dim, "spk_dim": spk_dim,
                       "hidden_dim": hidden_dim, "num_res_blocks": num_res_blocks}
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


class SinusoidalTimeEmbedding(nn.Module):
    """t ∈ [0,1] -> vector sin/cos nhiều tần số (chuẩn của diffusion / flow-matching)."""
    def __init__(self, dim=256):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(1, half - 1))
        ang = (t.float() * 1000.0)[:, None] * freqs[None, :]
        return torch.cat([ang.sin(), ang.cos()], dim=-1)


class FiLMResBlock1D(nn.Module):
    """Khối residual v2:
       h += Conv1x1(c)                         (điều kiện nội dung c đưa vào TỪNG khối)
       x = DilatedConv(h) -> MaskedGroupNorm -> FiLM(t, speaker) -> GELU -> Conv -> + h
    FiLM: x * (1 + scale) + shift, scale/shift sinh từ embedding toàn cục (t + speaker)."""
    def __init__(self, channels, cond_dim, glob_dim, dilation=1):
        super().__init__()
        self.cond_in = nn.Conv1d(cond_dim, channels, kernel_size=1)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm = MaskedGroupNorm(8, channels)
        self.film = nn.Linear(glob_dim, 2 * channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        # Khởi tạo 0 cho nhánh residual -> mỗi khối ban đầu là ánh xạ đồng nhất (train ổn định hơn)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, h, cond, glob, mask=None):
        m = mask.unsqueeze(1).to(h.dtype) if mask is not None else None
        x = h + self.cond_in(cond)
        if m is not None:
            x = x * m
        x = self.conv1(x)
        x = self.norm(x, mask)
        scale, shift = self.film(glob).unsqueeze(-1).chunk(2, dim=1)
        x = x * (1.0 + scale) + shift
        x = self.act(x)
        if m is not None:
            x = x * m
        x = self.conv2(x)
        out = h + x
        return out * m if m is not None else out


class CFMDecoderV2(nn.Module):
    """Decoder v2 (khuyến nghị):
    - t: sinusoidal -> MLP; speaker: Linear; cộng lại thành embedding toàn cục đưa vào MỌI khối qua FiLM
    - c (nội dung): nối ở đầu vào + Conv1x1 vào từng khối
    - Conv giãn (dilation 1,3,9,27 lặp lại) -> tầm nhìn ~±90 frame ≈ ±1 giây
    - 8 khối, hidden 512 (~20M tham số)"""
    def __init__(self, n_mels=100, cond_dim=512, spk_dim=192, hidden_dim=512, num_res_blocks=8,
                 dilations=(1, 3, 9, 27), time_dim=256, **_):
        super().__init__()
        self.n_mels = n_mels
        self.cond_dim = cond_dim
        self.spk_dim = spk_dim
        self.hidden_dim = hidden_dim
        dilations = list(dilations)
        self.config = {"arch": "v2", "n_mels": n_mels, "cond_dim": cond_dim, "spk_dim": spk_dim,
                       "hidden_dim": hidden_dim, "num_res_blocks": num_res_blocks,
                       "dilations": dilations, "time_dim": time_dim}
        glob_dim = hidden_dim
        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, glob_dim), nn.SiLU(), nn.Linear(glob_dim, glob_dim))
        self.spk_mlp = nn.Linear(spk_dim, glob_dim)
        self.in_proj = nn.Conv1d(n_mels + cond_dim, hidden_dim, kernel_size=3, padding=1)
        self.res_blocks = nn.ModuleList([
            FiLMResBlock1D(hidden_dim, cond_dim, glob_dim, dilation=dilations[i % len(dilations)])
            for i in range(num_res_blocks)
        ])
        self.out_norm = MaskedGroupNorm(8, hidden_dim)
        self.out_proj = nn.Conv1d(hidden_dim, n_mels, kernel_size=3, padding=1)

    def forward(self, x_t, t, cond, spk_emb, mask=None):
        m = mask.unsqueeze(1).to(x_t.dtype) if mask is not None else None
        glob = self.time_mlp(self.time_emb(t).to(x_t.dtype)) + self.spk_mlp(spk_emb.to(x_t.dtype))
        net_in = torch.cat([x_t, cond], dim=1)
        if m is not None:
            net_in = net_in * m
        h = self.in_proj(net_in)
        if m is not None:
            h = h * m
        for block in self.res_blocks:
            h = block(h, cond, glob, mask)
        h = F.gelu(self.out_norm(h, mask))
        if m is not None:
            h = h * m
        return self.out_proj(h)


DECODER_PRESETS = {
    "v1": {"arch": "v1", "hidden_dim": 384, "num_res_blocks": 4},
    "v2": {"arch": "v2", "hidden_dim": 512, "num_res_blocks": 8, "dilations": [1, 3, 9, 27], "time_dim": 256},
    "v2-small": {"arch": "v2", "hidden_dim": 384, "num_res_blocks": 4, "dilations": [1, 3, 9, 27], "time_dim": 256},
}
DEFAULT_DECODER = "v2"


def FlowMatchingTargetDecoder(n_mels=100, cond_dim=512, spk_dim=192, arch=None, preset=None, **kwargs):
    """Tạo decoder CFM. Dùng chung cho Stage 1, Stage 2 và eval (một định nghĩa duy nhất).
    - preset: "v1" | "v2" | "v2-small" (mặc định v2)
    - hoặc truyền thẳng cấu hình đã lưu trong checkpoint: FlowMatchingTargetDecoder(**decoder.config)"""
    cfg = dict(DECODER_PRESETS[preset or DEFAULT_DECODER]) if (preset or not arch) else {}
    cfg.update({k: v for k, v in kwargs.items() if v is not None})
    if arch:
        cfg["arch"] = arch
    a = cfg.pop("arch", "v2")
    cls = CFMDecoderV1 if a == "v1" else CFMDecoderV2
    return cls(n_mels=n_mels, cond_dim=cond_dim, spk_dim=spk_dim, **cfg)


def sample_t(batch, device, dtype, mode="uniform"):
    """Lấy mẫu thời điểm t khi train: 'uniform' hoặc 'logit_normal' (dồn về giữa, thường giúp chi tiết tốt hơn)."""
    if mode == "logit_normal":
        return torch.sigmoid(torch.randn(batch, device=device)).to(dtype)
    return torch.rand(batch, device=device, dtype=dtype)


def masked_mel_l1(pred_mel, ref_mel, mask):
    """Mel L1 giữa Mel sinh ra và Mel thật, chỉ trên frame thật (đo độ sát phổ âm)."""
    return masked_l1(pred_mel.float(), ref_mel.float(), mask).item()


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


# ---- CTC phụ trên văn bản tiếng Anh (chỉ dùng làm loss lúc train; suy luận KHÔNG cần text) ----
EN_CTC_CHARS = " abcdefghijklmnopqrstuvwxyz'0123456789"   # id 0 = blank, ký tự thứ k -> id k+1
EN_CTC_VOCAB = len(EN_CTC_CHARS) + 1
_EN_CTC_IDX = {c: i + 1 for i, c in enumerate(EN_CTC_CHARS)}


def en_text_to_ids(text):
    """Văn bản Anh -> list id ký tự (chuẩn hóa như ASR-BLEU; ký tự ngoài bảng bị bỏ)."""
    return [_EN_CTC_IDX[c] for c in normalize_en(text) if c in _EN_CTC_IDX]


def ctc_greedy_decode(ids):
    """Giải mã tham lam CTC: gộp ký tự lặp liên tiếp, bỏ blank -> chuỗi."""
    out, prev = [], 0
    for i in ids:
        i = int(i)
        if i != prev and i != 0:
            out.append(EN_CTC_CHARS[i - 1])
        prev = i
    return re.sub(r"\s+", " ", "".join(out)).strip()


def char_error_rate(hyps, refs):
    """CER gộp trên toàn tập (edit distance theo ký tự / tổng ký tự tham chiếu), sau normalize_en."""
    errs = tot = 0
    for h, r in zip(hyps, refs):
        h, r = normalize_en(h), normalize_en(r)
        prev = list(range(len(h) + 1))
        for i in range(1, len(r) + 1):
            cur = [i] + [0] * len(h)
            for j in range(1, len(h) + 1):
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
            prev = cur
        errs += prev[-1]
        tot += len(r)
    return errs / max(1, tot)


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
            try:   # tắt cảnh báo lặp "max_new_tokens ... max_length" làm rối log
                from transformers.utils import logging as hf_logging
                hf_logging.set_verbosity_error()
            except Exception:
                pass
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
# Nguồn dữ liệu audio cục bộ (manifest jsonl, VD PhoST) + trộn nhiều nguồn
# ------------------------------------------------------------------------------
class ManifestAudioPipeline:
    """Đọc audio 16kHz từ manifest jsonl (mỗi dòng có trường `audio_key`, đường dẫn tương đối so với
    thư mục chứa manifest). Xáo trộn lại mỗi lượt (seed + số lượt). Giao diện giống pipeline FLEURS:
    fetch_batch(batch_size) -> (audio_đệm [B, N], transcripts) và thuộc tính last_lengths."""
    def __init__(self, manifest, audio_key="en_wav", text_key="en_text", max_sec=12.0, shuffle_seed=0, name=None):
        import json
        self.root = os.path.dirname(os.path.abspath(manifest))
        with open(manifest, "r", encoding="utf-8") as f:
            self.rows = [json.loads(l) for l in f if l.strip()]
        if not self.rows:
            raise RuntimeError(f"Manifest rỗng: {manifest}")
        self.audio_key, self.text_key = audio_key, text_key
        self.max_samples = int(16000 * max_sec) if max_sec else None
        self.seed, self.pass_idx, self.pos = shuffle_seed, 0, 0
        self.name = name or os.path.basename(manifest)
        self.last_lengths = []
        self._shuffle()
        hours = sum(float(r.get("en_dur", 0)) for r in self.rows) / 3600.0
        print(f"✓ Manifest {self.name}: {len(self.rows)} câu, ~{hours:.1f} giờ")

    def _shuffle(self):
        import random
        self.order = list(range(len(self.rows)))
        random.Random(self.seed * 1000 + self.pass_idx).shuffle(self.order)
        self.pos = 0

    def next_item(self):
        import soundfile as sf
        for _ in range(50):
            if self.pos >= len(self.order):
                self.pass_idx += 1
                self._shuffle()
                print(f"   🔁 [{self.name}] Đã đọc hết -> bắt đầu lượt thứ {self.pass_idx + 1}")
            r = self.rows[self.order[self.pos]]
            self.pos += 1
            try:
                w, sr = sf.read(os.path.join(self.root, r[self.audio_key]), dtype="float32", always_2d=True)
            except Exception:
                continue           # file lỗi -> bỏ qua, lấy câu kế tiếp
            w = torch.from_numpy(w.mean(axis=1))
            if sr != 16000:
                raise RuntimeError(f"{r[self.audio_key]} có sample rate {sr}, cần 16000")
            if self.max_samples:
                w = w[: self.max_samples]
            if w.numel() >= 1600:
                return w, r.get(self.text_key, "")
        raise RuntimeError(f"Quá nhiều file lỗi liên tiếp trong {self.name}")

    def fetch_batch(self, batch_size=4):
        items = [self.next_item() for _ in range(batch_size)]
        return pad_batch(items, self)


def pad_batch(items, owner):
    """[(wave 1D, text)] -> (tensor đệm [B, N], [text]); ghi owner.last_lengths."""
    owner.last_lengths = [int(w.shape[0]) for w, _ in items]
    n = max(owner.last_lengths)
    return torch.stack([F.pad(w, (0, n - w.shape[0])) for w, _ in items]), [t for _, t in items]


class MixedAudioPipeline:
    """Trộn nhiều nguồn theo tỉ lệ: mỗi câu trong batch được lấy từ nguồn i với xác suất weights[i]
    (seed cố định -> tái lập được). Nguồn cần có next_item() hoặc fetch_batch(1)."""
    def __init__(self, sources, weights, seed=0):
        import random
        total = float(sum(weights))
        self.sources, self.weights = sources, [w / total for w in weights]
        self.rng = random.Random(seed)
        self.last_lengths = []
        self.counts = [0] * len(sources)

    def _one(self, src):
        if hasattr(src, "next_item"):
            return src.next_item()
        wav, txt = src.fetch_batch(batch_size=1)
        return wav[0, : src.last_lengths[0]], txt[0]

    def fetch_batch(self, batch_size=4):
        items = []
        for _ in range(batch_size):
            i = self.rng.choices(range(len(self.sources)), weights=self.weights)[0]
            self.counts[i] += 1
            items.append(self._one(self.sources[i]))
        return pad_batch(items, self)


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
