#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# HUẤN LUYỆN GIAI ĐOẠN 1: OPTIMAL TRANSPORT FLOW MATCHING (OT-CFM) ACOUSTIC DECODER
# Hỗ trợ: Huấn luyện tiếng Anh bản ngữ (en_us) làm Acoustic Teacher cho S2ST Vi -> En
# ==============================================================================

import os
import sys
import time
import math
import argparse
import json
import types
import importlib.machinery
import wave

# ==============================================================================
# RUNTIME RESILIENCE: TORCHAUDIO IN-MEMORY MOCK (PYTHON 3.13 ABI GUARD)
# ==============================================================================
def apply_torchaudio_mock_if_needed():
    """Bypass hoàn toàn lỗi C++ ABI undefined symbol của torchaudio trên Python 3.13."""
    try:
        import torchaudio
        _ = torchaudio.__version__
        return "native"
    except Exception:
        for mod in list(sys.modules.keys()):
            if mod.startswith("torchaudio"):
                del sys.modules[mod]

        def _create_mock(name: str, is_pkg: bool = False):
            mod = types.ModuleType(name)
            if is_pkg: mod.__path__ = []
            spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=is_pkg)
            spec.has_location = False
            mod.__spec__ = spec
            return mod

        ta = _create_mock("torchaudio", is_pkg=True)
        ta.__version__ = "2.5.0-shim"
        ta_func = _create_mock("torchaudio.functional", is_pkg=True)
        ta_func_func = _create_mock("torchaudio.functional.functional", is_pkg=False)
        ta_trans = _create_mock("torchaudio.transforms", is_pkg=False)
        ta_ext = _create_mock("torchaudio._extension", is_pkg=False)

        def _hz_to_mel(freq: float, mel_scale: str = "htk") -> float:
            return 2595.0 * math.log10(1.0 + (freq / 700.0))

        def _mel_to_hz(mel: float, mel_scale: str = "htk") -> float:
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        ta_func_func.hz_to_mel = _hz_to_mel
        ta_func_func.mel_to_hz = _mel_to_hz
        ta_func_func._hz_to_mel = _hz_to_mel
        ta_func_func._mel_to_hz = _mel_to_hz
        ta_func.hz_to_mel = _hz_to_mel
        ta_func.mel_to_hz = _mel_to_hz
        ta_func._hz_to_mel = _hz_to_mel
        ta_func._mel_to_hz = _mel_to_hz
        ta.functional = ta_func
        ta.transforms = ta_trans
        ta._extension = ta_ext

        sys.modules["torchaudio"] = ta
        sys.modules["torchaudio.functional"] = ta_func
        sys.modules["torchaudio.functional.functional"] = ta_func_func
        sys.modules["torchaudio.transforms"] = ta_trans
        sys.modules["torchaudio._extension"] = ta_ext
        return "mock"

_ = apply_torchaudio_mock_if_needed()

import torch
import torch.nn as nn
import torch.nn.functional as F

class MockSpectrogram(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.register_buffer("window", torch.empty(1024))
    def forward(self, x): return x

class MockMelScale(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.register_buffer("fb", torch.empty(513, 100))
    def forward(self, x): return x

class MockMelSpectrogram(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.spectrogram = MockSpectrogram()
        self.mel_scale = MockMelScale()
    def forward(self, x):
        return x

if "torchaudio.transforms" in sys.modules:
    sys.modules["torchaudio.transforms"].MelSpectrogram = MockMelSpectrogram
if "torchaudio" in sys.modules and hasattr(sys.modules["torchaudio"], "transforms"):
    sys.modules["torchaudio"].transforms.MelSpectrogram = MockMelSpectrogram

from train_utils import (
    FlowMatchingTargetDecoder, ResidualBlock1D, DECODER_PRESETS, sample_t, masked_mel_l1,
    encode_whisper, latent_lengths, mel_lengths, length_mask, masked_mse,
    EnglishASR, corpus_wer, resample_24k_to_16k_fft, sync_to_drive,
)

def load_vocos_safely(repo_id="charactr/vocos-mel-24khz", device="cpu"):
    """Nạp Vocos vocoder an toàn với strict=False và loại bỏ feature_extractor để chống lỗi shape mismatch."""
    from vocos import Vocos
    orig_load = Vocos.load_state_dict
    def safe_load(self, state_dict, strict=False):
        clean_sd = {k: v for k, v in state_dict.items() if not k.startswith("feature_extractor")}
        return orig_load(self, clean_sd, strict=False)
    Vocos.load_state_dict = safe_load
    try:
        vocos = Vocos.from_pretrained(repo_id)
    finally:
        Vocos.load_state_dict = orig_load
    return vocos.to(device)

class Logger:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_file = open(log_path, "a", encoding="utf-8", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self):
        return getattr(self.terminal, "isatty", lambda: False)()

    def fileno(self):
        return getattr(self.terminal, "fileno", lambda: 1)()

    def __getattr__(self, name):
        return getattr(self.terminal, name)

def save_wav_file(filepath, waveform, sample_rate):
    """Ghi audio thành file .wav chuẩn PCM 16-bit mà không cần phụ thuộc torchaudio C++."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if hasattr(waveform, "cpu"):
        wav_data = waveform.squeeze().cpu()
    else:
        wav_data = torch.as_tensor(waveform).squeeze().cpu()
    if wav_data.ndim > 1:
        wav_data = wav_data[0]
    wav_np = wav_data.float().numpy()
    max_val = max(1e-6, float(abs(wav_np).max()))
    if max_val > 1.0:
        wav_np = wav_np / max_val
    int_samples = (wav_np * 32767.0).astype("int16")
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int_samples.tobytes())

def load_yaml_config(config_path):
    default_config = {
        "model": {
            "whisper_model": "vinai/PhoWhisper-small",
            "encoder_dim": 768,
            "cond_dim": 512,
            "n_mels": 100,
            "spk_dim": 192,
            "decoder_hidden": 384,
            "num_res_blocks": 4,
            "decoder_preset": "v2",      # v1 (cũ) | v2 (FiLM + dilated) | v2-small
            "t_sampling": "uniform"      # uniform | logit_normal
        },
        "training": {
            "batch_size": 4,
            "learning_rate": 2.0e-4,
            "min_lr": 1.0e-5,
            "weight_decay": 1.0e-4,
            "epochs": 25,
            "steps_per_epoch": 100,
            "val_steps": 10,
            "val_clips": 32,
            "eval_wer_every": 5,
            "eval_wer_clips": 16,
            "max_grad_norm": 1.0,
            "bf16": True,
            "sync_drive": True
        },
        "data": {
            "dataset_name": "google/fleurs",
            "lang": "en_us",
            "sample_rate": 24000,
            "sample_rate_whisper": 16000,
            "max_audio_sec": 12.0,
            "streaming": True
        },
        "paths": {
            "checkpoint_dir": "checkpoints",
            "log_dir": "logs"
        }
    }
    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if loaded:
                    for k, v in loaded.items():
                        if isinstance(v, dict) and k in default_config:
                            default_config[k].update(v)
                        else:
                            default_config[k] = v
        except ImportError:
            pass
    return default_config

# ==============================================================================
# 1. BỘ XỬ LÝ ÂM HỌC THUẦN PYTORCH (100-MEL & RESAMPLER 16K -> 24K)
# ==============================================================================

def build_mel_filterbank(sr=24000, n_fft=1024, n_mels=100, fmin=0.0, fmax=12000.0, device='cpu'):
    """
    Ma trận Mel Filterbank tam giác liên tục chuẩn Torchaudio HTK (Khớp 100% với Vocos pre-trained).
    Loại bỏ hiện tượng rời rạc hóa tần số của hàm floor() để âm thanh trong trẻo, không bị buzz.
    """
    def hz_to_mel(hz): return 2595.0 * math.log10(1.0 + hz / 700.0)
    def mel_to_hz(mel): return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax if fmax is not None else sr / 2.0)
    mel_pts = torch.linspace(mel_min, mel_max, n_mels + 2, device=device)
    f_pts = torch.tensor([mel_to_hz(m.item()) for m in mel_pts], device=device)

    n_freqs = n_fft // 2 + 1
    all_freqs = torch.linspace(0, sr / 2.0, n_freqs, device=device)

    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)

    zero = torch.zeros(1, device=device, dtype=torch.float32)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]
    up_slopes = slopes[:, 2:] / f_diff[1:]
    fb = torch.max(zero, torch.min(down_slopes, up_slopes))
    return fb.T

def extract_100_mel(waveform_24k, mel_filterbank, n_fft=1024, hop_length=256):
    if waveform_24k.dim() == 1:
        waveform_24k = waveform_24k.unsqueeze(0)
    window = torch.hann_window(n_fft, device=waveform_24k.device)
    stft = torch.stft(
        waveform_24k, n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
        window=window, return_complex=True
    )
    magnitude = torch.abs(stft)
    mel_spectrogram = torch.matmul(mel_filterbank, magnitude)
    log_mel = torch.log(torch.clamp(mel_spectrogram, min=1e-5))
    return log_mel

def resample_16k_to_24k(waveform_16k):
    """
    Chuyển đổi tần số lấy mẫu từ 16kHz lên 24kHz bằng phép biến đổi Fourier (FFT Sinc Resampling).
    Loại bỏ hoàn toàn hiện tượng méo âm và nhiễu răng cưa (aliasing) do nội suy tuyến tính gây ra.
    """
    orig_shape = waveform_16k.shape
    if waveform_16k.dim() == 1:
        waveform_16k = waveform_16k.unsqueeze(0)
    elif waveform_16k.dim() > 2:
        waveform_16k = waveform_16k.view(-1, waveform_16k.shape[-1])

    B, T = waveform_16k.shape
    new_T = int(round(T * (24000.0 / 16000.0)))

    X = torch.fft.rfft(waveform_16k.float(), dim=-1)
    target_bins = new_T // 2 + 1
    pad_bins = target_bins - X.shape[-1]

    if pad_bins > 0:
        X_pad = F.pad(X, (0, pad_bins))
    else:
        X_pad = X[:, :target_bins]

    w_24k = torch.fft.irfft(X_pad, n=new_T, dim=-1) * (float(new_T) / float(T))

    if len(orig_shape) == 1:
        return w_24k.squeeze(0)
    elif len(orig_shape) == 3:
        return w_24k.view(orig_shape[0], orig_shape[1], new_T)
    return w_24k

# ==============================================================================
# 2. KIẾN TRÚC MÔ HÌNH: OT-CFM TARGET DECODER
# ==============================================================================

class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# ResidualBlock1D / FlowMatchingTargetDecoder: định nghĩa DUY NHẤT trong train_utils.py (dùng chung với Stage 2)

class AcousticSpeakerEncoder(nn.Module):
    def __init__(self, feat_dim=768, spk_dim=192):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feat_dim * 2, spk_dim),
            nn.GELU(),
            nn.Linear(spk_dim, spk_dim)
        )
    def forward(self, feats):
        mean = feats.mean(dim=1)
        std = feats.std(dim=1)
        stat = torch.cat([mean, std], dim=-1)
        spk_emb = self.proj(stat)
        return F.normalize(spk_emb, p=2, dim=-1)

class CFMStage1System(nn.Module):
    def __init__(self, encoder_dim=768, cond_dim=512, n_mels=100, spk_dim=192, hidden_dim=None,
                 decoder_preset="v2", decoder_config=None, t_sampling="uniform"):
        super().__init__()
        self.n_mels = n_mels
        self.t_sampling = t_sampling
        self.cond_proj = nn.Linear(encoder_dim, cond_dim)
        self.spk_encoder = AcousticSpeakerEncoder(feat_dim=encoder_dim, spk_dim=spk_dim)
        if decoder_config:   # dựng lại đúng kiến trúc đã lưu trong checkpoint
            self.decoder = FlowMatchingTargetDecoder(**decoder_config)
        else:
            self.decoder = FlowMatchingTargetDecoder(n_mels=n_mels, cond_dim=cond_dim, spk_dim=spk_dim,
                                                     preset=decoder_preset)

    def condition(self, whisper_features, T_target):
        """Điều kiện c [B, cond_dim, T_target] — Stage 2 dùng đúng hàm này làm NHÃN cho Bridge."""
        cond_512 = self.cond_proj(whisper_features)
        return F.interpolate(cond_512.transpose(1, 2), size=T_target, mode='linear', align_corners=False)

    def compute_loss(self, whisper_features, target_mel, mel_mask=None, t=None, x_0=None):
        """mel_mask [B, T]: bỏ qua phần đệm. t / x_0: truyền vào để val loss cố định (tái lập được)."""
        B, C, T_target = target_mel.shape
        cond_512 = self.condition(whisper_features, T_target)
        spk_emb = self.spk_encoder(whisper_features)

        if t is None:
            t = sample_t(B, target_mel.device, target_mel.dtype, self.t_sampling)
        if x_0 is None:
            x_0 = torch.randn_like(target_mel)
        x_1 = target_mel
        t_expand = t.view(B, 1, 1)
        x_t = (1.0 - t_expand) * x_0 + t_expand * x_1
        u_t = x_1 - x_0

        v_pred = self.decoder(x_t, t, cond_512, spk_emb, mask=mel_mask)
        return masked_mse(v_pred, u_t, mel_mask)

    @torch.no_grad()
    def inference(self, whisper_features, num_steps=32, target_len=None):
        self.eval()
        B, T_src, _ = whisper_features.shape
        if target_len is None:
            target_len = int(T_src * (24000.0 / 16000.0) * (320.0 / 256.0))
        cond_512 = self.cond_proj(whisper_features)
        cond_512 = F.interpolate(cond_512.transpose(1, 2), size=target_len, mode='linear', align_corners=False)
        spk_emb = self.spk_encoder(whisper_features)

        device = whisper_features.device
        dtype = whisper_features.dtype
        x = torch.randn(B, self.n_mels, target_len, device=device, dtype=dtype)
        dt = 1.0 / num_steps

        # Thuật toán giải ODE bậc 2 Runge-Kutta Midpoint: Triệt tiêu tích tụ sai số gây rè/rít
        for step in range(num_steps):
            t_val = step * dt
            t_cur = torch.full((B,), t_val, device=device, dtype=dtype)
            v1 = self.decoder(x, t_cur, cond_512, spk_emb)

            # Bước phụ tại điểm giữa t + 0.5*dt
            x_mid = x + v1 * (0.5 * dt)
            t_mid = torch.full((B,), min(1.0, t_val + 0.5 * dt), device=device, dtype=dtype)
            v2 = self.decoder(x_mid, t_mid, cond_512, spk_emb)

            x = x + v2 * dt

        # Chặn ngưỡng âm học chuẩn [-11.5, 3.0] của Vocos để triệt tiêu hoàn toàn nhiễu clipping
        x = torch.clamp(x, min=-11.5, max=3.0)
        return x

# ==============================================================================
# 3. PIPELINE DỮ LIỆU STREAMING FLEURS (HỖ TRỢ EN-US HOẶC VI-VN)
# ==============================================================================

class UniversalSpeechStreamingPipeline:
    def __init__(self, split='train', max_sec=12.0, lang='en_us', allow_synthetic=False, shuffle_seed=None):
        self.split = split
        self.max_sec = max_sec
        self.lang = lang
        self.is_online = False
        self.allow_synthetic = allow_synthetic
        self.shuffle_seed = shuffle_seed
        self.pass_idx = 0          # số lần đã đọc hết dataset (dùng để đổi thứ tự shuffle)
        self.last_lengths = []     # độ dài thật (mẫu 16kHz) của từng câu trong batch gần nhất

        try:
            from datasets import load_dataset, Audio
            print(f"📡 Đang kết nối Streaming Google FLEURS ({lang}, split='{split}')...")
            self.ds = load_dataset('google/fleurs', lang, split=split, streaming=True)
            try:
                self.ds = self.ds.cast_column('audio', Audio(decode=False))
            except Exception:
                pass
            if shuffle_seed is not None:
                # Shuffle có buffer cho streaming; set_epoch() đổi thứ tự sau mỗi lượt đọc hết
                self.ds = self.ds.shuffle(seed=shuffle_seed, buffer_size=500)
            self.iter_ds = iter(self.ds)
            self.is_online = True
            print(f"✓ Kết nối thành công FLEURS {lang} [{split}]!")
        except Exception as e:
            self.is_online = False
            if not allow_synthetic:
                raise RuntimeError(f"Không stream được FLEURS {lang} [{split}] ({e}) — dừng để tránh train trên dữ liệu giả.")
            print(f"[Cảnh báo Dữ liệu] Không thể stream ({e}). Dùng tín hiệu mô phỏng (chỉ cho smoke-test).")

    def _restart(self):
        """Đọc hết dataset -> bắt đầu lượt mới (đổi thứ tự shuffle nếu có)."""
        self.pass_idx += 1
        if self.shuffle_seed is not None and hasattr(self.ds, "set_epoch"):
            self.ds.set_epoch(self.pass_idx)
        self.iter_ds = iter(self.ds)
        print(f"   🔁 [{self.lang}/{self.split}] Đã đọc hết dữ liệu -> bắt đầu lượt thứ {self.pass_idx + 1}")

    def _extract_audio_tensor(self, audio_data):
        import io
        import soundfile as sf
        limit = int(16000 * self.max_sec) if self.max_sec else None
        if isinstance(audio_data, dict):
            if "bytes" in audio_data and audio_data["bytes"]:
                arr, sr = sf.read(io.BytesIO(audio_data["bytes"]))
                if arr.ndim > 1: arr = arr.mean(axis=-1)
                return torch.from_numpy(arr).float()[:limit]
            if "array" in audio_data and audio_data["array"] is not None:
                arr = audio_data["array"]
                if hasattr(arr, "ndim") and arr.ndim > 1: arr = arr.mean(axis=-1)
                return torch.from_numpy(arr).float()[:limit]
            if "path" in audio_data and os.path.exists(audio_data["path"]):
                arr, sr = sf.read(audio_data["path"])
                if arr.ndim > 1: arr = arr.mean(axis=-1)
                return torch.from_numpy(arr).float()[:limit]
        raise ValueError("Không tìm thấy audio hợp lệ trong mục dữ liệu")

    def fetch_batch(self, batch_size=4):
        audio_list = []
        transcripts = []

        if self.is_online:
            n_errors = 0
            while len(audio_list) < batch_size:
                try:
                    item = next(self.iter_ds)
                except StopIteration:
                    self._restart()
                    continue
                try:
                    w = self._extract_audio_tensor(item['audio'])
                    if w.numel() < 1600:   # < 0.1s -> bỏ
                        continue
                    audio_list.append(w)
                    transcripts.append(item.get('transcription', ''))
                except Exception as ex:
                    n_errors += 1       # câu lỗi giải mã -> bỏ qua, lấy câu kế tiếp (KHÔNG bịa dữ liệu)
                    if n_errors > 50:
                        raise RuntimeError(f"Quá nhiều câu lỗi giải mã liên tiếp ({ex})")

        if len(audio_list) < batch_size and not self.allow_synthetic:
            raise RuntimeError("Thiếu dữ liệu thật và không cho phép dữ liệu mô phỏng.")
        while len(audio_list) < batch_size:
            t_sec = torch.linspace(0, self.max_sec, int(16000 * self.max_sec))
            f0 = 130.0 + 30.0 * torch.sin(2 * math.pi * 1.5 * t_sec)
            w = torch.sin(2 * math.pi * f0 * t_sec) * 0.4
            audio_list.append(w)
            transcripts.append('synthetic speech sample')

        self.last_lengths = [int(w.shape[0]) for w in audio_list]
        max_len = max(self.last_lengths)
        padded_audio_16k = torch.stack([F.pad(w, (0, max_len - w.shape[0])) for w in audio_list], dim=0)
        return padded_audio_16k, transcripts

# ==============================================================================
# 4. CHƯƠNG TRÌNH HUẤN LUYỆN CHÍNH
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Huấn luyện OT-CFM Target Speech Decoder (Stage 1)")
    parser.add_argument("--config", type=str, default="configs/stage1_cfm_en.yaml")
    parser.add_argument("--exp-name", type=str, default="cfm_en")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--overfit-16", action="store_true",
                        help="Sanity Check #3: Overfit 500 bước trên 16 câu và xuất file .wav nghe thử")
    parser.add_argument("--decoder", type=str, default=None,
                        help="Ghi đè decoder_preset trong config: v1 | v2 | v2-small (thêm '+ln' = t logit-normal)")
    parser.add_argument("--compare-decoders", type=str, default=None,
                        help="So sánh kiến trúc, VD 'v1,v2-small,v2,v2+ln': train ngắn cùng dữ liệu/seed rồi đo Mel L1, WER, val loss")
    parser.add_argument("--compare-steps", type=int, default=3000, help="Số bước train cho mỗi biến thể khi so sánh")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    train_cfg = cfg["training"]
    paths = cfg["paths"]
    data_lang = cfg["data"].get("lang", "en_us")

    log_path = os.path.join(paths["log_dir"], f"{args.exp_name}.log")
    sys.stdout = Logger(log_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported() and train_cfg.get("bf16", True)) else torch.float32

    print("=" * 75)
    print(f"🚀 HUẤN LUYỆN STAGE 1: OT-CFM TARGET DECODER ({data_lang.upper()})")
    print(f"Thiết bị: {device} | Kiểu số thực: {dtype} | Batch Size: {train_cfg['batch_size']}")
    print("=" * 75)

    # Nạp Whisper
    whisper_id = cfg["model"]["whisper_model"]
    feat_extractor = None
    whisper_enc = None
    try:
        from transformers import WhisperFeatureExtractor, WhisperModel
        print(f"⏳ Đang nạp Whisper Encoder ({whisper_id})...")
        feat_extractor = WhisperFeatureExtractor.from_pretrained(whisper_id)
        whisper_enc = WhisperModel.from_pretrained(whisper_id).encoder.eval().to(device=device, dtype=dtype)
        for p in whisper_enc.parameters(): p.requires_grad = False
        print("✓ Whisper Encoder đã nạp và đóng băng an toàn!")
    except Exception as e:
        print(f"[Cảnh báo Whisper] {e}. Sẽ dùng đặc trưng mô phỏng.")

    def parse_variant(name):
        """'v2+ln' -> ('v2', 'logit_normal')"""
        base, _, suf = str(name).partition("+")
        if base not in DECODER_PRESETS:
            raise ValueError(f"Không có decoder preset '{base}'. Chọn trong: {list(DECODER_PRESETS)}")
        return base, ("logit_normal" if suf == "ln" else cfg["model"].get("t_sampling", "uniform"))

    def build_model(variant=None, decoder_config=None, t_sampling=None):
        preset, ts = parse_variant(variant or args.decoder or cfg["model"].get("decoder_preset", "v2"))
        return CFMStage1System(
            encoder_dim=cfg["model"]["encoder_dim"],
            cond_dim=cfg["model"]["cond_dim"],
            n_mels=cfg["model"]["n_mels"],
            spk_dim=cfg["model"]["spk_dim"],
            decoder_preset=preset, decoder_config=decoder_config,
            t_sampling=t_sampling or ts,
        ).to(device=device, dtype=dtype)

    model = build_model()
    n_params = sum(p.numel() for p in model.decoder.parameters())
    print(f"🧩 Decoder: {model.decoder.config['arch']} | {n_params/1e6:.2f}M tham số | t-sampling: {model.t_sampling}")

    mel_fb = build_mel_filterbank(sr=24000, n_mels=100, device=device)

    # --------------------------------------------------------------------------
    # CHẾ ĐỘ SANITY CHECK #3: OVERFIT 16 CÂU (500 BƯỚC) + XUẤT AUDIO
    # --------------------------------------------------------------------------
    if args.overfit_16:
        print("\n" + "=" * 75)
        print("🧪 BẮT ĐẦU SANITY CHECK #3: OVERFIT 500 BƯỚC TRÊN 16 CÂU TIẾNG ANH")
        print("Mục tiêu: Đảm bảo CFM học thuộc lòng âm học tiếng Anh và sinh ra file .wav rõ chữ!")
        print("=" * 75)
        pipeline = UniversalSpeechStreamingPipeline(split='train', max_sec=12.0, lang=data_lang,
                                                    allow_synthetic=args.smoke_test)
        fix_audio, fix_txt = pipeline.fetch_batch(batch_size=16)

        if feat_extractor and whisper_enc:
            in_p = feat_extractor(list(fix_audio.numpy()), sampling_rate=16000, return_tensors='pt', padding="max_length")
            f = in_p.input_features.to(device=device, dtype=dtype)
            if f.shape[-1] < 3000: f = F.pad(f, (0, 3000 - f.shape[-1]))
            elif f.shape[-1] > 3000: f = f[:, :, :3000]
            with torch.no_grad():
                fl = whisper_enc(f).last_hidden_state
            real_l = max(10, min(1500, int(fix_audio.shape[-1] / 320)))
            fix_latents = fl[:, :real_l, :]
            print(f"✓ Đã trích xuất đặc trưng âm học thật từ Whisper ({whisper_id}) [Shape: {list(fix_latents.shape)}]")
        else:
            print("⚠️ CẢNH BÁO: Whisper Encoder không tải được, đang dùng latents ngẫu nhiên!")
            fix_latents = torch.randn(16, 100, 768, device=device, dtype=dtype)

        fix_24k = resample_16k_to_24k(fix_audio.to(device=device, dtype=torch.float32))
        fix_mel = extract_100_mel(fix_24k, mel_fb).to(dtype=dtype)
        fix_mask = length_mask(mel_lengths(pipeline.last_lengths), fix_mel.shape[-1], device)

        num_overfit_steps = 800
        overfit_opt = torch.optim.AdamW(model.parameters(), lr=4.0e-4, weight_decay=1.0e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(overfit_opt, T_max=num_overfit_steps, eta_min=1.0e-5)
        model.train()
        t_start = time.time()
        initial_l = None
        final_l = None

        for step in range(1, num_overfit_steps + 1):
            overfit_opt.zero_grad()
            l = model.compute_loss(fix_latents, fix_mel, mel_mask=fix_mask)
            l.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            overfit_opt.step()
            scheduler.step()

            if step == 1: initial_l = l.item()
            if step % 50 == 0 or step == 1:
                print(f"   [Overfit Step {step:03d}/{num_overfit_steps}] CFM Loss: {l.item():.4f}")
            final_l = l.item()

        print("-" * 75)
        print(f"✅ Loss giảm từ {initial_l:.4f} xuống {final_l:.4f} (Giảm {(1.0 - final_l/initial_l)*100:.1f}%) trong {time.time()-t_start:.1f}s!")

        # Xuất toàn bộ 16 file .wav nghe thử qua Vocos
        out_dir = "outputs/sanity/cfm_overfit16"
        legacy_dir = "outputs/sanity"
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(legacy_dir, exist_ok=True)
        try:
            vocos = load_vocos_safely("charactr/vocos-mel-24khz", device=device)
            gen_mels = model.inference(fix_latents, num_steps=32, target_len=fix_mel.shape[-1])
            print(f"📏 Mel L1 (pred vs Mel thật, chỉ frame thật): {masked_mel_l1(gen_mels, fix_mel, fix_mask):.4f} "
                  "— càng thấp càng sát gt_mel_vocos")
            with torch.no_grad():
                print(f"\n🎧 ĐANG GIẢI MÃ VÀ XUẤT TOÀN BỘ {len(fix_audio)} CÂU AUDIO QUA VOCOS...")
                for idx in range(len(fix_audio)):
                    pred_wav = vocos.decode(gen_mels[idx:idx+1].float()).squeeze().cpu()
                    gt_mel_wav = vocos.decode(fix_mel[idx:idx+1].float()).squeeze().cpu()
                    gt_wav = fix_24k[idx].cpu()

                    out_pred = os.path.join(out_dir, f"sample_{idx:02d}_pred.wav")
                    out_gt_mel = os.path.join(out_dir, f"sample_{idx:02d}_gt_mel_vocos.wav")
                    out_gt = os.path.join(out_dir, f"sample_{idx:02d}_groundtruth.wav")

                    save_wav_file(out_pred, pred_wav, 24000)
                    save_wav_file(out_gt_mel, gt_mel_wav, 24000)
                    save_wav_file(out_gt, gt_wav, 24000)

                    if idx == 0:
                        save_wav_file(os.path.join(legacy_dir, "cfm_en_overfit16_sample0.wav"), pred_wav, 24000)

                    txt_preview = fix_txt[idx][:45] + "..." if len(fix_txt[idx]) > 45 else fix_txt[idx]
                    print(f"   [Mẫu #{idx:02d}/16] {out_pred} | \"{txt_preview}\"")
            print("-" * 75)
            print(f"✓ ĐÃ XUẤT TRỌN BỘ 16 MẪU ÂM THANH (CẢ FILE DỰ ĐOÁN, GT-MEL & FILE GỐC) TẠI:")
            print(f"  👉 Thư mục: {out_dir}/")
            print(f"  👉 Bạn có thể nghe so sánh sample_00_pred.wav với sample_00_gt_mel_vocos.wav!")
        except Exception as e:
            print(f"[Vocos Info] Không thể giải mã audio: {e}")
        print("=" * 75 + "\n")
        return

    # Huấn luyện thông thường
    if not (feat_extractor and whisper_enc) and not args.smoke_test:
        raise RuntimeError("Không nạp được Whisper encoder — dừng để tránh train trên latent ngẫu nhiên.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
        betas=(0.9, 0.999)
    )

    epochs = 1 if args.smoke_test else int(train_cfg["epochs"])
    steps_per_epoch = 2 if args.smoke_test else int(train_cfg["steps_per_epoch"])
    batch_size = int(train_cfg["batch_size"])
    total_steps = max(1, epochs * steps_per_epoch)
    min_lr_ratio = float(train_cfg.get("min_lr", 1e-5)) / float(train_cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda st: max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * min(1.0, st / total_steps)))))

    def featurize(audio_16k, lengths):
        """audio đệm -> (latent Whisper, mel đích, mask frame thật)."""
        if feat_extractor and whisper_enc:
            lat = encode_whisper(audio_16k, feat_extractor, whisper_enc, device, dtype)
        else:
            lat = torch.randn(audio_16k.shape[0], 100, 768, device=device, dtype=dtype)
        a24 = resample_16k_to_24k(audio_16k.to(device=device, dtype=torch.float32))
        mel = extract_100_mel(a24, mel_fb).to(dtype=dtype)
        mask = length_mask(mel_lengths(lengths), mel.shape[-1], device)
        return lat, mel, mask

    # ---- Tập validation CỐ ĐỊNH (cùng câu, cùng t / x0 mỗi epoch) -> val loss so sánh được ----
    val_clips = 4 if args.smoke_test else int(train_cfg.get("val_clips", 32))
    val_pipeline = UniversalSpeechStreamingPipeline(split='validation', max_sec=12.0, lang=data_lang,
                                                    allow_synthetic=args.smoke_test)
    val_set = []
    for _ in range((val_clips + batch_size - 1) // batch_size):
        v_audio, v_txt = val_pipeline.fetch_batch(batch_size=batch_size)
        val_set.append((v_audio, list(val_pipeline.last_lengths), v_txt))
    gen = torch.Generator(device="cpu").manual_seed(1234)
    val_noise = []
    for v_audio, v_lens, _ in val_set:
        T = mel_lengths([v_audio.shape[-1]])[0]
        val_noise.append((torch.rand(v_audio.shape[0], generator=gen),
                          torch.randn(v_audio.shape[0], cfg["model"]["n_mels"], T, generator=gen)))
    print(f"✓ Tập validation cố định: {sum(v[0].shape[0] for v in val_set)} câu (seed 1234)")

    # ---- Đo WER tái tạo: CFM sinh Mel từ latent -> Vocos -> Whisper-En -> WER ----
    eval_wer_every = int(train_cfg.get("eval_wer_every", 5))
    eval_wer_clips = int(train_cfg.get("eval_wer_clips", 16))
    vocos = None
    en_asr = EnglishASR(device)
    gt_wer = None

    def eval_resynthesis(n_clips=None, save_dir=None, seed=2024):
        """Sinh Mel từ latent của tập val (độ dài thật, noise cố định theo seed) rồi đo:
        - Mel L1 so với Mel thật (toàn bộ tập val, chỉ frame thật)
        - WER: Vocos -> Whisper-En nghe lại (n_clips câu đầu)
        save_dir: lưu pred / gt_mel_vocos / groundtruth của 4 câu đầu để nghe so sánh."""
        nonlocal vocos, gt_wer
        n_clips = eval_wer_clips if n_clips is None else n_clips
        if vocos is None:
            vocos = load_vocos_safely("charactr/vocos-mel-24khz", device=device)
            vocos.eval()
        torch.manual_seed(seed)
        hyps, refs, gt_hyps = [], [], []
        l1_sum, l1_n, done = 0.0, 0, 0
        for v_audio, v_lens, v_txt in val_set:
            lat, mel, msk = featurize(v_audio, v_lens)
            gen_mel = model.inference(lat, num_steps=32, target_len=mel.shape[-1])
            l1_sum += masked_mel_l1(gen_mel, mel, msk) * v_audio.shape[0]
            l1_n += v_audio.shape[0]
            for i in range(v_audio.shape[0]):
                if done >= n_clips:
                    break
                n_frames = mel_lengths([v_lens[i]])[0]
                with torch.no_grad():
                    wav24 = vocos.decode(gen_mel[i:i + 1, :, :n_frames].float()).squeeze().cpu()
                    if save_dir and done < 4:
                        os.makedirs(save_dir, exist_ok=True)
                        save_wav_file(os.path.join(save_dir, f"val_{done:02d}_pred.wav"), wav24, 24000)
                        gtv = vocos.decode(mel[i:i + 1, :, :n_frames].float()).squeeze().cpu()
                        save_wav_file(os.path.join(save_dir, f"val_{done:02d}_gt_mel_vocos.wav"), gtv, 24000)
                hyps.append(en_asr.transcribe(resample_24k_to_16k_fft(wav24)))
                refs.append(v_txt[i])
                if gt_wer is None:
                    gt_hyps.append(en_asr.transcribe(v_audio[i, :v_lens[i]]))
                done += 1
        if gt_wer is None:
            gt_wer = corpus_wer(gt_hyps, refs)
        return {"wer": corpus_wer(hyps, refs), "gt_wer": gt_wer, "mel_l1": l1_sum / max(1, l1_n),
                "examples": list(zip(refs[:2], hyps[:2]))}

    def eval_val_loss():
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for (v_audio, v_lens, _), (vt, vx0) in zip(val_set, val_noise):
                v_lat, v_mel, v_mask = featurize(v_audio, v_lens)
                vx0 = vx0[:, :, :v_mel.shape[-1]].to(device=device, dtype=dtype)
                tot += model.compute_loss(v_lat, v_mel, mel_mask=v_mask,
                                          t=vt.to(device=device, dtype=dtype), x_0=vx0).item()
                n += 1
        return tot / max(1, n)

    # ==========================================================================
    # CHẾ ĐỘ SO SÁNH KIẾN TRÚC DECODER (công bằng: cùng dữ liệu, cùng thứ tự, cùng seed, cùng số bước)
    # ==========================================================================
    if args.compare_decoders:
        variants = [v.strip() for v in args.compare_decoders.split(",") if v.strip()]
        n_steps = 20 if args.smoke_test else int(args.compare_steps)
        out_root = "outputs/sanity/decoder_compare"
        os.makedirs(out_root, exist_ok=True)
        results = []
        print("\n" + "=" * 75)
        print(f"⚖️ SO SÁNH DECODER: {variants} | {n_steps} bước/biến thể | batch {batch_size} | cùng seed & thứ tự dữ liệu")
        print("=" * 75)
        for v in variants:
            torch.manual_seed(0)
            model = build_model(v)
            n_par = sum(p.numel() for p in model.decoder.parameters())
            opt_c = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]),
                                      weight_decay=float(train_cfg["weight_decay"]), betas=(0.9, 0.999))
            sch_c = torch.optim.lr_scheduler.LambdaLR(
                opt_c, lambda st: max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * min(1.0, st / n_steps)))))
            pipe = UniversalSpeechStreamingPipeline(split='train', max_sec=12.0, lang=data_lang,
                                                    allow_synthetic=args.smoke_test,
                                                    shuffle_seed=None if args.smoke_test else 42)
            print(f"\n▶️ [{v}] decoder {model.decoder.config['arch']} | {n_par/1e6:.2f}M tham số | t: {model.t_sampling}")
            model.train()
            t0 = time.time()
            run_loss = 0.0
            for st in range(1, n_steps + 1):
                opt_c.zero_grad()
                audio_16k, _ = pipe.fetch_batch(batch_size=batch_size)
                lat, mel, msk = featurize(audio_16k, pipe.last_lengths)
                loss = model.compute_loss(lat, mel, mel_mask=msk)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(train_cfg["max_grad_norm"]))
                opt_c.step()
                sch_c.step()
                run_loss += loss.item()
                if st % 250 == 0 or st == n_steps:
                    print(f"   [{v}] bước {st}/{n_steps} | train loss TB 250 bước: {run_loss / min(250, st):.4f}")
                    run_loss = 0.0
            sec_per_step = (time.time() - t0) / n_steps
            val_l = eval_val_loss()
            ev = eval_resynthesis(save_dir=os.path.join(out_root, v.replace("+", "_")))
            res = {"variant": v, "params_M": round(n_par / 1e6, 2), "steps": n_steps, "sec_per_step": round(sec_per_step, 3),
                   "val_loss": round(val_l, 4), "mel_l1": round(ev["mel_l1"], 4), "wer": round(ev["wer"], 1),
                   "gt_wer": round(ev["gt_wer"], 1)}
            results.append(res)
            print(f"   ✅ [{v}] val loss {val_l:.4f} | Mel L1 {ev['mel_l1']:.4f} | WER {ev['wer']:.1f}% "
                  f"(audio gốc {ev['gt_wer']:.1f}%) | {sec_per_step:.2f}s/bước")
            for r, h in ev["examples"]:
                print(f"      • ref: {r[:70]}\n        cfm: {h[:70]}")
            del opt_c, sch_c
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Bảng tổng hợp + xếp hạng
        lines = ["# ⚖️ SO SÁNH KIẾN TRÚC DECODER (Stage 1 CFM)", "",
                 f"Cùng dữ liệu FLEURS {data_lang}, cùng thứ tự, seed 0, {n_steps} bước/biến thể, batch {batch_size}. "
                 f"Đánh giá trên {sum(x[0].shape[0] for x in val_set)} câu validation cố định "
                 f"(WER trên {eval_wer_clips} câu đầu; WER audio gốc qua cùng ASR = {results[0]['gt_wer'] if results else 'nan'}%).", "",
                 "| Biến thể | Tham số | Val loss ↓ | Mel L1 ↓ | WER tái tạo ↓ | s/bước |",
                 "| :--- | :---: | :---: | :---: | :---: | :---: |"]
        for r in results:
            lines.append(f"| {r['variant']} | {r['params_M']}M | {r['val_loss']:.4f} | {r['mel_l1']:.4f} | {r['wer']:.1f}% | {r['sec_per_step']:.2f} |")
        if results:
            best_wer = min(results, key=lambda r: r["wer"])
            best_l1 = min(results, key=lambda r: r["mel_l1"])
            lines += ["", f"- WER thấp nhất (rõ chữ nhất): **{best_wer['variant']}**",
                      f"- Mel L1 thấp nhất (sát phổ nhất): **{best_l1['variant']}**",
                      "- Chênh WER < ~3 điểm với 16 câu là trong vùng nhiễu → ưu tiên Mel L1 và nghe thử "
                      f"`{out_root}/<biến thể>/val_XX_pred.wav` so với `val_XX_gt_mel_vocos.wav`."]
        report = "\n".join(lines)
        with open(os.path.join(out_root, "compare_report.md"), "w", encoding="utf-8") as f:
            f.write(report)
        with open(os.path.join(out_root, "compare_results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("\n" + report)
        print(f"\n📁 Báo cáo: {out_root}/compare_report.md | audio nghe thử: {out_root}/<biến thể>/")
        print("👉 Chọn biến thể tốt nhất rồi đặt 'decoder_preset' (và 't_sampling') trong configs/stage1_cfm_en.yaml")
        return

    # ---- Resume ----
    os.makedirs(paths["checkpoint_dir"], exist_ok=True)
    ckpt_path = os.path.join(paths["checkpoint_dir"], f"{args.exp_name}-last.pth")
    best_ckpt = os.path.join(paths["checkpoint_dir"], f"{args.exp_name}-best.pth")
    start_epoch, global_step, best_val_loss, history = 0, 0, float("inf"), []
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        meta = ck.get("stage1_meta", {})
        if meta.get("lang") not in (None, data_lang):
            raise RuntimeError(f"{ckpt_path} là checkpoint ngôn ngữ {meta['lang']} — không resume.")
        if meta.get("decoder_config") and meta["decoder_config"] != model.decoder.config:
            print(f"[Resume] Checkpoint dùng decoder {meta['decoder_config'].get('arch')} -> dựng lại đúng kiến trúc đã lưu")
            model = build_model(decoder_config=meta["decoder_config"], t_sampling=meta.get("t_sampling"))
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]),
                                          weight_decay=float(train_cfg["weight_decay"]), betas=(0.9, 0.999))
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda st: max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * min(1.0, st / total_steps)))))
        model.load_state_dict(ck["full_system_state_dict"], strict=True)
        if "optimizer_state_dict" in ck:
            try:
                optimizer.load_state_dict(ck["optimizer_state_dict"])
            except Exception as e:
                print(f"[Resume] Khởi tạo lại optimizer ({e})")
        if "scheduler_state_dict" in ck:
            scheduler.load_state_dict(ck["scheduler_state_dict"])
        start_epoch = int(ck.get("epoch", -1)) + 1
        global_step = int(ck.get("step", 0))
        best_val_loss = float(ck.get("best_loss", float("inf")))
        history = ck.get("history", [])
        print(f"⚡ [Resume] Tiếp tục từ epoch {start_epoch} (step {global_step}, best val {best_val_loss:.4f})")
    elif args.resume:
        print(f"[Resume] Không thấy {ckpt_path} -> train từ đầu.")

    train_pipeline = UniversalSpeechStreamingPipeline(split='train', max_sec=12.0, lang=data_lang,
                                                      allow_synthetic=args.smoke_test,
                                                      shuffle_seed=None if args.smoke_test else 42)
    sync_enabled = bool(train_cfg.get("sync_drive", True)) and not args.smoke_test
    print(f"[Training Config] {epochs} epochs × {steps_per_epoch} steps × batch {batch_size} "
          f"| LR cosine {train_cfg['learning_rate']} → {train_cfg.get('min_lr', 1e-5)} | WER mỗi {eval_wer_every} epoch")

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for step in range(steps_per_epoch):
            optimizer.zero_grad()
            audio_16k, _ = train_pipeline.fetch_batch(batch_size=batch_size)
            latents, tgt_mel, mel_mask = featurize(audio_16k, train_pipeline.last_lengths)

            cfm_l = model.compute_loss(latents, tgt_mel, mel_mask=mel_mask)
            cfm_l.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(train_cfg["max_grad_norm"]))
            optimizer.step()
            scheduler.step()
            global_step += 1

            train_loss += cfm_l.item()
            if (step + 1) % 10 == 0:
                print(f"   [E{epoch:02d}|Step {step+1:03d}/{steps_per_epoch}] Train Loss: {cfm_l.item():.4f} "
                      f"| LR: {scheduler.get_last_lr()[0]:.2e}")

        avg_train = train_loss / steps_per_epoch
        print(f"==> Hoàn tất Epoch {epoch:02d} trong {time.time()-t0:.1f}s | Avg Train Loss: {avg_train:.4f}")

        # Validation cố định
        avg_val = eval_val_loss()
        print(f"   Val Loss (cố định): {avg_val:.4f} | Best: {min(best_val_loss, avg_val):.4f}")

        wer, mel_l1 = None, None
        if not args.smoke_test and eval_wer_every > 0 and ((epoch + 1) % eval_wer_every == 0 or epoch == epochs - 1):
            try:
                ev = eval_resynthesis(save_dir=f"outputs/sanity/{args.exp_name}_epoch{epoch:02d}")
                wer, mel_l1 = ev["wer"], ev["mel_l1"]
                print(f"   🎧 WER tái tạo: {wer:.1f}% (audio gốc qua cùng ASR: {gt_wer:.1f}%) | Mel L1: {mel_l1:.4f}")
                for r, h in ev["examples"]:
                    print(f"      • ref: {r[:70]}\n        cfm: {h[:70]}")
            except Exception as e:
                print(f"   [WER] Không đo được: {e}")
        history.append({"epoch": epoch, "train": avg_train, "val": avg_val, "wer": wer, "gt_wer": gt_wer,
                        "mel_l1": mel_l1})

        ckpt_data = {
            "epoch": epoch,
            "step": global_step,
            "best_loss": min(best_val_loss, avg_val),
            "model_state_dict": model.decoder.state_dict(),
            "full_system_state_dict": model.state_dict(),   # gồm cond_proj + spk_encoder cho Stage 2
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
            "stage1_meta": {"lang": data_lang, "whisper_model": whisper_id, "version": 3,
                            "decoder_config": model.decoder.config, "t_sampling": model.t_sampling,
                            "cond_dim": cfg["model"]["cond_dim"], "spk_dim": cfg["model"]["spk_dim"]},
            "config": cfg,
        }
        torch.save(ckpt_data, ckpt_path)
        synced = [ckpt_path]
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(ckpt_data, best_ckpt)
            synced.append(best_ckpt)
            print(f"⭐ [Epoch {epoch:02d}] Đã cập nhật Model Best: {best_val_loss:.4f}")
        sync_to_drive(synced, args.exp_name, enabled=sync_enabled)

    print("\n🎉 HUẤN LUYỆN STAGE 1 OT-CFM TIẾNG ANH HOÀN TẤT THÀNH CÔNG!")

if __name__ == "__main__":
    main()
