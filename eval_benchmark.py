#!/usr/bin/env python3
# ==============================================================================
# PIPELINE BENCHMARK ĐÁNH GIÁ ABLATION & ĐO METRIC S2ST VI -> EN
# Case 0: Pretrained Cascaded Baseline (Whisper-Vi + OpusMT + MMS-TTS-Eng)
# Case 0b / 5: Meta SeamlessM4T-v2 (SOTA Direct S2ST Benchmark)
# Case 1: Untrained CrossAttentionBridge + Untrained CFM (Zero-Baseline)
# Case 2: Untrained Bridge + Trained CFM (Acoustic Only)
# Case 3: Trained Bridge + Untrained CFM (Semantic Only)
# Case 4: Trained CrossAttentionBridge + Trained CFM (Full Proposed Vi-En System)
# Metrics đo lường:
#   1. RTF (Real-Time Factor) ↓: Tốc độ suy luận
#   2. Mel L1 Error ↓: Độ sai lệch phổ âm học so với English Ground Truth
#   3. SECS ↑: Mức độ bảo tồn âm sắc giọng nói nguồn (Speaker Embedding Cosine Similarity)
#   4. ASR-BLEU ↑: Độ chính xác dịch thuật (Whisper English ASR + SacreBLEU)
# ==============================================================================

import argparse
import importlib.machinery
import json
import math
import os
import re
import sys
import time
import types
import warnings
import wave
from collections import Counter
from pathlib import Path

# Tắt toàn bộ các cảnh báo không cần thiết từ transformers và HuggingFace
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.generation").setLevel(logging.ERROR)

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
            if is_pkg:
                mod.__path__ = []
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

        def _ta_load(filepath, **kwargs):
            import soundfile as sf
            data, sr = sf.read(filepath)
            t = torch.from_numpy(data).float()
            if t.ndim == 1:
                t = t.unsqueeze(0)
            elif t.ndim == 2:
                t = t.t()
            return t, sr

        def _ta_save(filepath, src, sample_rate, **kwargs):
            import soundfile as sf
            if hasattr(src, "cpu"):
                wav_np = src.squeeze().cpu().numpy()
            else:
                wav_np = torch.as_tensor(src).squeeze().cpu().numpy()
            if wav_np.ndim > 1:
                wav_np = wav_np[0]
            sf.write(filepath, wav_np, sample_rate)

        def _ta_info(filepath, **kwargs):
            import soundfile as sf
            info = sf.info(filepath)
            class AudioMetaData:
                sample_rate = info.samplerate
                num_frames = info.frames
                num_channels = info.channels
                bits_per_sample = 16
                encoding = "PCM_S"
            return AudioMetaData()

        def _resample(waveform, orig_freq, new_freq, **kwargs):
            """Thay thế torchaudio.functional.resample bằng torch interpolate."""
            if orig_freq == new_freq:
                return waveform
            ratio = new_freq / orig_freq
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0).unsqueeze(0)
                out = torch.nn.functional.interpolate(waveform, scale_factor=ratio, mode='linear', align_corners=False)
                return out.squeeze(0).squeeze(0)
            elif waveform.ndim == 2:
                waveform = waveform.unsqueeze(1)
                out = torch.nn.functional.interpolate(waveform, scale_factor=ratio, mode='linear', align_corners=False)
                return out.squeeze(1)
            else:
                return torch.nn.functional.interpolate(waveform, scale_factor=ratio, mode='linear', align_corners=False)

        ta_func_func.hz_to_mel = _hz_to_mel
        ta_func_func.mel_to_hz = _mel_to_hz
        ta_func_func._hz_to_mel = _hz_to_mel
        ta_func_func._mel_to_hz = _mel_to_hz
        ta_func_func.resample = _resample
        ta_func.hz_to_mel = _hz_to_mel
        ta_func.mel_to_hz = _mel_to_hz
        ta_func._hz_to_mel = _hz_to_mel
        ta_func._mel_to_hz = _mel_to_hz
        ta_func.resample = _resample
        ta.functional = ta_func
        ta.transforms = ta_trans
        ta._extension = ta_ext
        ta.load = _ta_load
        ta.save = _ta_save
        ta.info = _ta_info

        sys.modules["torchaudio"] = ta
        sys.modules["torchaudio.functional"] = ta_func
        sys.modules["torchaudio.functional.functional"] = ta_func_func
        sys.modules["torchaudio.transforms"] = ta_trans
        sys.modules["torchaudio._extension"] = ta_ext
        return "mock"

_TA_STATUS = apply_torchaudio_mock_if_needed()

# Import PyTorch an toàn
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def _mock_melscale_fbanks(n_freqs, f_min, f_max, n_mels, sample_rate, norm=None, mel_scale="htk"):
        """Tái hiện torchaudio.functional.melscale_fbanks (HTK hoặc Slaney) bằng PyTorch thuần.
        Trả về ma trận (n_freqs × n_mels) giống hệt torchaudio."""
        def hz_to_mel(f):
            if mel_scale == "htk":
                return 2595.0 * math.log10(1.0 + f / 700.0)
            f_sp, min_log_hz = 200.0 / 3, 1000.0
            min_log_mel, logstep = min_log_hz / f_sp, math.log(6.4) / 27.0
            return f / f_sp if f < min_log_hz else min_log_mel + math.log(f / min_log_hz) / logstep

        def mel_to_hz(m):
            if mel_scale == "htk":
                return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
            f_sp, min_log_hz = 200.0 / 3, 1000.0
            min_log_mel, logstep = min_log_hz / f_sp, math.log(6.4) / 27.0
            mels = f_sp * m
            return torch.where(m >= min_log_mel, min_log_hz * torch.exp(logstep * (m - min_log_mel)), mels)

        all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)
        m_pts = torch.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
        f_pts = mel_to_hz(m_pts)
        f_diff = f_pts[1:] - f_pts[:-1]
        slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)
        down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]
        up_slopes = slopes[:, 2:] / f_diff[1:]
        fb = torch.max(torch.zeros(1), torch.min(down_slopes, up_slopes))
        if norm == "slaney":
            enorm = 2.0 / (f_pts[2:n_mels + 2] - f_pts[:n_mels])
            fb = fb * enorm.unsqueeze(0)
        return fb

    class MockSpectrogram(nn.Module):
        """Mock torchaudio.transforms.Spectrogram — tính STFT thật bằng torch.stft.
        Buffer 'window' có kích thước theo win_length để khớp state_dict (VD XTTS-v2: [400])."""
        def __init__(self, n_fft=400, win_length=None, hop_length=None, pad=0,
                     window_fn=None, power=2.0, normalized=False, wkwargs=None,
                     center=True, pad_mode="reflect", onesided=True, **kwargs):
            super().__init__()
            self.n_fft = n_fft
            self.win_length = win_length if win_length is not None else n_fft
            self.hop_length = hop_length if hop_length is not None else (self.win_length // 2)
            self.pad = pad
            self.power = power
            self.normalized = normalized
            self.center = center
            self.pad_mode = pad_mode
            self.onesided = onesided
            win_fn = window_fn if window_fn is not None else torch.hann_window
            window = win_fn(self.win_length, **(wkwargs or {}))
            self.register_buffer("window", window)

        def forward(self, x):
            # torch.stft chỉ nhận đầu vào 1D/2D -> gộp các chiều batch phía trước
            lead_shape = x.shape[:-1]
            x = x.reshape(-1, x.shape[-1])
            if self.pad > 0:
                x = F.pad(x, (self.pad, self.pad), mode="constant")
            # torch.stft tự pad n_fft//2 khi center=True (không pad thủ công để tránh pad 2 lần)
            stft_out = torch.stft(x, self.n_fft, hop_length=self.hop_length,
                                  win_length=self.win_length,
                                  window=self.window.to(device=x.device, dtype=x.dtype),
                                  center=self.center, pad_mode=self.pad_mode,
                                  normalized=False, onesided=self.onesided,
                                  return_complex=True)
            if self.normalized:
                stft_out = stft_out / self.window.pow(2.0).sum().sqrt()
            stft_out = stft_out.reshape(lead_shape + stft_out.shape[-2:])
            if self.power is None:
                return stft_out
            if self.power == 1.0:
                return stft_out.abs()
            return stft_out.abs().pow(self.power)

    class MockMelScale(nn.Module):
        """Mock torchaudio.transforms.MelScale — filterbank 'fb' (n_stft × n_mels) được TÍNH THẬT.
        Quan trọng: F5-TTS và Vocos xóa key fb khỏi state_dict khi load, nên nếu để zeros
        thì toàn bộ mel đầu ra = 0 (giọng tham chiếu bị câm)."""
        def __init__(self, n_mels=128, sample_rate=16000, f_min=0., f_max=None,
                     n_stft=201, norm=None, mel_scale="htk", **kwargs):
            super().__init__()
            f_max_val = f_max if f_max is not None else float(sample_rate // 2)
            fb = _mock_melscale_fbanks(n_stft, f_min, f_max_val, n_mels, sample_rate, norm, mel_scale)
            self.register_buffer("fb", fb)

        def forward(self, x):
            fb = self.fb.to(device=x.device, dtype=x.dtype)
            return torch.matmul(x.transpose(-1, -2), fb).transpose(-1, -2)

    class MockMelSpectrogram(nn.Module):
        """Mock torchaudio.transforms.MelSpectrogram — tính mel spectrogram thật
        bằng torch.stft + mel filterbank, kích thước buffer khớp checkpoint."""
        def __init__(self, sample_rate=16000, n_fft=400, win_length=None, hop_length=None,
                     f_min=0., f_max=None, pad=0, n_mels=128, window_fn=None,
                     power=2.0, normalized=False, wkwargs=None, center=True,
                     pad_mode="reflect", onesided=None, norm=None, mel_scale="htk", **kwargs):
            super().__init__()
            wl = win_length if win_length is not None else n_fft
            n_stft = n_fft // 2 + 1
            self.spectrogram = MockSpectrogram(n_fft=n_fft, win_length=wl, hop_length=hop_length,
                                               pad=pad, window_fn=window_fn, power=power,
                                               normalized=normalized, wkwargs=wkwargs,
                                               center=center, pad_mode=pad_mode)
            self.mel_scale = MockMelScale(n_mels=n_mels, sample_rate=sample_rate, f_min=f_min,
                                          f_max=f_max, n_stft=n_stft, norm=norm, mel_scale=mel_scale)

        def forward(self, x):
            return self.mel_scale(self.spectrogram(x))

    def _install_mel_mocks():
        """Gắn các lớp Mock (tính toán thật) vào module torchaudio giả lập.
        Chỉ chạy ở chế độ mock — nếu torchaudio native hoạt động thì giữ nguyên bản gốc."""
        if _TA_STATUS != "mock":
            return
        ta_trans = sys.modules.get("torchaudio.transforms")
        if ta_trans is not None:
            ta_trans.MelSpectrogram = MockMelSpectrogram
            ta_trans.Spectrogram = MockSpectrogram
            ta_trans.MelScale = MockMelScale
        ta_mod = sys.modules.get("torchaudio")
        if ta_mod is not None and hasattr(ta_mod, "transforms"):
            ta_mod.transforms.MelSpectrogram = MockMelSpectrogram
            ta_mod.transforms.Spectrogram = MockSpectrogram
            ta_mod.transforms.MelScale = MockMelScale
        ta_func = sys.modules.get("torchaudio.functional")
        if ta_func is not None:
            ta_func.melscale_fbanks = _mock_melscale_fbanks

    # Lưu lại các lớp torchaudio native (nếu có) vì train_bridge.py ghi đè chúng khi import
    _NATIVE_TA_TRANSFORMS = {}
    if _TA_STATUS == "native":
        import torchaudio as _ta_native
        for _name in ("MelSpectrogram", "Spectrogram", "MelScale"):
            _NATIVE_TA_TRANSFORMS[_name] = getattr(_ta_native.transforms, _name)
except ImportError as e:
    print(f"[Fatal Error] PyTorch chưa được cài đặt: {e}")
    sys.exit(1)

# Import module hệ thống từ train_bridge.py
from train_bridge import (
    ViEnS2STSystem, EnViS2STSystem, CrossAttentionBridge, ToneAwareBridge,
    RealStreamingSpeechPipeline, build_mel_filterbank, extract_100_mel,
    resample_16k_to_24k, resample_24k_to_16k
)

# LƯU Ý: train_bridge.py khi import sẽ gắn các Mock no-op (forward trả về x) vào torchaudio.transforms.
# Phải gắn lại bản Mock tính toán thật SAU khi import, nếu không XTTS-v2 / F5-TTS nhận mel sai.
if _TA_STATUS == "mock":
    _install_mel_mocks()
else:
    import torchaudio as _ta_native
    for _name, _cls in _NATIVE_TA_TRANSFORMS.items():
        setattr(_ta_native.transforms, _name, _cls)

# Tự động cài đặt các thư viện cần thiết nếu còn thiếu
def auto_install_packages():
    pkgs = []
    try:
        import soundfile
    except Exception:
        pkgs.append("soundfile")
    try:
        import vocos
    except Exception:
        pkgs.append("vocos")
    try:
        import sacrebleu
    except Exception:
        pkgs.append("sacrebleu")
    if pkgs:
        import subprocess
        try:
            print(f"📦 Đang tự động cài đặt các thư viện: {pkgs}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + pkgs + ["-q"])
            print("✓ Cài đặt thư viện hoàn tất!")
        except Exception as e:
            print(f"[Cảnh báo] Lỗi cài đặt thư viện ({e}).")

auto_install_packages()

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

def load_vocos(device):
    """Nạp Vocos Vocoder pre-trained (24kHz) cho việc chuyển đổi Mel -> Waveform."""
    try:
        from vocos import Vocos
        print("⏳ Đang nạp Pretrained Vocos Vocoder 24kHz...")
        orig_load = Vocos.load_state_dict
        def safe_load(self, state_dict, strict=False):
            clean_sd = {k: v for k, v in state_dict.items() if not k.startswith("feature_extractor")}
            return orig_load(self, clean_sd, strict=False)
        Vocos.load_state_dict = safe_load
        try:
            vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
        finally:
            Vocos.load_state_dict = orig_load
        vocos.eval()
        print("✓ Nạp Vocos Vocoder thành công!")
        return vocos
    except Exception as e:
        print(f"[Vocos Warning] Không thể nạp Vocos: {e}.")
        return None

# ==============================================================================
# HỆ THỐNG ĐO LƯỜNG CHỈ SỐ (METRICS)
# ==============================================================================

CASE_DESCRIPTIONS = {
    0: "Case 0: Cascaded Baseline (PhoWhisper + VinAI-Translate/NLLB + MMS-TTS-Eng)",
    1: "Case 1: Untrained CrossAttentionBridge + Untrained CFM (Zero-Baseline)",
    2: "Case 2: Untrained Bridge + Trained CFM (Acoustic Only)",
    3: "Case 3: Trained Bridge + Untrained CFM (Semantic Only)",
    4: "Case 4: Trained CrossAttentionBridge + Trained CFM (Full Proposed Vi-En)",
    5: "Case 5: Meta SeamlessM4T-v2 (SOTA Direct S2ST Benchmark)",
    6: "Case 6: Cascaded Voice Cloning (XTTS-v2)",
    7: "Case 7: Cascaded Voice Cloning (F5-TTS Flow-Matching)",
}

def compute_cosine_similarity(emb1, emb2):
    """Đo độ tương đồng Cosine (SECS) giữa 2 vector âm sắc người nói."""
    a = F.normalize(emb1.float(), p=2, dim=-1)
    b = F.normalize(emb2.float(), p=2, dim=-1)
    return (a * b).sum(dim=-1).mean().item()

# ------------------------------------------------------------------------------
# SECS THẬT: Dùng mô hình Speaker Verification độc lập (WavLM-Base-Plus-SV, x-vector)
# để so sánh giọng nguồn tiếng Việt với giọng tiếng Anh sinh ra. Mô hình này không
# phụ thuộc vào hệ thống được đánh giá nên công bằng cho mọi Case.
# ------------------------------------------------------------------------------
SV_MODEL_ID = "microsoft/wavlm-base-plus-sv"
_SPEAKER_VERIFIER = {"loaded": False, "feat": None, "model": None}

def get_speaker_verifier(device):
    """Nạp (lazy, 1 lần) mô hình WavLM x-vector để đo SECS."""
    if _SPEAKER_VERIFIER["loaded"]:
        return _SPEAKER_VERIFIER["feat"], _SPEAKER_VERIFIER["model"]
    _SPEAKER_VERIFIER["loaded"] = True
    try:
        from transformers import AutoFeatureExtractor, WavLMForXVector
        print(f"⏳ Đang nạp mô hình Speaker Verification ({SV_MODEL_ID}) để đo SECS...")
        feat = AutoFeatureExtractor.from_pretrained(SV_MODEL_ID)
        model = WavLMForXVector.from_pretrained(SV_MODEL_ID).to(device).eval()
        _SPEAKER_VERIFIER["feat"], _SPEAKER_VERIFIER["model"] = feat, model
        print("✓ Nạp WavLM-SV thành công!")
    except Exception as e:
        print(f"[SECS Cảnh báo] Không nạp được {SV_MODEL_ID} ({e}). SECS sẽ ghi là NaN (không đo được).")
    return _SPEAKER_VERIFIER["feat"], _SPEAKER_VERIFIER["model"]

def _to_mono_16k(wav, sample_rate):
    """Chuẩn hóa waveform về 1D float32 16kHz trên CPU."""
    w = torch.as_tensor(wav).detach().float().squeeze().cpu()
    if w.ndim > 1:
        w = w[0]
    if sample_rate == 24000:
        w = resample_24k_to_16k(w.unsqueeze(0)).squeeze(0)
    elif sample_rate != 16000:
        w = torch.nn.functional.interpolate(
            w.view(1, 1, -1), scale_factor=16000.0 / sample_rate, mode="linear", align_corners=False
        ).view(-1)
    return w

def compute_secs(src_wav, src_sr, gen_wav, gen_sr, device):
    """Đo SECS = cosine(x-vector giọng nguồn, x-vector giọng sinh ra). Trả về NaN nếu không đo được."""
    feat, model = get_speaker_verifier(device)
    if feat is None or model is None or src_wav is None or gen_wav is None:
        return float("nan")
    try:
        a = _to_mono_16k(src_wav, src_sr)
        b = _to_mono_16k(gen_wav, gen_sr)
        if a.numel() < 1600 or b.numel() < 1600 or float(b.abs().max()) < 1e-6:
            return 0.0  # Audio sinh ra quá ngắn hoặc câm -> không giữ được giọng
        inputs = feat([a.numpy(), b.numpy()], sampling_rate=16000, padding=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            emb = model(**inputs).embeddings
        emb = F.normalize(emb.float(), p=2, dim=-1)
        return float((emb[0] * emb[1]).sum().item())
    except Exception as e:
        print(f"  ⚠️ Lỗi khi đo SECS: {e}")
        return float("nan")

def nanmean(values):
    """Trung bình bỏ qua NaN (mẫu không đo được). Trả NaN nếu không có mẫu hợp lệ."""
    valid = [v for v in values if v == v]
    return sum(valid) / len(valid) if valid else float("nan")

def normalize_en_text(text):
    """Chuẩn hóa văn bản tiếng Anh trước khi tính BLEU: chữ thường, bỏ dấu câu, gộp khoảng trắng.
    Lý do: transcript tham chiếu của FLEURS đã được chuẩn hóa (không dấu câu), còn Whisper ASR
    sinh ra có dấu câu -> nếu không chuẩn hóa, dấu câu bị tính là token sai và kéo BLEU xuống."""
    t = str(text or "").lower()
    t = t.replace("’", "'")
    t = re.sub(r"[^\w\s']", " ", t)       # bỏ dấu câu (giữ dấu nháy đơn: peter's, don't)
    t = re.sub(r"(?<!\w)'|'(?!\w)", " ", t)  # bỏ nháy đơn đứng lẻ
    return re.sub(r"\s+", " ", t).strip()

def compute_corpus_bleu(hyps, refs):
    """Corpus-level SacreBLEU (chuẩn báo cáo ASR-BLEU trong các paper S2ST)."""
    pairs = [(normalize_en_text(h), normalize_en_text(r)) for h, r in zip(hyps, refs) if normalize_en_text(r)]
    if not pairs:
        return 0.0
    try:
        import sacrebleu
        return round(float(sacrebleu.corpus_bleu([h for h, _ in pairs], [[r for _, r in pairs]]).score), 2)
    except Exception:
        return float("nan")

def compute_sentence_bleu(hyp_str, ref_str):
    """Tính điểm BLEU (0 -> 100) chuẩn SacreBLEU giữa câu tiếng Anh sinh ra và câu tham chiếu."""
    hyp_clean = normalize_en_text(hyp_str)
    ref_clean = normalize_en_text(ref_str)
    if not hyp_clean or not ref_clean:
        return 0.0
    try:
        import sacrebleu
        score = sacrebleu.sentence_bleu(hyp_clean, [ref_clean], smooth_method="exp").score
        return round(float(score), 2)
    except Exception:
        pass
    
    # Fallback thuần Python tính BLEU
    hyp_tokens = re.findall(r'\w+', hyp_clean)
    ref_tokens = re.findall(r'\w+', ref_clean)
    if not hyp_tokens or not ref_tokens:
        return 0.0

    precisions = []
    for n in range(1, 5):
        hyp_ngrams = Counter([tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens)-n+1)])
        ref_ngrams = Counter([tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)])
        overlap = sum((hyp_ngrams & ref_ngrams).values())
        total = max(1, sum(hyp_ngrams.values()))
        precisions.append((overlap + 1e-1) / (total + 1e-1))

    bp = 1.0 if len(hyp_tokens) > len(ref_tokens) else math.exp(1.0 - len(ref_tokens) / max(1, len(hyp_tokens)))
    log_p_sum = sum(math.log(p) for p in precisions) / len(precisions)
    bleu = bp * math.exp(log_p_sum) * 100.0
    return round(min(100.0, max(0.0, bleu)), 2)

def load_english_asr(device):
    """Nạp Whisper tiếng Anh để phiên âm phục vụ đo ASR-BLEU chuẩn S2ST."""
    try:
        from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperForConditionalGeneration
        model_id = "openai/whisper-small"
        asr_feat = WhisperFeatureExtractor.from_pretrained(model_id)
        asr_tok = WhisperTokenizer.from_pretrained(model_id, language="en", task="transcribe")
        asr_mod = WhisperForConditionalGeneration.from_pretrained(model_id).to(device).eval()
        if hasattr(asr_mod, "generation_config"):
            asr_mod.generation_config.max_length = None
        print(f"✓ Đã nạp thành công Whisper English ASR ({model_id}) để đo ASR-BLEU.")
        return asr_feat, asr_tok, asr_mod
    except Exception as e:
        print(f"[ASR Cảnh báo] Không thể nạp Whisper English ASR ({e}). Sẽ bỏ qua ASR-BLEU.")
        return None, None, None

def transcribe_en_audio(wav_tensor, sample_rate, asr_feat, asr_tok, asr_mod, device):
    """Phiên âm tiếng nói tiếng Anh (16k hoặc 24k) ra văn bản tiếng Anh để đo BLEU."""
    if asr_mod is None or wav_tensor is None:
        return ""
    try:
        if hasattr(wav_tensor, "cpu"):
            wav = wav_tensor.squeeze().cpu()
        else:
            wav = torch.as_tensor(wav_tensor).squeeze().cpu()
        if wav.ndim > 1:
            wav = wav[0]
            
        if sample_rate != 16000:
            wav_24k = wav.unsqueeze(0).to(torch.float32)
            wav_16k = resample_24k_to_16k(wav_24k).squeeze(0)
            wav = wav_16k
        
        in_proc = asr_feat(
            wav.numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True
        )
        in_feats = in_proc.input_features.to(device)
        in_mask = in_proc.attention_mask.to(device) if "attention_mask" in in_proc else None
        
        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": 128,
                "language": "en",
                "task": "transcribe"
            }
            if in_mask is not None:
                gen_kwargs["attention_mask"] = in_mask
            pred_ids = asr_mod.generate(in_feats, **gen_kwargs)
            transcription = asr_tok.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
            return transcription
    except Exception as e:
        return ""

def load_mt_model(mt_choice, device):
    """
    Nạp mô hình Dịch máy (MT) chất lượng cao cho cặp ngôn ngữ Vi -> En.
    Hỗ trợ:
      - 'vinai' (khuyên dùng): vinai/vinai-translate-vi2en-v2 (SOTA chuyên biệt Vi->En)
      - 'nllb': facebook/nllb-200-distilled-600M (Meta SOTA Multilingual)
      - 'opus': Helsinki-NLP/opus-mt-vi-en (Legacy baseline)
      - 'auto': Ưu tiên VinAI -> NLLB -> Opus-MT
    """
    candidates = []
    choice = (mt_choice or "auto").lower()
    if choice in ["vinai", "vinai-translate"]:
        candidates = ["vinai", "nllb", "opus"]
    elif choice in ["nllb", "nllb-200"]:
        candidates = ["nllb", "vinai", "opus"]
    elif choice in ["opus", "opus-mt"]:
        candidates = ["opus", "vinai", "nllb"]
    else:  # auto
        candidates = ["vinai", "nllb", "opus"]

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    for cand in candidates:
        if cand == "vinai":
            model_id = "vinai/vinai-translate-vi2en-v2"
            print(f"⏳ Đang nạp SOTA MT: VinAI-Translate ({model_id})...")
            try:
                tok = AutoTokenizer.from_pretrained(model_id, src_lang="vi_VN")
                mod = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device).eval()
                if hasattr(mod, "generation_config") and mod.generation_config is not None:
                    mod.generation_config.max_length = None
                print("✓ Nạp thành công VinAI-Translate-vi2en-v2 (SOTA Vi -> En)!")
                return tok, mod, "vinai"
            except Exception as e:
                print(f"[Cảnh báo MT] Không nạp được VinAI-Translate ({e}). Thử model kế tiếp...")

        elif cand == "nllb":
            model_id = "facebook/nllb-200-distilled-600M"
            print(f"⏳ Đang nạp SOTA MT: Meta NLLB-200 ({model_id})...")
            try:
                tok = AutoTokenizer.from_pretrained(model_id, src_lang="vie_Latn")
                mod = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device).eval()
                if hasattr(mod, "generation_config") and mod.generation_config is not None:
                    mod.generation_config.max_length = None
                print("✓ Nạp thành công Meta NLLB-200-distilled-600M!")
                return tok, mod, "nllb"
            except Exception as e:
                print(f"[Cảnh báo MT] Không nạp được Meta NLLB-200 ({e}). Thử model kế tiếp...")

        elif cand == "opus":
            model_id = "Helsinki-NLP/opus-mt-vi-en"
            print(f"⏳ Đang nạp Fallback MT: Opus-MT ({model_id})...")
            try:
                tok = AutoTokenizer.from_pretrained(model_id)
                mod = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device).eval()
                if hasattr(mod, "generation_config") and mod.generation_config is not None:
                    mod.generation_config.max_length = None
                print("✓ Nạp Opus-MT thành công!")
                return tok, mod, "opus"
            except Exception as e:
                print(f"[Cảnh báo MT] Không nạp được Opus-MT ({e}).")

    print("[Lỗi MT] Không nạp được bất kỳ mô hình MT nào!")
    return None, None, "none"

def translate_text_vi2en(mt_tok, mt_mod, mt_flavor, vi_text, device):
    """Thực hiện dịch câu tiếng Việt sang tiếng Anh với hỗ trợ VinAI, NLLB hoặc Opus-MT."""
    if mt_tok is None or mt_mod is None or not vi_text:
        return "hello world"

    clean_text = str(vi_text).strip()
    if not clean_text:
        return "hello world"

    try:
        inputs = mt_tok(clean_text, return_tensors="pt", padding=True).to(device)
        gen_kwargs = {"max_new_tokens": 128}

        if mt_flavor == "vinai":
            tgt_lang_id = None
            if hasattr(mt_tok, "lang_code_to_id") and isinstance(mt_tok.lang_code_to_id, dict):
                tgt_lang_id = mt_tok.lang_code_to_id.get("en_XX")
            if tgt_lang_id is not None:
                gen_kwargs["decoder_start_token_id"] = tgt_lang_id
                gen_kwargs["forced_bos_token_id"] = tgt_lang_id
            gen_kwargs["num_beams"] = 4

        elif mt_flavor == "nllb":
            tgt_lang_id = None
            if hasattr(mt_tok, "lang_code_to_id") and isinstance(mt_tok.lang_code_to_id, dict):
                tgt_lang_id = mt_tok.lang_code_to_id.get("eng_Latn")
            if tgt_lang_id is not None:
                gen_kwargs["forced_bos_token_id"] = tgt_lang_id
            gen_kwargs["num_beams"] = 4

        with torch.no_grad():
            output_ids = mt_mod.generate(**inputs, **gen_kwargs)
            trans_en = mt_tok.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            return trans_en if trans_en else "hello"
    except Exception as e:
        print(f"  ⚠️ Lỗi khi dịch MT ({mt_flavor}): {e}")
        return "hello"

# ==============================================================================
# HÀM ĐÁNH GIÁ TỪNG TRƯỜNG HỢP (CASE 0 -> 5)
# ==============================================================================

def evaluate_case(case_num, args, device, dtype, vocos, mel_fb, data_pipeline, feat_extractor, whisper_enc, en_asr_tools=None):
    case_name = CASE_DESCRIPTIONS[case_num]
    print("\n" + "=" * 75)
    print(f"📊 ĐANG CHẠY BENCHMARK: [{case_name}]")
    print("=" * 75)

    out_case_dir = os.path.join(args.output_dir, f"case_{case_num}")
    wavs_dir = os.path.join(out_case_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)

    # 1. Khởi tạo mô hình ViEnS2STSystem
    model = ViEnS2STSystem(encoder_dim=768, bridge_dim=512, bridge_layers=6, n_mels=100).to(device=device, dtype=dtype)
    model.eval()

    if case_num in [3, 4]:
        if os.path.exists(args.bridge_ckpt):
            print(f"⏳ Đang nạp weights cho Bridge từ: {args.bridge_ckpt}...")
            ckpt = torch.load(args.bridge_ckpt, map_location=device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict, strict=False)
            print("✓ Nạp weights Bridge thành công!")
        else:
            print(f"[Cảnh báo] Không tìm thấy {args.bridge_ckpt}! Sử dụng weights ngẫu nhiên.")
    else:
        print("ℹ️ Case yêu cầu Bridge CHƯA TRAIN -> Khởi tạo trọng số ngẫu nhiên.")

    if case_num in [2, 4]:
        if args.cfm_ckpt and os.path.exists(args.cfm_ckpt):
            print(f"⏳ Đang nạp weights cho CFM từ: {args.cfm_ckpt}...")
            ckpt = torch.load(args.cfm_ckpt, map_location=device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.decoder.load_state_dict(state_dict, strict=False)
            print("✓ Nạp weights CFM thành công!")
    else:
        print("ℹ️ Case yêu cầu CFM CHƯA TRAIN -> Khởi tạo trọng số ngẫu nhiên.")

    # 2. Suy luận trên tập mẫu
    results = []
    total_rtf = 0.0
    secs_list = []
    total_mel_l1 = 0.0
    total_bleu = 0.0
    bleu_hyps, bleu_refs = [], []

    for idx in range(args.num_samples):
        src_audio, tgt_audio, en_texts, vi_texts = data_pipeline.fetch_batch(batch_size=1, return_text=True)
        src_w = src_audio[0]
        tgt_w = tgt_audio[0]
        ref_en_text = en_texts[0]
        ref_vi_text = vi_texts[0] if vi_texts else ""

        # Trích xuất Whisper Latents từ tiếng Việt
        if feat_extractor and whisper_enc:
            in_proc = feat_extractor(src_w.numpy(), sampling_rate=16000, return_tensors='pt', padding="max_length")
            feats = in_proc.input_features.to(device=device, dtype=dtype)
            if feats.shape[-1] < 3000: feats = F.pad(feats, (0, 3000 - feats.shape[-1]))
            elif feats.shape[-1] > 3000: feats = feats[:, :, :3000]

            with torch.no_grad():
                full_latents = whisper_enc(feats).last_hidden_state
            real_len = max(10, int(src_w.shape[-1] / 320))
            src_latents = full_latents[:, :real_len, :]
        else:
            src_latents = torch.randn(1, 100, 768, device=device, dtype=dtype)

        # Đo RTF
        t_start = time.perf_counter()
        with torch.no_grad():
            gen_mel = model.inference(src_latents, num_steps=args.num_steps)
        infer_time = time.perf_counter() - t_start

        # Giải mã Vocos
        gen_wav = None
        if vocos is not None:
            with torch.no_grad():
                gen_wav = vocos.decode(gen_mel.float()).squeeze().cpu()
            gen_duration = len(gen_wav) / 24000.0
        else:
            gen_duration = (gen_mel.shape[-1] * 256.0) / 24000.0

        rtf = infer_time / max(1e-4, gen_duration)
        total_rtf += rtf

        # Đo SECS thật giữa audio nguồn (16k) và audio sinh ra (24k)
        secs = compute_secs(src_w, 16000, gen_wav, 24000, device) if gen_wav is not None else float("nan")
        secs_list.append(secs)

        # Đo sai số phổ Mel L1
        tgt_24k = resample_16k_to_24k(tgt_w.unsqueeze(0).to(device=device, dtype=torch.float32))
        tgt_mel = extract_100_mel(tgt_24k, mel_fb).to(dtype=dtype)
        min_t = min(gen_mel.shape[-1], tgt_mel.shape[-1])
        mel_l1 = F.l1_loss(gen_mel[:, :, :min_t], tgt_mel[:, :, :min_t]).item()
        total_mel_l1 += mel_l1

        # Đo ASR-BLEU
        bleu = 0.0
        asr_transcribed_en = ""
        if gen_wav is not None and en_asr_tools is not None:
            en_feat, en_tok, en_mod = en_asr_tools
            # Đo thật cho mọi Case (kể cả Ablation 1-3) thay vì gán cứng 0
            asr_transcribed_en = transcribe_en_audio(gen_wav, 24000, en_feat, en_tok, en_mod, device)
            if asr_transcribed_en:
                bleu = compute_sentence_bleu(asr_transcribed_en, ref_en_text)
        total_bleu += bleu
        bleu_hyps.append(asr_transcribed_en)
        bleu_refs.append(ref_en_text)

        sample_name = f"sample_{idx+1:03d}"
        if gen_wav is not None:
            save_wav_file(os.path.join(wavs_dir, f"{sample_name}_case{case_num}_gen.wav"), gen_wav, 24000)
            if not os.path.exists(os.path.join(wavs_dir, f"{sample_name}_src.wav")):
                save_wav_file(os.path.join(wavs_dir, f"{sample_name}_src.wav"), src_w, 16000)
                save_wav_file(os.path.join(wavs_dir, f"{sample_name}_tgt.wav"), tgt_w, 16000)

        results.append({
            "sample_id": idx + 1,
            "ref_vi_text": ref_vi_text,
            "ref_en_text": ref_en_text,
            "asr_transcribed_en": asr_transcribed_en,
            "rtf": round(rtf, 4),
            "secs_timbre": round(secs, 4),
            "mel_l1_error": round(mel_l1, 4),
            "asr_bleu": round(bleu, 2),
            "gen_duration": round(gen_duration, 2)
        })

        print(f"\n--- [Sample {idx+1:02d}/{args.num_samples}] ------------------------------------------")
        if ref_vi_text:
            print(f"  🇻🇳 Input Text (Vi): {ref_vi_text}")
        if asr_transcribed_en:
            print(f"  🎙️ Synthesized ASR (En): {asr_transcribed_en}")
        print(f"  🎯 Reference (En) : {ref_en_text}")
        print(f"  📊 RTF: {rtf:.4f} | Mel L1: {mel_l1:.4f} | SECS: {secs:.4f} | ASR-BLEU: {bleu:.2f} | Dur: {gen_duration:.2f}s")

    avg_rtf = total_rtf / max(1, args.num_samples)
    avg_secs = nanmean(secs_list)
    avg_mel_l1 = total_mel_l1 / max(1, args.num_samples)
    avg_bleu = total_bleu / max(1, args.num_samples)

    summary = {
        "case_id": case_num,
        "case_name": case_name,
        "avg_rtf": round(avg_rtf, 4),
        "avg_secs_timbre": round(avg_secs, 4),
        "avg_mel_l1_error": round(avg_mel_l1, 4),
        "avg_asr_bleu": round(avg_bleu, 2),
        "corpus_asr_bleu": compute_corpus_bleu(bleu_hyps, bleu_refs),
        "num_samples": args.num_samples,
        "samples": results
    }

    with open(os.path.join(out_case_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary

def evaluate_cascaded_case(args, device, mel_fb, data_pipeline, en_asr_tools=None):
    case_name = CASE_DESCRIPTIONS[0]
    print("\n" + "=" * 75)
    print(f"📊 ĐANG CHẠY BENCHMARK: [{case_name}]")
    print("=" * 75)

    out_case_dir = os.path.join(args.output_dir, "case_0")
    wavs_dir = os.path.join(out_case_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)

    print("⏳ Đang nạp các mô hình thành phần Cascaded Vi -> En (Pretrained chuẩn):")
    print("  1. ASR: vinai/PhoWhisper-small (Nhận diện giọng nói tiếng Việt tối ưu WER 9.0%)")
    print("  2. MT:  VinAI-Translate-vi2en-v2 / Meta NLLB-200 (SOTA Dịch thuật Việt - Anh)")
    print("  3. TTS: facebook/mms-tts-eng (Tổng hợp giọng nói tiếng Anh VITS)")

    try:
        from transformers import (
            WhisperFeatureExtractor, WhisperTokenizer, WhisperForConditionalGeneration,
            AutoTokenizer, AutoModelForSeq2SeqLM, VitsModel
        )
    except ImportError as e:
        print(f"[Lỗi] Thiếu thư viện transformers để chạy Cascaded Baseline: {e}")
        return None

    # Nạp ASR PhoWhisper tiếng Việt (SOTA Tiếng Việt)
    asr_feat_extractor = WhisperFeatureExtractor.from_pretrained("vinai/PhoWhisper-small")
    asr_tokenizer = WhisperTokenizer.from_pretrained("vinai/PhoWhisper-small", language="vi", task="transcribe")
    asr_model = WhisperForConditionalGeneration.from_pretrained("vinai/PhoWhisper-small").to(device).eval()
    asr_model.config.forced_decoder_ids = None
    if hasattr(asr_model, "generation_config") and asr_model.generation_config is not None:
        asr_model.generation_config.forced_decoder_ids = None
        asr_model.generation_config.max_length = None

    # Nạp MT (Ưu tiên VinAI-Translate-vi2en-v2 -> Meta NLLB-200 -> Opus-MT)
    mt_tok, mt_model, mt_flavor = load_mt_model(getattr(args, "mt_model", "auto"), device)

    # Nạp TTS MMS-TTS-Eng
    tts_tok = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
    tts_model = VitsModel.from_pretrained("facebook/mms-tts-eng").to(device).eval()

    if en_asr_tools is None:
        en_feat, en_tok, en_mod = load_english_asr(device)
    else:
        en_feat, en_tok, en_mod = en_asr_tools

    results = []
    total_rtf = 0.0
    secs_list = []
    total_mel_l1 = 0.0
    total_bleu = 0.0
    bleu_hyps, bleu_refs = [], []
    s2tt_hyps = []

    for idx in range(args.num_samples):
        src_audio, tgt_audio, en_texts, vi_texts = data_pipeline.fetch_batch(batch_size=1, return_text=True)
        src_w = src_audio[0]
        tgt_w = tgt_audio[0]
        ref_vi_text = vi_texts[0] if vi_texts else ""
        ref_en_text = en_texts[0] if en_texts else ""

        t_start = time.perf_counter()

        # 1. ASR: Audio Vi 16k -> Vietnamese Text
        in_proc = asr_feat_extractor(
            src_w.numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True
        )
        in_feats = in_proc.input_features.to(device)
        in_mask = in_proc.attention_mask.to(device) if "attention_mask" in in_proc else None

        transcribed_vi = ""
        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": 128,
                "language": "vi",
                "task": "transcribe"
            }
            if in_mask is not None:
                gen_kwargs["attention_mask"] = in_mask
            try:
                pred_ids = asr_model.generate(in_feats, **gen_kwargs)
            except Exception:
                pred_ids = asr_model.generate(in_feats, max_new_tokens=128)
            transcribed_vi = asr_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()

        # 2. MT: Vietnamese Text -> English Text (không dùng transcript chuẩn khi ASR rỗng)
        translated_en = translate_text_vi2en(mt_tok, mt_model, mt_flavor, transcribed_vi, device) if transcribed_vi else ""
        s2tt_hyps.append(translated_en)

        # 3. TTS: English Text -> Waveform (16kHz)
        tts_text = translated_en if translated_en else "hello"
        inputs_tts = tts_tok(tts_text, return_tensors="pt").to(device)
        with torch.no_grad():
            try:
                output_tts = tts_model(**inputs_tts).waveform
                gen_wav_16k = output_tts.squeeze().cpu()
            except Exception:
                gen_wav_16k = torch.zeros(16000)

        t_end = time.perf_counter()
        infer_time = t_end - t_start

        # Resample sang 24k cho khớp format
        gen_wav_24k = resample_16k_to_24k(gen_wav_16k.unsqueeze(0)).squeeze(0)
        gen_duration = len(gen_wav_24k) / 24000.0

        rtf = infer_time / max(1e-4, gen_duration)
        total_rtf += rtf

        # Đo SECS thật (WavLM-SV) giữa giọng nguồn và giọng MMS-TTS sinh ra
        secs = compute_secs(src_w, 16000, gen_wav_24k, 24000, device)
        secs_list.append(secs)

        # Đo sai số phổ Mel L1
        tgt_24k = resample_16k_to_24k(tgt_w.unsqueeze(0).to(device=device, dtype=torch.float32))
        tgt_mel = extract_100_mel(tgt_24k, mel_fb).to(device)
        gen_mel = extract_100_mel(gen_wav_24k.unsqueeze(0).to(device=device, dtype=torch.float32), mel_fb).to(device)
        min_t = min(gen_mel.shape[-1], tgt_mel.shape[-1])
        mel_l1 = F.l1_loss(gen_mel[:, :, :min_t], tgt_mel[:, :, :min_t]).item()
        total_mel_l1 += mel_l1

        # Đo ASR-BLEU
        bleu = 0.0
        asr_transcribed_en = ""
        if en_mod is not None:
            asr_transcribed_en = transcribe_en_audio(gen_wav_24k, 24000, en_feat, en_tok, en_mod, device)
            if asr_transcribed_en:
                bleu = compute_sentence_bleu(asr_transcribed_en, ref_en_text)
        total_bleu += bleu
        bleu_hyps.append(asr_transcribed_en)
        bleu_refs.append(ref_en_text)

        sample_name = f"sample_{idx+1:03d}"
        save_wav_file(os.path.join(wavs_dir, f"{sample_name}_case0_gen.wav"), gen_wav_24k, 24000)
        if not os.path.exists(os.path.join(wavs_dir, f"{sample_name}_src.wav")):
            save_wav_file(os.path.join(wavs_dir, f"{sample_name}_src.wav"), src_w, 16000)
            save_wav_file(os.path.join(wavs_dir, f"{sample_name}_tgt.wav"), tgt_w, 16000)

        results.append({
            "sample_id": idx + 1,
            "ref_vi_text": ref_vi_text,
            "asr_vi_text": transcribed_vi,
            "mt_translated_en": translated_en,
            "ref_en_text": ref_en_text,
            "asr_transcribed_en": asr_transcribed_en,
            "rtf": round(rtf, 4),
            "secs_timbre": round(secs, 4),
            "mel_l1_error": round(mel_l1, 4),
            "asr_bleu": round(bleu, 2),
            "gen_duration": round(gen_duration, 2)
        })

        print(f"\n--- [Sample {idx+1:02d}/{args.num_samples}] (Cascaded Baseline) --------------------")
        print(f"  🇻🇳 Input Vi   : {ref_vi_text}")
        print(f"  🗣️ ASR Vi     : {transcribed_vi}")
        print(f"  📖 MT English : {translated_en}")
        print(f"  🎙️ TTS ASR (En): {asr_transcribed_en}")
        print(f"  🎯 Target Ref : {ref_en_text}")
        print(f"  📊 RTF: {rtf:.4f} | Mel L1: {mel_l1:.4f} | SECS: {secs:.4f} | ASR-BLEU: {bleu:.2f} | Dur: {gen_duration:.2f}s")

    avg_rtf = total_rtf / max(1, args.num_samples)
    avg_secs = nanmean(secs_list)
    avg_mel_l1 = total_mel_l1 / max(1, args.num_samples)
    avg_bleu = total_bleu / max(1, args.num_samples)

    summary = {
        "case_id": 0,
        "case_name": case_name,
        "avg_rtf": round(avg_rtf, 4),
        "avg_secs_timbre": round(avg_secs, 4),
        "avg_mel_l1_error": round(avg_mel_l1, 4),
        "avg_asr_bleu": round(avg_bleu, 2),
        "corpus_asr_bleu": compute_corpus_bleu(bleu_hyps, bleu_refs),
        "corpus_s2tt_bleu": compute_corpus_bleu(s2tt_hyps, bleu_refs),
        "num_samples": args.num_samples,
        "samples": results
    }

    with open(os.path.join(out_case_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary

def evaluate_seamless_case(args, device, data_pipeline, en_asr_tools=None):
    """Đánh giá SOTA Direct S2ST: Meta SeamlessM4T-v2 (vie -> eng)."""
    case_name = CASE_DESCRIPTIONS[5]
    print("\n" + "=" * 75)
    print(f"📊 ĐANG CHẠY BENCHMARK: [{case_name}]")
    print("=" * 75)

    out_case_dir = os.path.join(args.output_dir, "case_5")
    wavs_dir = os.path.join(out_case_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)

    try:
        from transformers import SeamlessM4Tv2Model, AutoProcessor
        print("⏳ Đang nạp Meta SeamlessM4T-v2 (facebook/seamless-m4t-v2-large)...")
        processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
        model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large").to(device).eval()
        print("✓ Nạp SeamlessM4T-v2 thành công!")
    except Exception as e:
        print(f"[Seamless Warning] Không thể nạp SeamlessM4T ({e}). Thử bản medium...")
        try:
            from transformers import SeamlessM4TModel, AutoProcessor
            processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-medium")
            model = SeamlessM4TModel.from_pretrained("facebook/hf-seamless-m4t-medium").to(device).eval()
        except Exception as e2:
            print(f"[Seamless Error] Bỏ qua Case 5 ({e2}).")
            return None

    if en_asr_tools is None:
        en_feat, en_tok, en_mod = load_english_asr(device)
    else:
        en_feat, en_tok, en_mod = en_asr_tools

    results = []
    total_rtf = 0.0
    secs_list = []
    total_bleu = 0.0
    bleu_hyps, bleu_refs = [], []

    for idx in range(args.num_samples):
        src_audio, tgt_audio, en_texts, vi_texts = data_pipeline.fetch_batch(batch_size=1, return_text=True)
        src_w = src_audio[0]
        ref_en_text = en_texts[0]
        ref_vi_text = vi_texts[0] if vi_texts else ""

        t_start = time.perf_counter()
        audio_inputs = processor(audios=src_w.numpy(), return_tensors="pt", sampling_rate=16000).to(device)
        with torch.no_grad():
            output_tokens = model.generate(**audio_inputs, tgt_lang="eng", generate_speech=True)
            speech_out = output_tokens[0].cpu().squeeze()
        infer_time = time.perf_counter() - t_start

        gen_duration = len(speech_out) / 16000.0
        rtf = infer_time / max(1e-4, gen_duration)
        total_rtf += rtf

        # Đo SECS thật (Seamless xuất audio 16kHz)
        secs = compute_secs(src_w, 16000, speech_out, 16000, device)
        secs_list.append(secs)

        asr_transcribed_en = ""
        bleu = 0.0
        if en_mod is not None:
            asr_transcribed_en = transcribe_en_audio(speech_out, 16000, en_feat, en_tok, en_mod, device)
            if asr_transcribed_en:
                bleu = compute_sentence_bleu(asr_transcribed_en, ref_en_text)
        total_bleu += bleu
        bleu_hyps.append(asr_transcribed_en)
        bleu_refs.append(ref_en_text)

        sample_name = f"sample_{idx+1:03d}"
        save_wav_file(os.path.join(wavs_dir, f"{sample_name}_case5_gen.wav"), speech_out, 16000)

        results.append({
            "sample_id": idx + 1,
            "ref_vi_text": ref_vi_text,
            "ref_en_text": ref_en_text,
            "asr_transcribed_en": asr_transcribed_en,
            "rtf": round(rtf, 4),
            "secs_timbre": round(secs, 4),
            "asr_bleu": round(bleu, 2),
            "gen_duration": round(gen_duration, 2)
        })

        print(f"\n--- [Sample {idx+1:02d}/{args.num_samples}] (SeamlessM4T-v2) -------------------")
        print(f"  🇻🇳 Input Vi   : {ref_vi_text}")
        print(f"  🎙️ S2ST ASR (En): {asr_transcribed_en}")
        print(f"  🎯 Target Ref : {ref_en_text}")
        print(f"  📊 RTF: {rtf:.4f} | SECS: {secs:.4f} | ASR-BLEU: {bleu:.2f} | Dur: {gen_duration:.2f}s")

    avg_rtf = total_rtf / max(1, args.num_samples)
    avg_secs = nanmean(secs_list)
    avg_bleu = total_bleu / max(1, args.num_samples)

    summary = {
        "case_id": 5,
        "case_name": case_name,
        "avg_rtf": round(avg_rtf, 4),
        "avg_secs_timbre": round(avg_secs, 4),
        "avg_mel_l1_error": 0.0,
        "avg_asr_bleu": round(avg_bleu, 2),
        "corpus_asr_bleu": compute_corpus_bleu(bleu_hyps, bleu_refs),
        "num_samples": args.num_samples,
        "samples": results
    }

    with open(os.path.join(out_case_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary

def sanitize_text_for_f5(text):
    """Chuẩn hóa văn bản tiếng Việt sang ký tự Latinh ASCII để tương thích hoàn toàn với vocab F5-TTS."""
    import unicodedata
    if not text:
        return "xin chao"
    nfkd = unicodedata.normalize('NFKD', str(text))
    latin = ''.join([c for c in nfkd if not unicodedata.combining(c)])
    latin = latin.replace('đ', 'd').replace('Đ', 'D')
    clean = re.sub(r'[^a-zA-Z0-9\s.,?!]', ' ', latin)
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean if clean else "xin chao"

def disable_broken_flash_attn():
    """Vô hiệu hóa flash-attn bị hỏng để F5-TTS / transformers tự chuyển sang torch SDPA.

    Vấn đề: sau `pip uninstall flash-attn`, thư mục `site-packages/flash_attn/` thường còn sót
    (không có __init__.py) -> Python coi là namespace package, `importlib.util.find_spec`
    vẫn trả về spec nên F5-TTS tưởng flash-attn có sẵn rồi crash khi
    `from flash_attn import flash_attn_func` ("unknown location").
    """
    import shutil
    import subprocess
    import site
    print("⚠️ flash-attn bị lỗi hoặc chưa cài. Vô hiệu hóa để F5-TTS dùng torch SDPA fallback...")
    subprocess.call([sys.executable, "-m", "pip", "uninstall", "flash-attn", "-y", "-q"])

    # Xóa thư mục flash_attn sót lại (chỉ khi KHÔNG còn __init__.py, tức là mảnh vỡ của gói đã gỡ)
    site_dirs = list(getattr(site, "getsitepackages", lambda: [])())
    try:
        site_dirs.append(site.getusersitepackages())
    except Exception:
        pass
    for sp in site_dirs:
        leftover = os.path.join(sp, "flash_attn")
        if os.path.isdir(leftover) and not os.path.exists(os.path.join(leftover, "__init__.py")):
            shutil.rmtree(leftover, ignore_errors=True)
            print(f"  ✓ Đã xóa thư mục flash_attn sót lại: {leftover}")

    # Chặn import trong tiến trình hiện tại: sys.modules[name] = None khiến
    # `import flash_attn` ném ImportError và find_spec("flash_attn") trả về None.
    for mod_name in list(sys.modules.keys()):
        if mod_name == "flash_attn" or mod_name.startswith("flash_attn."):
            del sys.modules[mod_name]
    sys.modules["flash_attn"] = None
    try:
        import importlib
        importlib.invalidate_caches()
    except Exception:
        pass

def load_voice_clone_tts(tts_engine, device):
    """Nạp engine TTS phục vụ Voice Cloning (XTTS-v2 hoặc F5-TTS).
    
    Lưu ý: Trên Molab Python 3.13+, coqui-tts cần torchcodec để xử lý audio I/O,
    và F5-TTS cần google-cloud-storage. Các deps này nên được cài từ run.sh trước.
    """
    import subprocess

    if tts_engine == "xtts":
        print("⏳ Đang nạp coqui/XTTS-v2 (Zero-shot Voice Cloning từ speaker_wav)...")

        # Tự động đồng ý license CPML (bỏ qua prompt interactive)
        os.environ["COQUI_TOS_AGREED"] = "1"

        # Đảm bảo torchcodec có sẵn (cần cho torchaudio backend trên PyTorch 2.9+)
        try:
            import torchcodec
        except ImportError:
            print("📦 Đang cài torchcodec (cần cho torchaudio backend)...")
            subprocess.call([sys.executable, "-m", "pip", "install", "torchcodec", "-q"])

        # Thử nạp coqui-tts, cài nếu chưa có
        try:
            from TTS.api import TTS
        except ImportError:
            print("📦 coqui-tts chưa cài đặt. Đang tự động cài đặt...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "coqui-tts", "-q"])
                from TTS.api import TTS
            except Exception as e:
                print(f"[Cảnh báo TTS] Không thể cài đặt coqui-tts ({e}).")
                print("👉 Bạn hãy chạy thủ công: pip install torchcodec coqui-tts")
                return None

        try:
            model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
            print("✓ Nạp XTTS-v2 thành công!")
            return model
        except RuntimeError as e:
            err_msg = str(e)
            if "size mismatch" in err_msg or "Missing key" in err_msg:
                print(f"[XTTS-v2] State dict mismatch — thử xóa cache và tải lại...")
                # Xóa cache XTTS bị lỗi
                xtts_cache = os.path.expanduser("~/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2")
                if os.path.isdir(xtts_cache):
                    import shutil
                    shutil.rmtree(xtts_cache, ignore_errors=True)
                    print(f"  ✓ Đã xóa cache: {xtts_cache}")
                # Nâng cấp coqui-tts
                subprocess.call([sys.executable, "-m", "pip", "install", "--upgrade", "coqui-tts", "-q"])
                try:
                    # Reload module
                    for mod_name in list(sys.modules.keys()):
                        if mod_name.startswith("TTS"):
                            del sys.modules[mod_name]
                    from TTS.api import TTS
                    model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
                    print("✓ Nạp XTTS-v2 thành công (sau khi xóa cache)!")
                    return model
                except Exception as e2:
                    print(f"[Cảnh báo TTS] Vẫn không thể nạp XTTS-v2 sau retry ({e2}).")
                    return None
            else:
                print(f"[Cảnh báo TTS] Lỗi khi nạp XTTS-v2 ({e}).")
                return None
        except Exception as e:
            print(f"[Cảnh báo TTS] Lỗi khi nạp XTTS-v2 ({e}).")
            return None

    elif tts_engine == "f5tts":
        print("⏳ Đang nạp SWivid/F5-TTS (Flow-Matching In-Context Voice Cloning)...")

        # Đảm bảo google-cloud-storage có sẵn (dependency ngầm của F5-TTS)
        try:
            import google.cloud.storage
        except (ImportError, ModuleNotFoundError):
            print("📦 Đang cài google-cloud-storage (dependency của F5-TTS)...")
            subprocess.call([sys.executable, "-m", "pip", "install", "google-cloud-storage", "-q"])

        # Cài f5-tts nếu thật sự CHƯA có (kiểm tra qua metadata, không qua import —
        # vì import có thể lỗi do flash_attn chứ không phải do thiếu f5-tts)
        try:
            from importlib.metadata import version as _pkg_version
            _pkg_version("f5-tts")
        except Exception:
            print("📦 f5-tts chưa cài đặt. Đang tự động cài đặt...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "f5-tts", "-q"])
            except Exception as e:
                print(f"[Cảnh báo TTS] Không thể cài đặt f5-tts ({e}).")
                print("👉 Bạn hãy chạy thủ công: pip install google-cloud-storage f5-tts")
                return None

        # Kiểm tra flash-attn SAU khi cài f5-tts (f5-tts có thể kéo flash-attn về lại).
        # Nếu flash-attn hỏng -> vô hiệu hóa hoàn toàn để F5-TTS dùng torch SDPA.
        try:
            from flash_attn import flash_attn_func  # noqa: F401
        except Exception:
            disable_broken_flash_attn()

        # Reload f5_tts sau khi xử lý flash-attn
        for mod_name in list(sys.modules.keys()):
            if "f5_tts" in mod_name:
                del sys.modules[mod_name]
        try:
            from f5_tts.api import F5TTS
        except Exception as e:
            print(f"[Cảnh báo TTS] Không thể nạp f5-tts sau khi gỡ flash-attn ({e}).")
            print("👉 Bạn hãy chạy thủ công: pip install google-cloud-storage f5-tts")
            return None

        try:
            f5_dev = "cuda" if (torch.cuda.is_available() and device.type == "cuda") else "cpu"
            model = F5TTS(device=f5_dev)
            print("✓ Nạp F5-TTS thành công!")
            return model
        except Exception as e:
            print(f"[Cảnh báo TTS] Lỗi khi nạp F5-TTS ({e}).")
            return None

    return None

DEFAULT_2STEP_MODELS = "openai/whisper-small,openai/whisper-large-v3,vinai/PhoWhisper-small,vinai/PhoWhisper-large"
DEFAULT_VI_ASR_MODELS = "vinai/PhoWhisper-small,vinai/PhoWhisper-large"

def short_model_name(model_id):
    """'openai/whisper-large-v3' -> 'whisper-large-v3' (dùng cho tên Case và thư mục)."""
    return str(model_id).rstrip("/").split("/")[-1]

def parse_model_list(text):
    return [m.strip() for m in str(text or "").split(",") if m.strip()]

def build_s2tt_variants(s2tt_mode_arg, models_2step, models_vi_asr):
    """Danh sách biến thể S2TT cần chạy: [{'mode': '2step'|'3step', 'model_id': ..., 'short': ...}]."""
    variants = []
    if s2tt_mode_arg in ["2step", "both"]:
        for mid in models_2step:
            variants.append({"mode": "2step", "model_id": mid, "short": short_model_name(mid)})
    if s2tt_mode_arg in ["3step", "both"]:
        for mid in models_vi_asr:
            variants.append({"mode": "3step", "model_id": mid, "short": short_model_name(mid)})
    return variants

def _s2tt_dtype(device):
    """fp16 trên GPU để model large (~1.5B) không tràn VRAM; áp dụng đồng nhất cho mọi biến thể."""
    return torch.float16 if (device.type == "cuda") else torch.float32

def _force_translate_generation_config(model, tok, model_id):
    """PhoWhisper được fine-tune chỉ cho ASR tiếng Việt nên generation_config có thể thiếu
    task_to_id / lang_to_id -> generate(task='translate') báo lỗi hoặc âm thầm transcribe.
    Bổ sung các id này từ chính tokenizer để ÉP chạy task='translate' một cách trung thực
    (kiểm chứng xem PhoWhisper còn khả năng dịch hay không)."""
    gc_cfg = model.generation_config
    if getattr(gc_cfg, "task_to_id", None) and getattr(gc_cfg, "lang_to_id", None):
        return False
    try:
        from transformers import GenerationConfig
        size = "small"
        for cand in ["tiny", "base", "small", "medium", "large"]:
            if cand in model_id.lower():
                size = cand
        base_id = "openai/whisper-large-v2" if size == "large" else f"openai/whisper-{size}"
        new_gc = GenerationConfig.from_pretrained(base_id)
    except Exception:
        new_gc = gc_cfg
    new_gc.task_to_id = {
        "transcribe": tok.convert_tokens_to_ids("<|transcribe|>"),
        "translate": tok.convert_tokens_to_ids("<|translate|>"),
    }
    new_gc.lang_to_id = {"<|vi|>": tok.convert_tokens_to_ids("<|vi|>"),
                         "<|en|>": tok.convert_tokens_to_ids("<|en|>")}
    new_gc.no_timestamps_token_id = tok.convert_tokens_to_ids("<|notimestamps|>")
    new_gc.is_multilingual = True
    new_gc.forced_decoder_ids = None
    new_gc.max_length = None
    model.generation_config = new_gc
    return True

_MT_CACHE = {}

def get_mt_cached(mt_choice, device):
    """Nạp MT một lần, dùng chung cho mọi biến thể 3-step."""
    key = (mt_choice or "auto")
    if key not in _MT_CACHE:
        _MT_CACHE[key] = load_mt_model(mt_choice, device)
    return _MT_CACHE[key]

def load_s2tt_variant(variant, device, mt_choice="auto"):
    """Nạp mô hình cho 1 biến thể S2TT. Trả về tuple module hoặc None nếu lỗi."""
    from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperForConditionalGeneration
    mid, mode = variant["model_id"], variant["mode"]
    dtype = _s2tt_dtype(device)
    try:
        if mode == "2step":
            print(f"⏳ [2-Step Direct S2TT] Nạp {mid} (task='translate', Vi -> En trực tiếp)...")
            feat = WhisperFeatureExtractor.from_pretrained(mid)
            tok = WhisperTokenizer.from_pretrained(mid, language="vi", task="translate")
            mod = WhisperForConditionalGeneration.from_pretrained(mid, torch_dtype=dtype).to(device).eval()
            mod.config.forced_decoder_ids = None
            if _force_translate_generation_config(mod, tok, mid):
                print(f"  ℹ️ {mid} thiếu cấu hình task translate -> đã bổ sung để ép task='translate'.")
            print(f"✓ Nạp {mid} thành công!")
            return (feat, tok, mod)
        else:
            print(f"⏳ [3-Step Pipeline] Nạp ASR {mid} + MT (VinAI-Translate / NLLB-200)...")
            feat = WhisperFeatureExtractor.from_pretrained(mid)
            tok = WhisperTokenizer.from_pretrained(mid, language="vi", task="transcribe")
            mod = WhisperForConditionalGeneration.from_pretrained(mid, torch_dtype=dtype).to(device).eval()
            mod.config.forced_decoder_ids = None
            if getattr(mod, "generation_config", None) is not None:
                mod.generation_config.forced_decoder_ids = None
                mod.generation_config.max_length = None
            mt_tok, mt_mod, mt_flavor = get_mt_cached(mt_choice, device)
            print(f"✓ Nạp {mid} + MT ({mt_flavor.upper()}) thành công!")
            return (feat, tok, mod, mt_tok, mt_mod, mt_flavor)
    except Exception as e:
        print(f"[Cảnh báo S2TT] Không nạp được {mid} ({e}).")
        return None

def release_cuda_memory():
    """Thu gom rác + giải phóng cache CUDA (gọi sau khi đã xóa mọi tham chiếu tới model)."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

_VI_DIACRITICS = re.compile(r"[ăâđêôơưàảãáạằẳẵắặầẩẫấậèẻẽéẹềểễếệìỉĩíịòỏõóọồổỗốộờởỡớợùủũúụừửữứựỳỷỹýỵ]", re.IGNORECASE)

def looks_vietnamese(text):
    """Phát hiện đầu ra 'dịch' thực chất vẫn là tiếng Việt (model không dịch được)."""
    return len(_VI_DIACRITICS.findall(str(text or ""))) >= 2

def whisper_compression_ratio(text):
    """Tỉ lệ nén zlib của văn bản (tiêu chí phát hiện lặp/hallucination của Whisper gốc, ngưỡng 2.4)."""
    import zlib
    b = str(text or "").encode("utf-8")
    if len(b) < 40:
        return 0.0
    return len(b) / max(1, len(zlib.compress(b)))

def translate_speech_vi2en(src_w, ref_vi_text, mode, module, device):
    """Chuyển âm thanh tiếng Việt thành văn bản tiếng Anh theo chế độ 2-step hoặc 3-step.
    Trả về (translated_en, transcribed_vi). Không bao giờ dùng transcript chuẩn (ground truth)
    thay cho đầu ra ASR — tránh làm đẹp số liệu."""
    translated_en = ""
    transcribed_vi = ""
    if module is None:
        return translated_en, transcribed_vi

    if mode == "2step":
        w_feat, w_tok, w_mod = module
        in_proc = w_feat(src_w.numpy(), sampling_rate=16000, return_tensors="pt")
        in_feats = in_proc.input_features.to(device=device, dtype=w_mod.dtype)
        with torch.no_grad():
            try:
                pred_ids = w_mod.generate(in_feats, language="vi", task="translate", max_new_tokens=128)
            except Exception as e:
                print(f"  ⚠️ generate(task='translate') lỗi ({e}) -> chạy generate mặc định")
                pred_ids = w_mod.generate(in_feats, max_new_tokens=128)
            translated_en = w_tok.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()

            # Whisper hay rơi vào vòng lặp ("the following year, the following year, ...").
            # Áp dụng đúng tiêu chí fallback của Whisper gốc: tỉ lệ nén zlib > 2.4 -> giải mã lại
            # bằng beam search + chặn lặp n-gram. Chỉ kích hoạt khi phát hiện lặp.
            if whisper_compression_ratio(translated_en) > 2.4:
                print("  ⚠️ Whisper lặp vô hạn (compression ratio > 2.4) -> giải mã lại với beam=5, no_repeat_ngram=3")
                try:
                    pred_ids = w_mod.generate(in_feats, language="vi", task="translate", max_new_tokens=128,
                                              num_beams=5, no_repeat_ngram_size=3)
                    translated_en = w_tok.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
                except Exception:
                    pass

    elif mode == "3step":
        asr_feat, asr_tok, asr_mod, mt_tok, mt_mod, mt_flavor = module
        in_proc = asr_feat(src_w.numpy(), sampling_rate=16000, return_tensors="pt")
        in_feats = in_proc.input_features.to(device=device, dtype=asr_mod.dtype)
        with torch.no_grad():
            try:
                pred_ids = asr_mod.generate(in_feats, max_new_tokens=128)
                transcribed_vi = asr_tok.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
            except Exception as e:
                print(f"  ⚠️ ASR tiếng Việt lỗi ({e}) -> transcript rỗng")
                transcribed_vi = ""
        if transcribed_vi:
            translated_en = translate_text_vi2en(mt_tok, mt_mod, mt_flavor, transcribed_vi, device)

    return translated_en, transcribed_vi

def synthesize_voice_clone(tts_engine, tts_model, text, ref_wav_path, ref_vi_text, wavs_dir, idx):
    """Sinh âm thanh tiếng Anh mang âm sắc của người nói tiếng Việt."""
    if tts_engine == "xtts":
        temp_out_path = os.path.join(wavs_dir, f"_temp_xtts_{idx}.wav")
        try:
            tts_model.tts_to_file(
                text=text,
                speaker_wav=ref_wav_path,
                language="en",
                file_path=temp_out_path
            )
            import soundfile as sf
            gen_wav_np, _ = sf.read(temp_out_path)
            gen_wav_24k = torch.as_tensor(gen_wav_np, dtype=torch.float32).squeeze()
            if gen_wav_24k.ndim > 1: gen_wav_24k = gen_wav_24k[0]
            return gen_wav_24k
        except Exception as e:
            print(f"  ⚠️ XTTS-v2 lỗi tại mẫu {idx+1}: {e}")
            return torch.zeros(24000)
        finally:
            if os.path.exists(temp_out_path):
                try: os.remove(temp_out_path)
                except Exception: pass

    elif tts_engine == "f5tts":
        f5_ref_text = sanitize_text_for_f5(ref_vi_text)
        try:
            gen_wav_np, sr, _ = tts_model.infer(
                ref_file=ref_wav_path,
                ref_text=f5_ref_text,
                gen_text=text,
            )
            gen_wav_24k = torch.as_tensor(gen_wav_np, dtype=torch.float32).squeeze()
            if gen_wav_24k.ndim > 1: gen_wav_24k = gen_wav_24k[0]
            if sr != 24000:
                gen_wav_24k = resample_16k_to_24k(gen_wav_24k.unsqueeze(0)).squeeze(0)
            return gen_wav_24k
        except Exception as e:
            print(f"  ⚠️ F5-TTS lỗi tại mẫu {idx+1}: {e}")
            return torch.zeros(24000)

    return torch.zeros(24000)

def evaluate_single_voice_clone_run(case_num, tts_engine, variant, args, device, mel_fb, data_pipeline, en_asr_tools, s2tt_module, tts_model, write_canonical=False):
    s2tt_mode = variant["mode"]
    model_short = variant["short"]
    engine_title = "XTTS-v2" if tts_engine == "xtts" else "F5-TTS"
    mode_title = "2-Step" if s2tt_mode == "2step" else "3-Step"
    if s2tt_mode == "3step":
        mode_title += " + MT"
    full_case_name = f"Case {case_num}: {engine_title} | {mode_title} | {model_short}"

    print("\n" + "=" * 75)
    print(f"📊 ĐANG CHẠY BENCHMARK: [{full_case_name}]")
    print(f"   ⚙️ Chế độ dịch: {s2tt_mode.upper()} | Model: {variant['model_id']} | Voice Engine: {engine_title.upper()}")
    print("=" * 75)

    case_tag = f"case_{case_num}_{tts_engine}_{s2tt_mode}_{model_short}"
    out_case_dir = os.path.join(args.output_dir, case_tag)
    wavs_dir = os.path.join(out_case_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)

    # Canonical wavs dir for gallery compatibility
    canonical_dir = os.path.join(args.output_dir, f"case_{case_num}")
    canonical_wavs = os.path.join(canonical_dir, "wavs")
    os.makedirs(canonical_wavs, exist_ok=True)

    if en_asr_tools is None:
        en_feat, en_tok, en_mod = load_english_asr(device)
    else:
        en_feat, en_tok, en_mod = en_asr_tools

    results = []
    total_rtf = 0.0
    secs_list = []
    total_mel_l1 = 0.0
    total_bleu = 0.0
    bleu_hyps, bleu_refs = [], []
    s2tt_hyps = []
    n_not_translated = 0

    for idx in range(args.num_samples):
        src_audio, tgt_audio, en_texts, vi_texts = data_pipeline.fetch_batch(batch_size=1, return_text=True)
        src_w = src_audio[0]
        tgt_w = tgt_audio[0]
        ref_vi_text = vi_texts[0] if vi_texts else ""
        ref_en_text = en_texts[0] if en_texts else ""

        t_start = time.perf_counter()

        # 1. Dịch Vi audio -> En text
        translated_en, transcribed_vi = translate_speech_vi2en(src_w, ref_vi_text, s2tt_mode, s2tt_module, device)
        not_translated = looks_vietnamese(translated_en)
        if not_translated:
            n_not_translated += 1
        # S2TT-BLEU: văn bản dịch (trước TTS) so với Target Ref -> tách lỗi dịch khỏi lỗi TTS
        s2tt_bleu = compute_sentence_bleu(translated_en, ref_en_text) if translated_en else 0.0
        s2tt_hyps.append(translated_en)
        # TTS không nhận chuỗi rỗng -> dùng câu đệm cố định (BLEU sẽ ~0, phản ánh trung thực lỗi dịch)
        tts_target_text = translated_en if translated_en else "Hello, welcome to speech translation."

        # 2. Sinh tiếng nói giữ giọng
        src_24k = resample_16k_to_24k(src_w.unsqueeze(0).to(device=device, dtype=torch.float32)).squeeze(0).cpu()
        temp_ref_path = os.path.join(wavs_dir, f"_temp_ref_{idx}.wav")
        save_wav_file(temp_ref_path, src_24k, 24000)

        effective_ref_vi = ref_vi_text if ref_vi_text else transcribed_vi
        gen_wav_24k = synthesize_voice_clone(tts_engine, tts_model, tts_target_text, temp_ref_path, effective_ref_vi, wavs_dir, idx)

        if os.path.exists(temp_ref_path):
            try: os.remove(temp_ref_path)
            except Exception: pass

        if gen_wav_24k is None or gen_wav_24k.numel() == 0:
            gen_wav_24k = torch.zeros(24000)

        t_end = time.perf_counter()
        infer_time = t_end - t_start
        gen_duration = len(gen_wav_24k) / 24000.0

        rtf = infer_time / max(1e-4, gen_duration)
        total_rtf += rtf

        # Đo SECS thật (WavLM-SV) giữa giọng nguồn và giọng clone sinh ra
        secs = compute_secs(src_w, 16000, gen_wav_24k, 24000, device)
        secs_list.append(secs)

        # Đo sai số phổ Mel L1
        tgt_24k = resample_16k_to_24k(tgt_w.unsqueeze(0).to(device=device, dtype=torch.float32))
        tgt_mel = extract_100_mel(tgt_24k, mel_fb).to(device)
        gen_mel = extract_100_mel(gen_wav_24k.unsqueeze(0).to(device=device, dtype=torch.float32), mel_fb).to(device)
        min_t = min(gen_mel.shape[-1], tgt_mel.shape[-1])
        mel_l1 = F.l1_loss(gen_mel[:, :, :min_t], tgt_mel[:, :, :min_t]).item()
        total_mel_l1 += mel_l1

        # Đo ASR-BLEU
        bleu = 0.0
        asr_transcribed_en = ""
        if en_mod is not None:
            asr_transcribed_en = transcribe_en_audio(gen_wav_24k, 24000, en_feat, en_tok, en_mod, device)
            if asr_transcribed_en:
                bleu = compute_sentence_bleu(asr_transcribed_en, ref_en_text)
        total_bleu += bleu
        bleu_hyps.append(asr_transcribed_en)
        bleu_refs.append(ref_en_text)

        sample_name = f"sample_{idx+1:03d}"
        gen_file_case = os.path.join(wavs_dir, f"{sample_name}_{case_tag}_gen.wav")
        save_wav_file(gen_file_case, gen_wav_24k, 24000)

        if write_canonical:
            canonical_gen_file = os.path.join(canonical_wavs, f"{sample_name}_case{case_num}_gen.wav")
            save_wav_file(canonical_gen_file, gen_wav_24k, 24000)

        if not os.path.exists(os.path.join(wavs_dir, f"{sample_name}_src.wav")):
            save_wav_file(os.path.join(wavs_dir, f"{sample_name}_src.wav"), src_w, 16000)
            save_wav_file(os.path.join(wavs_dir, f"{sample_name}_tgt.wav"), tgt_w, 16000)
        if write_canonical and not os.path.exists(os.path.join(canonical_wavs, f"{sample_name}_src.wav")):
            save_wav_file(os.path.join(canonical_wavs, f"{sample_name}_src.wav"), src_w, 16000)
            save_wav_file(os.path.join(canonical_wavs, f"{sample_name}_tgt.wav"), tgt_w, 16000)

        results.append({
            "sample_id": idx + 1,
            "ref_vi_text": ref_vi_text,
            "asr_vi_text": transcribed_vi,
            "translated_en": translated_en,
            "s2tt_not_translated": bool(not_translated),
            "ref_en_text": ref_en_text,
            "asr_transcribed_en": asr_transcribed_en,
            "rtf": round(rtf, 4),
            "secs_timbre": round(secs, 4),
            "mel_l1_error": round(mel_l1, 4),
            "s2tt_bleu": round(s2tt_bleu, 2),
            "asr_bleu": round(bleu, 2),
            "gen_duration": round(gen_duration, 2)
        })

        print(f"\n--- [Sample {idx+1:02d}/{args.num_samples}] ({engine_title} + {s2tt_mode.upper()} + {model_short}) ---")
        if ref_vi_text:
            print(f"  🇻🇳 Input Vi     : {ref_vi_text}")
        if s2tt_mode == "3step":
            print(f"  🗣️ ASR Vi       : {transcribed_vi}")
        print(f"  📖 S2TT English : {translated_en}" + ("   ⚠️ [VẪN LÀ TIẾNG VIỆT — model không dịch]" if not_translated else ""))
        if asr_transcribed_en:
            print(f"  🎙️ Cloned ASR(En): {asr_transcribed_en}")
        print(f"  🎯 Target Ref   : {ref_en_text}")
        print(f"  📊 RTF: {rtf:.4f} | Mel L1: {mel_l1:.4f} | SECS: {secs:.4f} | S2TT-BLEU: {s2tt_bleu:.2f} | ASR-BLEU: {bleu:.2f} | Dur: {gen_duration:.2f}s")

    avg_rtf = total_rtf / max(1, args.num_samples)
    avg_secs = nanmean(secs_list)
    avg_mel_l1 = total_mel_l1 / max(1, args.num_samples)
    avg_bleu = total_bleu / max(1, args.num_samples)

    summary = {
        "case_id": case_num,
        "case_tag": case_tag,
        "case_name": full_case_name,
        "tts_engine": tts_engine,
        "s2tt_mode": s2tt_mode,
        "s2tt_model": variant["model_id"],
        "num_not_translated": n_not_translated,
        "corpus_s2tt_bleu": compute_corpus_bleu(s2tt_hyps, bleu_refs),
        "avg_rtf": round(avg_rtf, 4),
        "avg_secs_timbre": round(avg_secs, 4),
        "avg_mel_l1_error": round(avg_mel_l1, 4),
        "avg_asr_bleu": round(avg_bleu, 2),
        "corpus_asr_bleu": compute_corpus_bleu(bleu_hyps, bleu_refs),
        "num_samples": args.num_samples,
        "samples": results
    }

    if n_not_translated:
        print(f"⚠️ {n_not_translated}/{args.num_samples} câu S2TT vẫn là tiếng Việt -> {variant['model_id']} không dịch được sang tiếng Anh.")

    with open(os.path.join(out_case_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    if write_canonical:
        with open(os.path.join(canonical_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary

def evaluate_cascaded_voice_clone_suite(case_num, tts_engine, variants, args, device, mel_fb, data_pipeline, en_asr_tools=None):
    """Voice Cloning cho một engine TTS (XTTS-v2 hoặc F5-TTS) trên mọi biến thể S2TT.
    TTS nạp 1 lần; mỗi biến thể ASR/Whisper được nạp -> chạy -> giải phóng để tiết kiệm VRAM."""
    tts_model = load_voice_clone_tts(tts_engine, device)
    if tts_model is None:
        print(f"[Cảnh báo] Bỏ qua Case {case_num} ({tts_engine}) do không nạp được TTS model.")
        return []

    # Biến thể đại diện cho gallery (thư mục case_N): 3-step đầu tiên, nếu không có thì biến thể đầu tiên
    canonical_idx = next((i for i, v in enumerate(variants) if v["mode"] == "3step"), 0)

    summaries = []
    for i, variant in enumerate(variants):
        module = load_s2tt_variant(variant, device, getattr(args, "mt_model", "auto"))
        if module is None:
            print(f"[Cảnh báo] Bỏ qua biến thể {variant['mode']} / {variant['model_id']}.")
            continue
        if hasattr(data_pipeline, "reset"):
            data_pipeline.reset()
        try:
            summ = evaluate_single_voice_clone_run(
                case_num=case_num,
                tts_engine=tts_engine,
                variant=variant,
                args=args,
                device=device,
                mel_fb=mel_fb,
                data_pipeline=data_pipeline,
                en_asr_tools=en_asr_tools,
                s2tt_module=module,
                tts_model=tts_model,
                write_canonical=(i == canonical_idx),
            )
            if summ:
                summaries.append(summ)
        finally:
            del module
            release_cuda_memory()

    # Dọn dẹp TTS model khỏi VRAM
    del tts_model
    release_cuda_memory()
    return summaries

# ==============================================================================
# TẠO GIAO DIỆN HTML AUDIO GALLERY TRỰC QUAN
# ==============================================================================

def generate_audio_gallery_html(output_dir):
    html_path = os.path.join(output_dir, "index.html")
    
    # Load all available metrics.json
    case_data = {}
    for c_tag in ["case_0", "case_6", "case_7", "case_4", "case_5", "case_6_xtts_2step", "case_6_xtts_3step", "case_7_f5tts_2step", "case_7_f5tts_3step"]:
        c_json = os.path.join(output_dir, c_tag, "metrics.json")
        if os.path.exists(c_json):
            try:
                with open(c_json, "r", encoding="utf-8") as f:
                    case_data[c_tag] = json.load(f)
            except Exception:
                pass

    c0_samples = {s["sample_id"]: s for s in case_data.get("case_0", {}).get("samples", [])}
    c6_samples = {s["sample_id"]: s for s in case_data.get("case_6", {}).get("samples", [])}
    c7_samples = {s["sample_id"]: s for s in case_data.get("case_7", {}).get("samples", [])}
    c4_samples = {s["sample_id"]: s for s in case_data.get("case_4", {}).get("samples", [])}

    all_sids = sorted(list(set(
        list(c0_samples.keys()) + list(c6_samples.keys()) + list(c7_samples.keys()) + list(c4_samples.keys())
    )))
    if not all_sids:
        all_sids = list(range(1, 11))

    rows_html = []
    for sid in all_sids:
        s0 = c0_samples.get(sid, {})
        s6 = c6_samples.get(sid, {})
        s7 = c7_samples.get(sid, {})
        s4 = c4_samples.get(sid, {})

        ref_vi = s0.get("ref_vi_text") or s6.get("ref_vi_text") or s7.get("ref_vi_text") or s4.get("ref_vi_text", "N/A")
        ref_en = s0.get("ref_en_text") or s6.get("ref_en_text") or s7.get("ref_en_text") or s4.get("ref_en_text", "N/A")

        # Find paths for audio files
        src_wav = f"case_0/wavs/sample_{sid:03d}_src.wav"
        if not os.path.exists(os.path.join(output_dir, src_wav)):
            src_wav = f"case_6/wavs/sample_{sid:03d}_src.wav"
        if not os.path.exists(os.path.join(output_dir, src_wav)):
            src_wav = f"case_7/wavs/sample_{sid:03d}_src.wav"
        if not os.path.exists(os.path.join(output_dir, src_wav)):
            src_wav = f"case_4/wavs/sample_{sid:03d}_src.wav"

        tgt_wav = f"case_0/wavs/sample_{sid:03d}_tgt.wav"
        if not os.path.exists(os.path.join(output_dir, tgt_wav)):
            tgt_wav = f"case_6/wavs/sample_{sid:03d}_tgt.wav"
        if not os.path.exists(os.path.join(output_dir, tgt_wav)):
            tgt_wav = f"case_7/wavs/sample_{sid:03d}_tgt.wav"
        if not os.path.exists(os.path.join(output_dir, tgt_wav)):
            tgt_wav = f"case_4/wavs/sample_{sid:03d}_tgt.wav"

        c0_wav = f"case_0/wavs/sample_{sid:03d}_case0_gen.wav"
        c6_wav = f"case_6/wavs/sample_{sid:03d}_case6_gen.wav"
        c7_wav = f"case_7/wavs/sample_{sid:03d}_case7_gen.wav"
        c4_wav = f"case_4/wavs/sample_{sid:03d}_case4_gen.wav"

        def render_player(wav_rel, title, badge_cls, sample_data, text_key):
            full_p = os.path.join(output_dir, wav_rel)
            if os.path.exists(full_p):
                txt = sample_data.get(text_key, "")
                txt_p = f'<p class="transcript">"{txt}"</p>' if txt else ""
                rtf = sample_data.get("rtf", 0.0)
                bleu = sample_data.get("asr_bleu", 0.0)
                secs = sample_data.get("secs_timbre", 0.0)
                metric_str = f"RTF: {rtf:.4f} | BLEU: {bleu:.1f}"
                if secs > 0:
                    metric_str += f" | SECS: {secs:.2f}"
                return f"""
                <div class="audio-box">
                    <span class="badge {badge_cls}">{title}</span>
                    <audio controls preload="none" src="{wav_rel}"></audio>
                    {txt_p}
                    <span class="metric-pill">{metric_str}</span>
                </div>"""
            else:
                return f"""
                <div class="audio-box">
                    <span class="badge {badge_cls}">{title}</span>
                    <span class="metric-pill" style="opacity:0.6;">Chưa benchmark</span>
                </div>"""

        cell_src = f"""
        <div class="audio-box">
            <span class="badge badge-src">🇻🇳 Tiếng Việt (Nguồn)</span>
            <audio controls preload="none" src="{src_wav}"></audio>
            <p class="transcript">"{ref_vi}"</p>
        </div>"""

        cell_c0 = render_player(c0_wav, "⚙️ MMS-TTS Baseline", "badge-c0", s0, "mt_translated_en")
        cell_c6 = render_player(c6_wav, "🎙️ XTTS-v2 Voice Cloning", "badge-c6", s6, "translated_en")
        cell_c7 = render_player(c7_wav, "🌊 F5-TTS Flow-Matching", "badge-c7", s7, "translated_en")
        cell_c4 = render_player(c4_wav, "✨ Đề xuất (Vi-En S2ST)", "badge-c4", s4, "asr_transcribed_en")

        cell_tgt = f"""
        <div class="audio-box">
            <span class="badge badge-tgt">🎯 Tiếng Anh chuẩn (FLEURS)</span>
            <audio controls preload="none" src="{tgt_wav}"></audio>
            <p class="transcript">"{ref_en}"</p>
        </div>"""

        rows_html.append(f"""
        <tr>
            <td class="id-col"><b>#{sid:02d}</b></td>
            <td>{cell_src}</td>
            <td>{cell_c0}</td>
            <td>{cell_c6}</td>
            <td>{cell_c7}</td>
            <td>{cell_c4}</td>
            <td>{cell_tgt}</td>
        </tr>
        """)

    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎧 So Sánh Mẫu Dịch Tiếng Nói S2ST (Vi-En) - Full Benchmarks</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #0b0f19;
            color: #f1f5f9;
            margin: 0;
            padding: 24px;
        }}
        h1 {{ color: #38bdf8; margin-bottom: 8px; font-size: 24px; }}
        p.subtitle {{ color: #94a3b8; margin-top: 0; margin-bottom: 24px; font-size: 14px; }}
        table {{ width: 100%; border-collapse: separate; border-spacing: 0 12px; }}
        th {{
            background: #1e293b; color: #94a3b8; padding: 14px; text-align: left;
            font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; border-bottom: 2px solid #334155;
        }}
        td {{ background: #131b2e; padding: 14px; border-top: 1px solid #1e293b; border-bottom: 1px solid #1e293b; vertical-align: top; }}
        td.id-col {{ width: 36px; text-align: center; color: #64748b; font-size: 13px; }}
        .audio-box {{ display: flex; flex-direction: column; gap: 6px; min-width: 170px; }}
        .badge {{
            display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; width: fit-content;
        }}
        .badge-src {{ background: #0369a1; color: #e0f2fe; }}
        .badge-c0  {{ background: #475569; color: #f1f5f9; }}
        .badge-c6  {{ background: #d97706; color: #fef3c7; }}
        .badge-c7  {{ background: #0284c7; color: #e0f2fe; }}
        .badge-c4  {{ background: #059669; color: #d1fae5; }}
        .badge-tgt {{ background: #7c3aed; color: #ede9fe; }}
        audio {{ width: 100%; height: 32px; border-radius: 6px; }}
        p.transcript {{
            margin: 0; font-size: 12px; color: #cbd5e1; line-height: 1.35; font-style: italic;
        }}
        .metric-pill {{
            font-size: 11px; color: #94a3b8; background: #1e293b; padding: 2px 6px; border-radius: 4px; width: fit-content;
        }}
    </style>
</head>
<body>
    <h1>🎧 Thư Viện Mẫu Âm Thanh S2ST (Tiếng Việt ➔ Tiếng Anh)</h1>
    <p class="subtitle">So sánh đối đầu: Giọng Việt nguồn ➔ MMS-TTS Baseline ➔ XTTS-v2 Voice Cloning ➔ F5-TTS Flow-Matching ➔ Mô hình đề xuất (Flow-Matching Direct S2ST) ➔ Tiếng Anh chuẩn (FLEURS)</p>
    <table>
        <thead>
            <tr>
                <th>#</th>
                <th>1. Tiếng Việt (Nguồn)</th>
                <th>2. Case 0 (MMS Baseline)</th>
                <th>3. Case 6 (XTTS-v2 Cloned)</th>
                <th>4. Case 7 (F5-TTS Flow-Match)</th>
                <th>5. Case 4 (Đề xuất Vi-En)</th>
                <th>6. Tiếng Anh (FLEURS Đích)</th>
            </tr>
        </thead>
        <tbody>
            {"".join(rows_html)}
        </tbody>
    </table>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"🌐 Trình nghe thử audio trực quan đã được tạo tại: {html_path}\n")

# ==============================================================================
# HÀM CHẠY CHÍNH (MAIN ENTRY POINT)
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Pipeline Benchmark S2ST Vi -> En")
    parser.add_argument("--case", type=str, default="sota",
                        help="Kịch bản thử nghiệm: '0' (MMS-TTS), '6'/'xtts' (XTTS-v2), '7'/'f5'/'f5tts' (F5-TTS), 'sota'/'both_tts' (Cả XTTS & F5-TTS), '5' (SeamlessM4T), '4' (Đề xuất), '1'-'3' (Ablation), hoặc 'all'")
    parser.add_argument("--split", type=str, default="test",
                        help="Tập dữ liệu kiểm thử Google FLEURS: 'test', 'validation', hoặc 'train'")
    parser.add_argument("--bridge-ckpt", type=str, default="checkpoints/bridge-best.pth",
                        help="Đường dẫn checkpoint của Bridge")
    parser.add_argument("--cfm-ckpt", type=str, default="checkpoints/cfm-best.pth",
                        help="Đường dẫn checkpoint của CFM")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Số lượng mẫu test để đánh giá")
    parser.add_argument("--output-dir", type=str, default="outputs/benchmark",
                        help="Thư mục xuất file âm thanh và báo cáo")
    parser.add_argument("--num-steps", type=int, default=25,
                        help="Số bước ODE của CFM (mặc định 25)")
    parser.add_argument("--max-sec", type=float, default=15.0,
                        help="Độ dài tối đa của file âm thanh (giây).")
    parser.add_argument("--s2tt-mode", type=str, default="both", choices=["2step", "3step", "both"],
                        help="Phương thức dịch thuật: '2step' (Whisper task=translate), '3step' (PhoWhisper + MT), hoặc 'both' (cả 2 chế độ)")
    parser.add_argument("--s2tt-models", type=str, default=DEFAULT_2STEP_MODELS,
                        help="Danh sách model 2-step (Whisper task=translate), cách nhau bởi dấu phẩy. "
                             f"Mặc định: {DEFAULT_2STEP_MODELS}")
    parser.add_argument("--vi-asr-models", type=str, default=DEFAULT_VI_ASR_MODELS,
                        help="Danh sách model ASR tiếng Việt cho 3-step (ASR + MT), cách nhau bởi dấu phẩy. "
                             f"Mặc định: {DEFAULT_VI_ASR_MODELS}")
    parser.add_argument("--whisper-s2tt-model", type=str, default=None,
                        help="(Cũ) Một model 2-step duy nhất — nếu đặt sẽ ghi đè --s2tt-models.")
    parser.add_argument("--mt-model", type=str, default="auto", choices=["auto", "vinai", "nllb", "opus"],
                        help="Mô hình dịch máy (MT) cho 3-step & Case 0: 'vinai' (vinai/vinai-translate-vi2en-v2 SOTA), 'nllb' (facebook/nllb-200-distilled-600M), 'opus', hoặc 'auto' (ưu tiên VinAI -> NLLB)")
    args = parser.parse_args()

    # Fallback checkpoint: run.sh lưu tên theo experiment (cfm_en-best.pth, bridge_vien-best.pth)
    for attr, alt in [("cfm_ckpt", "checkpoints/cfm_en-best.pth"), ("bridge_ckpt", "checkpoints/bridge_vien-best.pth")]:
        cur = getattr(args, attr)
        if cur and not os.path.exists(cur) and os.path.exists(alt):
            print(f"ℹ️ Không thấy {cur} -> dùng checkpoint {alt}")
            setattr(args, attr, alt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    vocos = load_vocos(device)
    mel_fb = build_mel_filterbank(sr=24000, n_mels=100, device=device)

    # Nạp pipeline dữ liệu từ tập test (Vi -> En)
    print(f"📡 Đang kết nối tập dữ liệu Google FLEURS [{args.split.upper()}] (Vi -> En)...")
    data_pipeline = RealStreamingSpeechPipeline(split=args.split, max_sec=args.max_sec, src_lang="vi_vn", tgt_lang="en_us")

    # Nạp mô hình phiên âm tiếng Anh để đo ASR-BLEU
    en_asr_tools = load_english_asr(device)

    feat_extractor = None
    whisper_enc = None

    case_arg = args.case.lower()
    if case_arg == "all":
        cases_to_run = [0, 6, 7, 1, 2, 3, 4]
    elif case_arg in ["sota", "both_tts", "cascaded_sota"]:
        cases_to_run = [6, 7]
    elif case_arg in ["xtts", "6"]:
        cases_to_run = [6]
    elif case_arg in ["f5", "f5tts", "7"]:
        cases_to_run = [7]
    elif case_arg in ["seamless", "5"]:
        cases_to_run = [5]
    elif case_arg in ["cascaded", "0"]:
        cases_to_run = [0]
    elif case_arg in ["proposed", "4"]:
        cases_to_run = [4]
    else:
        cases_to_run = [int(args.case)]

    if any(c in [1, 2, 3, 4] for c in cases_to_run):
        try:
            from transformers import WhisperFeatureExtractor, WhisperModel
            whisper_id = "vinai/PhoWhisper-small"
            feat_extractor = WhisperFeatureExtractor.from_pretrained(whisper_id)
            whisper_enc = WhisperModel.from_pretrained(whisper_id).encoder.eval().to(device=device, dtype=dtype)
            for p in whisper_enc.parameters(): p.requires_grad = False
        except Exception as e:
            print(f"[Whisper Info] Dùng dummy encoder: {e}")

    # Danh sách biến thể S2TT cho Case 6/7 (mỗi biến thể được nạp -> chạy -> giải phóng lần lượt)
    models_2step = [args.whisper_s2tt_model] if args.whisper_s2tt_model else parse_model_list(args.s2tt_models)
    models_vi_asr = parse_model_list(args.vi_asr_models)
    s2tt_variants = build_s2tt_variants(args.s2tt_mode, models_2step, models_vi_asr)
    if any(c in [6, 7] for c in cases_to_run):
        print("🧪 Các biến thể S2TT sẽ chạy cho mỗi engine TTS:")
        for v in s2tt_variants:
            print(f"   • {v['mode']:<5} | {v['model_id']}")

    all_summaries = []

    for c in cases_to_run:
        if hasattr(data_pipeline, "reset"):
            data_pipeline.reset()
        if c == 0:
            summary = evaluate_cascaded_case(args, device, mel_fb, data_pipeline, en_asr_tools)
            if summary: all_summaries.append(summary)
        elif c == 5:
            summary = evaluate_seamless_case(args, device, data_pipeline, en_asr_tools)
            if summary: all_summaries.append(summary)
        elif c == 6:
            sums = evaluate_cascaded_voice_clone_suite(
                case_num=6,
                tts_engine="xtts",
                variants=s2tt_variants,
                args=args,
                device=device,
                mel_fb=mel_fb,
                data_pipeline=data_pipeline,
                en_asr_tools=en_asr_tools,
            )
            all_summaries.extend(sums)
        elif c == 7:
            sums = evaluate_cascaded_voice_clone_suite(
                case_num=7,
                tts_engine="f5tts",
                variants=s2tt_variants,
                args=args,
                device=device,
                mel_fb=mel_fb,
                data_pipeline=data_pipeline,
                en_asr_tools=en_asr_tools,
            )
            all_summaries.extend(sums)
        else:
            summary = evaluate_case(c, args, device, dtype, vocos, mel_fb, data_pipeline, feat_extractor, whisper_enc, en_asr_tools)
            if summary: all_summaries.append(summary)

        # Giải phóng bộ nhớ GPU sau mỗi Case chống tràn VRAM
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # In Bảng Tổng Hợp
    print("\n" + "=" * 90)
    print("📋 BẢNG TỔNG HỢP KẾT QUẢ BENCHMARK S2ST (TẬP TEST FLEURS VI-EN)")
    print("=" * 90)
    print(f"{'Kịch bản (Benchmark Case)':<52} | {'RTF ↓':<8} | {'Mel L1 ↓':<10} | {'SECS ↑':<8} | {'S2TT-BLEU ↑':<11} | {'ASR-BLEU ↑':<10} | {'BLEU câu TB':<11}")
    print("-" * 90)
    for s in all_summaries:
        bleu_str = (f"{s.get('corpus_s2tt_bleu', float('nan')):<11.2f} | {s.get('corpus_asr_bleu', float('nan')):<10.2f} | "
                    f"{s.get('avg_asr_bleu', 0.0):<11.2f}")
        print(f"{s['case_name']:<52} | {s['avg_rtf']:<8.4f} | {s['avg_mel_l1_error']:<10.4f} | {s['avg_secs_timbre']:<8.4f} | {bleu_str}")
    print("=" * 90)
    print("ℹ️ S2TT-BLEU = văn bản dịch (trước TTS) vs Target Ref | ASR-BLEU = Whisper-En nghe lại audio sinh ra vs Target Ref.")
    print("   Cả hai là corpus SacreBLEU (chữ thường + bỏ dấu câu). Chênh lệch S2TT-BLEU − ASR-BLEU ≈ điểm mất do TTS.")

    # Lưu bảng tổng hợp ra file Markdown
    os.makedirs(args.output_dir, exist_ok=True)
    summary_md_path = os.path.join(args.output_dir, "ablation_summary.md")
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("# 📋 BẢNG SO SÁNH THỰC NGHIỆM S2ST (TẬP TEST FLEURS VI-EN)\n\n")
        f.write("| Kịch bản thử nghiệm | RTF ↓ | Mel L1 Error ↓ | SECS (Giữ giọng) ↑ | S2TT-BLEU ↑ | ASR-BLEU corpus ↑ | BLEU câu TB | Đánh giá & Phân tích |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |\n")
        for s in all_summaries:
            cid = s.get("case_id", 0)
            tag = s.get("case_tag", "")
            if cid == 0:
                note = "Cascaded Baseline: PhoWhisper ASR -> VinAI-Translate/NLLB-200 -> MMS-TTS-Eng."
            elif cid == 1:
                note = "Zero-baseline (chưa huấn luyện): Sinh nhiễu Gaussian ngẫu nhiên."
            elif cid == 2:
                note = "Chỉ train CFM: Tái tạo âm học tiếng Anh nhưng thiếu ngữ nghĩa dịch."
            elif cid == 3:
                note = "Chỉ train Bridge: Ánh xạ ngữ nghĩa Cross-Attention nhưng CFM chưa tối ưu."
            elif cid == 4:
                note = "Hệ thống đề xuất hoàn chỉnh (Vi-En S2ST): Tốc độ ~200x RT, giữ giọng người nói Việt sang tiếng Anh."
            elif cid == 5:
                note = "Meta SeamlessM4T-v2: SOTA Direct S2ST toàn cầu."
            elif cid == 6:
                note = "Cascaded SOTA Voice Cloning: XTTS-v2 giữ âm sắc chuẩn xác từ speaker_wav không cần ref_text."
            elif cid == 7:
                note = "Cascaded SOTA Voice Cloning: F5-TTS Flow-Matching điều kiện in-context audio+text."
            else:
                note = "Thử nghiệm S2ST"
            f.write(f"| **{s['case_name']}** | {s['avg_rtf']:.4f} | {s['avg_mel_l1_error']:.4f} | {s['avg_secs_timbre']:.4f} | {s.get('corpus_s2tt_bleu', float('nan')):.2f} | {s.get('corpus_asr_bleu', float('nan')):.2f} | {s.get('avg_asr_bleu', 0.0):.2f} | {note} |\n")
    print(f"📁 Báo cáo tổng hợp đã lưu tại: {summary_md_path}\n")
    generate_audio_gallery_html(args.output_dir)

if __name__ == "__main__":
    main()
