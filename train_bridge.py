#!/usr/bin/env python3
# ==============================================================================
# S2ST VI-EN: HUẤN LUYỆN CROSS-ATTENTION BRIDGE & FLOW MATCHING
# Tuân thủ nghiêm ngặt quy chuẩn SKILL2.md cho GPU Cloud Runtime (Molab)
# Hỗ trợ: Dữ liệu thật Streaming Google FLEURS (Vi -> En) + Whisper Encoder (Frozen)
# ==============================================================================

import argparse
import importlib.machinery
import math
import os
import random
import sys
import time
import types
import warnings
import wave

# Tắt cảnh báo phiền toái từ transformers và huggingface_hub
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

        def _resample(waveform, orig_freq, new_freq, **kwargs):
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

        sys.modules["torchaudio"] = ta
        sys.modules["torchaudio.functional"] = ta_func
        sys.modules["torchaudio.functional.functional"] = ta_func_func
        sys.modules["torchaudio.transforms"] = ta_trans
        sys.modules["torchaudio._extension"] = ta_ext
        return "mock"

_ = apply_torchaudio_mock_if_needed()

from train_utils import (
    FlowMatchingTargetDecoder, ResidualBlock1D,
    encode_whisper, latent_lengths, mel_lengths, length_mask, masked_mse, masked_l1,
    EnglishASR, corpus_wer, resample_24k_to_16k_fft, sync_to_drive,
    EN_CTC_VOCAB, en_text_to_ids, ctc_greedy_decode, char_error_rate,
)

# Tee output để vừa hiển thị terminal vừa ghi log file
class TeeLogger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.log = open(filepath, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return getattr(self.terminal, "isatty", lambda: False)()

    def fileno(self):
        return getattr(self.terminal, "fileno", lambda: 1)()

    def __getattr__(self, name):
        return getattr(self.terminal, name)

def load_yaml_config(config_path):
    """Đọc file YAML an toàn với fallback nếu chưa cài pyyaml."""
    default_config = {
        "model": {
            "whisper_model": "vinai/PhoWhisper-small",
            "encoder_dim": 768,
            "bridge_dim": 512,
            "bridge_layers": 6,
            "bridge_heads": 8,
            "n_mels": 100,
            "spk_dim": 192,
            "decoder_hidden": 384,
            "use_tone_head": False,
            "num_tones": 0,
        },
        "loss_weights": {
            "cfm_weight": 1.0,
            "dur_weight": 0.1,
            "cond_weight": 1.0,
            "ctc_weight": 0.0,
        },
        "training": {
            "batch_size": 4,
            "gradient_accumulation_steps": 2,
            "learning_rate": 2.0e-4,
            "min_lr": 1.0e-5,
            "weight_decay": 1.0e-4,
            "epochs": 20,
            "warmup_steps": 150,
            "max_grad_norm": 1.0,
            "bf16": True,
            "save_interval_steps": 250,
            "log_interval_steps": 20,
            "val_pairs": 16,
            "max_train_pairs": 4000,
            "sync_drive": True,
        },
        "data": {
            "dataset_name": "google/fleurs",
            "src_lang": "vi_vn",
            "tgt_lang": "en_us",
            "sample_rate_src": 16000,
            "sample_rate_tgt": 24000,
            "max_audio_sec": 12.0,
            "streaming": True,
        },
        "paths": {
            "checkpoint_dir": "checkpoints",
            "log_dir": "logs",
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
            print(f"[Config] Đã nạp thành công cấu hình từ: {config_path}")
        except ImportError:
            print("[Config Cảnh báo] Chưa cài 'pyyaml', sử dụng cấu hình mặc định tương thích.")
        except Exception as e:
            print(f"[Config Cảnh báo] Lỗi đọc {config_path}: {e}, dùng cấu hình mặc định.")
    else:
        print(f"[Config] Không tìm thấy {config_path}, dùng cấu hình mặc định.")
        
    return default_config

# Import PyTorch an toàn
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class MockSpectrogram(nn.Module):
        def __init__(self, n_fft=1024, win_length=None, hop_length=None, power=None,
                     normalized=False, wkwargs=None, center=True, pad_mode="reflect",
                     onesided=True, **kwargs):
            super().__init__()
            wl = win_length if win_length is not None else n_fft
            self.register_buffer("window", torch.empty(wl))
        def forward(self, x): return x

    class MockMelScale(nn.Module):
        def __init__(self, n_mels=100, sample_rate=24000, f_min=0., f_max=None,
                     n_stft=None, norm=None, mel_scale="htk", **kwargs):
            super().__init__()
            n_stft_val = n_stft if n_stft is not None else 513
            self.register_buffer("fb", torch.empty(n_stft_val, n_mels))
        def forward(self, x): return x

    class MockMelSpectrogram(nn.Module):
        def __init__(self, sample_rate=24000, n_fft=1024, win_length=None, hop_length=None,
                     f_min=0., f_max=None, pad=0, n_mels=100, window_fn=None,
                     power=None, normalized=False, wkwargs=None, center=True,
                     pad_mode="reflect", onesided=True, norm=None, mel_scale="htk", **kwargs):
            super().__init__()
            wl = win_length if win_length is not None else n_fft
            n_stft = n_fft // 2 + 1
            self.spectrogram = MockSpectrogram(n_fft=n_fft, win_length=wl)
            self.mel_scale = MockMelScale(n_mels=n_mels, n_stft=n_stft, sample_rate=sample_rate)
        def forward(self, x):
            return x

    if "torchaudio.transforms" in sys.modules:
        sys.modules["torchaudio.transforms"].MelSpectrogram = MockMelSpectrogram
        sys.modules["torchaudio.transforms"].Spectrogram = MockSpectrogram
        sys.modules["torchaudio.transforms"].MelScale = MockMelScale
    if "torchaudio" in sys.modules and hasattr(sys.modules["torchaudio"], "transforms"):
        sys.modules["torchaudio"].transforms.MelSpectrogram = MockMelSpectrogram
        sys.modules["torchaudio"].transforms.Spectrogram = MockSpectrogram
        sys.modules["torchaudio"].transforms.MelScale = MockMelScale

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
except ImportError as e:
    print(f"[Fatal Error] PyTorch chưa được cài đặt: {e}")
    sys.exit(1)

# ==============================================================================
# 1. PURE PYTORCH AUDIO UTILITIES (100-MEL CHO VOCOS 24KHZ)
# ==============================================================================

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

def build_mel_filterbank(sr=24000, n_fft=1024, n_mels=100, fmin=0.0, fmax=12000.0, device="cpu"):
    """
    Ma trận Mel Filterbank tam giác liên tục chuẩn Torchaudio HTK (Khớp 100% với Vocos pre-trained).
    Loại bỏ hiện tượng rời rạc hóa tần số để âm thanh trong trẻo, không bị méo hài âm.
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

def extract_100_mel(waveform_24k, mel_fb, n_fft=1024, hop_length=256):
    """Trích xuất 100-kênh log-mel spectrogram chuẩn Vocos 24kHz bằng thuần PyTorch."""
    if waveform_24k.dim() == 1:
        waveform_24k = waveform_24k.unsqueeze(0)
    window = torch.hann_window(n_fft, device=waveform_24k.device)
    stft = torch.stft(
        waveform_24k, n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
        window=window, return_complex=True
    )
    spec = torch.abs(stft)
    mel = torch.matmul(mel_fb.to(waveform_24k.device), spec)
    return torch.log(torch.clamp(mel, min=1e-5))

def resample_16k_to_24k(audio_16k):
    """FFT Sinc Resampling từ 16kHz -> 24kHz bằng PyTorch thuần (Triệt tiêu nhiễu aliasing)."""
    orig_shape = audio_16k.shape
    if audio_16k.dim() == 1:
        audio_16k = audio_16k.unsqueeze(0)
    elif audio_16k.dim() > 2:
        audio_16k = audio_16k.view(-1, audio_16k.shape[-1])

    B, T = audio_16k.shape
    new_T = int(round(T * (24000.0 / 16000.0)))

    X = torch.fft.rfft(audio_16k.float(), dim=-1)
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

def resample_24k_to_16k(audio_24k):
    """FFT Sinc Downsampling từ 24kHz -> 16kHz bằng PyTorch thuần."""
    orig_shape = audio_24k.shape
    if audio_24k.dim() == 1:
        audio_24k = audio_24k.unsqueeze(0)
    elif audio_24k.dim() > 2:
        audio_24k = audio_24k.view(-1, audio_24k.shape[-1])

    B, T = audio_24k.shape
    new_T = int(round(T * (16000.0 / 24000.0)))

    X = torch.fft.rfft(audio_24k.float(), dim=-1)
    target_bins = new_T // 2 + 1
    X_cut = X[:, :target_bins]

    w_16k = torch.fft.irfft(X_cut, n=new_T, dim=-1) * (float(new_T) / float(T))
    if len(orig_shape) == 1:
        return w_16k.squeeze(0)
    elif len(orig_shape) == 3:
        return w_16k.view(orig_shape[0], orig_shape[1], new_T)
    return w_16k

# ==============================================================================
# 2. KIẾN TRÚC MÔ HÌNH: CROSS-ATTENTION BRIDGE & OT-CFM TARGET DECODER
# ==============================================================================

class DurationPredictor(nn.Module):
    """Dự đoán tỷ lệ co dãn thời lượng (duration scale) từ tiếng Việt sang tiếng Anh."""
    def __init__(self, in_dim=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(256, 1, kernel_size=1)
        )
    def forward(self, x, pad_mask=None):
        """x [B, T, C]; pad_mask [B, T] (True = phần đệm) -> trung bình chỉ trên frame thật."""
        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)   # conv k=3 ở frame biên không "nhìn thấy" phần đệm
        h = self.conv(x.transpose(1, 2)).squeeze(1)          # [B, T]
        if pad_mask is not None:
            keep = (~pad_mask).to(h.dtype)
            h_mean = (h * keep).sum(dim=-1) / keep.sum(dim=-1).clamp(min=1.0)
        else:
            h_mean = h.mean(dim=-1)
        return torch.sigmoid(h_mean) * 1.4 + 0.8  # scale trong khoảng [0.8, 2.2]

class CrossAttentionBridge(nn.Module):
    """Cầu nối dịch thuật Non-Autoregressive sử dụng Cross-Attention Transformer Decoder.
    Giải quyết triệt để vấn đề đảo ngữ pháp (non-monotonic word reordering) giữa Tiếng Việt và Tiếng Anh.
    """
    def __init__(self, in_dim=768, hidden_dim=512, num_layers=6, n_heads=8, query_mode="interp",
                 norm_mode="prenorm_skip", ctc_vocab=0):
        """query_mode:
          'interp'  (mặc định): query đích = memory nguồn nội suy theo thời gian sang độ dài đích + mã hóa vị trí
                    TƯƠNG ĐỐI (0→1) ở cả 2 phía -> ngay từ đầu frame i đã "nhìn" đúng vùng tương ứng của câu nguồn,
                    cross-attention học thêm phần đảo trật tự.
          'learned' (cũ, giữ làm ablation): query = positional embedding học từ đầu, nguồn không có vị trí tương
                    đối -> attention dễ sụp thành trung bình đều (overfit 16 câu: L1(c*) ≈ mốc đoán trung bình).
        norm_mode:
          'prenorm_skip' (mặc định): Transformer PRE-norm (norm_first) + LayerNorm cuối + đường tắt skip_proj(query)
                    cộng thẳng vào đầu ra -> thông tin theo từng frame của query đi thẳng ra, không bị rửa trôi.
          'postnorm' (cũ, chỉ để nạp checkpoint cũ): post-norm 6 tầng SỤP về hằng số theo thời gian khi train
                    (đo 26/11/2026: std theo thời gian của đầu ra 0.0002, L1 = đúng mốc trung bình, kể cả khi
                    mục tiêu chỉ là ánh xạ gần đồng nhất). Pre-norm + skip: L1 = 18% mốc trên cùng bài."""
        super().__init__()
        if norm_mode not in ("prenorm_skip", "postnorm"):
            raise ValueError(f"norm_mode không hợp lệ: {norm_mode}")
        self.hidden_dim = hidden_dim
        self.query_mode = query_mode
        self.norm_mode = norm_mode
        pre = norm_mode == "prenorm_skip"
        self.src_proj = nn.Linear(in_dim, hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        # Hệ số học được cho mã hóa vị trí tương đối (khởi tạo vừa phải để không lấn át nội dung)
        self.rel_pos_scale = nn.Parameter(torch.tensor(1.0))

        # Encoder tự chú ý trên đặc trưng nguồn (tiếng Việt)
        src_enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            batch_first=True, activation="gelu", norm_first=pre
        )
        self.src_encoder = nn.TransformerEncoder(src_enc_layer, num_layers=max(2, num_layers // 2),
                                                 norm=nn.LayerNorm(hidden_dim) if pre else None,
                                                 enable_nested_tensor=False)

        # Decoder chú ý chéo (Cross-Attention) sang đặc trưng đích (tiếng Anh)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            batch_first=True, activation="gelu", norm_first=pre
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers,
                                             norm=nn.LayerNorm(hidden_dim) if pre else None)

        # Bộ dự đoán tỷ lệ co dãn thời lượng
        self.duration_predictor = DurationPredictor(in_dim=hidden_dim)

        # Positional Embeddings học được cho Query đích
        self.pos_emb = nn.Embedding(4096, hidden_dim)

        # Lớp chiếu đầu ra: đầu ra TransformerDecoder (post-norm) đã qua LayerNorm -> mỗi frame bị ép cùng
        # trung bình/phương sai, KHÔNG khớp được c* (độ lớn khác nhau giữa frame im lặng và frame có tiếng).
        # Linear tự do phía sau gỡ bỏ giới hạn đó (overfit cũ kẹt L1(c*) ≈ 0.555).
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        # Đường tắt: query (đã căn chỉnh theo thời gian) -> đầu ra, để Bridge không thể "quên" cấu trúc từng frame
        self.skip_proj = nn.Linear(hidden_dim, hidden_dim) if pre else None
        # Đầu CTC phụ: từ trạng thái ẩn của decoder dự đoán chuỗi KÝ TỰ tiếng Anh (không cần căn chỉnh theo frame)
        # -> ép Bridge mã hóa NỘI DUNG câu đích. Chỉ dùng làm loss lúc train (kiểu Translatotron 2 / UnitY).
        self.ctc_vocab = int(ctc_vocab or 0)
        self.ctc_head = nn.Linear(hidden_dim, self.ctc_vocab) if self.ctc_vocab > 0 else None

    def _rel_pos(self, lengths, T, device, dtype):
        """Mã hóa vị trí TƯƠNG ĐỐI: frame j của câu dài L có vị trí p = j/(L-1) ∈ [0, 1];
        sin/cos nhiều tần số của p -> [B, T, hidden]. Cùng p ở nguồn và đích => cùng mã -> attention dễ căn chỉnh."""
        B = len(lengths)
        lens = torch.as_tensor(lengths, device=device, dtype=torch.float32).view(B, 1)
        p = torch.arange(T, device=device, dtype=torch.float32).view(1, T) / (lens - 1).clamp(min=1.0)
        p = p.clamp(max=1.0)
        half = self.hidden_dim // 2
        freqs = torch.exp(torch.linspace(0.0, math.log(256.0), half, device=device)) * math.pi
        ang = p.unsqueeze(-1) * freqs.view(1, 1, half)
        return (torch.cat([ang.sin(), ang.cos()], dim=-1) * self.rel_pos_scale).to(dtype)

    @staticmethod
    def _interp_to(memory, src_lengths, tgt_lengths, T_tgt):
        """Nội suy tuyến tính phần thật của memory từng câu (độ dài src_len) sang độ dài đích tgt_len, đệm tới T_tgt."""
        B, _, D = memory.shape
        out = memory.new_zeros(B, T_tgt, D)
        for b in range(B):
            ls, lt = int(src_lengths[b]), int(tgt_lengths[b])
            seg = memory[b:b + 1, :ls].transpose(1, 2)                       # [1, D, ls]
            out[b, :lt] = F.interpolate(seg, size=lt, mode="linear", align_corners=True)[0].transpose(0, 1)
        return out

    @staticmethod
    def _pad_mask(lengths, T, device):
        if lengths is None:
            return None
        lens = torch.as_tensor(lengths, device=device).view(-1, 1)
        return torch.arange(T, device=device).view(1, -1) >= lens       # True = phần đệm

    def forward(self, whisper_features, target_len=None, src_lengths=None, tgt_lengths=None, return_ctc=False):
        """
        whisper_features: [B, T_src, in_dim]
        target_len: Số frame Mel đích T_tgt (lúc train) hoặc None (lúc inference)
        src_lengths / tgt_lengths: độ dài thật (frame) từng câu -> attention bỏ qua phần đệm,
            nên một câu trong batch đệm cho kết quả như khi chạy riêng (train/suy luận nhất quán).
        """
        B, T_src, _ = whisper_features.shape
        dev = whisper_features.device
        src_pad = self._pad_mask(src_lengths, T_src, dev)
        src_lens = list(src_lengths) if src_lengths is not None else [T_src] * B
        src_h = self.src_proj(whisper_features)
        if self.query_mode == "interp":
            src_h = src_h + self._rel_pos(src_lens, T_src, dev, src_h.dtype)
        memory = self.src_encoder(src_h, src_key_padding_mask=src_pad)

        dur_scale = self.duration_predictor(memory, src_pad)  # [B]

        if target_len is None:
            # Lúc Inference: Ước lượng T_tgt từ dur_scale
            avg_scale = float(dur_scale.mean().item())
            target_len = max(20, min(2048, int(T_src * avg_scale)))

        tgt_lens = [int(x) for x in tgt_lengths] if tgt_lengths is not None else [target_len] * B
        if self.query_mode == "interp":
            # Query = nội dung nguồn kéo giãn theo thời gian (căn chỉnh đơn điệu ban đầu) + vị trí tương đối
            queries = self.query_proj(self._interp_to(memory, src_lens, tgt_lens, target_len))
            queries = queries + self._rel_pos(tgt_lens, target_len, dev, queries.dtype)
        else:
            positions = torch.arange(target_len, device=dev).clamp(max=4095)
            queries = self.pos_emb(positions).unsqueeze(0).expand(B, -1, -1)

        # Cross-Attention: Cho phép từ tiếng Anh đảo vị trí tùy ý so với tiếng Việt
        tgt_pad = self._pad_mask(tgt_lengths, target_len, dev)
        latents = self.decoder(tgt=queries, memory=memory,
                               tgt_key_padding_mask=tgt_pad, memory_key_padding_mask=src_pad)
        ctc_logits = self.ctc_head(latents) if (return_ctc and self.ctc_head is not None) else None
        latents = self.out_proj(latents)
        if self.skip_proj is not None:
            latents = latents + self.skip_proj(queries)
        if return_ctc:
            return latents, dur_scale, ctc_logits     # ctc_logits: [B, T_tgt, V] hoặc None
        return latents, dur_scale

# Alias tương thích ngược
ToneAwareBridge = CrossAttentionBridge

class AcousticSpeakerEncoder(nn.Module):
    """Trích xuất d-vector âm sắc (Speaker Timbre Embedding - 192d) qua Mean/Std statistics."""
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

# ResidualBlock1D / FlowMatchingTargetDecoder: định nghĩa DUY NHẤT trong train_utils.py (dùng chung với Stage 1).
# Trước đây hai file định nghĩa lệch tên tham số -> Stage 2 không nạp được CFM Stage 1.

class ViEnS2STSystem(nn.Module):
    """Hệ thống tích hợp toàn diện Vi -> En: Whisper -> CrossAttentionBridge -> CFM Decoder -> 100 Mel."""
    def __init__(self, encoder_dim=768, bridge_dim=512, bridge_layers=6, n_mels=100, spk_dim=192,
                 bridge_query_mode="interp", bridge_norm="prenorm_skip", bridge_ctc_vocab=0):
        super().__init__()
        self.n_mels = n_mels
        self.bridge = CrossAttentionBridge(in_dim=encoder_dim, hidden_dim=bridge_dim, num_layers=bridge_layers,
                                           query_mode=bridge_query_mode, norm_mode=bridge_norm,
                                           ctc_vocab=bridge_ctc_vocab)
        self.spk_encoder = AcousticSpeakerEncoder(feat_dim=encoder_dim, spk_dim=spk_dim)
        # Kiến trúc mặc định = preset mặc định của train_utils; khi nạp Stage 1 sẽ dựng lại ĐÚNG kiến trúc đã lưu
        self.decoder = FlowMatchingTargetDecoder(n_mels=n_mels, cond_dim=bridge_dim, spk_dim=spk_dim)
        # cond_proj của Stage 1 (ĐÓNG BĂNG): biến đặc trưng Whisper của audio TIẾNG ANH đích thành điều kiện c*
        # mà CFM decoder đã quen -> làm NHÃN trực tiếp cho Bridge (loss chưng cất L_cond).
        self.teacher_cond_proj = nn.Linear(encoder_dim, bridge_dim)
        self.has_teacher = False
        self.target_encoder_id = None   # None = cùng encoder với phía nguồn
        # Mixed precision: TRỌNG SỐ giữ fp32 (cập nhật nhỏ không bị làm tròn mất), PHÉP TÍNH chạy bf16 qua autocast.
        # Trước đây model.to(bfloat16) -> LR 2e-4 bị làm tròn ở bf16 -> Bridge "đứng yên" (overfit kẹt ở mốc trung bình).
        self.amp_dtype = None           # đặt torch.bfloat16 khi train trên GPU

    def _amp(self):
        dev = next(self.parameters()).device.type
        return torch.autocast(device_type=dev, dtype=self.amp_dtype or torch.float32,
                              enabled=self.amp_dtype is not None)

    def _in(self, x):
        """Đưa tensor đầu vào về đúng dtype của trọng số (fp32 khi train, bf16 nếu cả model ở bf16 như eval)."""
        return x.to(next(self.parameters()).dtype) if torch.is_tensor(x) and x.is_floating_point() else x

    def speaker(self, feats):
        """Speaker embedding từ đặc trưng encoder (tự khớp dtype / autocast)."""
        with self._amp():
            return self.spk_encoder(self._in(feats))

    def teacher_condition(self, tgt_whisper, T_tgt):
        """c* [B, bridge_dim, T_tgt] = interp(cond_proj_stage1(Whisper(audio En đích)))."""
        with self._amp():
            c = self.teacher_cond_proj(self._in(tgt_whisper))
        return F.interpolate(c.float().transpose(1, 2), size=T_tgt, mode='linear', align_corners=False)

    def compute_loss(self, src_whisper, target_mel, speaker_emb=None, target_dur_ratio=None, loss_weights=None,
                     mel_mask=None, cond_target=None, t=None, x_0=None, src_lengths=None, ctc_targets=None, **kwargs):
        """Trả về (total, cfm, dur, cond). CTC loss (nếu bật) đã cộng vào total và lưu ở self.last_ctc_loss;
        log-prob CTC lưu ở self.last_ctc_logp [B, T, V] (để giải mã kiểm tra).
        ctc_targets: list id ký tự tiếng Anh từng câu (en_text_to_ids) hoặc None = tắt.
        speaker_emb: lúc TRAIN nên là embedding của audio ĐÍCH (người đọc tiếng Anh) để decoder tái tạo đúng Mel đích.
        cond_target: c* từ teacher_condition() -> loss chưng cất L1 (None = tắt).
        mel_mask: [B, T_tgt] bỏ qua phần đệm. t / x_0: truyền vào để val loss cố định."""
        weights = {"cfm_weight": 1.0, "dur_weight": 0.1, "cond_weight": 0.0, "ctc_weight": 0.0}
        weights.update(loss_weights or {})
        B, _, T_tgt = target_mel.shape
        src_whisper, target_mel = self._in(src_whisper), self._in(target_mel)
        speaker_emb, t, x_0 = self._in(speaker_emb), self._in(t), self._in(x_0)
        with self._amp():
            return self._compute_loss(src_whisper, target_mel, speaker_emb, target_dur_ratio, weights,
                                      mel_mask, cond_target, t, x_0, src_lengths, B, T_tgt, ctc_targets)

    def _compute_loss(self, src_whisper, target_mel, speaker_emb, target_dur_ratio, weights,
                      mel_mask, cond_target, t, x_0, src_lengths, B, T_tgt, ctc_targets=None):

        # Truyền target_len=T_tgt để Cross-Attention sinh đúng kích thước Mel đích; độ dài thật -> mask phần đệm
        tgt_lengths = mel_mask.sum(dim=-1).long() if mel_mask is not None else None
        use_ctc = (ctc_targets is not None and weights.get("ctc_weight", 0) > 0
                   and getattr(self.bridge, "ctc_head", None) is not None)
        translated_latents, dur_scale, ctc_logits = self.bridge(src_whisper, target_len=T_tgt, src_lengths=src_lengths,
                                                                tgt_lengths=tgt_lengths, return_ctc=True)

        if speaker_emb is None:
            speaker_emb = self.spk_encoder(src_whisper)

        cond = translated_latents.transpose(1, 2)  # [B, bridge_dim, T_tgt]

        # 1. CFM Optimal Transport Loss (gradient đi xuyên CFM decoder đóng băng)
        if t is None:
            t = torch.rand(B, device=target_mel.device, dtype=target_mel.dtype)
        if x_0 is None:
            x_0 = torch.randn_like(target_mel)
        x_1 = target_mel
        t_exp = t.view(B, 1, 1)
        x_t = (1.0 - t_exp) * x_0 + t_exp * x_1
        u_t = x_1 - x_0

        v_pred = self.decoder(x_t, t, cond, speaker_emb, mask=mel_mask)
        cfm_loss = masked_mse(v_pred, u_t, mel_mask)

        # 2. Duration Loss
        if target_dur_ratio is not None:
            dur_loss = F.smooth_l1_loss(dur_scale.float().view(-1), target_dur_ratio.float().view(-1))
        else:
            dur_loss = torch.tensor(0.0, device=target_mel.device, dtype=target_mel.dtype)

        # 3. Loss chưng cất điều kiện: Bridge phải sinh ra đúng c* của audio tiếng Anh
        if cond_target is not None and weights["cond_weight"] > 0:
            cond_loss = masked_l1(cond, cond_target, mel_mask)
        else:
            cond_loss = torch.tensor(0.0, device=target_mel.device, dtype=target_mel.dtype)

        total_loss = (weights["cfm_weight"] * cfm_loss + weights["dur_weight"] * dur_loss
                      + weights["cond_weight"] * cond_loss)

        # 4. CTC phụ: chuỗi ký tự tiếng Anh từ trạng thái ẩn của Bridge (không cần căn chỉnh frame)
        self.last_ctc_loss, self.last_ctc_logp = None, None
        if use_ctc:
            logp = F.log_softmax(ctc_logits.float(), dim=-1)                       # [B, T, V], fp32
            in_lens = (tgt_lengths if tgt_lengths is not None
                       else torch.full((B,), T_tgt, device=logp.device)).long().cpu()
            tgt_lens = torch.tensor([len(x) for x in ctc_targets], dtype=torch.long)
            flat = torch.tensor([i for x in ctc_targets for i in x], dtype=torch.long)
            ctc_loss = F.ctc_loss(logp.transpose(0, 1), flat, in_lens, tgt_lens,
                                  blank=0, reduction="mean", zero_infinity=True)
            total_loss = total_loss + weights["ctc_weight"] * ctc_loss
            self.last_ctc_loss, self.last_ctc_logp = ctc_loss.detach(), logp.detach()
            self.last_ctc_lens = in_lens
        return total_loss, cfm_loss, dur_loss, cond_loss

    @torch.no_grad()
    def inference(self, src_whisper, speaker_emb=None, target_len=None, num_steps=32):
        self.eval()
        src_whisper, speaker_emb = self._in(src_whisper), self._in(speaker_emb)
        with self._amp():
            out = self._inference(src_whisper, speaker_emb, target_len, num_steps)
        return out.float()

    def _inference(self, src_whisper, speaker_emb, target_len, num_steps):
        B = src_whisper.shape[0]
        if speaker_emb is None:
            speaker_emb = self.spk_encoder(src_whisper)

        translated_latents, dur_scale = self.bridge(src_whisper, target_len=target_len)
        T_tgt = translated_latents.shape[1]
        cond = translated_latents.transpose(1, 2)

        x_t = torch.randn(B, self.n_mels, T_tgt, device=src_whisper.device, dtype=src_whisper.dtype)
        dt = 1.0 / num_steps

        # 2nd-Order Midpoint ODE Solver
        for step in range(num_steps):
            t_val = step * dt
            cur_t = torch.full((B,), t_val, device=src_whisper.device, dtype=src_whisper.dtype)
            v1 = self.decoder(x_t, cur_t, cond, speaker_emb)

            x_mid = x_t + v1 * (0.5 * dt)
            t_mid = torch.full((B,), min(1.0, t_val + 0.5 * dt), device=src_whisper.device, dtype=src_whisper.dtype)
            v2 = self.decoder(x_mid, t_mid, cond, speaker_emb)

            x_t = x_t + v2 * dt

        # Chặn ngưỡng âm học chuẩn [-11.5, 3.0] chống clipping
        x_t = torch.clamp(x_t, min=-11.5, max=3.0)
        return x_t

# Alias tương thích ngược
EnViS2STSystem = ViEnS2STSystem

def load_stage1_into(model, ckpt_path, device, allow_legacy=False):
    """Nạp checkpoint Stage 1 (train_cfm.py) vào ViEnS2STSystem:
       decoder <- decoder, teacher_cond_proj <- cond_proj, spk_encoder <- spk_encoder.
    Trả về True nếu có đủ teacher (cond_proj) để bật loss chưng cất."""
    ck = torch.load(ckpt_path, map_location=device)
    meta = ck.get("stage1_meta") if isinstance(ck, dict) else None
    if meta is None and not allow_legacy:
        raise RuntimeError(
            f"{ckpt_path} không có 'stage1_meta' -> có thể là checkpoint CŨ (hướng En->Vi). "
            "Hãy train lại Stage 1 bằng 'bash run.sh start cfm_en' hoặc thêm --allow-legacy-cfm nếu chắc chắn.")
    if meta is not None and meta.get("lang") != "en_us":
        raise RuntimeError(f"{ckpt_path} là CFM ngôn ngữ '{meta.get('lang')}', cần 'en_us'.")
    full = ck.get("full_system_state_dict") if isinstance(ck, dict) else None
    dec = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    if full:
        dec = {k[len("decoder."):]: v for k, v in full.items() if k.startswith("decoder.")} or dec
    dec_cfg = (meta or {}).get("decoder_config")
    if dec_cfg and dec_cfg != getattr(model.decoder, "config", None):
        # Dựng lại decoder đúng kiến trúc Stage 1 đã train (v1 / v2 / v2-small ...)
        dev_dtype = next(model.parameters())
        model.decoder = FlowMatchingTargetDecoder(**dec_cfg).to(device=dev_dtype.device, dtype=dev_dtype.dtype)
    model.decoder.load_state_dict(dec, strict=True)
    has_teacher = False
    if full:
        cp = {k[len("cond_proj."):]: v for k, v in full.items() if k.startswith("cond_proj.")}
        sp = {k[len("spk_encoder."):]: v for k, v in full.items() if k.startswith("spk_encoder.")}
        if cp:
            model.teacher_cond_proj.load_state_dict(cp, strict=True)
            has_teacher = True
        if sp:
            model.spk_encoder.load_state_dict(sp, strict=True)
    model.has_teacher = has_teacher
    # Encoder phía đích mà Stage 1 đã dùng (c* và speaker PHẢI tính bằng đúng encoder này)
    model.target_encoder_id = (meta or {}).get("target_encoder") or (meta or {}).get("whisper_model")
    print(f"✓ Nạp Stage 1 từ {ckpt_path}: decoder ✓ | cond_proj (teacher) {'✓' if has_teacher else '✗'} "
          f"| spk_encoder {'✓' if full and any(k.startswith('spk_encoder.') for k in full) else '✗'}")
    return has_teacher

def load_stage2_into(model, ckpt_path, device):
    """Nạp checkpoint Stage 2 (Bridge + decoder đóng băng + spk_encoder) — dựng decoder đúng kiến trúc đã lưu."""
    ck = torch.load(ckpt_path, map_location=device)
    meta = ck.get("stage2_meta") if isinstance(ck, dict) else None
    if meta is None:
        raise RuntimeError(f"{ckpt_path} không có stage2_meta (bản cũ En->Vi?) — train lại Stage 2.")
    dec_cfg = meta.get("decoder_config")
    ref = next(model.parameters())
    if dec_cfg and dec_cfg != getattr(model.decoder, "config", None):
        model.decoder = FlowMatchingTargetDecoder(**dec_cfg).to(device=ref.device, dtype=ref.dtype)
    qm = meta.get("bridge_query_mode", "learned")   # checkpoint trước khi có tùy chọn này dùng query 'learned'
    nm = meta.get("bridge_norm", "postnorm")         # checkpoint trước 26/11/2026 dùng post-norm
    cv = int(meta.get("bridge_ctc_vocab", 0))        # checkpoint chưa có CTC = 0
    if qm != model.bridge.query_mode or nm != model.bridge.norm_mode or cv != model.bridge.ctc_vocab:
        b = model.bridge
        model.bridge = CrossAttentionBridge(in_dim=b.src_proj.in_features, hidden_dim=b.hidden_dim,
                                            num_layers=len(b.decoder.layers), query_mode=qm,
                                            norm_mode=nm, ctc_vocab=cv).to(device=ref.device, dtype=ref.dtype)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.has_teacher = bool(meta.get("has_teacher"))
    model.target_encoder_id = meta.get("target_encoder")
    return meta

# ==============================================================================
# 3. PIPELINE DỮ LIỆU ÂM THANH THẬT STREAMING (GOOGLE FLEURS VI -> EN)
# ==============================================================================

class RealStreamingSpeechPipeline:
    """Nạp dữ liệu âm thanh thật trực tuyến Vi-En với căn chỉnh song song bằng trường 'id' (FLoRes)."""
    def __init__(self, split='train', max_sec=12.0, src_lang='vi_vn', tgt_lang='en_us',
                 allow_synthetic=False, max_pairs=None):
        self.split = split
        self.max_sec = max_sec
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.is_online = False
        self.allow_synthetic = allow_synthetic
        self.max_pairs = max_pairs
        self._parallel_pairs = []
        self._pair_index = 0
        self.last_src_lengths = []   # độ dài thật (mẫu 16kHz) từng câu nguồn trong batch gần nhất
        self.last_tgt_lengths = []   # độ dài thật (mẫu 16kHz) từng câu đích trong batch gần nhất

        try:
            import soundfile
        except ImportError:
            try:
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "soundfile", "-q"])
            except Exception:
                pass

        try:
            from datasets import load_dataset, Audio
            print(f"📡 Đang kết nối Streaming Google FLEURS (Src: {src_lang.upper()} -> Tgt: {tgt_lang.upper()}, split='{split}')...")
            self.ds_src = load_dataset('google/fleurs', src_lang, split=split, streaming=True)
            self.ds_tgt = load_dataset('google/fleurs', tgt_lang, split=split, streaming=True)

            try:
                self.ds_src = self.ds_src.cast_column('audio', Audio(decode=False))
                self.ds_tgt = self.ds_tgt.cast_column('audio', Audio(decode=False))
            except Exception:
                pass

            self.is_online = True
            print("✓ Kết nối dữ liệu thật thành công qua Streaming!")
            self._build_parallel_pairs()
        except Exception as e:
            self.is_online = False
            if not self.allow_synthetic:
                raise RuntimeError(f"Không stream được FLEURS ({e}) — dừng để tránh chạy trên dữ liệu giả lập.")
            print(f"[Data Warning] Không thể stream online ({e}). Dùng tín hiệu mô phỏng (chỉ cho smoke-test).")

    def _build_parallel_pairs(self):
        """Buffer các mục của ngôn ngữ đích theo 'id', rồi match từng mục của ngôn ngữ nguồn."""
        print(f"🔗 Đang căn chỉnh song song {self.src_lang} ↔ {self.tgt_lang} bằng trường 'id' (FLoRes sentence ID)...")
        tgt_by_id = {}
        count_tgt = 0
        limit_tgt = 800 if self.split == 'train' else 600
        limit_pairs = 600 if self.split == 'train' else 400
        if self.max_pairs:
            # Cho phép dùng nhiều dữ liệu train hơn (FLEURS train có vài nghìn cặp)
            limit_pairs = int(self.max_pairs)
            limit_tgt = max(limit_tgt, int(self.max_pairs))

        for item in self.ds_tgt:
            fid = item.get('id')
            if fid is not None and fid not in tgt_by_id:
                tgt_by_id[fid] = item
                count_tgt += 1
            if count_tgt >= limit_tgt:
                break
        print(f"   → Đã nạp {count_tgt} câu đích ({self.tgt_lang}) độc lập vào bộ đệm.")

        pairs = []
        for item in self.ds_src:
            fid = item.get('id')
            if fid is not None and fid in tgt_by_id:
                pairs.append((item, tgt_by_id[fid]))
                if len(pairs) >= limit_pairs:
                    break

        self._parallel_pairs = pairs
        self._pair_index = 0
        print(f"   ✓ Căn chỉnh thành công {len(pairs)} cặp song song {self.src_lang} ↔ {self.tgt_lang} chuẩn xác!")

        if pairs:
            print("\n📋 [KIỂM CHỨNG DỮ LIỆU SONG SONG]:")
            for i in range(min(3, len(pairs))):
                src_txt = pairs[i][0].get('transcription', '').strip()
                tgt_txt = pairs[i][1].get('transcription', '').strip()
                cid = pairs[i][0].get('id', 'N/A')
                print(f"   🔹 Cặp #{i+1:02d} [FLoRes ID {cid}]:")
                print(f"      🇻🇳 Vi (Nguồn): \"{src_txt}\"")
                print(f"      🇬🇧 En (Đích) : \"{tgt_txt}\"")
            print("-" * 65 + "\n")
        else:
            self.is_online = False
            if not self.allow_synthetic:
                raise RuntimeError("Không tìm được cặp song song nào trong FLEURS (trường 'id').")
            print("[Cảnh báo] Không tìm được cặp song song nào! Kiểm tra trường 'id' trong dataset.")

    def reset(self):
        self._pair_index = 0

    def shuffle(self):
        if self._parallel_pairs:
            random.shuffle(self._parallel_pairs)
            self._pair_index = 0

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
        elif isinstance(audio_data, str) and os.path.exists(audio_data):
            arr, sr = sf.read(audio_data)
            if arr.ndim > 1: arr = arr.mean(axis=-1)
            return torch.from_numpy(arr).float()[:limit]
        raise ValueError("Không tìm thấy audio hợp lệ")

    def fetch_batch(self, batch_size=4, return_text=True):
        src_waves = []
        tgt_waves = []
        transcripts = []      # Target transcripts (English)
        src_transcripts = []  # Source transcripts (Vietnamese)

        if self.is_online and self._parallel_pairs:
            n_errors = 0
            while len(src_waves) < batch_size:
                if self._pair_index >= len(self._parallel_pairs):
                    self._pair_index = 0
                item_src, item_tgt = self._parallel_pairs[self._pair_index]
                self._pair_index += 1
                try:
                    w_src = self._extract_audio_tensor(item_src['audio'])
                    w_tgt = self._extract_audio_tensor(item_tgt['audio'])
                except Exception as ex:
                    # Cặp lỗi giải mã -> bỏ qua, lấy cặp kế tiếp (KHÔNG bịa dữ liệu giả)
                    n_errors += 1
                    if n_errors > 50:
                        raise RuntimeError(f"Quá nhiều cặp lỗi giải mã liên tiếp ({ex})")
                    continue
                src_waves.append(w_src)
                tgt_waves.append(w_tgt)
                transcripts.append(item_tgt.get('transcription', ''))
                src_transcripts.append(item_src.get('transcription', ''))

        if len(src_waves) < batch_size and not self.allow_synthetic:
            raise RuntimeError("Thiếu dữ liệu thật và không cho phép dữ liệu mô phỏng.")
        while len(src_waves) < batch_size:
            t_sec = torch.linspace(0, self.max_sec, int(16000 * self.max_sec))
            f0_vi = 160.0 + 40.0 * torch.sin(2 * math.pi * 2.0 * t_sec)
            src_waves.append(torch.sin(2 * math.pi * f0_vi * t_sec))
            f0_en = 130.0 + 30.0 * torch.sin(2 * math.pi * 1.5 * t_sec)
            tgt_waves.append(torch.sin(2 * math.pi * f0_en * t_sec))
            transcripts.append("synthetic english translation")
            src_transcripts.append("âm thanh mô phỏng tiếng việt")

        self.last_src_lengths = [int(w.shape[-1]) for w in src_waves]
        self.last_tgt_lengths = [int(w.shape[-1]) for w in tgt_waves]
        max_src_len = max(w.shape[-1] for w in src_waves)
        max_tgt_len = max(w.shape[-1] for w in tgt_waves)
        pad_src = torch.stack([F.pad(w, (0, max_src_len - w.shape[-1])) for w in src_waves])
        pad_tgt = torch.stack([F.pad(w, (0, max_tgt_len - w.shape[-1])) for w in tgt_waves])

        if return_text:
            return pad_src, pad_tgt, transcripts, src_transcripts
        return pad_src, pad_tgt

# ==============================================================================
# 4. CHƯƠNG TRÌNH HUẤN LUYỆN CHÍNH (MAIN TRAINER)
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Huấn luyện CrossAttentionBridge S2ST Vi -> En")
    parser.add_argument("--config", type=str, default="configs/stage2_bridge.yaml")
    parser.add_argument("--exp-name", type=str, default="bridge_vien")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sanity-check", action="store_true",
                        help="Kiểm tra gradient nhanh 50 bước trên 4 mẫu câu")
    parser.add_argument("--overfit-16", action="store_true",
                        help="Sanity Check #3: Huấn luyện overfit 500 bước trên 16 cặp câu Vi-En và xuất audio")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--cfm-ckpt", type=str, default="checkpoints/cfm_en-best.pth")
    parser.add_argument("--no-freeze-cfm", action="store_true")
    parser.add_argument("--overfit-steps", type=int, default=800,
                        help="Số bước overfit 16 câu (mặc định 800; kiểm tra khả năng học sâu: 3000)")
    parser.add_argument("--overfit-lr", type=float, default=None,
                        help="LR đỉnh khi overfit (mặc định = training.learning_rate); warmup 5% rồi cosine về 1e-5")
    parser.add_argument("--overfit-cond-only", action="store_true",
                        help="Chẩn đoán A: chỉ train L1(Bridge, c*) (+Dur), BỎ CFM loss -> Bridge có học thuộc được c* không?")
    parser.add_argument("--overfit-oracle-src", action="store_true",
                        help="Chẩn đoán B: đưa PhoWhisper của audio TIẾNG ANH (đích) vào Bridge thay cho audio Việt "
                             "(gần như ánh xạ đồng nhất) -> học không được = lỗi kiến trúc")
    parser.add_argument("--overfit-no-dropout", action="store_true",
                        help="Tắt dropout trong Bridge khi overfit (học thuộc 16 câu không bị nhiễu dropout cản)")
    parser.add_argument("--overfit-tag", type=str, default="",
                        help="Hậu tố thư mục audio overfit (outputs/sanity/bridge_overfit16<_tag>/) để các biến thể không ghi đè nhau")
    parser.add_argument("--allow-legacy-cfm", action="store_true",
                        help="Cho phép nạp CFM không có stage1_meta (KHÔNG khuyến nghị: dễ nạp nhầm bản En->Vi cũ)")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    train_cfg = cfg["training"]
    paths = cfg["paths"]
    loss_weights = cfg.get("loss_weights", {"cfm_weight": 1.0, "dur_weight": 0.1})

    log_file = os.path.join(paths["log_dir"], f"{args.exp_name}.log")
    sys.stdout = TeeLogger(log_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported() and train_cfg.get("bf16", True)) else torch.float32

    print("=" * 75)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN S2ST VIỆT -> ANH (CROSS-ATTENTION BRIDGE & OT-CFM)")
    print(f"   • Device: {device} | Precision: {dtype} | Batch Size: {train_cfg['batch_size']}")
    print(f"   • Cặp ngôn ngữ: {cfg['data'].get('src_lang', 'vi_vn')} -> {cfg['data'].get('tgt_lang', 'en_us')}")
    print("=" * 75)

    # 1. Nạp Whisper Feature Extractor & Encoder
    whisper_id = cfg["model"]["whisper_model"]
    feat_extractor = None
    whisper_enc = None
    try:
        from transformers import WhisperFeatureExtractor, WhisperModel
        print(f"⏳ Đang nạp Whisper Feature Extractor & Encoder ({whisper_id})...")
        feat_extractor = WhisperFeatureExtractor.from_pretrained(whisper_id)
        whisper_enc = WhisperModel.from_pretrained(whisper_id).encoder.eval().to(device=device, dtype=dtype)
        for p in whisper_enc.parameters():
            p.requires_grad = False
        print("✓ Whisper Encoder đã được đóng băng (Frozen) an toàn!")
    except Exception as e:
        print(f"[Whisper Warning] Chưa thể nạp transformers ({e}). Sẽ dùng đặc trưng mô phỏng.")

    # 2. Khởi tạo mô hình ViEnS2STSystem
    bridge_query_mode = cfg["model"].get("bridge_query_mode", "interp")
    bridge_norm = cfg["model"].get("bridge_norm", "prenorm_skip")
    use_ctc = float(loss_weights.get("ctc_weight", 0.0)) > 0
    bridge_ctc_vocab = EN_CTC_VOCAB if use_ctc else 0
    model = ViEnS2STSystem(
        encoder_dim=cfg["model"]["encoder_dim"],
        bridge_dim=cfg["model"]["bridge_dim"],
        bridge_layers=cfg["model"]["bridge_layers"],
        n_mels=cfg["model"]["n_mels"],
        spk_dim=cfg["model"]["spk_dim"],
        bridge_query_mode=bridge_query_mode,
        bridge_norm=bridge_norm,
        bridge_ctc_vocab=bridge_ctc_vocab,
    ).to(device=device)                                  # trọng số fp32
    model.amp_dtype = dtype if dtype == torch.bfloat16 else None
    print(f"⚙️ Mixed precision: trọng số fp32 | tính toán {'bf16 (autocast)' if model.amp_dtype else 'fp32'}")
    print(f"🌉 Bridge query mode: {bridge_query_mode} | norm: {bridge_norm} | "
          f"CTC phụ (ký tự Anh): {'BẬT, ' + str(bridge_ctc_vocab) + ' ký hiệu' if use_ctc else 'tắt'}")

    mel_fb = build_mel_filterbank(sr=24000, n_mels=100, device=device)

    # Nạp Stage 1 (CFM decoder + cond_proj teacher + spk_encoder)
    found_cfm = None
    for cp in [args.cfm_ckpt, "checkpoints/cfm_en-best.pth", "checkpoints/cfm_en-last.pth"]:
        if cp and os.path.exists(cp):
            found_cfm = cp
            break
    has_teacher = False
    if found_cfm:
        has_teacher = load_stage1_into(model, found_cfm, device, allow_legacy=args.allow_legacy_cfm)
        model.to(device=device, dtype=torch.float32)     # giữ trọng số fp32 sau khi nạp Stage 1
    else:
        print("[Cảnh báo] Không tìm thấy CFM Stage 1 (checkpoints/cfm_en-best.pth)! CFM sẽ dùng weights ngẫu nhiên.")
        if not (args.overfit_16 or args.sanity_check or args.smoke_test):
            raise RuntimeError("Train Stage 2 cần CFM Stage 1 đã train. Chạy: bash run.sh restore cfm_en  (hoặc start cfm_en).")

    # Encoder phía đích (nghe audio tiếng Anh -> c*, speaker). Khác encoder nguồn thì nạp riêng.
    tgt_enc_id = getattr(model, "target_encoder_id", None) or whisper_id
    feat_tgt, enc_tgt = feat_extractor, whisper_enc
    if tgt_enc_id != whisper_id:
        from transformers import WhisperFeatureExtractor, WhisperModel
        print(f"⏳ Nạp encoder PHÍA ĐÍCH {tgt_enc_id} (Stage 1 dùng encoder này; nguồn Vi vẫn là {whisper_id})...")
        feat_tgt = WhisperFeatureExtractor.from_pretrained(tgt_enc_id)
        enc_tgt = WhisperModel.from_pretrained(tgt_enc_id).encoder.eval().to(device=device, dtype=dtype)
        for p in enc_tgt.parameters():
            p.requires_grad = False
    print(f"ℹ️ Encoder nguồn (Vi -> Bridge): {whisper_id} | Encoder đích (c*, speaker): {tgt_enc_id}")

    def speaker_from_audio(audio_16k_padded, lengths):
        """Speaker embedding = spk_encoder(encoder_đích(audio)), tính TỪNG câu (không lẫn phần đệm).
        Lúc suy luận áp lên audio NGUỒN (giọng người nói Việt) -> giữ giọng."""
        embs = []
        for i, n in enumerate(lengths):
            lat = encode_whisper(audio_16k_padded[i:i + 1, :n], feat_tgt, enc_tgt, device, dtype)
            embs.append(model.speaker(lat))
        return torch.cat(embs, dim=0)

    if not has_teacher and float(loss_weights.get("cond_weight", 0)) > 0:
        print("⚠️ Không có cond_proj Stage 1 -> TẮT loss chưng cất L_cond (cond_weight = 0).")
        loss_weights = dict(loss_weights, cond_weight=0.0)

    should_freeze_cfm = not args.no_freeze_cfm
    if (args.overfit_16 or args.sanity_check) and not found_cfm:
        should_freeze_cfm = False
        print("💡 [Overfit Mode] Chưa có CFM pre-trained -> Mở toàn bộ trọng số (không freeze CFM) để End-to-End Bridge+CFM cùng overfit 16 câu!")

    if should_freeze_cfm:
        for p in model.decoder.parameters():
            p.requires_grad = False
        print("🔒 Đã ĐÓNG BĂNG (FREEZE) CFM Decoder. Toàn bộ áp lực học tập trung 100% vào CrossAttentionBridge!")
    # Teacher cond_proj luôn đóng băng; spk_encoder đóng băng khi đã có từ Stage 1
    for p in model.teacher_cond_proj.parameters():
        p.requires_grad = False
    if has_teacher:
        for p in model.spk_encoder.parameters():
            p.requires_grad = False
        print("🔒 Đóng băng spk_encoder Stage 1 | Train: speaker = audio ĐÍCH (En) | Suy luận: speaker = giọng NGUỒN (Vi)")
    print(f"ℹ️ Loss: {loss_weights.get('cfm_weight', 1.0)}·CFM + {loss_weights.get('dur_weight', 0.1)}·Dur "
          f"+ {loss_weights.get('cond_weight', 0.0)}·L1(Bridge, c*) + {loss_weights.get('ctc_weight', 0.0)}·CTC(ký tự Anh)")

    def ctc_ids(texts):
        return [en_text_to_ids(x) for x in texts] if use_ctc else None

    def ctc_decode_batch(n=None):
        """Giải mã tham lam CTC từ lần compute_loss gần nhất -> list chuỗi (độ dài thật theo mask)."""
        lp = getattr(model, "last_ctc_logp", None)
        if lp is None:
            return None
        ids = lp.argmax(-1).cpu()
        lens = getattr(model, "last_ctc_lens", None)
        k = ids.shape[0] if n is None else min(n, ids.shape[0])
        return [ctc_greedy_decode(ids[i, :int(lens[i])] if lens is not None else ids[i]) for i in range(k)]

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    num_total = sum(p.numel() for p in model.parameters())
    print(f"ℹ️ Số tham số huấn luyện (Trainable): {num_trainable:,} / {num_total:,} ({num_trainable/num_total*100:.1f}%)")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
        betas=(0.9, 0.98)
    )

    def featurize(src_16k, tgt_16k, src_lens, tgt_lens):
        """-> (latent Vi, Mel En đích, mask, dur_ratio từng câu, c* hoặc None, speaker của audio ĐÍCH hoặc None)."""
        if feat_extractor is not None and whisper_enc is not None:
            s_lat = encode_whisper(src_16k, feat_extractor, whisper_enc, device, dtype)
        else:
            s_lat = torch.randn(src_16k.shape[0], 100, 768, device=device, dtype=dtype)
        t24 = resample_16k_to_24k(tgt_16k.to(device=device, dtype=torch.float32))
        t_mel = extract_100_mel(t24, mel_fb).to(dtype=dtype)
        m_lens = mel_lengths(tgt_lens)
        mask = length_mask(m_lens, t_mel.shape[-1], device)
        # Tỉ lệ thời lượng THẬT của từng câu: frame Mel đích / frame latent nguồn
        l_lens = latent_lengths(src_lens)
        d_ratio = torch.tensor([m / max(1, l) for m, l in zip(m_lens, l_lens)], device=device, dtype=dtype)
        c_star, spk_tgt = None, None
        if has_teacher and feat_tgt is not None and enc_tgt is not None:
            t_lat = encode_whisper(tgt_16k, feat_tgt, enc_tgt, device, dtype)
            with torch.no_grad():
                c_star = model.teacher_condition(t_lat, t_mel.shape[-1])
                spk_tgt = model.speaker(t_lat)
        return s_lat, t_mel, mask, d_ratio, c_star, spk_tgt

    start_epoch = 0
    global_step = 0
    best_loss = float("inf")
    history = []

    ckpt_path = os.path.join(paths["checkpoint_dir"], f"{args.exp_name}-last.pth")
    if args.resume and os.path.exists(ckpt_path):
        print(f"[Resume] Đang khôi phục checkpoint từ: {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device)
        if "stage2_meta" not in ckpt:
            raise RuntimeError(f"{ckpt_path} không có stage2_meta (bản cũ?) — không resume. Dùng 'bash run.sh start {args.exp_name} --fresh'.")
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                print(f"[Optimizer Info] Khởi tạo optimizer mới ({e}).")
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("step", 0)
        best_loss = ckpt.get("best_loss", float("inf"))
        history = ckpt.get("history", [])
        print(f"✓ [Resume] Đã nạp thành công từ Epoch {start_epoch - 1}, Step {global_step}!")

    # 3. Pipeline Dữ liệu
    max_sec = cfg["data"].get("max_audio_sec", 12.0)
    src_l = cfg["data"].get("src_lang", "vi_vn")
    tgt_l = cfg["data"].get("tgt_lang", "en_us")
    data_pipeline = RealStreamingSpeechPipeline(
        split='train', max_sec=max_sec, src_lang=src_l, tgt_lang=tgt_l,
        allow_synthetic=args.smoke_test,
        max_pairs=None if (args.overfit_16 or args.sanity_check or args.smoke_test) else train_cfg.get("max_train_pairs", 4000))

    # 4. Chế độ Sanity Check & Overfit 16 mẫu
    if args.sanity_check or args.overfit_16:
        is_deep = args.overfit_16
        num_samples = 16 if is_deep else 4
        num_steps = int(args.overfit_steps) if is_deep else 50
        print("\n" + "=" * 75)
        print(f"🧪 BẮT ĐẦU KIỂM TRA OVERFIT ({num_steps} BƯỚC TRÊN {num_samples} CẶP CÂU VI -> EN)")
        print("Mục tiêu: Đảm bảo Gradient chảy thông suốt qua Cross-Attention Bridge và kiểm tra khả năng học.")
        print("=" * 75)
        model.train()
        fix_src, fix_tgt, fix_en_texts, fix_vi_texts = data_pipeline.fetch_batch(batch_size=num_samples)
        s_lat, tgt_m, fix_mask, d_ratio, c_star, spk_tgt = featurize(
            fix_src, fix_tgt, data_pipeline.last_src_lengths, data_pipeline.last_tgt_lengths)
        fix_src_lat_lens = latent_lengths(data_pipeline.last_src_lengths)
        print(f"✓ Latent tiếng Việt [Shape: {list(s_lat.shape)}] | Teacher c*: {'có' if c_star is not None else 'không'}")

        # ---- Các biến thể chẩn đoán ----
        variant = []
        if args.overfit_oracle_src:
            if feat_tgt is None or enc_tgt is None:
                raise RuntimeError("--overfit-oracle-src cần encoder phía đích (Stage 1 checkpoint).")
            s_lat = encode_whisper(fix_tgt, feat_tgt, enc_tgt, device, dtype)
            fix_src_lat_lens = latent_lengths(data_pipeline.last_tgt_lengths)
            d_ratio = torch.tensor([m / max(1, l) for m, l in zip(mel_lengths(data_pipeline.last_tgt_lengths),
                                                                    fix_src_lat_lens)], device=device, dtype=dtype)
            variant.append("ORACLE: đầu vào Bridge = PhoWhisper(audio TIẾNG ANH)")
            print(f"🔬 [Chẩn đoán B] Đầu vào Bridge = PhoWhisper(audio TIẾNG ANH đích) [Shape: {list(s_lat.shape)}] "
                  f"-> gần như ánh xạ đồng nhất, phải học được dễ dàng.")
        ovf_weights = dict(loss_weights)
        if args.overfit_cond_only:
            if c_star is None:
                raise RuntimeError("--overfit-cond-only cần c* (Stage 1 checkpoint có teacher cond_proj).")
            ovf_weights.update(cfm_weight=0.0, cond_weight=1.0)
            variant.append("CHỈ L1(c*) (bỏ CFM loss)")
            print("🔬 [Chẩn đoán A] Chỉ train L1(Bridge, c*) + 0.1·Dur, BỎ CFM loss.")
        if args.overfit_no_dropout:
            n_drop = 0
            for m_ in model.bridge.modules():
                if isinstance(m_, nn.Dropout):
                    m_.p = 0.0
                    n_drop += 1
                elif isinstance(m_, nn.MultiheadAttention):
                    m_.dropout = 0.0
                    n_drop += 1
            variant.append("không dropout")
            print(f"🔬 Đã tắt dropout trong Bridge ({n_drop} lớp).")
        print(f"ℹ️ Biến thể overfit: {', '.join(variant) if variant else 'mặc định (đủ loss, Vi -> Bridge)'} | "
              f"loss = {ovf_weights.get('cfm_weight', 1.0)}·CFM + {ovf_weights.get('dur_weight', 0.1)}·Dur "
              f"+ {ovf_weights.get('cond_weight', 0.0)}·L1(c*)")

        # Mốc so sánh cho Cond: L1 nếu Bridge chỉ đoán trung bình c* theo thời gian của từng câu
        cond_trivial0 = None
        if c_star is not None:
            with torch.no_grad():
                m3_ = fix_mask.unsqueeze(1).to(c_star.dtype)
                c_mean0 = (c_star * m3_).sum(-1, keepdim=True) / m3_.sum(-1, keepdim=True).clamp(min=1.0)
                cond_trivial0 = masked_l1(c_mean0.expand_as(c_star), c_star, fix_mask).item()
            print(f"ℹ️ Mốc Cond 'đoán trung bình theo thời gian' = {cond_trivial0:.4f} "
                  f"(Cond phải xuống RÕ dưới mốc này; học thuộc thật -> gần 0)")

        peak_lr = float(args.overfit_lr) if args.overfit_lr else float(train_cfg["learning_rate"])
        for g_ in optimizer.param_groups:
            g_["lr"] = peak_lr
        warm = max(1, int(0.05 * num_steps))
        min_ratio = 1e-5 / peak_lr

        def _ovf_lr(st):   # warmup tuyến tính 5% rồi cosine về 1e-5
            if st < warm:
                return (st + 1) / warm
            prog = (st - warm) / max(1, num_steps - warm)
            return max(min_ratio, 0.5 * (1.0 + math.cos(math.pi * prog)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _ovf_lr)
        print(f"ℹ️ Overfit: {num_steps} bước | LR đỉnh {peak_lr:.1e} (warmup {warm} bước, cosine về 1e-5)")
        initial_loss = None
        final_loss = None
        t_start = time.time()
        for st in range(num_steps):
            optimizer.zero_grad()
            l_tot, cfm_l, d_l, c_l = model.compute_loss(
                s_lat, tgt_m, speaker_emb=spk_tgt, target_dur_ratio=d_ratio, loss_weights=ovf_weights,
                mel_mask=fix_mask, cond_target=c_star, src_lengths=fix_src_lat_lens, ctc_targets=ctc_ids(fix_en_texts))
            l_tot.backward()
            nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if st == 0: initial_loss = l_tot.item()
            log_interval = max(50, num_steps // 16) if is_deep else 10
            if (st + 1) % log_interval == 0 or st == 0:
                ctc_s = f" | CTC: {model.last_ctc_loss.item():.3f}" if model.last_ctc_loss is not None else ""
                print(f"   [Overfit Step {st+1:04d}/{num_steps}] Total: {l_tot.item():.4f} | CFM: {cfm_l.item():.4f} "
                      f"| Dur: {d_l.item():.4f} | Cond: {c_l.item():.4f}{ctc_s} | LR: {scheduler.get_last_lr()[0]:.1e}")
            final_loss = l_tot.item()

        print("-" * 75)
        if use_ctc:
            dec = ctc_decode_batch()
            if dec:
                cer = char_error_rate(dec, fix_en_texts[:len(dec)])
                print(f"🔤 CTC (giải mã tham lam, train 16 câu): CER = {cer*100:.1f}%")
                for i in range(min(2, len(dec))):
                    print(f"      ref : {fix_en_texts[i][:90]}\n      ctc : {dec[i][:90]}")
        print(f"📉 Loss tổng: {initial_loss:.4f} -> {final_loss:.4f} (giảm {(1.0 - final_loss/initial_loss)*100.0:.1f}%) "
              f"trong {time.time()-t_start:.1f}s. (CFM loss có sàn nhiễu nên KHÔNG dùng % này để kết luận — xem phần dưới)")

        # Kiểm tra điều kiện c (trung bình 8 lần lấy t/x0 cố định để bớt nhiễu):
        #   L(c*)   = CFM loss với điều kiện ĐÚNG (mốc tốt nhất decoder có thể đạt)
        #   L(0)    = CFM loss khi bỏ hẳn điều kiện (mốc tệ nhất)
        #   L(Bridge) nằm giữa -> "% khoảng cách đã thu hẹp" = (L(0) - L(Bridge)) / (L(0) - L(c*))
        model.eval()
        with torch.no_grad(), model._amp():
            g = torch.Generator(device="cpu").manual_seed(0)
            lat, _ = model.bridge(model._in(s_lat), target_len=tgt_m.shape[-1],
                                  src_lengths=fix_src_lat_lens, tgt_lengths=fix_mask.sum(-1).long())
            c_bridge = lat.transpose(1, 2)
            spk = model._in(spk_tgt) if spk_tgt is not None else model.speaker(s_lat)
            c_mean = None
            if c_star is not None:   # mốc trung bình: trung bình c* theo thời gian (chỉ frame THẬT) của từng câu
                m3 = fix_mask.unsqueeze(1).to(c_star.dtype)
                c_mean = (c_star * m3).sum(-1, keepdim=True) / m3.sum(-1, keepdim=True).clamp(min=1.0)
            sums = {"c*": 0.0, "Bridge": 0.0, "TB(c*)": 0.0, "0": 0.0}
            for _ in range(8):
                tt = torch.rand(num_samples, generator=g).to(device=device, dtype=dtype)
                x0 = torch.randn(tgt_m.shape, generator=g).to(device=device, dtype=dtype)
                xt = (1 - tt.view(-1, 1, 1)) * x0 + tt.view(-1, 1, 1) * tgt_m
                ut = tgt_m - x0
                c_mean_full = c_mean.expand_as(c_star) if c_mean is not None else None
                for name, cc in [("c*", c_star), ("Bridge", c_bridge), ("TB(c*)", c_mean_full),
                                 ("0", torch.zeros_like(c_bridge))]:
                    if cc is None:
                        continue
                    sums[name] += masked_mse(model.decoder(model._in(xt), model._in(tt), model._in(cc), spk, mask=fix_mask),
                                             ut, fix_mask).item() / 8
            cond_trivial = masked_l1(c_mean.expand_as(c_star), c_star, fix_mask).item() \
                if c_star is not None else float("nan")
            cond_bridge = masked_l1(c_bridge, c_star, fix_mask).item() if c_star is not None else float("nan")
        l_star, l_br, l_0, l_mean = sums["c*"], sums["Bridge"], sums["0"], sums["TB(c*)"]
        print(f"🔎 CFM loss: với c* (đúng) = {l_star:.4f} | với c của Bridge = {l_br:.4f} | "
              f"với trung bình c* = {l_mean:.4f} | với c = 0 = {l_0:.4f}")
        if c_star is not None:
            if l_br < l_mean - 0.005:
                print("   ✓ Bridge TỐT HƠN mốc 'trung bình c*' -> đã học được thông tin theo từng frame (thời gian).")
            else:
                print("   ⚠️ Bridge chưa tốt hơn mốc 'trung bình c*' -> mới học được thông tin mức câu, chưa theo từng frame.")
        if c_star is not None and l_0 - l_star > 1e-4:
            closed = (l_0 - l_br) / (l_0 - l_star) * 100.0
            print(f"   Bridge thu hẹp {closed:.0f}% khoảng cách giữa 'không có điều kiện' và 'điều kiện đúng' "
                  f"(decoder dùng c: L(0)/L(c*) = {l_0 / max(1e-8, l_star):.2f}x)")
            print(f"   L1(Bridge, c*) = {cond_bridge:.4f} | mốc 'đoán trung bình theo thời gian' = {cond_trivial:.4f}")
        if c_star is not None:
            ratio = cond_bridge / max(1e-8, cond_trivial)
            if ratio < 0.5:
                verdict = "✅ OVERFIT PASS: Bridge HỌC THUỘC được c* (L1 < 50% mốc trung bình)."
            elif ratio < 0.9:
                verdict = "🟡 Bridge học được một phần c* (L1 50–90% mốc trung bình)."
            else:
                verdict = "❌ OVERFIT FAIL: L1(Bridge, c*) không xuống dưới mốc trung bình -> Bridge KHÔNG học thuộc được c*."
            print(f"🏁 KẾT LUẬN [{', '.join(variant) if variant else 'mặc định'}]: L1/mốc = {ratio:.2f} -> {verdict}")

        # Nếu là Overfit sâu 800 bước -> Xuất trọn bộ 16 file audio nghe thử qua Vocos
        if is_deep:
            out_dir = "outputs/sanity/bridge_overfit16" + (f"_{args.overfit_tag}" if args.overfit_tag else "")
            legacy_dir = "outputs/sanity"
            os.makedirs(out_dir, exist_ok=True)
            os.makedirs(legacy_dir, exist_ok=True)
            try:
                vocos = load_vocos_safely("charactr/vocos-mel-24khz", device=device)
                # Suy luận đúng như lúc dùng thật: speaker = giọng NGUỒN, độ dài = DurationPredictor
                with torch.no_grad():
                    spk_src = speaker_from_audio(fix_src, data_pipeline.last_src_lengths)
                gen_mels = model.inference(s_lat, speaker_emb=spk_src, num_steps=32)
                gen_mels_oracle = model.inference(s_lat, speaker_emb=spk_src, target_len=tgt_m.shape[-1], num_steps=32)
                with torch.no_grad():
                    print(f"\n🎧 ĐANG GIẢI MÃ VÀ XUẤT TOÀN BỘ {len(fix_src)} FILE AUDIO VI -> EN QUA VOCOS...")
                    for idx in range(len(fix_src)):
                        pred_wav = vocos.decode(gen_mels[idx:idx+1].float()).squeeze().cpu()
                        pred_oracle_wav = vocos.decode(gen_mels_oracle[idx:idx+1].float()).squeeze().cpu()
                        vocos_upper_wav = vocos.decode(tgt_m[idx:idx+1].float()).squeeze().cpu()
                        gt_wav = resample_16k_to_24k(fix_tgt[idx:idx+1].float()).squeeze().cpu()
                        src_wav_16k = fix_src[idx].cpu()

                        save_wav_file(os.path.join(out_dir, f"sample_{idx:02d}_vi_input.wav"), src_wav_16k, 16000)
                        save_wav_file(os.path.join(out_dir, f"sample_{idx:02d}_vi_to_en_pred.wav"), pred_wav, 24000)
                        save_wav_file(os.path.join(out_dir, f"sample_{idx:02d}_vi_to_en_pred_oracle.wav"), pred_oracle_wav, 24000)
                        save_wav_file(os.path.join(out_dir, f"sample_{idx:02d}_en_mel_vocos.wav"), vocos_upper_wav, 24000)
                        save_wav_file(os.path.join(out_dir, f"sample_{idx:02d}_en_groundtruth.wav"), gt_wav, 24000)

                        if idx == 0:
                            save_wav_file(os.path.join(legacy_dir, "bridge_vi_en_overfit16_sample0.wav"), pred_wav, 24000)

                        vi_p = fix_vi_texts[idx][:35] + "..." if len(fix_vi_texts[idx]) > 35 else fix_vi_texts[idx]
                        en_p = fix_en_texts[idx][:35] + "..." if len(fix_en_texts[idx]) > 35 else fix_en_texts[idx]
                        print(f"   [Cặp #{idx:02d}/16] Vi: \"{vi_p}\" -> En: \"{en_p}\"")
                print("-" * 75)
                print(f"✓ ĐÃ XUẤT TRỌN BỘ 16 MẪU ÂM THANH DỊCH VI -> EN (INPUT, PRED, PRED_ORACLE, MEL_VOCOS & GT) TẠI:")
                print(f"  👉 Thư mục: {out_dir}/")
            except Exception as e:
                print(f"[Vocos Info] Chưa xuất được audio: {e}")

        print("=" * 75 + "\n")
        return

    if feat_extractor is None or whisper_enc is None:
        if not args.smoke_test:
            raise RuntimeError("Không nạp được Whisper encoder — dừng để tránh train trên latent ngẫu nhiên.")

    epochs = 1 if args.smoke_test else train_cfg["epochs"]
    batch_size = train_cfg["batch_size"]
    num_pairs = len(data_pipeline._parallel_pairs) if data_pipeline._parallel_pairs else 100
    auto_steps = max(20, num_pairs // batch_size)
    steps_per_epoch = 2 if args.smoke_test else auto_steps

    total_train_steps = epochs * steps_per_epoch
    warmup_steps = min(train_cfg.get("warmup_steps", 150), max(10, total_train_steps // 6))

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_train_steps - warmup_steps))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    for _ in range(global_step):     # resume: đưa LR về đúng vị trí
        scheduler.step()

    # ---- Tập validation CỐ ĐỊNH (FLEURS validation, cùng t / x0 mỗi epoch) -> chọn best khách quan ----
    val_pairs = 4 if args.smoke_test else int(train_cfg.get("val_pairs", 16))
    val_pipeline = RealStreamingSpeechPipeline(split='validation', max_sec=max_sec, src_lang=src_l, tgt_lang=tgt_l,
                                               allow_synthetic=args.smoke_test)
    val_set = []
    g = torch.Generator(device="cpu").manual_seed(1234)
    for _ in range((val_pairs + batch_size - 1) // batch_size):
        v_src, v_tgt, v_en, _ = val_pipeline.fetch_batch(batch_size=batch_size)
        feats = featurize(v_src, v_tgt, val_pipeline.last_src_lengths, val_pipeline.last_tgt_lengths)
        tt = torch.rand(v_src.shape[0], generator=g)
        x0 = torch.randn(feats[1].shape, generator=g)
        val_set.append((v_src, list(val_pipeline.last_src_lengths), v_en, feats, tt, x0))
    print(f"✓ Tập validation cố định: {sum(v[0].shape[0] for v in val_set)} cặp (FLEURS validation, seed 1234)")

    eval_bleu_every = int(train_cfg.get("eval_asr_every", 5))
    en_asr = EnglishASR(device)
    vocos_eval = None
    sync_enabled = bool(train_cfg.get("sync_drive", True)) and not args.smoke_test

    print(f"\n[Training Config] Bắt đầu: {epochs} Epochs | {steps_per_epoch} Steps/Epoch (~{num_pairs} câu) | Batch: {batch_size}")
    print(f"   → Tổng cộng: {total_train_steps} Steps | Warmup: {warmup_steps} Steps | LR Scheduler: Cosine Warmup")
    print("-" * 75)

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step in range(steps_per_epoch):
            optimizer.zero_grad()
            src_audio_16k, tgt_audio_16k, en_texts, vi_texts = data_pipeline.fetch_batch(batch_size=batch_size)
            src_latents, tgt_mel, mel_mask, dur_ratio, c_star, spk_tgt = featurize(
                src_audio_16k, tgt_audio_16k, data_pipeline.last_src_lengths, data_pipeline.last_tgt_lengths)

            total_loss, cfm_l, dur_l, cond_l = model.compute_loss(
                src_latents, tgt_mel, speaker_emb=spk_tgt, target_dur_ratio=dur_ratio,
                loss_weights=loss_weights, mel_mask=mel_mask, cond_target=c_star,
                src_lengths=latent_lengths(data_pipeline.last_src_lengths), ctc_targets=ctc_ids(en_texts)
            )

            total_loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, max_norm=float(train_cfg["max_grad_norm"]))
            optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_loss += total_loss.item()

            if global_step % 10 == 0 or args.smoke_test:
                cur_lr = scheduler.get_last_lr()[0]
                ctc_s = f" | CTC: {model.last_ctc_loss.item():.3f}" if model.last_ctc_loss is not None else ""
                print(f"   [E{epoch:02d}|Step {global_step:04d}] Total: {total_loss.item():.4f} | "
                      f"CFM: {cfm_l.item():.4f} | Dur: {dur_l.item():.4f} | Cond: {cond_l.item():.4f}{ctc_s} | LR: {cur_lr:.2e}")

        avg_loss = epoch_loss / max(1, steps_per_epoch)
        elapsed = time.time() - t0
        print(f"==> Hoàn tất Epoch {epoch:02d} trong {elapsed:.1f}s | Avg Train Loss: {avg_loss:.4f}")

        data_pipeline.shuffle()

        # Validation cố định
        model.eval()
        v_tot = v_cfm = v_dur = v_cond = v_ctc = 0.0
        v_ctc_hyp, v_ctc_ref = [], []
        with torch.no_grad():
            for v_src, v_lens, v_en, (s_lat, t_mel, msk, dr, cs, spk), tt, x0 in val_set:
                lt, lc, ld, lco = model.compute_loss(
                    s_lat, t_mel, speaker_emb=spk, target_dur_ratio=dr, loss_weights=loss_weights,
                    mel_mask=msk, cond_target=cs, t=tt.to(device=device, dtype=dtype), x_0=x0.to(device=device, dtype=dtype),
                    src_lengths=latent_lengths(v_lens), ctc_targets=ctc_ids(v_en))
                v_tot += lt.item(); v_cfm += lc.item(); v_dur += ld.item(); v_cond += lco.item()
                if model.last_ctc_loss is not None:
                    v_ctc += model.last_ctc_loss.item()
                    v_ctc_hyp += ctc_decode_batch(); v_ctc_ref += list(v_en)
        nv = max(1, len(val_set))
        val_loss = v_tot / nv
        ctc_s = ""
        if v_ctc_hyp:
            v_cer = char_error_rate(v_ctc_hyp, v_ctc_ref)
            ctc_s = f" | CTC {v_ctc/nv:.3f} (CER {v_cer*100:.1f}%)"
        print(f"   Val (cố định): Total {val_loss:.4f} | CFM {v_cfm/nv:.4f} | Dur {v_dur/nv:.4f} | Cond {v_cond/nv:.4f}{ctc_s} "
              f"| Best: {min(best_loss, val_loss):.4f}")
        if v_ctc_hyp:
            # Bridge "hiểu" câu đích đến đâu (độc lập với chất lượng âm thanh): giải mã CTC 1 câu val
            print(f"      ctc ref : {v_ctc_ref[0][:90]}\n      ctc pred: {v_ctc_hyp[0][:90]}")

        # ASR-BLEU nhanh trên tập val (suy luận như thật: độ dài dự đoán + giọng nguồn)
        val_bleu = None
        if not args.smoke_test and eval_bleu_every > 0 and ((epoch + 1) % eval_bleu_every == 0 or epoch == epochs - 1):
            try:
                if vocos_eval is None:
                    vocos_eval = load_vocos_safely("charactr/vocos-mel-24khz", device=device)
                    vocos_eval.eval()
                hyps, refs = [], []
                for v_src, v_lens, v_en, (s_lat, *_), _, _ in val_set:
                    for i in range(v_src.shape[0]):
                        li = latent_lengths([v_lens[i]])[0]
                        with torch.no_grad():
                            spk_i = speaker_from_audio(v_src[i:i + 1], [v_lens[i]])
                        mel = model.inference(s_lat[i:i + 1, :max(10, li)], speaker_emb=spk_i, num_steps=32)
                        with torch.no_grad():
                            w24 = vocos_eval.decode(mel.float()).squeeze().cpu()
                        hyps.append(en_asr.transcribe(resample_24k_to_16k_fft(w24)))
                        refs.append(v_en[i])
                try:
                    import sacrebleu
                    from train_utils import normalize_en
                    val_bleu = sacrebleu.corpus_bleu([normalize_en(h) for h in hyps], [[normalize_en(r) for r in refs]]).score
                except Exception:
                    val_bleu = None
                wer = corpus_wer(hyps, refs)
                print(f"   🎧 Val ASR-BLEU: {val_bleu if val_bleu is not None else float('nan'):.2f} | WER: {wer:.1f}%")
                for h, r in list(zip(hyps, refs))[:2]:
                    print(f"      • ref : {r[:70]}\n        pred: {h[:70]}")
            except Exception as e:
                print(f"   [Val ASR] Không đo được: {e}")
        history.append({"epoch": epoch, "train": avg_loss, "val": val_loss, "val_cond": v_cond / nv,
                        "val_cfm": v_cfm / nv, "val_bleu": val_bleu})

        # Lưu Checkpoint Last & Best (best theo VAL cố định, không theo train loss)
        os.makedirs(paths["checkpoint_dir"], exist_ok=True)
        ckpt_data = {
            "epoch": epoch,
            "step": global_step,
            "best_loss": min(best_loss, val_loss),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "stage2_meta": {"src": src_l, "tgt": tgt_l, "cfm_ckpt": found_cfm, "has_teacher": has_teacher,
                            "source_encoder": whisper_id, "target_encoder": tgt_enc_id,
                            "bridge_query_mode": bridge_query_mode, "bridge_norm": bridge_norm,
                            "bridge_ctc_vocab": bridge_ctc_vocab,
                            "decoder_config": model.decoder.config,
                            "loss_weights": loss_weights, "version": 2},
            "config": cfg,
        }
        last_p = os.path.join(paths["checkpoint_dir"], f"{args.exp_name}-last.pth")
        torch.save(ckpt_data, last_p)
        synced = [last_p]
        if val_loss < best_loss:
            best_loss = val_loss
            best_p = os.path.join(paths["checkpoint_dir"], f"{args.exp_name}-best.pth")
            torch.save(ckpt_data, best_p)
            synced.append(best_p)
            print(f"⭐ [Epoch {epoch:02d}] Đã cập nhật Model Best (val): {best_loss:.4f}")
        sync_to_drive(synced, args.exp_name, enabled=sync_enabled)

    print("\n🎉 HUẤN LUYỆN CROSS-ATTENTION BRIDGE (VI -> EN) HOÀN TẤT THÀNH CÔNG!")

if __name__ == "__main__":
    main()
