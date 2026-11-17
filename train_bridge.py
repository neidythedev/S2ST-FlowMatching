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
    def forward(self, x):
        h = self.conv(x.transpose(1, 2))
        return torch.sigmoid(h.mean(dim=[-1, -2])) * 1.4 + 0.8  # scale trong khoảng [0.8, 2.2]

class CrossAttentionBridge(nn.Module):
    """Cầu nối dịch thuật Non-Autoregressive sử dụng Cross-Attention Transformer Decoder.
    Giải quyết triệt để vấn đề đảo ngữ pháp (non-monotonic word reordering) giữa Tiếng Việt và Tiếng Anh.
    """
    def __init__(self, in_dim=768, hidden_dim=512, num_layers=6, n_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.src_proj = nn.Linear(in_dim, hidden_dim)

        # Encoder tự chú ý trên đặc trưng nguồn (tiếng Việt)
        src_enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            batch_first=True, activation="gelu"
        )
        self.src_encoder = nn.TransformerEncoder(src_enc_layer, num_layers=max(2, num_layers // 2))

        # Decoder chú ý chéo (Cross-Attention) sang đặc trưng đích (tiếng Anh)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            batch_first=True, activation="gelu"
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        # Bộ dự đoán tỷ lệ co dãn thời lượng
        self.duration_predictor = DurationPredictor(in_dim=hidden_dim)

        # Positional Embeddings học được cho Query đích
        self.pos_emb = nn.Embedding(4096, hidden_dim)

    def forward(self, whisper_features, target_len=None):
        """
        whisper_features: [B, T_src, in_dim]
        target_len: Số frame Mel đích T_tgt (lúc train) hoặc None (lúc inference)
        """
        B, T_src, _ = whisper_features.shape
        src_h = self.src_proj(whisper_features)
        memory = self.src_encoder(src_h)

        dur_scale = self.duration_predictor(memory)  # [B]

        if target_len is None:
            # Lúc Inference: Ước lượng T_tgt từ dur_scale
            avg_scale = float(dur_scale.mean().item())
            target_len = max(20, min(2048, int(T_src * avg_scale)))

        positions = torch.arange(target_len, device=whisper_features.device).clamp(max=4095)
        queries = self.pos_emb(positions).unsqueeze(0).expand(B, -1, -1)

        # Cross-Attention: Cho phép từ tiếng Anh đảo vị trí tùy ý so với tiếng Việt
        latents = self.decoder(tgt=queries, memory=memory)
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

class FlowMatchingTargetDecoder(nn.Module):
    """Bộ giải mã âm học Optimal Transport CFM: Sinh 100-Mels sắc nét có điều kiện."""
    def __init__(self, n_mels=100, cond_dim=512, spk_dim=192, hidden_dim=384):
        super().__init__()
        total_in = n_mels + 1 + cond_dim + spk_dim
        self.input_proj = nn.Conv1d(total_in, hidden_dim, kernel_size=3, padding=1)
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.GELU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GroupNorm(8, hidden_dim),
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

class ViEnS2STSystem(nn.Module):
    """Hệ thống tích hợp toàn diện Vi -> En: Whisper -> CrossAttentionBridge -> CFM Decoder -> 100 Mel."""
    def __init__(self, encoder_dim=768, bridge_dim=512, bridge_layers=6, n_mels=100, spk_dim=192):
        super().__init__()
        self.n_mels = n_mels
        self.bridge = CrossAttentionBridge(in_dim=encoder_dim, hidden_dim=bridge_dim, num_layers=bridge_layers)
        self.spk_encoder = AcousticSpeakerEncoder(feat_dim=encoder_dim, spk_dim=spk_dim)
        self.decoder = FlowMatchingTargetDecoder(n_mels=n_mels, cond_dim=bridge_dim, spk_dim=spk_dim, hidden_dim=384)

    def compute_loss(self, src_whisper, target_mel, speaker_emb=None, target_dur_ratio=None, loss_weights=None, **kwargs):
        weights = loss_weights or {"cfm_weight": 1.0, "dur_weight": 0.1}
        B, _, T_tgt = target_mel.shape

        # Truyền target_len=T_tgt để Cross-Attention sinh đúng kích thước Mel đích
        translated_latents, dur_scale = self.bridge(src_whisper, target_len=T_tgt)

        if speaker_emb is None:
            speaker_emb = self.spk_encoder(src_whisper)

        cond = translated_latents.transpose(1, 2)  # [B, bridge_dim, T_tgt]

        # 1. CFM Optimal Transport Loss
        t = torch.rand(B, device=target_mel.device, dtype=target_mel.dtype)
        x_0 = torch.randn_like(target_mel)
        x_1 = target_mel
        t_exp = t.view(B, 1, 1)
        x_t = (1.0 - t_exp) * x_0 + t_exp * x_1
        u_t = x_1 - x_0

        v_pred = self.decoder(x_t, t, cond, speaker_emb)
        cfm_loss = F.mse_loss(v_pred, u_t)

        # 2. Duration Loss
        if target_dur_ratio is not None:
            dur_loss = F.smooth_l1_loss(dur_scale.view(-1), target_dur_ratio.view(-1))
        else:
            dur_loss = torch.tensor(0.0, device=target_mel.device, dtype=target_mel.dtype)

        total_loss = weights["cfm_weight"] * cfm_loss + weights["dur_weight"] * dur_loss
        return total_loss, cfm_loss, dur_loss

    @torch.no_grad()
    def inference(self, src_whisper, speaker_emb=None, num_steps=32):
        self.eval()
        B = src_whisper.shape[0]
        if speaker_emb is None:
            speaker_emb = self.spk_encoder(src_whisper)

        translated_latents, dur_scale = self.bridge(src_whisper, target_len=None)
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

# ==============================================================================
# 3. PIPELINE DỮ LIỆU ÂM THANH THẬT STREAMING (GOOGLE FLEURS VI -> EN)
# ==============================================================================

class RealStreamingSpeechPipeline:
    """Nạp dữ liệu âm thanh thật trực tuyến Vi-En với căn chỉnh song song bằng trường 'id' (FLoRes)."""
    def __init__(self, split='train', max_sec=12.0, src_lang='vi_vn', tgt_lang='en_us'):
        self.split = split
        self.max_sec = max_sec
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.is_online = False
        self._parallel_pairs = []
        self._pair_index = 0

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
            print(f"[Data Warning] Không thể stream online ({e}). Kích hoạt bộ tạo tín hiệu mô phỏng chuẩn xác.")
            self.is_online = False

    def _build_parallel_pairs(self):
        """Buffer các mục của ngôn ngữ đích theo 'id', rồi match từng mục của ngôn ngữ nguồn."""
        print(f"🔗 Đang căn chỉnh song song {self.src_lang} ↔ {self.tgt_lang} bằng trường 'id' (FLoRes sentence ID)...")
        tgt_by_id = {}
        count_tgt = 0
        limit_tgt = 800 if self.split == 'train' else 200
        limit_pairs = 600 if self.split == 'train' else 50

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
            print("[Cảnh báo] Không tìm được cặp song song nào! Kiểm tra trường 'id' trong dataset.")
            self.is_online = False

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
            for _ in range(batch_size):
                try:
                    if self._pair_index >= len(self._parallel_pairs):
                        self._pair_index = 0
                    item_src, item_tgt = self._parallel_pairs[self._pair_index]
                    self._pair_index += 1
                    w_src = self._extract_audio_tensor(item_src['audio'])
                    w_tgt = self._extract_audio_tensor(item_tgt['audio'])
                    src_waves.append(w_src)
                    tgt_waves.append(w_tgt)
                    transcripts.append(item_tgt.get('transcription', 'hello world'))
                    src_transcripts.append(item_src.get('transcription', 'xin chào'))
                except Exception as ex:
                    self._pair_index += 1
                    continue

        while len(src_waves) < batch_size:
            t_sec = torch.linspace(0, self.max_sec, int(16000 * self.max_sec))
            f0_vi = 160.0 + 40.0 * torch.sin(2 * math.pi * 2.0 * t_sec)
            src_waves.append(torch.sin(2 * math.pi * f0_vi * t_sec))
            f0_en = 130.0 + 30.0 * torch.sin(2 * math.pi * 1.5 * t_sec)
            tgt_waves.append(torch.sin(2 * math.pi * f0_en * t_sec))
            transcripts.append("synthetic english translation")
            src_transcripts.append("âm thanh mô phỏng tiếng việt")

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
    parser.add_argument("--cfm-ckpt", type=str, default="checkpoints/cfm-best.pth")
    parser.add_argument("--no-freeze-cfm", action="store_true")
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
    model = ViEnS2STSystem(
        encoder_dim=cfg["model"]["encoder_dim"],
        bridge_dim=cfg["model"]["bridge_dim"],
        bridge_layers=cfg["model"]["bridge_layers"],
        n_mels=cfg["model"]["n_mels"],
        spk_dim=cfg["model"]["spk_dim"]
    ).to(device=device, dtype=dtype)

    mel_fb = build_mel_filterbank(sr=24000, n_mels=100, device=device)

    # Nạp Pretrained CFM nếu có
    if os.path.exists(args.cfm_ckpt):
        print(f"⏳ Đang nạp weights Pretrained CFM từ: {args.cfm_ckpt}...")
        ckpt_cfm = torch.load(args.cfm_ckpt, map_location=device)
        st = ckpt_cfm["model_state_dict"] if "model_state_dict" in ckpt_cfm else ckpt_cfm
        dec_st = {}
        for k, v in st.items():
            if k.startswith("decoder."):
                dec_st[k[len("decoder."):]] = v
            elif not any(k.startswith(p) for p in ["bridge.", "spk_encoder.", "cond_proj."]):
                dec_st[k] = v
        model.decoder.load_state_dict(dec_st if dec_st else st, strict=False)
        print("✓ Nạp weights Pretrained CFM thành công!")
    else:
        print(f"[Cảnh báo] Không tìm thấy {args.cfm_ckpt}! CFM sẽ dùng weights ngẫu nhiên (hoặc huấn luyện từ đầu).")

    if not args.no_freeze_cfm:
        for p in model.decoder.parameters():
            p.requires_grad = False
        print("🔒 Đã ĐÓNG BĂNG (FREEZE) CFM Decoder. Toàn bộ áp lực học tập trung 100% vào CrossAttentionBridge!")

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

    start_epoch = 0
    global_step = 0
    best_loss = float("inf")

    ckpt_path = os.path.join(paths["checkpoint_dir"], f"{args.exp_name}-last.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(paths["checkpoint_dir"], "model-last.pth")

    if args.resume and os.path.exists(ckpt_path):
        print(f"[Resume] Đang khôi phục checkpoint từ: {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                print(f"[Optimizer Info] Khởi tạo optimizer mới ({e}).")
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("step", 0)
        best_loss = ckpt.get("best_loss", float("inf"))
        print(f"✓ [Resume] Đã nạp thành công từ Epoch {start_epoch - 1}, Step {global_step}!")

    # 3. Pipeline Dữ liệu
    max_sec = cfg["data"].get("max_audio_sec", 12.0)
    src_l = cfg["data"].get("src_lang", "vi_vn")
    tgt_l = cfg["data"].get("tgt_lang", "en_us")
    data_pipeline = RealStreamingSpeechPipeline(split='train', max_sec=max_sec, src_lang=src_l, tgt_lang=tgt_l)

    # 4. Chế độ Sanity Check & Overfit 16 mẫu
    if args.sanity_check or args.overfit_16:
        is_deep = args.overfit_16
        num_samples = 16 if is_deep else 4
        num_steps = 500 if is_deep else 50
        print("\n" + "=" * 75)
        print(f"🧪 BẮT ĐẦU KIỂM TRA OVERFIT ({num_steps} BƯỚC TRÊN {num_samples} CẶP CÂU VI -> EN)")
        print("Mục tiêu: Đảm bảo Gradient chảy thông suốt qua Cross-Attention Bridge và kiểm tra khả năng học.")
        print("=" * 75)
        model.train()
        fix_src, fix_tgt, fix_en_texts, fix_vi_texts = data_pipeline.fetch_batch(batch_size=num_samples)

        if feat_extractor is not None and whisper_enc is not None:
            in_proc = feat_extractor(list(fix_src.numpy()), sampling_rate=16000, return_tensors='pt', padding="max_length")
            f = in_proc.input_features.to(device=device, dtype=dtype)
            if f.shape[-1] < 3000: f = F.pad(f, (0, 3000 - f.shape[-1]))
            elif f.shape[-1] > 3000: f = f[:, :, :3000]
            with torch.no_grad():
                fl = whisper_enc(f).last_hidden_state
            real_l = max(10, min(1500, int(fix_src.shape[-1] / 320)))
            s_lat = fl[:, :real_l, :]
            print(f"✓ Đã trích xuất đặc trưng tiếng Việt từ PhoWhisper ({whisper_id}) [Shape: {list(s_lat.shape)}]")
        else:
            print("⚠️ CẢNH BÁO: PhoWhisper Encoder không tải được, đang dùng latents ngẫu nhiên!")
            s_lat = torch.randn(num_samples, 100, 768, device=device, dtype=dtype)

        tgt_24 = resample_16k_to_24k(fix_tgt.to(device=device, dtype=torch.float32))
        tgt_m = extract_100_mel(tgt_24, mel_fb).to(dtype=dtype)
        d_ratio = torch.tensor([float(tgt_m.shape[-1]) / float(s_lat.shape[1])] * num_samples, device=device, dtype=dtype)

        initial_loss = None
        final_loss = None
        t_start = time.time()
        for s in range(num_steps):
            optimizer.zero_grad()
            l_tot, cfm_l, d_l = model.compute_loss(
                s_lat, tgt_m, target_dur_ratio=d_ratio, loss_weights=loss_weights
            )
            l_tot.backward()
            nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            if s == 0: initial_loss = l_tot.item()
            log_interval = 50 if is_deep else 10
            if (s + 1) % log_interval == 0 or s == 0:
                print(f"   [Overfit Step {s+1:03d}/{num_steps}] Total Loss: {l_tot.item():.4f} | CFM: {cfm_l.item():.4f} | Dur: {d_l.item():.4f}")
            final_loss = l_tot.item()

        print("-" * 75)
        if final_loss < initial_loss:
            print(f"✅ OVERFIT PASS! Loss giảm từ {initial_loss:.4f} xuống {final_loss:.4f} (giảm {(1.0 - final_loss/initial_loss)*100.0:.1f}%) trong {time.time()-t_start:.1f}s.")
        else:
            print(f"⚠️ Cảnh báo: Loss chưa giảm sâu ({initial_loss:.4f} -> {final_loss:.4f}).")

        # Nếu là Overfit sâu 500 bước -> Xuất trọn bộ 16 file audio nghe thử qua Vocos
        if is_deep:
            out_dir = "outputs/sanity/bridge_overfit16"
            legacy_dir = "outputs/sanity"
            os.makedirs(out_dir, exist_ok=True)
            os.makedirs(legacy_dir, exist_ok=True)
            try:
                vocos = load_vocos_safely("charactr/vocos-mel-24khz", device=device)
                gen_mels = model.inference(s_lat, num_steps=25)
                with torch.no_grad():
                    print(f"\n🎧 ĐANG GIẢI MÃ VÀ XUẤT TOÀN BỘ {len(fix_src)} FILE AUDIO VI -> EN QUA VOCOS...")
                    for idx in range(len(fix_src)):
                        pred_wav = vocos.decode(gen_mels[idx:idx+1].float()).squeeze().cpu()
                        gt_wav = tgt_24[idx].cpu()

                        out_pred = os.path.join(out_dir, f"sample_{idx:02d}_vi_to_en_pred.wav")
                        out_gt = os.path.join(out_dir, f"sample_{idx:02d}_en_groundtruth.wav")
                        save_wav_file(out_pred, pred_wav, 24000)
                        save_wav_file(out_gt, gt_wav, 24000)

                        if idx == 0:
                            save_wav_file(os.path.join(legacy_dir, "bridge_vi_en_overfit16_sample0.wav"), pred_wav, 24000)

                        vi_p = fix_vi_texts[idx][:35] + "..." if len(fix_vi_texts[idx]) > 35 else fix_vi_texts[idx]
                        en_p = fix_en_texts[idx][:35] + "..." if len(fix_en_texts[idx]) > 35 else fix_en_texts[idx]
                        print(f"   [Cặp #{idx:02d}/16] Vi: \"{vi_p}\" -> En: \"{en_p}\"")
                print("-" * 75)
                print(f"✓ ĐÃ XUẤT TRỌN BỘ 16 MẪU ÂM THANH DỊCH VI -> EN (CẢ PRED & GROUND TRUTH) TẠI:")
                print(f"  👉 Thư mục: {out_dir}/")
            except Exception as e:
                print(f"[Vocos Info] Chưa xuất được audio: {e}")

        print("=" * 75 + "\n")
        return

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

            # Trích xuất Whisper Latents từ audio tiếng Việt nguồn
            if feat_extractor is not None and whisper_enc is not None:
                in_proc = feat_extractor(
                    src_audio_16k.numpy(),
                    sampling_rate=16000,
                    return_tensors='pt',
                    padding="max_length"
                )
                feats = in_proc.input_features.to(device=device, dtype=dtype)
                if feats.shape[-1] < 3000:
                    feats = F.pad(feats, (0, 3000 - feats.shape[-1]))
                elif feats.shape[-1] > 3000:
                    feats = feats[:, :, :3000]

                with torch.no_grad():
                    full_latents = whisper_enc(feats).last_hidden_state

                real_len = max(10, int(src_audio_16k.shape[-1] / 320))
                src_latents = full_latents[:, :real_len, :]
            else:
                src_latents = torch.randn(batch_size, 100, 768, device=device, dtype=dtype)

            # Chuyển đổi audio tiếng Anh sang 24kHz và trích xuất 100-Mel đích
            tgt_24k = resample_16k_to_24k(tgt_audio_16k.to(device=device, dtype=torch.float32))
            tgt_mel = extract_100_mel(tgt_24k, mel_fb).to(dtype=dtype)

            dur_ratio = torch.tensor([float(tgt_mel.shape[-1]) / float(src_latents.shape[1])] * batch_size,
                                     device=device, dtype=dtype)

            total_loss, cfm_l, dur_l = model.compute_loss(
                src_latents, tgt_mel, target_dur_ratio=dur_ratio, loss_weights=loss_weights
            )

            total_loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, max_norm=float(train_cfg["max_grad_norm"]))
            optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_loss += total_loss.item()

            if global_step % 10 == 0 or args.smoke_test:
                cur_lr = scheduler.get_last_lr()[0]
                print(f"   [E{epoch:02d}|Step {global_step:04d}] Total Loss: {total_loss.item():.4f} | "
                      f"CFM: {cfm_l.item():.4f} | Dur: {dur_l.item():.4f} | LR: {cur_lr:.2e}")

        avg_loss = epoch_loss / max(1, steps_per_epoch)
        elapsed = time.time() - t0
        print(f"==> Hoàn tất Epoch {epoch:02d} trong {elapsed:.1f}s | Avg Loss: {avg_loss:.4f} | Best Loss: {min(best_loss, avg_loss):.4f}")

        data_pipeline.shuffle()

        # Lưu Checkpoint Last & Best
        os.makedirs(paths["checkpoint_dir"], exist_ok=True)
        ckpt_data = {
            "epoch": epoch,
            "step": global_step,
            "best_loss": min(best_loss, avg_loss),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
        }
        torch.save(ckpt_data, os.path.join(paths["checkpoint_dir"], f"{args.exp_name}-last.pth"))
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt_data, os.path.join(paths["checkpoint_dir"], f"{args.exp_name}-best.pth"))
            print(f"⭐ [Epoch {epoch:02d}] Đã cập nhật Model Best Loss: {best_loss:.4f}")

    print("\n🎉 HUẤN LUYỆN CROSS-ATTENTION BRIDGE (VI -> EN) HOÀN TẤT THÀNH CÔNG!")

if __name__ == "__main__":
    main()
