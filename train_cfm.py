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
        self.register_buffer("window", torch.empty(0))
    def forward(self, x): return x

class MockMelScale(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.register_buffer("fb", torch.empty(0))
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
    """Nạp Vocos vocoder an toàn với strict=False để chống lỗi buffer torchaudio."""
    from vocos import Vocos
    orig_load = Vocos.load_state_dict
    Vocos.load_state_dict = lambda self, sd, strict=False: orig_load(self, sd, strict=False)
    vocos = Vocos.from_pretrained(repo_id)
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
            "whisper_model": "openai/whisper-small",
            "encoder_dim": 768,
            "cond_dim": 512,
            "n_mels": 100,
            "spk_dim": 192,
            "decoder_hidden": 384,
            "num_res_blocks": 4
        },
        "training": {
            "batch_size": 4,
            "learning_rate": 2.0e-4,
            "min_lr": 1.0e-5,
            "weight_decay": 1.0e-4,
            "epochs": 25,
            "steps_per_epoch": 100,
            "val_steps": 10,
            "max_grad_norm": 1.0,
            "bf16": True
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
    def hz_to_mel(hz): return 2595.0 * math.log10(1.0 + hz / 700.0)
    def mel_to_hz(mel): return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax if fmax is not None else sr / 2.0)
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2, device=device)
    hz_points = mel_to_hz(mel_points)
    bin_points = torch.floor((n_fft + 1) * hz_points / sr).long()
    filterbank = torch.zeros(n_mels, n_fft // 2 + 1, device=device)
    for i in range(n_mels):
        left, center, right = bin_points[i], bin_points[i + 1], bin_points[i + 2]
        if center > left:
            filterbank[i, left:center] = (torch.arange(left, center, device=device) - left) / (center - left)
        if right > center:
            filterbank[i, center:right] = (right - torch.arange(center, right, device=device)) / (right - center)
    return filterbank

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
    if waveform_16k.dim() == 1:
        waveform_16k = waveform_16k.unsqueeze(0).unsqueeze(0)
    elif waveform_16k.dim() == 2:
        waveform_16k = waveform_16k.unsqueeze(1)
    new_len = int(waveform_16k.shape[-1] * (24000.0 / 16000.0))
    resampled = F.interpolate(waveform_16k, size=new_len, mode='linear', align_corners=False)
    return resampled.squeeze(1)

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

class ResidualBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.conv2(x)
        return x + res

class FlowMatchingTargetDecoder(nn.Module):
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

    def forward(self, x_t, t, cond, spk_emb):
        B, C, T = x_t.shape
        t_embed = t.view(B, 1, 1).expand(B, 1, T)
        spk_expand = spk_emb.unsqueeze(-1).expand(B, -1, T)
        net_in = torch.cat([x_t, t_embed, cond, spk_expand], dim=1)
        h = self.in_proj(net_in)
        for block in self.res_blocks:
            h = block(h)
        v = self.out_proj(h)
        return v

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
    def __init__(self, encoder_dim=768, cond_dim=512, n_mels=100, spk_dim=192, hidden_dim=384):
        super().__init__()
        self.n_mels = n_mels
        self.cond_proj = nn.Linear(encoder_dim, cond_dim)
        self.spk_encoder = AcousticSpeakerEncoder(feat_dim=encoder_dim, spk_dim=spk_dim)
        self.decoder = FlowMatchingTargetDecoder(
            n_mels=n_mels, cond_dim=cond_dim, spk_dim=spk_dim, hidden_dim=hidden_dim
        )

    def compute_loss(self, whisper_features, target_mel):
        B, C, T_target = target_mel.shape
        cond_512 = self.cond_proj(whisper_features)
        cond_512 = F.interpolate(cond_512.transpose(1, 2), size=T_target, mode='linear', align_corners=False)
        spk_emb = self.spk_encoder(whisper_features)

        t = torch.rand(B, device=target_mel.device, dtype=target_mel.dtype)
        x_0 = torch.randn_like(target_mel)
        x_1 = target_mel
        t_expand = t.view(B, 1, 1)
        x_t = (1.0 - t_expand) * x_0 + t_expand * x_1
        u_t = x_1 - x_0

        v_pred = self.decoder(x_t, t, cond_512, spk_emb)
        loss = F.mse_loss(v_pred, u_t)
        return loss

    @torch.no_grad()
    def inference(self, whisper_features, num_steps=25):
        self.eval()
        B, T_src, _ = whisper_features.shape
        T_tgt = int(T_src * (24000.0 / 16000.0) * (320.0 / 256.0))
        cond_512 = self.cond_proj(whisper_features)
        cond_512 = F.interpolate(cond_512.transpose(1, 2), size=T_tgt, mode='linear', align_corners=False)
        spk_emb = self.spk_encoder(whisper_features)

        device = whisper_features.device
        dtype = whisper_features.dtype
        x = torch.randn(B, self.n_mels, T_tgt, device=device, dtype=dtype)
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t_val = step * dt
            t = torch.full((B,), t_val, device=device, dtype=dtype)
            v = self.decoder(x, t, cond_512, spk_emb)
            x = x + v * dt
        return x

# ==============================================================================
# 3. PIPELINE DỮ LIỆU STREAMING FLEURS (HỖ TRỢ EN-US HOẶC VI-VN)
# ==============================================================================

class UniversalSpeechStreamingPipeline:
    def __init__(self, split='train', max_sec=12.0, lang='en_us'):
        self.split = split
        self.max_sec = max_sec
        self.lang = lang
        self.is_online = False

        try:
            from datasets import load_dataset, Audio
            print(f"📡 Đang kết nối Streaming Google FLEURS ({lang}, split='{split}')...")
            self.ds = load_dataset('google/fleurs', lang, split=split, streaming=True)
            try:
                self.ds = self.ds.cast_column('audio', Audio(decode=False))
            except Exception:
                pass
            self.iter_ds = iter(self.ds)
            self.is_online = True
            print(f"✓ Kết nối thành công FLEURS {lang} [{split}]!")
        except Exception as e:
            print(f"[Cảnh báo Dữ liệu] Không thể stream ({e}). Kích hoạt bộ tạo tín hiệu mô phỏng.")
            self.is_online = False

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
        return torch.randn(int(16000 * 4.0))

    def fetch_batch(self, batch_size=4):
        audio_list = []
        transcripts = []

        if self.is_online:
            for _ in range(batch_size):
                try:
                    item = next(self.iter_ds)
                    w = self._extract_audio_tensor(item['audio'])
                    audio_list.append(w)
                    transcripts.append(item.get('transcription', 'speech sample'))
                except Exception:
                    self.iter_ds = iter(self.ds)
                    break

        while len(audio_list) < batch_size:
            t_sec = torch.linspace(0, self.max_sec, int(16000 * self.max_sec))
            f0 = 130.0 + 30.0 * torch.sin(2 * math.pi * 1.5 * t_sec)
            w = torch.sin(2 * math.pi * f0 * t_sec) * 0.4
            audio_list.append(w)
            transcripts.append('synthetic speech sample')

        max_len = max(w.shape[0] for w in audio_list)
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

    model = CFMStage1System(
        encoder_dim=cfg["model"]["encoder_dim"],
        cond_dim=cfg["model"]["cond_dim"],
        n_mels=cfg["model"]["n_mels"],
        spk_dim=cfg["model"]["spk_dim"],
        hidden_dim=cfg["model"]["decoder_hidden"]
    ).to(device=device, dtype=dtype)

    mel_fb = build_mel_filterbank(sr=24000, n_mels=100, device=device)

    # --------------------------------------------------------------------------
    # CHẾ ĐỘ SANITY CHECK #3: OVERFIT 16 CÂU (500 BƯỚC) + XUẤT AUDIO
    # --------------------------------------------------------------------------
    if args.overfit_16:
        print("\n" + "=" * 75)
        print("🧪 BẮT ĐẦU SANITY CHECK #3: OVERFIT 500 BƯỚC TRÊN 16 CÂU TIẾNG ANH")
        print("Mục tiêu: Đảm bảo CFM học thuộc lòng âm học tiếng Anh và sinh ra file .wav rõ chữ!")
        print("=" * 75)
        pipeline = UniversalSpeechStreamingPipeline(split='train', max_sec=12.0, lang=data_lang)
        fix_audio, fix_txt = pipeline.fetch_batch(batch_size=16)

        if feat_extractor and whisper_enc:
            in_p = feat_extractor(fix_audio.numpy(), sampling_rate=16000, return_tensors='pt', padding="max_length")
            f = in_p.input_features.to(device=device, dtype=dtype)
            if f.shape[-1] < 3000: f = F.pad(f, (0, 3000 - f.shape[-1]))
            elif f.shape[-1] > 3000: f = f[:, :, :3000]
            with torch.no_grad():
                fl = whisper_enc(f).last_hidden_state
            real_l = max(10, int(fix_audio.shape[-1] / 320))
            fix_latents = fl[:, :real_l, :]
        else:
            fix_latents = torch.randn(16, 100, 768, device=device, dtype=dtype)

        fix_24k = resample_16k_to_24k(fix_audio.to(device=device, dtype=torch.float32))
        fix_mel = extract_100_mel(fix_24k, mel_fb).to(dtype=dtype)

        overfit_opt = torch.optim.AdamW(model.parameters(), lr=3.0e-4)
        model.train()
        t_start = time.time()
        initial_l = None
        final_l = None

        for step in range(1, 501):
            overfit_opt.zero_grad()
            l = model.compute_loss(fix_latents, fix_mel)
            l.backward()
            overfit_opt.step()

            if step == 1: initial_l = l.item()
            if step % 50 == 0 or step == 1:
                print(f"   [Overfit Step {step:03d}/500] CFM Loss: {l.item():.4f}")
            final_l = l.item()

        print("-" * 75)
        print(f"✅ Loss giảm từ {initial_l:.4f} xuống {final_l:.4f} (Giảm {(1.0 - final_l/initial_l)*100:.1f}%) trong {time.time()-t_start:.1f}s!")

        # Xuất file .wav nghe thử qua Vocos
        out_dir = "outputs/sanity"
        os.makedirs(out_dir, exist_ok=True)
        try:
            vocos = load_vocos_safely("charactr/vocos-mel-24khz", device=device)
            gen_mel = model.inference(fix_latents[:1], num_steps=25)
            with torch.no_grad():
                gen_wav = vocos.decode(gen_mel.float()).squeeze().cpu()
            out_wav = os.path.join(out_dir, "cfm_en_overfit16_sample0.wav")
            save_wav_file(out_wav, gen_wav, 24000)
            print(f"🎧 Đã xuất file audio kiểm tra khả năng học tại: {out_wav}")
            print(f"   Văn bản câu 0: \"{fix_txt[0]}\"")
        except Exception as e:
            print(f"[Vocos Info] Không thể giải mã audio: {e}")
        print("=" * 75 + "\n")
        return

    # Huấn luyện thông thường
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
        betas=(0.9, 0.999)
    )

    epochs = 1 if args.smoke_test else train_cfg["epochs"]
    steps_per_epoch = 2 if args.smoke_test else train_cfg["steps_per_epoch"]
    val_steps = 1 if args.smoke_test else train_cfg["val_steps"]
    batch_size = train_cfg["batch_size"]

    train_pipeline = UniversalSpeechStreamingPipeline(split='train', max_sec=12.0, lang=data_lang)
    val_pipeline = UniversalSpeechStreamingPipeline(split='validation', max_sec=12.0, lang=data_lang)

    best_val_loss = float("inf")
    ckpt_path = os.path.join(paths["checkpoint_dir"], f"{args.exp_name}-last.pth")
    os.makedirs(paths["checkpoint_dir"], exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for step in range(steps_per_epoch):
            optimizer.zero_grad()
            audio_16k, _ = train_pipeline.fetch_batch(batch_size=batch_size)

            if feat_extractor and whisper_enc:
                in_p = feat_extractor(audio_16k.numpy(), sampling_rate=16000, return_tensors='pt', padding="max_length")
                f = in_p.input_features.to(device=device, dtype=dtype)
                if f.shape[-1] < 3000: f = F.pad(f, (0, 3000 - f.shape[-1]))
                elif f.shape[-1] > 3000: f = f[:, :, :3000]
                with torch.no_grad():
                    fl = whisper_enc(f).last_hidden_state
                real_len = max(10, int(audio_16k.shape[-1] / 320))
                latents = fl[:, :real_len, :]
            else:
                latents = torch.randn(batch_size, 100, 768, device=device, dtype=dtype)

            audio_24k = resample_16k_to_24k(audio_16k.to(device=device, dtype=torch.float32))
            tgt_mel = extract_100_mel(audio_24k, mel_fb).to(dtype=dtype)

            cfm_l = model.compute_loss(latents, tgt_mel)
            cfm_l.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(train_cfg["max_grad_norm"]))
            optimizer.step()

            train_loss += cfm_l.item()
            if (step + 1) % 10 == 0:
                print(f"   [E{epoch:02d}|Step {step+1:03d}/{steps_per_epoch}] Train Loss: {cfm_l.item():.4f}")

        avg_train = train_loss / steps_per_epoch
        print(f"==> Hoàn tất Epoch {epoch:02d} trong {time.time()-t0:.1f}s | Avg Train Loss: {avg_train:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for _ in range(val_steps):
                v_audio, _ = val_pipeline.fetch_batch(batch_size=batch_size)
                if feat_extractor and whisper_enc:
                    in_p = feat_extractor(v_audio.numpy(), sampling_rate=16000, return_tensors='pt', padding="max_length")
                    f = in_p.input_features.to(device=device, dtype=dtype)
                    if f.shape[-1] < 3000: f = F.pad(f, (0, 3000 - f.shape[-1]))
                    elif f.shape[-1] > 3000: f = f[:, :, :3000]
                    v_fl = whisper_enc(f).last_hidden_state
                    v_lat = v_fl[:, :max(10, int(v_audio.shape[-1] / 320)), :]
                else:
                    v_lat = torch.randn(batch_size, 100, 768, device=device, dtype=dtype)

                v_24k = resample_16k_to_24k(v_audio.to(device=device, dtype=torch.float32))
                v_mel = extract_100_mel(v_24k, mel_fb).to(dtype=dtype)
                val_loss += model.compute_loss(v_lat, v_mel).item()

        avg_val = val_loss / max(1, val_steps)
        print(f"   Val Loss: {avg_val:.4f} | Best Val Loss: {min(best_val_loss, avg_val):.4f}")

        ckpt_data = {
            "epoch": epoch,
            "best_loss": min(best_val_loss, avg_val),
            "model_state_dict": model.decoder.state_dict(),
            "full_system_state_dict": model.state_dict(),
        }
        torch.save(ckpt_data, ckpt_path)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_ckpt = os.path.join(paths["checkpoint_dir"], f"{args.exp_name}-best.pth")
            torch.save(ckpt_data, best_ckpt)
            print(f"⭐ [Epoch {epoch:02d}] Đã cập nhật Model Best: {best_val_loss:.4f}")

    print("\n🎉 HUẤN LUYỆN STAGE 1 OT-CFM TIẾNG ANH HOÀN TẤT THÀNH CÔNG!")

if __name__ == "__main__":
    main()
