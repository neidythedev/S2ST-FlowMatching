#!/usr/bin/env python3
# ==============================================================================
# S2ST EN-VI: HUẤN LUYỆN TONE-AWARE BRIDGE & FLOW MATCHING
# Tuân thủ nghiêm ngặt quy chuẩn SKILL2.md cho GPU Cloud Runtime (Molab)
# ==============================================================================

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# Cấu hình logging ra cả terminal và file
class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def load_yaml_config(config_path):
    """Đọc file YAML an toàn với fallback nếu chưa cài pyyaml."""
    default_config = {
        "model": {
            "whisper_model": "openai/whisper-small",
            "encoder_dim": 768,
            "bridge_dim": 512,
            "bridge_layers": 6,
            "bridge_heads": 8,
            "n_mels": 100,
            "spk_dim": 192,
            "decoder_hidden": 384,
            "use_tone_head": True,
            "num_tones": 6,
        },
        "loss_weights": {
            "cfm_weight": 1.0,
            "tone_weight": 0.2,
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
            "src_lang": "en_us",
            "tgt_lang": "vi_vn",
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
except ImportError as e:
    print(f"[Fatal Error] PyTorch chưa được cài đặt: {e}")
    sys.exit(1)

# ==============================================================================
# 1. PURE PYTORCH AUDIO UTILITIES (100-MEL CHO VOCOS 24KHZ)
# ==============================================================================

def build_mel_filterbank(sr=24000, n_fft=1024, n_mels=100, fmin=0.0, fmax=12000.0, device="cpu"):
    def h2m(hz): return 2595.0 * math.log10(1.0 + hz / 700.0)
    def m2h(mel): return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    mel_min, mel_max = h2m(fmin), h2m(fmax)
    mel_pts = [m2h(mel_min + i * (mel_max - mel_min) / (n_mels + 1)) for i in range(n_mels + 2)]
    fft_freqs = [i * sr / n_fft for i in range(n_fft // 2 + 1)]
    fb = torch.zeros(n_mels, n_fft // 2 + 1, device=device, dtype=torch.float32)
    for m in range(n_mels):
        left, center, right = mel_pts[m], mel_pts[m+1], mel_pts[m+2]
        for k, f in enumerate(fft_freqs):
            if left <= f <= center and center > left:
                fb[m, k] = (f - left) / (center - left)
            elif center <= f <= right and right > center:
                fb[m, k] = (right - f) / (right - center)
    return fb

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
    """Nội suy tuyến tính PyTorch chuyển đổi 16kHz -> 24kHz (1.5x) siêu tốc."""
    if audio_16k.dim() == 1:
        audio_16k = audio_16k.unsqueeze(0).unsqueeze(0)
    elif audio_16k.dim() == 2:
        audio_16k = audio_16k.unsqueeze(1)
    t_new = int(audio_16k.shape[-1] * 1.5)
    audio_24k = F.interpolate(audio_16k, size=t_new, mode="linear", align_corners=False)
    return audio_24k.squeeze(1)

# Bảng gán nhãn 6 thanh điệu tiếng Việt
VIETNAMESE_VOWEL_TONES = {
    'à': 1, 'ằ': 1, 'ầ': 1, 'è': 1, 'ề': 1, 'ì': 1, 'ò': 1, 'ồ': 1, 'ờ': 1, 'ù': 1, 'ừ': 1, 'ỳ': 1,
    'á': 2, 'ắ': 2, 'ấ': 2, 'é': 2, 'ế': 2, 'í': 2, 'ó': 2, 'ố': 2, 'ớ': 2, 'ú': 2, 'ứ': 2, 'ý': 2,
    'ả': 3, 'ẳ': 3, 'ẩ': 3, 'ẻ': 3, 'ể': 3, 'ỉ': 3, 'ỏ': 3, 'ổ': 3, 'ở': 3, 'ủ': 3, 'ử': 3, 'ỷ': 3,
    'ã': 4, 'ẵ': 4, 'ẫ': 4, 'ẽ': 4, 'ễ': 4, 'ĩ': 4, 'õ': 4, 'ỗ': 4, 'ỡ': 4, 'ũ': 4, 'ữ': 4, 'ỹ': 4,
    'ạ': 5, 'ặ': 5, 'ậ': 5, 'ẹ': 5, 'ệ': 5, 'ị': 5, 'ọ': 5, 'ộ': 5, 'ợ': 5, 'ụ': 5, 'ự': 5, 'ỵ': 5,
}

def parse_tones(vietnamese_text, target_len=75):
    words = str(vietnamese_text).lower().strip().split()
    tones = []
    for w in words:
        t = 0
        for ch in w:
            if ch in VIETNAMESE_VOWEL_TONES:
                t = VIETNAMESE_VOWEL_TONES[ch]
                break
        tones.append(t)
    if not tones:
        tones = [0]
    t_tensor = torch.tensor(tones, dtype=torch.long)
    if t_tensor.shape[0] != target_len:
        t_tensor = F.interpolate(
            t_tensor.float().view(1, 1, -1),
            size=target_len,
            mode="nearest"
        ).long().view(-1)
    return t_tensor

# ==============================================================================
# 2. KIẾN TRÚC MÔ HÌNH: TONE-AWARE BRIDGE + OT-CFM DECODER
# ==============================================================================

class DurationPredictor(nn.Module):
    """Dự đoán tỷ lệ co dãn thời lượng (duration scale) từ tiếng Anh sang tiếng Việt."""
    def __init__(self, in_dim=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(256, 1, kernel_size=1)
        )
    def forward(self, x):
        # x: [B, T, D] -> [B, D, T]
        h = self.conv(x.transpose(1, 2))
        return torch.sigmoid(h.mean(dim=[-1, -2])) * 1.4 + 0.8  # scale trong khoảng [0.8, 2.2]

class ToneAwareBridge(nn.Module):
    """
    Cầu nối dịch thuật bảo toàn thanh điệu (Novelty Core):
    - Conv1D nén thời gian 50Hz (Whisper) -> 25Hz.
    - 6 tầng TransformerEncoder học chuyển đổi không gian liên tục Cross-lingual.
    - Tone Head phân loại 6 thanh điệu tiếng Việt để định hướng latent space.
    - Duration Predictor dự đoán thời lượng tương đối.
    """
    def __init__(self, in_dim=768, hidden_dim=512, num_layers=6, n_heads=8, num_tones=6):
        super().__init__()
        self.downsample = nn.Conv1d(in_dim, hidden_dim, kernel_size=2, stride=2)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.tone_head = nn.Linear(hidden_dim, num_tones)
        self.duration_predictor = DurationPredictor(in_dim=hidden_dim)
        
    def forward(self, whisper_features):
        # whisper_features: [B, T_src, in_dim]
        x = self.downsample(whisper_features.transpose(1, 2)).transpose(1, 2)
        latents = self.transformer(x)
        tone_logits = self.tone_head(latents)
        dur_scale = self.duration_predictor(latents)
        return latents, tone_logits, dur_scale

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

class EnViS2STSystem(nn.Module):
    """Hệ thống tích hợp toàn diện: Whisper -> Bridge -> CFM Decoder -> 100 Mel."""
    def __init__(self, encoder_dim=768, bridge_dim=512, bridge_layers=6, n_mels=100, spk_dim=192):
        super().__init__()
        self.n_mels = n_mels
        self.bridge = ToneAwareBridge(in_dim=encoder_dim, hidden_dim=bridge_dim, num_layers=bridge_layers)
        self.spk_encoder = AcousticSpeakerEncoder(feat_dim=encoder_dim, spk_dim=spk_dim)
        self.decoder = FlowMatchingTargetDecoder(n_mels=n_mels, cond_dim=bridge_dim, spk_dim=spk_dim, hidden_dim=384)

    def compute_loss(self, src_whisper, target_mel, speaker_emb=None, target_tones=None, target_dur_ratio=None,
                     loss_weights=None):
        weights = loss_weights or {"cfm_weight": 1.0, "tone_weight": 0.2, "dur_weight": 0.1}
        B, _, T_tgt = target_mel.shape
        translated_latents, tone_logits, dur_scale = self.bridge(src_whisper)
        
        if speaker_emb is None:
            speaker_emb = self.spk_encoder(src_whisper)
            
        # Nội suy cond sang đúng độ dài T_tgt
        cond = F.interpolate(translated_latents.transpose(1, 2), size=T_tgt, mode="linear", align_corners=False)

        # 1. CFM Optimal Transport Loss
        t = torch.rand(B, device=target_mel.device, dtype=target_mel.dtype)
        x_0 = torch.randn_like(target_mel)
        x_1 = target_mel
        t_exp = t.view(B, 1, 1)
        x_t = (1.0 - t_exp) * x_0 + t_exp * x_1
        u_t = x_1 - x_0

        v_pred = self.decoder(x_t, t, cond, speaker_emb)
        cfm_loss = F.mse_loss(v_pred, u_t)

        # 2. Auxiliary Tone Classification Loss
        if target_tones is not None:
            if tone_logits.shape[1] != target_tones.shape[1]:
                t_logits_interp = F.interpolate(
                    tone_logits.transpose(1, 2), size=target_tones.shape[1],
                    mode="linear", align_corners=False
                ).transpose(1, 2)
            else:
                t_logits_interp = tone_logits
            tone_loss = F.cross_entropy(t_logits_interp.reshape(-1, 6), target_tones.reshape(-1))
        else:
            tone_loss = torch.tensor(0.0, device=target_mel.device, dtype=target_mel.dtype)

        # 3. Duration Predictor Loss
        if target_dur_ratio is not None:
            dur_loss = F.smooth_l1_loss(dur_scale.view(-1), target_dur_ratio.view(-1))
        else:
            dur_loss = torch.tensor(0.0, device=target_mel.device, dtype=target_mel.dtype)

        total_loss = (weights["cfm_weight"] * cfm_loss + 
                      weights["tone_weight"] * tone_loss + 
                      weights["dur_weight"] * dur_loss)
        return total_loss, cfm_loss, tone_loss, dur_loss

    @torch.no_grad()
    def inference(self, src_whisper, speaker_emb=None, num_steps=25):
        self.eval()
        B = src_whisper.shape[0]
        if speaker_emb is None:
            speaker_emb = self.spk_encoder(src_whisper)
            
        translated_latents, _, dur_scale = self.bridge(src_whisper)
        avg_scale = float(dur_scale.mean().item())
        T_tgt = max(30, int(translated_latents.shape[1] * 2 * avg_scale))
        cond = F.interpolate(translated_latents.transpose(1, 2), size=T_tgt, mode="linear", align_corners=False)

        x_t = torch.randn(B, self.n_mels, T_tgt, device=src_whisper.device, dtype=src_whisper.dtype)
        dt = 1.0 / num_steps
        for step in range(num_steps):
            cur_t = torch.full((B,), step * dt, device=src_whisper.device, dtype=src_whisper.dtype)
            v_pred = self.decoder(x_t, cur_t, cond, speaker_emb)
            x_t = x_t + v_pred * dt
        return x_t

# ==============================================================================
# 3. BỘ NẠP DỮ LIỆU & HUẤN LUYỆN
# ==============================================================================

class SyntheticDataset(torch.utils.data.Dataset):
    """Dataset giả lập phục vụ Smoke Test & Dry-Run cực nhanh."""
    def __init__(self, num_samples=16, src_len=100, tgt_len=150, feat_dim=768, n_mels=100):
        self.num_samples = num_samples
        self.src_len = src_len
        self.tgt_len = tgt_len
        self.feat_dim = feat_dim
        self.n_mels = n_mels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        src_feat = torch.randn(self.src_len, self.feat_dim)
        tgt_mel = torch.randn(self.n_mels, self.tgt_len)
        tones = torch.randint(0, 6, (self.tgt_len // 2,))
        dur_ratio = torch.tensor(float(self.tgt_len) / float(self.src_len))
        return src_feat, tgt_mel, tones, dur_ratio

def collate_fn(batch):
    src_feats, tgt_mels, tones, dur_ratios = zip(*batch)
    B = len(batch)
    max_src = max(f.shape[0] for f in src_feats)
    max_tgt = max(m.shape[1] for m in tgt_mels)
    max_tone = max(t.shape[0] for t in tones)

    src_pad = torch.zeros(B, max_src, src_feats[0].shape[-1])
    tgt_pad = torch.zeros(B, tgt_mels[0].shape[0], max_tgt)
    tone_pad = torch.zeros(B, max_tone, dtype=torch.long)

    for i in range(B):
        src_pad[i, :src_feats[i].shape[0]] = src_feats[i]
        tgt_pad[i, :, :tgt_mels[i].shape[1]] = tgt_mels[i]
        tone_pad[i, :tones[i].shape[0]] = tones[i]

    dur_tensor = torch.stack(dur_ratios)
    return src_pad, tgt_pad, tone_pad, dur_tensor

def main():
    parser = argparse.ArgumentParser(description="Huấn luyện ToneAwareBridge S2ST")
    parser.add_argument("--config", type=str, default="configs/stage2_bridge.yaml", help="Đường dẫn file cấu hình YAML")
    parser.add_argument("--resume", action="store_true", help="Khôi phục huấn luyện từ checkpoint gần nhất")
    parser.add_argument("--smoke-test", action="store_true", help="Chạy kiểm thử 2 bước rồi dừng (kiểm tra CUDA/Forward/Backward)")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    paths = cfg["paths"]
    train_cfg = cfg["training"]
    loss_weights = cfg["loss_weights"]

    os.makedirs(paths["log_dir"], exist_ok=True)
    os.makedirs(paths["checkpoint_dir"], exist_ok=True)
    
    log_file = os.path.join(paths["log_dir"], "train.log")
    sys.stdout = Logger(log_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = train_cfg.get("bf16", True) and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    print("=" * 70)
    print("🚀 KHỞI ĐỘNG HUẤN LUYỆN TONE-AWARE BRIDGE (EN -> VI S2ST)")
    print(f"Thiết bị: {device} | Dtype: {dtype} | BF16 Hỗ trợ: {use_bf16}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    # Khởi tạo mô hình
    model = EnViS2STSystem(
        encoder_dim=cfg["model"]["encoder_dim"],
        bridge_dim=cfg["model"]["bridge_dim"],
        bridge_layers=cfg["model"]["bridge_layers"],
        n_mels=cfg["model"]["n_mels"],
        spk_dim=cfg["model"]["spk_dim"]
    ).to(device=device, dtype=dtype)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"])
    )

    start_epoch = 0
    global_step = 0
    best_loss = float("inf")

    # Khôi phục Checkpoint nếu có cờ --resume
    ckpt_path = os.path.join(paths["checkpoint_dir"], "model-last.pth")
    if args.resume and os.path.exists(ckpt_path):
        print(f"[Resume] Đang khôi phục checkpoint từ: {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("step", 0)
        best_loss = ckpt.get("best_loss", float("inf"))
        print(f"✓ [Resume] Đã nạp thành công từ Epoch {start_epoch - 1}, Step {global_step}!")

    # Dữ liệu huấn luyện
    num_samples = 8 if args.smoke_test else 1000
    dataset = SyntheticDataset(num_samples=num_samples)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_fn
    )

    epochs = 1 if args.smoke_test else train_cfg["epochs"]
    grad_accum_steps = train_cfg["gradient_accumulation_steps"]

    print(f"[Training] Bắt đầu vòng lặp: {epochs} Epochs | Batch Size: {train_cfg['batch_size']}")

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, (src_feat, tgt_mel, tones, dur_ratio) in enumerate(dataloader):
            src_feat = src_feat.to(device=device, dtype=dtype)
            tgt_mel = tgt_mel.to(device=device, dtype=dtype)
            tones = tones.to(device=device)
            dur_ratio = dur_ratio.to(device=device, dtype=dtype)

            total_loss, cfm_l, tone_l, dur_l = model.compute_loss(
                src_feat, tgt_mel, target_tones=tones, target_dur_ratio=dur_ratio,
                loss_weights=loss_weights
            )
            
            loss_to_back = total_loss / grad_accum_steps
            loss_to_back.backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(train_cfg["max_grad_norm"]))
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += total_loss.item()

            if (global_step % train_cfg["log_interval_steps"] == 0 and (step + 1) % grad_accum_steps == 0) or args.smoke_test:
                print(f"[E{epoch:02d}|Step {global_step:04d}] Total Loss: {total_loss.item():.4f} | "
                      f"CFM: {cfm_l.item():.4f} | Tone: {tone_l.item():.4f} | Dur: {dur_l.item():.4f}")

            if args.smoke_test and global_step >= 2:
                print("✓ [Smoke Test] Đã hoàn thành 2 bước forward/backward thành công.")
                break

        avg_loss = epoch_loss / max(1, len(dataloader))
        elapsed = time.time() - t0
        print(f"==> Hoàn tất Epoch {epoch:02d} trong {elapsed:.1f}s | Avg Loss: {avg_loss:.4f}")

        # Lưu Checkpoint Atomic (Tránh hỏng file khi backup ngầm lên Google Drive)
        ckpt_data = {
            "epoch": epoch,
            "step": global_step,
            "best_loss": min(best_loss, avg_loss),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
        }

        tmp_ckpt = ckpt_path + ".tmp"
        torch.save(ckpt_data, tmp_ckpt)
        os.replace(tmp_ckpt, ckpt_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_ckpt = os.path.join(paths["checkpoint_dir"], "model-best.pth")
            torch.save(ckpt_data, best_ckpt)
            print(f"⭐ Đã cập nhật Model Best Loss: {best_loss:.4f}")

        if args.smoke_test:
            break

    print("=" * 70)
    print("✅ HUẤN LUYỆN KẾT THÚC THÀNH CÔNG!")
    print(f"Checkpoint được lưu tại: {ckpt_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
