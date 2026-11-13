#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Huấn luyện Giai đoạn 1: Optimal Transport Conditional Flow Matching (OT-CFM) Target Decoder
Mô hình chuyên gia sinh phổ Mel 100-band tiếng Việt từ biểu diễn điều kiện và âm sắc (Timbre).
Bao gồm cả vòng lặp Huấn luyện (Train) và Đánh giá (Validation) trên Google FLEURS vi_vn.
"""

import os
import sys
import time
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 0. HỆ THỐNG GHI LOG CHUẨN
# ==============================================================================

class Logger:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log_file = open(log_path, "a", encoding="utf-8", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

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
            "steps_per_epoch": 30,
            "val_steps": 6,
            "max_grad_norm": 1.0,
            "bf16": True
        },
        "data": {
            "dataset_name": "google/fleurs",
            "lang": "vi_vn",
            "sample_rate": 24000,
            "sample_rate_whisper": 16000,
            "max_audio_sec": 10.0,
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
    def hz_to_mel(hz):
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax if fmax is not None else sr / 2.0)
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2, device=device)
    hz_points = mel_to_hz(mel_points)
    bin_points = torch.floor((n_fft + 1) * hz_points / sr).long()

    filters = torch.zeros(n_mels, n_fft // 2 + 1, device=device)
    for i in range(n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]
        if center > left:
            filters[i, left:center] = (torch.arange(left, center, device=device) - left) / (center - left)
        if right > center:
            filters[i, center:right] = (right - torch.arange(center, right, device=device)) / (right - center)
    return filters

def extract_100_mel(audio_24k, mel_fb, n_fft=1024, hop_length=256):
    B, T = audio_24k.shape
    window = torch.hann_window(n_fft, device=audio_24k.device)
    stft = torch.stft(
        audio_24k,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True
    )
    mag = torch.clamp(stft.abs(), min=1e-5)
    mel_fb_dtype = mel_fb.to(dtype=mag.dtype)
    mel = torch.matmul(mel_fb_dtype, mag)
    return torch.log(mel)

def resample_16k_to_24k(audio_16k):
    B, T = audio_16k.shape
    audio_3d = audio_16k.unsqueeze(1)
    target_len = int(T * 24000.0 / 16000.0)
    resampled_3d = F.interpolate(audio_3d, size=target_len, mode="linear", align_corners=False)
    return resampled_3d.squeeze(1)

# ==============================================================================
# 2. KIẾN TRÚC MÔ HÌNH: OT-CFM TARGET DECODER
# ==============================================================================

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
    def __init__(self, n_mels=100, cond_dim=512, spk_dim=192, hidden_dim=384, num_res_blocks=4):
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
            ) for _ in range(num_res_blocks)
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

class CFMTargetSystem(nn.Module):
    """Hệ thống huấn luyện CFM Độc lập (Stage 1):
       Điều kiện hóa biểu diễn ngữ nghĩa Whisper (768d -> 512d) + Speaker Timbre (192d)
       Học tái tạo phân phối phổ 100-Mel tiếng Việt qua Optimal Transport Flow Matching.
    """
    def __init__(self, encoder_dim=768, cond_dim=512, n_mels=100, spk_dim=192, hidden_dim=384, num_res_blocks=4):
        super().__init__()
        self.n_mels = n_mels
        # Projector chuyển đổi Whisper latents sang cond_dim (512d)
        self.cond_proj = nn.Sequential(
            nn.Linear(encoder_dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim)
        )
        self.spk_encoder = AcousticSpeakerEncoder(feat_dim=encoder_dim, spk_dim=spk_dim)
        self.decoder = FlowMatchingTargetDecoder(
            n_mels=n_mels,
            cond_dim=cond_dim,
            spk_dim=spk_dim,
            hidden_dim=hidden_dim,
            num_res_blocks=num_res_blocks
        )

    def compute_loss(self, vi_whisper_latents, target_mel):
        B, _, T_tgt = target_mel.shape
        # Trích xuất speaker timbre
        speaker_emb = self.spk_encoder(vi_whisper_latents)
        # Biểu diễn điều kiện khớp độ dài T_tgt
        cond_latents = self.cond_proj(vi_whisper_latents)  # [B, T_lat, 512]
        cond = F.interpolate(cond_latents.transpose(1, 2), size=T_tgt, mode="linear", align_corners=False)

        # Optimal Transport Conditional Flow Matching (OT-CFM)
        t = torch.rand(B, device=target_mel.device, dtype=target_mel.dtype)
        x_0 = torch.randn_like(target_mel)
        x_1 = target_mel
        t_exp = t.view(B, 1, 1)
        x_t = (1.0 - t_exp) * x_0 + t_exp * x_1
        u_t = x_1 - x_0

        v_pred = self.decoder(x_t, t, cond, speaker_emb)
        cfm_loss = F.mse_loss(v_pred, u_t)
        return cfm_loss

    @torch.no_grad()
    def inference(self, cond_512, speaker_emb, num_steps=25):
        self.eval()
        B, _, T = cond_512.shape
        device = cond_512.device
        dtype = cond_512.dtype

        # Euler ODE Solver từ t=0 đến t=1
        x = torch.randn(B, self.n_mels, T, device=device, dtype=dtype)
        dt = 1.0 / num_steps

        for step in range(num_steps):
            t_val = step * dt
            t = torch.full((B,), t_val, device=device, dtype=dtype)
            v = self.decoder(x, t, cond_512, speaker_emb)
            x = x + v * dt
        return x

# ==============================================================================
# 3. PIPELINE DỮ LIỆU STREAMING FLEURS TIẾNG VIỆT (CẢ TRAIN VÀ VALIDATION)
# ==============================================================================

class VietnameseStreamingPipeline:
    def __init__(self, split='train', max_sec=10.0):
        self.split = split
        self.max_sec = max_sec
        self.is_online = False

        try:
            from datasets import load_dataset
            print(f"📡 Đang kết nối Streaming Google FLEURS (vi_vn, split='{split}')...")
            self.ds = load_dataset('google/fleurs', 'vi_vn', split=split, streaming=True)
            self.iter_ds = iter(self.ds)
            self.is_online = True
            print(f"✓ Kết nối thành công FLEURS vi_vn [{split}]!")
        except Exception as e:
            print(f"[Cảnh báo Dữ liệu] Không thể stream ({e}). Kích hoạt bộ tạo tín hiệu tổng hợp.")
            self.is_online = False

    def fetch_batch(self, batch_size=4):
        audio_list = []
        transcripts = []

        if self.is_online:
            for _ in range(batch_size):
                try:
                    item = next(self.iter_ds)
                    arr = item['audio']['array']
                    w = torch.from_numpy(arr).float()[:int(16000 * self.max_sec)]
                    audio_list.append(w)
                    transcripts.append(item.get('transcription', 'tiếng việt'))
                except Exception:
                    self.iter_ds = iter(self.ds)
                    break

        while len(audio_list) < batch_size:
            t_sec = torch.linspace(0, self.max_sec, int(16000 * self.max_sec))
            f0 = 180.0 + 40.0 * torch.sin(2 * math.pi * 2.0 * t_sec)
            w = torch.sin(2 * math.pi * f0 * t_sec) * 0.4
            audio_list.append(w)
            transcripts.append('phát âm tiếng việt chuẩn')

        max_len = max(w.shape[0] for w in audio_list)
        padded_audio_16k = torch.stack([F.pad(w, (0, max_len - w.shape[0])) for w in audio_list], dim=0)
        return padded_audio_16k, transcripts

# ==============================================================================
# 4. TIẾN TRÌNH HUẤN LUYỆN CHÍNH (TRAIN + VALIDATION)
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Huấn luyện OT-CFM Target Speech Decoder")
    parser.add_argument("--config", type=str, default="configs/stage1_cfm.yaml", help="File cấu hình YAML")
    parser.add_argument("--exp-name", type=str, default="cfm", help="Tên thử nghiệm")
    parser.add_argument("--resume", action="store_true", help="Khôi phục checkpoint gần nhất")
    parser.add_argument("--smoke-test", action="store_true", help="Chạy test nhanh 2 bước rồi dừng")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    paths = cfg["paths"]
    train_cfg = cfg["training"]

    os.makedirs(paths["log_dir"], exist_ok=True)
    os.makedirs(paths["checkpoint_dir"], exist_ok=True)

    log_file = os.path.join(paths["log_dir"], f"{args.exp_name}.log")
    sys.stdout = Logger(log_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = train_cfg.get("bf16", True) and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    print("=" * 75)
    print(f"🌊 KHỞI ĐỘNG HUẤN LUYỆN OT-CFM TARGET DECODER: [{args.exp_name.upper()}]")
    print(f"Thiết bị: {device} | Dtype: {dtype} | BF16 Hỗ trợ: {use_bf16}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 75)

    # 1. Nạp Whisper-Small Encoder để trích xuất đặc trưng điều kiện
    whisper_id = cfg["model"].get("whisper_model", "openai/whisper-small")
    feat_extractor = None
    whisper_enc = None

    try:
        from transformers import WhisperFeatureExtractor, WhisperModel
        print(f"⏳ Đang nạp Whisper Encoder ({whisper_id})...")
        feat_extractor = WhisperFeatureExtractor.from_pretrained(whisper_id)
        whisper_enc = WhisperModel.from_pretrained(whisper_id).encoder.eval().to(device=device, dtype=dtype)
        for p in whisper_enc.parameters():
            p.requires_grad = False
        print("✓ Whisper Encoder đã được đóng băng (Frozen) an toàn!")
    except Exception as e:
        print(f"[Cảnh báo] Chưa thể nạp transformers ({e}). Sử dụng đặc trưng ngẫu nhiên.")

    # 2. Khởi tạo mô hình CFM
    model = CFMTargetSystem(
        encoder_dim=cfg["model"]["encoder_dim"],
        cond_dim=cfg["model"]["cond_dim"],
        n_mels=cfg["model"]["n_mels"],
        spk_dim=cfg["model"]["spk_dim"],
        hidden_dim=cfg["model"]["decoder_hidden"],
        num_res_blocks=cfg["model"]["num_res_blocks"]
    ).to(device=device, dtype=dtype)

    mel_fb = build_mel_filterbank(sr=24000, n_mels=100, device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
        betas=(0.9, 0.98)
    )

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    ckpt_path = os.path.join(paths["checkpoint_dir"], f"{args.exp_name}-last.pth")
    if args.resume and os.path.exists(ckpt_path):
        print(f"[Resume] Đang nạp checkpoint từ: {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"✓ [Resume] Đã nạp thành công từ Epoch {start_epoch - 1}, Step {global_step} | Best Val Loss: {best_val_loss:.4f}")

    # 3. Dữ liệu: Cả Train và Validation
    max_sec = cfg["data"].get("max_audio_sec", 10.0)
    train_pipeline = VietnameseStreamingPipeline(split='train', max_sec=max_sec)
    val_pipeline = VietnameseStreamingPipeline(split='validation', max_sec=max_sec)

    epochs = 1 if args.smoke_test else train_cfg["epochs"]
    steps_per_epoch = 2 if args.smoke_test else train_cfg.get("steps_per_epoch", 30)
    val_steps = 1 if args.smoke_test else train_cfg.get("val_steps", 6)
    batch_size = train_cfg["batch_size"]

    print(f"\n[Training Config] {epochs} Epochs | {steps_per_epoch} Steps/Epoch | {val_steps} Val Steps | Batch: {batch_size}")

    for epoch in range(start_epoch, epochs):
        # ------------------- VÒNG LẶP TRAIN -------------------
        model.train()
        train_epoch_loss = 0.0
        t0 = time.time()

        for step in range(steps_per_epoch):
            optimizer.zero_grad()

            audio_16k, _ = train_pipeline.fetch_batch(batch_size=batch_size)

            # Trích xuất Whisper Latents
            if feat_extractor is not None and whisper_enc is not None:
                in_proc = feat_extractor(
                    audio_16k.numpy(), sampling_rate=16000, return_tensors='pt', padding="max_length"
                )
                feats = in_proc.input_features.to(device=device, dtype=dtype)
                if feats.shape[-1] < 3000:
                    feats = F.pad(feats, (0, 3000 - feats.shape[-1]))
                elif feats.shape[-1] > 3000:
                    feats = feats[:, :, :3000]

                with torch.no_grad():
                    full_latents = whisper_enc(feats).last_hidden_state
                real_len = max(10, int(audio_16k.shape[-1] / 320))
                latents = full_latents[:, :real_len, :]
            else:
                latents = torch.randn(batch_size, 100, 768, device=device, dtype=dtype)

            # Audio 24kHz -> 100-Mel Spectrogram
            audio_24k = resample_16k_to_24k(audio_16k.to(device=device, dtype=torch.float32))
            tgt_mel = extract_100_mel(audio_24k, mel_fb).to(dtype=dtype)

            # Loss OT-CFM
            cfm_loss = model.compute_loss(latents, tgt_mel)
            cfm_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(train_cfg["max_grad_norm"]))
            optimizer.step()

            global_step += 1
            train_epoch_loss += cfm_loss.item()

            if global_step % 5 == 0 or args.smoke_test:
                print(f"[E{epoch:02d}|Step {global_step:04d}] Train CFM Loss: {cfm_loss.item():.4f}")

        avg_train_loss = train_epoch_loss / max(1, steps_per_epoch)
        train_time = time.time() - t0

        # ------------------- VÒNG LẶP VALIDATION -------------------
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for v_idx in range(val_steps):
                v_audio_16k, _ = val_pipeline.fetch_batch(batch_size=batch_size)

                if feat_extractor is not None and whisper_enc is not None:
                    in_proc = feat_extractor(
                        v_audio_16k.numpy(), sampling_rate=16000, return_tensors='pt', padding="max_length"
                    )
                    v_feats = in_proc.input_features.to(device=device, dtype=dtype)
                    if v_feats.shape[-1] < 3000:
                        v_feats = F.pad(v_feats, (0, 3000 - v_feats.shape[-1]))
                    elif v_feats.shape[-1] > 3000:
                        v_feats = v_feats[:, :, :3000]

                    v_latents = whisper_enc(v_feats).last_hidden_state
                    v_real_len = max(10, int(v_audio_16k.shape[-1] / 320))
                    v_latents = v_latents[:, :v_real_len, :]
                else:
                    v_latents = torch.randn(batch_size, 100, 768, device=device, dtype=dtype)

                v_audio_24k = resample_16k_to_24k(v_audio_16k.to(device=device, dtype=torch.float32))
                v_tgt_mel = extract_100_mel(v_audio_24k, mel_fb).to(dtype=dtype)

                v_loss = model.compute_loss(v_latents, v_tgt_mel)
                val_loss_total += v_loss.item()

        avg_val_loss = val_loss_total / max(1, val_steps)

        print("-" * 75)
        print(f"📊 [Epoch {epoch:02d} Tổng kết] Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Thời gian: {train_time:.1f}s")

        # Lưu Checkpoint Atomic
        ckpt_data = {
            "epoch": epoch,
            "step": global_step,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "best_val_loss": min(best_val_loss, avg_val_loss),
            "model_state_dict": model.decoder.state_dict(),  # Lưu riêng decoder weights để cắm vào mọi kịch bản
            "full_system_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
        }

        tmp_ckpt = ckpt_path + ".tmp"
        torch.save(ckpt_data, tmp_ckpt)
        os.replace(tmp_ckpt, ckpt_path)

        # Cập nhật Best Checkpoint dựa trên Validation Loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_ckpt = os.path.join(paths["checkpoint_dir"], f"{args.exp_name}-best.pth")
            torch.save(ckpt_data, best_ckpt)
            print(f"⭐ [Mới] Đã cập nhật Model Best (Dựa trên Val Loss): {best_val_loss:.4f}")
        print("-" * 75)

    print("=" * 75)
    print("✅ HUẤN LUYỆN OT-CFM TARGET DECODER KẾT THÚC THÀNH CÔNG!")
    print(f"Checkpoint tốt nhất lưu tại: {os.path.join(paths['checkpoint_dir'], f'{args.exp_name}-best.pth')}")
    print("=" * 75)

if __name__ == "__main__":
    main()
