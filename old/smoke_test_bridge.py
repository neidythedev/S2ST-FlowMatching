#!/usr/bin/env python3
# ==============================================================================
# SMOKE TEST: TONE-AWARE BRIDGE & ENVIS2ST SYSTEM
# Tuân thủ nghiêm ngặt quy chuẩn SKILL2.md:
# Phải in ra "ALL_TESTS_PASSED" khi mọi kiểm thử CUDA/Forward/Backward hoàn tất.
# ==============================================================================

import os
import sys
import tempfile
import time
from pathlib import Path

def run_smoke_test():
    print("=" * 65)
    print("🧪 BẮT ĐẦU SMOKE TEST: TONE-AWARE BRIDGE PIPELINE")
    print("=" * 65)

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as e:
        print(f"[FAIL] Thiếu thư viện PyTorch: {e}")
        return False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1/6] Kiểm tra thiết bị: {device}")
    if torch.cuda.is_available():
        print(f"      GPU: {torch.cuda.get_device_name(0)}")

    # Import mô hình từ train_bridge
    try:
        from train_bridge import (
            ToneAwareBridge,
            FlowMatchingTargetDecoder,
            AcousticSpeakerEncoder,
            EnViS2STSystem,
            parse_tones
        )
        print("[2/6] Import các module cốt lõi: THÀNH CÔNG")
    except Exception as e:
        print(f"[FAIL] Không thể import module từ train_bridge.py: {e}")
        return False

    # Test 1: Kiểm tra cấu trúc ToneAwareBridge
    B, T_src, D_in = 2, 80, 768
    src_feat = torch.randn(B, T_src, D_in, device=device)
    bridge = ToneAwareBridge(in_dim=D_in, hidden_dim=512, num_layers=2, num_tones=6).to(device)

    try:
        latents, tone_logits, dur_scale = bridge(src_feat)
        assert latents.shape == (B, T_src // 2, 512), f"Sai shape latents: {latents.shape}"
        assert tone_logits.shape == (B, T_src // 2, 6), f"Sai shape tone_logits: {tone_logits.shape}"
        assert dur_scale.shape == (B,), f"Sai shape dur_scale: {dur_scale.shape}"
        print(f"[3/6] Forward ToneAwareBridge: THÀNH CÔNG (latents: {latents.shape}, tones: {tone_logits.shape})")
    except Exception as e:
        print(f"[FAIL] Lỗi Forward ToneAwareBridge: {e}")
        return False

    # Test 2: Kiểm tra EnViS2STSystem (Forward, Loss, Backward)
    T_tgt = 120
    tgt_mel = torch.randn(B, 100, T_tgt, device=device)
    dummy_tones = torch.randint(0, 6, (B, T_tgt // 2), device=device)
    dur_ratio = torch.tensor([1.5, 1.5], device=device)

    system = EnViS2STSystem(encoder_dim=D_in, bridge_dim=512, bridge_layers=2, n_mels=100).to(device)
    optimizer = torch.optim.AdamW(system.parameters(), lr=1e-4)

    try:
        total_loss, cfm_l, tone_l, dur_l = system.compute_loss(
            src_feat, tgt_mel, target_tones=dummy_tones, target_dur_ratio=dur_ratio
        )
        assert not torch.isnan(total_loss), "Total Loss bị NaN!"
        assert not torch.isnan(cfm_l), "CFM Loss bị NaN!"
        assert not torch.isnan(tone_l), "Tone Loss bị NaN!"
        assert not torch.isnan(dur_l), "Duration Loss bị NaN!"

        optimizer.zero_grad()
        total_loss.backward()

        # Kiểm tra gradient có tồn tại không
        has_grad = False
        for p in system.bridge.parameters():
            if p.grad is not None:
                has_grad = True
                break
        assert has_grad, "Không có gradient chảy về ToneAwareBridge!"

        optimizer.step()
        print(f"[4/6] Tính toán Multi-Task Loss & Backward: THÀNH CÔNG")
        print(f"      Total: {total_loss.item():.4f} | CFM: {cfm_l.item():.4f} | Tone: {tone_l.item():.4f} | Dur: {dur_l.item():.4f}")
    except Exception as e:
        print(f"[FAIL] Lỗi trong khâu tính Loss/Backward: {e}")
        return False

    # Test 3: Kiểm tra Inference Flow Matching (5 bước ODE)
    try:
        with torch.no_grad():
            gen_mel = system.inference(src_feat[:1], num_steps=5)
            assert gen_mel.shape[1] == 100, f"Sai số kênh mel: {gen_mel.shape}"
            assert not torch.isnan(gen_mel).any(), "Sinh phổ mel bị NaN!"
        print(f"[5/6] Sinh âm thanh thử nghiệm (ODE 5 steps): THÀNH CÔNG (Output Mel: {gen_mel.shape})")
    except Exception as e:
        print(f"[FAIL] Lỗi Inference ODE: {e}")
        return False

    # Test 4: Kiểm tra Checkpoint Save & Resume Atomic
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = os.path.join(tmp_dir, "model-test.pth")
            torch.save({
                "model_state_dict": system.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": 42
            }, ckpt_path)

            loaded = torch.load(ckpt_path, map_location=device)
            system.load_state_dict(loaded["model_state_dict"])
            assert loaded["step"] == 42
        print("[6/6] Kiểm tra Lưu & Nạp Checkpoint: THÀNH CÔNG")
    except Exception as e:
        print(f"[FAIL] Lỗi Checkpoint Save/Load: {e}")
        return False

    print("=" * 65)
    print("ALL_TESTS_PASSED")
    print("=" * 65)
    return True

if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
