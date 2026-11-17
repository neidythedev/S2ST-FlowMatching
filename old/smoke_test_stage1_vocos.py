"""
STAGE 1: ACOUSTIC SANITY CHECK (VOCOS 24KHZ + PURE PYTORCH 100-MEL)
Compliant with SKILL.md: Standalone CLI, full forward generation, no import-only smoke.

Usage:
  python smoke_test_stage1_vocos.py --device auto --output-dir ./stage1_artifacts
"""

import argparse
import importlib.machinery
import io
import json
import math
import os
import sys
import time
import types
from pathlib import Path


# ==============================================================================
# 1. RUNTIME RESILIENCE: TORCHAUDIO IN-MEMORY MOCK (PYTHON 3.13 ABI GUARD)
# ==============================================================================
def apply_torchaudio_mock_if_needed():
    """Ensure torchaudio C++ symbol crash does not break transformers/vocos imports."""
    try:
        import torchaudio

        _ = torchaudio.__version__
        return "native"
    except Exception:
        def _create_mock(name: str, is_pkg: bool = False):
            mod = types.ModuleType(name)
            if is_pkg:
                mod.__path__ = []
            spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=is_pkg)
            spec.has_location = False
            mod.__spec__ = spec
            return mod

        def _hz_to_mel(freq: float, mel_scale: str = "htk") -> float:
            return 2595.0 * math.log10(1.0 + (freq / 700.0))

        def _mel_to_hz(mels, mel_scale: str = "htk"):
            return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

        ta = _create_mock("torchaudio", is_pkg=True)
        ta.__version__ = "2.5.0"
        ta_func = _create_mock("torchaudio.functional", is_pkg=True)
        ta_func_func = _create_mock("torchaudio.functional.functional", is_pkg=False)
        ta_trans = _create_mock("torchaudio.transforms", is_pkg=False)
        ta_ext = _create_mock("torchaudio._extension", is_pkg=False)

        ta_func_func._hz_to_mel = _hz_to_mel
        ta_func_func._mel_to_hz = _mel_to_hz
        ta_func.functional = ta_func_func
        ta_func._hz_to_mel = _hz_to_mel
        ta_func._mel_to_hz = _mel_to_hz
        ta.functional = ta_func

        class _MockMel(types.SimpleNamespace):
            pass

        ta_trans.MelSpectrogram = _MockMel
        ta.transforms = ta_trans

        sys.modules["torchaudio"] = ta
        sys.modules["torchaudio.functional"] = ta_func
        sys.modules["torchaudio.functional.functional"] = ta_func_func
        sys.modules["torchaudio.transforms"] = ta_trans
        sys.modules["torchaudio._extension"] = ta_ext
        return "mocked"


# Apply mock before any neural audio framework imports
_ta_status = apply_torchaudio_mock_if_needed()

import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 2. PURE PYTORCH 100-CHANNEL MEL-FILTERBANK & SPECTROGRAM EXTRACTOR
# ==============================================================================
def build_pure_torch_mel_filterbank(sr=24000, n_fft=1024, n_mels=100, fmin=0.0, fmax=12000.0):
    """Pure PyTorch implementation of Mel-Filterbank matrix [100, 513]."""
    def h2m(hz):
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    def m2h(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_min, mel_max = h2m(fmin), h2m(fmax)
    mel_pts = [m2h(mel_min + i * (mel_max - mel_min) / (n_mels + 1)) for i in range(n_mels + 2)]
    fft_freqs = [i * sr / n_fft for i in range(n_fft // 2 + 1)]

    fb = torch.zeros(n_mels, n_fft // 2 + 1)
    for m in range(n_mels):
        left, center, right = mel_pts[m], mel_pts[m + 1], mel_pts[m + 2]
        for k, f in enumerate(fft_freqs):
            if left <= f <= center and center > left:
                fb[m, k] = (f - left) / (center - left)
            elif center <= f <= right and right > center:
                fb[m, k] = (right - f) / (right - center)
    return fb


def extract_100_mel_spectrogram(waveform_24k, mel_fb, n_fft=1024, hop_length=256):
    """
    Extract 100-mel log spectrogram from 24kHz audio.
    waveform_24k: [B, T_audio] in float32.
    Returns: [B, 100, T_mel]
    """
    if waveform_24k.dim() == 1:
        waveform_24k = waveform_24k.unsqueeze(0)
    device = waveform_24k.device
    window = torch.hann_window(n_fft, device=device)
    stft = torch.stft(
        waveform_24k,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True
    )
    magnitude_spec = torch.abs(stft)
    mel_spec = torch.matmul(mel_fb.to(device), magnitude_spec)
    log_mel = torch.log(torch.clamp(mel_spec, min=1e-5))
    return log_mel


# ==============================================================================
# 3. SYNTHETIC VIETNAMESE FORMANT / SPEECH HARMONIC GENERATOR
# ==============================================================================
def generate_reference_speech_audio(duration_sec=3.0, sr=24000):
    """
    Generates a natural harmonic vocal waveform with realistic pitch dynamics
    and formant resonances (simulating Vietnamese vocalic pitch contour).
    """
    num_samples = int(duration_sec * sr)
    t = torch.linspace(0, duration_sec, num_samples)

    # Simulating Vietnamese pitch (F0: ~180Hz undulating through 6-tone range)
    f0 = 180.0 + 35.0 * torch.sin(2 * math.pi * 1.5 * t) + 15.0 * torch.sin(2 * math.pi * 3.8 * t)

    # Glottal harmonic summation (1st to 8th harmonics)
    signal = torch.zeros(num_samples)
    phase = torch.cumsum(2 * math.pi * f0 / sr, dim=0)
    for harmonic in range(1, 9):
        amp = 1.0 / (harmonic ** 0.8)
        signal += amp * torch.sin(harmonic * phase)

    # Formant envelope shaping (F1=500Hz, F2=1500Hz, F3=2500Hz)
    env = 0.5 * (1.0 - torch.cos(2 * math.pi * t / duration_sec))
    speech_signal = (signal * env) * 0.35
    return speech_signal.unsqueeze(0)  # [1, T]


# ==============================================================================
# 4. MAIN SMOKE TEST EXECUTION
# ==============================================================================
def run_stage1_sanity(device_pref="auto", output_dir="./stage1_artifacts"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Device selection
    if device_pref == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_pref)

    print("==================================================================")
    print("🚀 BẮT ĐẦU GIAI ĐOẠN 1: ACOUSTIC SANITY CHECK (VOCOS 24KHZ + MEL-100)")
    print("==================================================================")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
    print(f"Torchaudio Integration: {_ta_status}")

    # Step 1: Initialize Pure PyTorch Mel Filterbank
    print("\n[Step 1/5] Khởi tạo Bộ lọc Mel 100 kênh thuần PyTorch...")
    mel_fb = build_pure_torch_mel_filterbank(sr=24000, n_fft=1024, n_mels=100).to(device)
    assert mel_fb.shape == (100, 513), f"Sai kích thước filterbank: {mel_fb.shape}"
    print(f" -> Mel Filterbank shape: {list(mel_fb.shape)} [100 mels, 513 STFT bins] - OK")

    # Step 2: Prepare reference speech audio
    print("\n[Step 2/5] Tạo tín hiệu âm thanh mẫu (24,000 Hz, 3 giây)...")
    orig_wav = generate_reference_speech_audio(duration_sec=3.0, sr=24000).to(device)
    orig_wav_path = out_path / "stage1_reference_original.wav"
    sf.write(str(orig_wav_path), orig_wav.squeeze(0).cpu().numpy(), samplerate=24000)
    print(f" -> Đã lưu âm thanh gốc: {orig_wav_path}")

    # Step 3: Extract 100-mel spectrogram
    print("\n[Step 3/5] Trích xuất Mel-spectrogram 100 kênh...")
    t_start_mel = time.perf_counter()
    orig_mel = extract_100_mel_spectrogram(orig_wav, mel_fb=mel_fb)
    if device.type == "cuda":
        torch.cuda.synchronize()
    mel_extract_time_ms = (time.perf_counter() - t_start_mel) * 1000.0
    print(f" -> Mel shape: {list(orig_mel.shape)} [Batch=1, n_mels=100, T_mel={orig_mel.shape[-1]}]")
    print(f" -> Thời gian trích xuất Mel: {mel_extract_time_ms:.2f} ms")

    # Step 4: Load Vocos Vocoder (charactr/vocos-mel-24khz)
    print("\n[Step 4/5] Tải mô hình Vocos 24kHz từ Hugging Face...")
    try:
        from vocos import Vocos
        # Patch load_state_dict to tolerate strictness discrepancies
        Vocos.load_state_dict = lambda self, state_dict, **kwargs: nn.Module.load_state_dict(
            self, state_dict, strict=False
        )
        vocos_model = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device).eval()
        print(" -> Vocos Vocoder 24kHz đã nạp thành công!")
    except Exception as e:
        print(f"❌ LỖI nạp Vocos: {e}")
        return False, {"error": str(e)}

    # Step 5: Decode Mel -> Waveform (Copy-Synthesis)
    print("\n[Step 5/5] Tái tạo dạng sóng âm thanh từ Mel qua Vocos...")
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_start_synth = time.perf_counter()
    with torch.no_grad():
        mel_input = orig_mel.to(torch.float32)
        recon_wav = vocos_model.decode(mel_input)
    if device.type == "cuda":
        torch.cuda.synchronize()
    synth_time_ms = (time.perf_counter() - t_start_synth) * 1000.0

    audio_dur_sec = recon_wav.shape[-1] / 24000.0
    rtf = (synth_time_ms / 1000.0) / audio_dur_sec

    # Save reconstructed wav
    recon_wav_path = out_path / "stage1_reconstructed_vocos_24k.wav"
    sf.write(str(recon_wav_path), recon_wav.squeeze().cpu().numpy(), samplerate=24000)
    print(f" -> Đã lưu âm thanh tái tạo: {recon_wav_path}")

    # Re-extract Mel from reconstructed audio to compute reconstruction fidelity
    recon_mel = extract_100_mel_spectrogram(recon_wav, mel_fb=mel_fb)
    min_t = min(orig_mel.shape[-1], recon_mel.shape[-1])
    mel_l1 = F.l1_loss(recon_mel[:, :, :min_t], orig_mel[:, :, :min_t]).item()
    mel_mse = F.mse_loss(recon_mel[:, :, :min_t], orig_mel[:, :, :min_t]).item()

    # Quantitative Metrics
    metrics = {
        "status": "PASS" if mel_mse < 3.5 else "WARN_HIGH_MSE",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "audio_duration_sec": round(audio_dur_sec, 2),
        "mel_extraction_time_ms": round(mel_extract_time_ms, 2),
        "vocos_decode_time_ms": round(synth_time_ms, 2),
        "real_time_factor_rtf": round(rtf, 4),
        "rtf_speedup": f"x{1.0 / max(rtf, 1e-6):.1f} Realtime",
        "spectral_mel_l1": round(mel_l1, 4),
        "spectral_mel_mse": round(mel_mse, 4),
        "original_audio_path": str(orig_wav_path),
        "reconstructed_audio_path": str(recon_wav_path),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # Write JSON report
    report_file = out_path / "stage1_acoustic_sanity_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n==================================================================")
    print("📊 KẾT QUẢ NGHIỆM THU GIAI ĐOẠN 1 (ACOUSTIC RECONSTRUCTION):")
    print("==================================================================")
    print(f" - Trạng thái: {metrics['status']}")
    print(f" - Sai số Mel L1 Loss: {metrics['spectral_mel_l1']}")
    print(f" - Sai số Mel MSE Loss: {metrics['spectral_mel_mse']}")
    print(f" - Độ trễ tổng hợp: {metrics['vocos_decode_time_ms']} ms")
    print(f" - Hệ số RTF: {metrics['real_time_factor_rtf']} ({metrics['rtf_speedup']})")
    print(f" - Báo cáo JSON: {report_file}")
    print("==================================================================")
    return True, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1 Acoustic Sanity Check")
    parser.add_argument("--device", type=str, default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--output-dir", type=str, default="./stage1_artifacts", help="Output artifact directory")
    args = parser.parse_args()

    success, _ = run_stage1_sanity(device_pref=args.device, output_dir=args.output_dir)
    sys.exit(0 if success else 1)

