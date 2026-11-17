#!/usr/bin/env python3
# ==============================================================================
# BỘ CÔNG CỤ EDA VÀ SANITY CHECK TOÀN DIỆN (S2ST VI -> EN)
# Thực hiện chuẩn xác 4 bước kiểm tra sinh tử trước khi huấn luyện:
#   1. EDA Dữ liệu (Độ dài, Sample Rate, Tỷ lệ En/Vi, lọc outlier)
#   2. Sanity Check #1: Vocos Round-Trip Test (Kiểm tra hàm Mel 100 kênh)
#   3. Sanity Check #2: Kiểm tra độ khỏe của Whisper trên Tiếng Việt (WER/CER)
# ==============================================================================

import argparse
import io
import math
import os
import sys
import time
import types
import importlib.machinery
import wave

# Tắt cảnh báo transformers
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# RUNTIME RESILIENCE: TORCHAUDIO IN-MEMORY MOCK (PYTHON 3.13 ABI GUARD)
# ==============================================================================
def apply_torchaudio_mock_if_needed():
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
import torch.nn.functional as F

from train_bridge import (
    build_mel_filterbank, extract_100_mel, resample_16k_to_24k, resample_24k_to_16k,
    RealStreamingSpeechPipeline
)

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

# ==============================================================================
# BƯỚC 1: EDA DỮ LIỆU (DATA INTEGRITY & DISTRIBUTION CHECK)
# ==============================================================================

def run_eda_check(num_samples=30):
    print("\n" + "=" * 75)
    print("📊 BƯỚC 1: EDA DỮ LIỆU & KIỂM TRA PHÂN PHỐI ĐỘ DÀI (GOOGLE FLEURS VI -> EN)")
    print("=" * 75)
    pipeline = RealStreamingSpeechPipeline(split='train', max_sec=20.0, src_lang="vi_vn", tgt_lang="en_us")

    if not pipeline._parallel_pairs:
        print("[Lỗi] Không thể nạp cặp dữ liệu song song để phân tích EDA.")
        return

    n = min(num_samples, len(pipeline._parallel_pairs))
    vi_durations = []
    en_durations = []
    ratios = []
    anomalies = 0

    print(f"🔍 Đang phân tích thống kê trên {n} cặp câu song song...")
    for i in range(n):
        item_vi, item_en = pipeline._parallel_pairs[i]
        try:
            w_vi = pipeline._extract_audio_tensor(item_vi['audio'])
            w_en = pipeline._extract_audio_tensor(item_en['audio'])
            dur_vi = len(w_vi) / 16000.0
            dur_en = len(w_en) / 16000.0
            ratio = dur_en / max(0.1, dur_vi)

            vi_durations.append(dur_vi)
            en_durations.append(dur_en)
            ratios.append(ratio)

            # Cảnh báo bất thường
            if dur_vi < 0.5 or dur_en < 0.5:
                print(f"  ⚠️ Cảnh báo cặp #{i+1}: Audio quá ngắn (<0.5s): Vi={dur_vi:.2f}s, En={dur_en:.2f}s")
                anomalies += 1
            elif dur_vi > 15.0 or dur_en > 15.0:
                print(f"  ⚠️ Cảnh báo cặp #{i+1}: Audio quá dài (>15s): Vi={dur_vi:.2f}s, En={dur_en:.2f}s (Có nguy cơ OOM)")
                anomalies += 1
            elif ratio > 3.0 or ratio < 0.33:
                print(f"  ⚠️ Cảnh báo cặp #{i+1}: Tỷ lệ độ dài bất thường (En/Vi = {ratio:.2f}x)")
                anomalies += 1
        except Exception as e:
            print(f"  [Lỗi đọc cặp #{i+1}]: {e}")

    avg_vi = sum(vi_durations) / len(vi_durations)
    avg_en = sum(en_durations) / len(en_durations)
    avg_ratio = sum(ratios) / len(ratios)

    print("-" * 75)
    print("📈 KẾT QUẢ PHÂN TÍCH THỐNG KÊ EDA:")
    print(f"  • Độ dài Tiếng Việt (Nguồn): Trung bình {avg_vi:.2f}s | Min: {min(vi_durations):.2f}s | Max: {max(vi_durations):.2f}s")
    print(f"  • Độ dài Tiếng Anh  (Đích) : Trung bình {avg_en:.2f}s | Min: {min(en_durations):.2f}s | Max: {max(en_durations):.2f}s")
    print(f"  • Tỷ lệ độ dài En / Vi    : Trung bình {avg_ratio:.2f}x (Hợp lý trong khoảng 0.7x ~ 1.5x)")
    print(f"  • Số lượng mẫu bất thường : {anomalies} / {n} ({anomalies/n*100:.1f}%)")

    if anomalies == 0:
        print("✅ EDA PASS: Dữ liệu sạch, phân phối độ dài rất chuẩn cho bài toán Vi -> En!")
    else:
        print("ℹ️ Dữ liệu có vài câu ngoại lai, cơ chế max_audio_sec=12.0s trong pipeline sẽ tự động cắt gọt an toàn.")
    print("=" * 75 + "\n")

# ==============================================================================
# BƯỚC 2: SANITY CHECK #1 - VOCOS ROUND-TRIP TEST
# ==============================================================================

def run_vocos_roundtrip_test(device):
    print("\n" + "=" * 75)
    print("🧪 BƯỚC 2: SANITY CHECK #1 - VOCOS ROUND-TRIP TEST (KIỂM TRA HÀM MEL 100)")
    print("Mục tiêu: Đảm bảo Mel 100-band qua Vocos tái tạo âm thanh trong trẻo, không bị rè/méo.")
    print("=" * 75)

    try:
        from vocos import Vocos
        vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
        vocos.eval()
        print("✓ Nạp thành công Vocos Vocoder (24kHz)!")
    except Exception as e:
        print(f"[Thất bại] Không thể nạp Vocos: {e}. Vui lòng cài đặt: pip install vocos")
        return

    pipeline = RealStreamingSpeechPipeline(split='train', max_sec=10.0, src_lang="vi_vn", tgt_lang="en_us")
    src_audio, tgt_audio = pipeline.fetch_batch(batch_size=5, return_text=False)

    mel_fb = build_mel_filterbank(sr=24000, n_mels=100, device=device)
    out_dir = "outputs/sanity/vocos_roundtrip"
    os.makedirs(out_dir, exist_ok=True)

    mel_l1_errors = []
    for idx in range(min(5, tgt_audio.shape[0])):
        w_16k = tgt_audio[idx]  # Lấy audio tiếng Anh
        w_24k = resample_16k_to_24k(w_16k.unsqueeze(0).to(device=device, dtype=torch.float32))

        # 1. Trích xuất Mel 100 kênh bằng hàm của chúng ta
        mel = extract_100_mel(w_24k, mel_fb)

        # 2. Đưa thẳng Mel vào Vocos giải mã
        with torch.no_grad():
            rec_wav_24k = vocos.decode(mel).squeeze()

        # 3. Trích xuất lại Mel từ audio tái tạo để đo độ sai lệch phổ
        rec_mel = extract_100_mel(rec_wav_24k.unsqueeze(0), mel_fb)
        min_t = min(mel.shape[-1], rec_mel.shape[-1])
        mel_err = F.l1_loss(mel[:, :, :min_t], rec_mel[:, :, :min_t]).item()
        mel_l1_errors.append(mel_err)

        # Lưu file để nghe đối chiếu
        orig_path = os.path.join(out_dir, f"sample_{idx+1:02d}_original_24k.wav")
        rec_path = os.path.join(out_dir, f"sample_{idx+1:02d}_vocos_reconstructed.wav")
        save_wav_file(orig_path, w_24k, 24000)
        save_wav_file(rec_path, rec_wav_24k, 24000)

        print(f"  • Mẫu #{idx+1:02d}: Mel L1 Reconstruction Error = {mel_err:.4f}")

    avg_mel_err = sum(mel_l1_errors) / len(mel_l1_errors)
    print("-" * 75)
    print(f"📊 Mel L1 Tái Tạo Trung Bình: {avg_mel_err:.4f}")
    if avg_mel_err < 0.40:
        print("✅ SANITY CHECK #1 PASS! Hàm tính Mel 100 khớp hoàn hảo với Vocos Vocoder.")
        print(f"   🎧 Nghe thử file âm thanh tái tạo tại: {out_dir}")
    else:
        print("⚠️ Cảnh báo: Sai lệch phổ Mel cao (> 0.40). Hãy kiểm tra lại n_fft, hop_length hoặc fmin/fmax.")
    print("=" * 75 + "\n")

# ==============================================================================
# BƯỚC 3: SANITY CHECK #2 - ĐỘ KHỎE CỦA WHISPER TRÊN TIẾNG VIỆT
# ==============================================================================

def run_whisper_vietnamese_health_check(device, model_id="openai/whisper-small", num_samples=10):
    print("\n" + "=" * 75)
    print(f"🧪 BƯỚC 3: SANITY CHECK #2 - KIỂM TRA ĐỘ KHỎE CỦA WHISPER TRÊN TIẾNG VIỆT ({model_id})")
    print("Mục tiêu: Đảm bảo Whisper Encoder nghe tiếng Việt chuẩn xác trước khi đóng băng làm nguồn.")
    print("=" * 75)

    try:
        from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperForConditionalGeneration
        print(f"⏳ Đang nạp Whisper ({model_id})...")
        feat_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
        tokenizer = WhisperTokenizer.from_pretrained(model_id, language="vi", task="transcribe")
        model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device).eval()
        if hasattr(model, "generation_config"):
            model.generation_config.max_length = None
        print("✓ Nạp Whisper hoàn tất!")
    except Exception as e:
        print(f"[Thất bại] Không thể nạp Whisper: {e}")
        return

    pipeline = RealStreamingSpeechPipeline(split='train', max_sec=12.0, src_lang="vi_vn", tgt_lang="en_us")
    src_audio, tgt_audio, en_texts, vi_texts = pipeline.fetch_batch(batch_size=num_samples, return_text=True)

    def simple_wer(ref, hyp):
        r = ref.lower().strip().split()
        h = hyp.lower().strip().split()
        if not r: return 0.0
        # Tính edit distance
        d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
        for i in range(len(r) + 1): d[i][0] = i
        for j in range(len(h) + 1): d[0][j] = j
        for i in range(1, len(r) + 1):
            for j in range(1, len(h) + 1):
                cost = 0 if r[i-1] == h[j-1] else 1
                d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
        return min(1.0, d[len(r)][len(h)] / len(r))

    wers = []
    print(f"🎙️ Đang cho Whisper nhận diện thử {num_samples} câu tiếng Việt thực tế...\n")
    for idx in range(min(num_samples, src_audio.shape[0])):
        w_16k = src_audio[idx]
        ref_txt = vi_texts[idx]

        in_p = feat_extractor(w_16k.numpy(), sampling_rate=16000, return_tensors='pt', return_attention_mask=True)
        in_feats = in_p.input_features.to(device)
        in_mask = in_p.attention_mask.to(device) if "attention_mask" in in_p else None

        with torch.no_grad():
            gen_kwargs = {"max_new_tokens": 128, "language": "vi", "task": "transcribe"}
            if in_mask is not None: gen_kwargs["attention_mask"] = in_mask
            pred_ids = model.generate(in_feats, **gen_kwargs)
            hyp_txt = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()

        wer = simple_wer(ref_txt, hyp_txt)
        wers.append(wer)

        print(f"  🔹 Cặp #{idx+1:02d} [WER: {wer*100:.1f}%]:")
        print(f"     🎯 Chuẩn (Ref): \"{ref_txt}\"")
        print(f"     🗣️ Nhận diện   : \"{hyp_txt}\"")

    avg_wer = sum(wers) / len(wers) * 100.0
    print("-" * 75)
    print(f"📊 Word Error Rate (WER) Trung Bình: {avg_wer:.1f}%")
    if avg_wer < 20.0:
        print(f"✅ SANITY CHECK #2 PASS! Whisper Encoder ({model_id}) nghe tiếng Việt rất khỏe ({avg_wer:.1f}% WER).")
        print("   Đảm bảo trích xuất đặc trưng ngữ nghĩa dày dặn và chính xác cho Bridge!")
    elif avg_wer < 35.0:
        print(f"⚠️ Chấp nhận được ({avg_wer:.1f}% WER). Để đạt độ chính xác cao nhất cho bài báo, nên cân nhắc vinai/PhoWhisper-base hoặc whisper-large-v3.")
    else:
        print(f"❌ CẢNH BÁO: Whisper ({model_id}) nghe tiếng Việt kém ({avg_wer:.1f}% WER). Cần nâng cấp Encoder!")
    print("=" * 75 + "\n")

# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Bộ công cụ EDA & Sanity Check Toàn Diện (S2ST Vi -> En)")
    parser.add_argument("--all", action="store_true", help="Chạy toàn bộ 3 bước kiểm tra tuần tự")
    parser.add_argument("--eda", action="store_true", help="Chạy Bước 1: EDA Dữ liệu và độ dài")
    parser.add_argument("--vocos-roundtrip", action="store_true", help="Chạy Bước 2: Vocos Round-Trip Test")
    parser.add_argument("--whisper-check", action="store_true", help="Chạy Bước 3: Whisper Vietnamese Health Check")
    parser.add_argument("--whisper-model", type=str, default="openai/whisper-small", help="Model Whisper cần test")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not any([args.all, args.eda, args.vocos_roundtrip, args.whisper_check]):
        args.all = True

    if args.all or args.eda:
        run_eda_check(num_samples=30)

    if args.all or args.vocos_roundtrip:
        run_vocos_roundtrip_test(device)

    if args.all or args.whisper_check:
        run_whisper_vietnamese_health_check(device, model_id=args.whisper_model, num_samples=10)

if __name__ == "__main__":
    main()
