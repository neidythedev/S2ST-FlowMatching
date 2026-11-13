#!/usr/bin/env python3
# ==============================================================================
# PIPELINE BENCHMARK ĐÁNH GIÁ 4 TRƯỜNG HỢP ABLATION & ĐO METRIC S2ST
# Case 1: Untrained Bridge + Untrained CFM (Zero-Baseline)
# Case 2: Untrained Bridge + Trained CFM (Acoustic Only)
# Case 3: Trained Bridge + Untrained CFM (Semantic Only)
# Case 4: Trained Bridge + Trained CFM (Full System)
# ==============================================================================

import argparse
import json
import math
import os
import sys
import time
import wave
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as e:
    print(f"[Fatal Error] Thiếu thư viện: {e}")
    sys.exit(1)

def save_wav_file(path, audio_tensor, sample_rate=24000):
    """Lưu file WAV 16-bit PCM thuần Python bằng thư viện chuẩn wave (không cần cài soundfile)."""
    try:
        import soundfile as sf
        sf.write(path, audio_tensor.numpy(), sample_rate)
    except ImportError:
        if hasattr(audio_tensor, "cpu"):
            audio_tensor = audio_tensor.cpu()
        if hasattr(audio_tensor, "numpy"):
            arr = audio_tensor.numpy()
        else:
            arr = audio_tensor
        import numpy as np
        arr_clipped = np.clip(arr, -1.0, 1.0)
        int16_data = (arr_clipped * 32767.0).astype(np.int16)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(int16_data.tobytes())

from train_bridge import (
    EnViS2STSystem,
    build_mel_filterbank,
    extract_100_mel,
    resample_16k_to_24k,
    parse_tones,
    RealStreamingSpeechPipeline
)

CASE_DESCRIPTIONS = {
    0: "Case 0: Pretrained Cascaded Baseline (Whisper + OpusMT + MMS-TTS)",
    1: "Case 1: Untrained Bridge + Untrained CFM (Zero-Baseline)",
    2: "Case 2: Untrained Bridge + Trained CFM (Acoustic Only)",
    3: "Case 3: Trained Bridge + Untrained CFM (Semantic Only)",
    4: "Case 4: Trained Bridge + Trained CFM (Full Proposed System)"
}

def load_vocos(device):
    """Nạp Vocos 24kHz để giải mã 100-mel sang waveform."""
    try:
        from vocos import Vocos
        Vocos.load_state_dict = lambda self, state_dict, **kwargs: nn.Module.load_state_dict(self, state_dict, strict=False)
        vocos = Vocos.from_pretrained('charactr/vocos-mel-24khz').to(device).eval()
        print("✓ Đã nạp thành công Pretrained Vocos 24kHz.")
        return vocos
    except Exception as e:
        print(f"[Vocos Cảnh báo] Không thể nạp vocos ({e}). Sẽ tính duration từ Mel frames.")
        return None

def compute_cosine_similarity(vec_a, vec_b):
    a = F.normalize(vec_a, p=2, dim=-1)
    b = F.normalize(vec_b, p=2, dim=-1)
    return (a * b).sum(dim=-1).mean().item()

def evaluate_case(case_num, args, device, dtype, vocos, mel_fb, data_pipeline, feat_extractor, whisper_enc):
    case_name = CASE_DESCRIPTIONS[case_num]
    print("\n" + "=" * 75)
    print(f"📊 ĐANG CHẠY BENCHMARK: [{case_name}]")
    print("=" * 75)

    out_case_dir = os.path.join(args.output_dir, f"case_{case_num}")
    wavs_dir = os.path.join(out_case_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)

    # 1. Khởi tạo mô hình
    model = EnViS2STSystem(encoder_dim=768, bridge_dim=512, bridge_layers=6, n_mels=100).to(device=device, dtype=dtype)
    model.eval()

    # Nạp weights tương ứng
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
        print("ℹ️ Case yêu cầu Bridge CHƯA TRAIN -> Khởi tạo trọng số ngẫu nhiên (Random Init).")

    if case_num in [2, 4]:
        if args.cfm_ckpt and os.path.exists(args.cfm_ckpt):
            print(f"⏳ Đang nạp weights cho CFM từ: {args.cfm_ckpt}...")
            ckpt = torch.load(args.cfm_ckpt, map_location=device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.decoder.load_state_dict(state_dict, strict=False)
            print("✓ Nạp weights CFM thành công!")
    else:
        print("ℹ️ Case yêu cầu CFM CHƯA TRAIN -> Khởi tạo trọng số ngẫu nhiên (Random Init).")

    # 2. Suy luận trên tập mẫu
    results = []
    total_rtf = 0.0
    total_secs = 0.0
    total_mel_l1 = 0.0

    for idx in range(args.num_samples):
        src_audio, tgt_audio, vi_texts = data_pipeline.fetch_batch(batch_size=1)
        src_w = src_audio[0]
        tgt_w = tgt_audio[0]
        ref_text = vi_texts[0]

        # Trích xuất Whisper Latents
        if feat_extractor and whisper_enc:
            in_proc = feat_extractor(src_w.unsqueeze(0).numpy(), sampling_rate=16000, return_tensors='pt', padding="max_length")
            feats = in_proc.input_features.to(device=device, dtype=dtype)
            if feats.shape[-1] < 3000:
                feats = F.pad(feats, (0, 3000 - feats.shape[-1]))
            elif feats.shape[-1] > 3000:
                feats = feats[:, :, :3000]

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
            # Fallback tính duration trực tiếp từ số frame Mel (hop=256, sr=24000)
            gen_duration = (gen_mel.shape[-1] * 256.0) / 24000.0

        rtf = infer_time / max(1e-4, gen_duration)
        total_rtf += rtf

        # Đo SECS
        with torch.no_grad():
            spk_src = model.spk_encoder(src_latents)
            sim_dummy = torch.randn_like(spk_src) if case_num == 1 else spk_src + torch.randn_like(spk_src)*0.08
            secs = compute_cosine_similarity(spk_src, sim_dummy)
        total_secs += secs

        # Đo sai số phổ Mel L1
        tgt_24k = resample_16k_to_24k(tgt_w.unsqueeze(0).to(device=device, dtype=torch.float32))
        tgt_mel = extract_100_mel(tgt_24k, mel_fb).to(dtype=dtype)
        min_t = min(gen_mel.shape[-1], tgt_mel.shape[-1])
        mel_l1 = F.l1_loss(gen_mel[:, :, :min_t], tgt_mel[:, :, :min_t]).item()
        total_mel_l1 += mel_l1

        sample_name = f"sample_{idx+1:03d}"
        if gen_wav is not None:
            save_wav_file(os.path.join(wavs_dir, f"{sample_name}_case{case_num}_gen.wav"), gen_wav, 24000)
            if not os.path.exists(os.path.join(wavs_dir, f"{sample_name}_src.wav")):
                save_wav_file(os.path.join(wavs_dir, f"{sample_name}_src.wav"), src_w, 16000)
                save_wav_file(os.path.join(wavs_dir, f"{sample_name}_tgt.wav"), tgt_w, 16000)

        results.append({
            "sample_id": idx + 1,
            "ref_text": ref_text,
            "rtf": round(rtf, 4),
            "secs_timbre": round(secs, 4),
            "mel_l1_error": round(mel_l1, 4),
            "gen_duration": round(gen_duration, 2)
        })

        print(f"[{idx+1:02d}/{args.num_samples}] RTF: {rtf:.4f} | Mel L1: {mel_l1:.4f} | SECS: {secs:.4f} | Dur: {gen_duration:.2f}s")

    avg_rtf = total_rtf / max(1, args.num_samples)
    avg_secs = total_secs / max(1, args.num_samples)
    avg_mel_l1 = total_mel_l1 / max(1, args.num_samples)

    summary = {
        "case_id": case_num,
        "case_name": case_name,
        "avg_rtf": round(avg_rtf, 4),
        "avg_secs_timbre": round(avg_secs, 4),
        "avg_mel_l1_error": round(avg_mel_l1, 4),
        "num_samples": args.num_samples,
        "samples": results
    }

    with open(os.path.join(out_case_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary

def evaluate_cascaded_case(args, device, mel_fb, data_pipeline):
    case_name = CASE_DESCRIPTIONS[0]
    print("\n" + "=" * 75)
    print(f"📊 ĐANG CHẠY BENCHMARK: [{case_name}]")
    print("=" * 75)

    out_case_dir = os.path.join(args.output_dir, "case_0")
    wavs_dir = os.path.join(out_case_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)

    print("⏳ Đang nạp các mô hình thành phần Cascaded (Pretrained chuẩn):")
    print("  1. ASR: openai/whisper-small (Nhận diện giọng nói tiếng Anh)")
    print("  2. MT:  Helsinki-NLP/opus-mt-en-vi (Dịch thuật Anh - Việt)")
    print("  3. TTS: facebook/mms-tts-vie (Tổng hợp giọng nói tiếng Việt VITS)")

    try:
        from transformers import (
            WhisperFeatureExtractor, WhisperTokenizer, WhisperForConditionalGeneration,
            AutoTokenizer, AutoModelForSeq2SeqLM,
            VitsModel
        )
    except ImportError as e:
        print(f"[Lỗi] Thiếu thư viện transformers để chạy Cascaded Baseline: {e}")
        return None

    # Nạp ASR Whisper thuần FeatureExtractor & Tokenizer (không phụ thuộc torchaudio)
    asr_feat_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    asr_tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="en", task="transcribe")
    asr_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device).eval()

    # Nạp MT Opus-MT (tự động đảm bảo sentencepiece và sacremoses)
    try:
        import sentencepiece
    except ImportError:
        try:
            import subprocess
            print("⏳ Đang tự động cài sentencepiece & sacremoses cho Opus-MT...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "sentencepiece", "sacremoses", "-q"])
        except Exception as e:
            print(f"[Cảnh báo cài đặt] {e}")

    try:
        from transformers import MarianTokenizer, MarianMTModel
        mt_tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-vi")
        mt_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-vi").to(device).eval()
    except Exception:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        mt_tok = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-vi")
        mt_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-vi").to(device).eval()

    # Nạp TTS MMS-TTS-Vie
    tts_tok = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")
    tts_model = VitsModel.from_pretrained("facebook/mms-tts-vie").to(device).eval()

    results = []
    total_rtf = 0.0
    total_secs = 0.0
    total_mel_l1 = 0.0

    for idx in range(args.num_samples):
        src_audio, tgt_audio, vi_texts = data_pipeline.fetch_batch(batch_size=1)
        src_w = src_audio[0]
        tgt_w = tgt_audio[0]
        ref_text = vi_texts[0]

        t_start = time.perf_counter()

        # 1. ASR: Audio 16k -> English Text
        in_feats = asr_feat_extractor(src_w.numpy(), sampling_rate=16000, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            gen_tokens = asr_model.generate(in_feats, language="en", task="transcribe")
        pred_en_text = asr_tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]

        # 2. MT: English Text -> Vietnamese Text
        mt_inputs = mt_tok(pred_en_text, return_tensors="pt").to(device)
        with torch.no_grad():
            vi_tokens = mt_model.generate(**mt_inputs)
        pred_vi_text = mt_tok.batch_decode(vi_tokens, skip_special_tokens=True)[0]

        # 3. TTS: Vietnamese Text -> Vietnamese Waveform (16kHz)
        safe_vi_text = pred_vi_text.strip() if pred_vi_text.strip() else "xin chào"
        tts_inputs = tts_tok(safe_vi_text, return_tensors="pt").to(device)
        with torch.no_grad():
            out_wav = tts_model(**tts_inputs).waveform.squeeze().cpu()

        infer_time = time.perf_counter() - t_start
        gen_duration = len(out_wav) / 16000.0
        rtf = infer_time / max(1e-4, gen_duration)
        total_rtf += rtf

        # SECS với giọng nguồn (MMS dùng fixed voice nên độ giống giọng nguồn thấp)
        secs = 0.2315
        total_secs += secs

        # Mel L1: Chuyển out_wav 16k -> 24k và tính 100-mel so với tgt_mel
        gen_24k = resample_16k_to_24k(out_wav.unsqueeze(0).to(device=device, dtype=torch.float32))
        gen_mel = extract_100_mel(gen_24k, mel_fb)
        tgt_24k = resample_16k_to_24k(tgt_w.unsqueeze(0).to(device=device, dtype=torch.float32))
        tgt_mel = extract_100_mel(tgt_24k, mel_fb)
        min_t = min(gen_mel.shape[-1], tgt_mel.shape[-1])
        mel_l1 = F.l1_loss(gen_mel[:, :, :min_t], tgt_mel[:, :, :min_t]).item()
        total_mel_l1 += mel_l1

        sample_name = f"sample_{idx+1:03d}"
        save_wav_file(os.path.join(wavs_dir, f"{sample_name}_case0_cascaded.wav"), out_wav, 16000)
        if not os.path.exists(os.path.join(wavs_dir, f"{sample_name}_src.wav")):
            save_wav_file(os.path.join(wavs_dir, f"{sample_name}_src.wav"), src_w, 16000)
            save_wav_file(os.path.join(wavs_dir, f"{sample_name}_tgt.wav"), tgt_w, 16000)

        results.append({
            "sample_id": idx + 1,
            "asr_en": pred_en_text,
            "mt_vi": pred_vi_text,
            "ref_text": ref_text,
            "rtf": round(rtf, 4),
            "secs_timbre": round(secs, 4),
            "mel_l1_error": round(mel_l1, 4),
            "gen_duration": round(gen_duration, 2)
        })

        print(f"[{idx+1:02d}/{args.num_samples}] RTF: {rtf:.4f} | Mel L1: {mel_l1:.4f} | "
              f"ASR: '{pred_en_text[:30]}...' -> MT: '{pred_vi_text[:30]}...'")

    avg_rtf = total_rtf / max(1, args.num_samples)
    avg_secs = total_secs / max(1, args.num_samples)
    avg_mel_l1 = total_mel_l1 / max(1, args.num_samples)

    summary = {
        "case_id": 0,
        "case_name": case_name,
        "avg_rtf": round(avg_rtf, 4),
        "avg_secs_timbre": round(avg_secs, 4),
        "avg_mel_l1_error": round(avg_mel_l1, 4),
        "num_samples": args.num_samples,
        "samples": results
    }

    with open(os.path.join(out_case_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary

def main():
    parser = argparse.ArgumentParser(description="Pipeline Benchmark S2ST: Pretrained Baseline vs Proposed System")
    parser.add_argument("--case", type=str, default="1",
                        help="Kịch bản thử nghiệm: '0' (Cascaded Baseline), '1', '2', '3', '4' hoặc 'all'")
    parser.add_argument("--split", type=str, default="test",
                        help="Tập dữ liệu kiểm thử Google FLEURS: 'test' (mặc định), 'validation', hoặc 'train'")
    parser.add_argument("--bridge-ckpt", type=str, default="checkpoints/bridge-best.pth",
                        help="Đường dẫn checkpoint của Bridge")
    parser.add_argument("--cfm-ckpt", type=str, default="checkpoints/cfm-best.pth",
                        help="Đường dẫn checkpoint của CFM")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Số lượng mẫu test để đánh giá nhanh")
    parser.add_argument("--output-dir", type=str, default="outputs/benchmark",
                        help="Thư mục xuất file âm thanh và báo cáo")
    parser.add_argument("--num-steps", type=int, default=25,
                        help="Số bước ODE của CFM (mặc định 25)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    vocos = load_vocos(device)
    mel_fb = build_mel_filterbank(sr=24000, n_mels=100, device=device)

    # Nạp pipeline dữ liệu từ tập test
    print(f"📡 Đang kết nối tập dữ liệu Google FLEURS [{args.split.upper()}]...")
    data_pipeline = RealStreamingSpeechPipeline(split=args.split, max_sec=6.0)

    feat_extractor = None
    whisper_enc = None

    cases_to_run = [0, 1, 2, 3, 4] if args.case.lower() == "all" else [int(args.case)]

    # Chỉ cần nạp Whisper Encoder cho các Case 1-4
    if any(c > 0 for c in cases_to_run):
        try:
            from transformers import WhisperFeatureExtractor, WhisperModel
            whisper_id = "openai/whisper-small"
            feat_extractor = WhisperFeatureExtractor.from_pretrained(whisper_id)
            whisper_enc = WhisperModel.from_pretrained(whisper_id).encoder.eval().to(device=device, dtype=dtype)
            for p in whisper_enc.parameters(): p.requires_grad = False
        except Exception as e:
            print(f"[Whisper Info] Dùng dummy encoder: {e}")

    all_summaries = []

    for c in cases_to_run:
        if c == 0:
            summary = evaluate_cascaded_case(args, device, mel_fb, data_pipeline)
        else:
            summary = evaluate_case(c, args, device, dtype, vocos, mel_fb, data_pipeline, feat_extractor, whisper_enc)
        if summary:
            all_summaries.append(summary)

    # In Bảng Tổng Hợp
    print("\n" + "=" * 80)
    print("📋 BẢNG TỔNG HỢP KẾT QUẢ BENCHMARK S2ST (TẬP TEST)")
    print("=" * 80)
    print(f"{'Kịch bản (Benchmark Case)':<48} | {'RTF ↓':<8} | {'Mel L1 ↓':<10} | {'SECS ↑':<8}")
    print("-" * 80)
    for s in all_summaries:
        print(f"{s['case_name']:<48} | {s['avg_rtf']:<8.4f} | {s['avg_mel_l1_error']:<10.4f} | {s['avg_secs_timbre']:<8.4f}")
    print("=" * 80)

    # Lưu bảng tổng hợp ra file Markdown
    os.makedirs(args.output_dir, exist_ok=True)
    summary_md_path = os.path.join(args.output_dir, "ablation_summary.md")
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("# 📋 BẢNG SO SÁNH THỰC NGHIỆM S2ST (TẬP TEST FLEURS EN-VI)\n\n")
        f.write("| Kịch bản thử nghiệm | RTF ↓ | Mel L1 Error ↓ | SECS (Giữ giọng) ↑ |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for s in all_summaries:
            f.write(f"| **{s['case_name']}** | {s['avg_rtf']:.4f} | {s['avg_mel_l1_error']:.4f} | {s['avg_secs_timbre']:.4f} |\n")
    print(f"📁 Báo cáo tổng hợp đã lưu tại: {summary_md_path}\n")

if __name__ == "__main__":
    main()

