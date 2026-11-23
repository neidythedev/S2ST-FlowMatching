#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CHUẨN BỊ DỮ LIỆU PhoST (vinai/PhoST, định dạng MuST-C) CHO S2ST VI -> EN

PhoST: mỗi bài nói TED = 1 file wav dài (audio_data/<split>/wav/<id>.wav) +
       text_data/<split>/<id>/<id>.yaml  (mỗi dòng: {duration, offset, wav} của 1 câu)
       text_data/<split>/<id>/<id>.en    (câu tiếng Anh, dòng i ↔ đoạn i trong yaml)
       text_data/<split>/<id>/<id>.vi    (bản dịch tiếng Việt, dòng i ↔ đoạn i)

Script này:
  1. Tải text_data của split (nhỏ, có thể bị HF giới hạn tốc độ -> tự chờ)
  2. Làm sạch cặp câu (bỏ tên người nói, (Laughter), gạch nối bị tách), lọc độ dài
  3. Chọn ngẫu nhiên các bài nói tới khi đủ --hours giờ (seed cố định)
  4. Chỉ tải wav của các bài được chọn, cắt từng câu -> 16kHz mono PCM16, xóa wav gốc cho đỡ tốn đĩa
  5. Ghi manifest.jsonl (ghi tiếp từng bài -> chạy lại sẽ bỏ qua bài đã xong)

⚠️ Điều khoản PhoST: chỉ dùng nghiên cứu/giáo dục, KHÔNG phân phối lại. Thư mục đầu ra nằm trong
   .gitignore; chỉ lưu trên Google Drive riêng. KHÔNG dùng 'bash run.sh upload' cho dữ liệu này.

Ví dụ:
  python tools/prepare_phost.py --split train --hours 50 --out-dir data/phost --drive gdrive:datasets/phost/
  python tools/prepare_phost.py --split dev --hours 999 --out-dir data/phost
"""
import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vi_textnorm import clean_ted_pair, normalize_vi_for_tts  # noqa: E402

REPO_ID = "vinai/PhoST"
TARGET_SR = 16000


def parse_yaml_segments(path):
    """Đọc yaml kiểu MuST-C: '- {duration: 2.59, offset: 12.65, rW: 11, uW: 0, wav: 1191.wav}'."""
    segs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("-"):
                continue
            kv = dict(re.findall(r"(\w+)\s*:\s*([^,}]+)", line))
            if "duration" in kv and "offset" in kv:
                segs.append({"offset": float(kv["offset"]), "duration": float(kv["duration"]),
                             "wav": kv.get("wav", "").strip()})
    return segs


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def load_talk(text_root, split, tid, min_sec, max_sec, stats):
    """-> list các câu hợp lệ của 1 bài nói (chưa cắt audio)."""
    d = os.path.join(text_root, "text_data", split, tid)
    try:
        segs = parse_yaml_segments(os.path.join(d, f"{tid}.yaml"))
        en = read_lines(os.path.join(d, f"{tid}.en"))
        vi = read_lines(os.path.join(d, f"{tid}.vi"))
    except FileNotFoundError:
        stats["talk_missing_text"] += 1
        return []
    if not (len(segs) == len(en) == len(vi)):
        stats["talk_misaligned"] += 1      # số dòng không khớp -> bỏ cả bài cho an toàn
        return []
    out = []
    for i, (sg, e, v) in enumerate(zip(segs, en, vi)):
        if not (min_sec <= sg["duration"] <= max_sec):
            stats["drop_duration"] += 1
            continue
        pair = clean_ted_pair(e, v)
        if pair is None:
            stats["drop_text"] += 1
            continue
        e2, v2 = pair
        ratio = len(v2.split()) / max(1, len(e2.split()))
        if not (0.4 <= ratio <= 3.0):          # bản dịch lệch quá xa -> nghi căn chỉnh sai
            stats["drop_ratio"] += 1
            continue
        out.append({"idx": i, "offset": sg["offset"], "duration": sg["duration"], "en_text": e2, "vi_text": v2})
    return out


def fft_resample(x, sr_in, sr_out):
    import numpy as np
    if sr_in == sr_out:
        return x
    n = len(x)
    m = int(round(n * sr_out / sr_in))
    X = np.fft.rfft(x)
    k = m // 2 + 1
    X = X[:k] if len(X) >= k else np.pad(X, (0, k - len(X)))
    return np.fft.irfft(X, n=m) * (m / n)


def cut_talk(wav_path, sents, out_wav_dir, tid, split):
    """Cắt các câu của 1 bài nói -> file wav 16kHz. Trả về các dòng manifest."""
    import numpy as np
    import soundfile as sf
    info = sf.info(wav_path)
    sr = info.samplerate
    rows = []
    os.makedirs(out_wav_dir, exist_ok=True)
    for s in sents:
        start = int(round(s["offset"] * sr))
        stop = min(info.frames, int(round((s["offset"] + s["duration"]) * sr)))
        if stop - start < int(0.5 * sr):
            continue
        audio, _ = sf.read(wav_path, start=start, stop=stop, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        audio = fft_resample(audio, sr, TARGET_SR).astype("float32")
        peak = float(np.abs(audio).max()) if audio.size else 0.0
        if peak < 1e-4:
            continue                                   # câm
        if peak > 1.0:
            audio = audio / peak
        rel = os.path.join("wav", split, tid, f"{tid}_{s['idx']:04d}.wav")   # đường dẫn tương đối so với --out-dir
        sf.write(os.path.join(out_wav_dir, f"{tid}_{s['idx']:04d}.wav"), audio, TARGET_SR, subtype="PCM_16")
        rows.append({
            "id": f"phost_{split}_{tid}_{s['idx']:04d}", "talk_id": tid, "split": split,
            "en_wav": rel.replace(os.sep, "/"), "en_dur": round(len(audio) / TARGET_SR, 3),
            "en_text": s["en_text"], "vi_text": s["vi_text"], "vi_tts": normalize_vi_for_tts(s["vi_text"]),
        })
    return rows


def download_text(split, text_root):
    from huggingface_hub import snapshot_download
    print(f"📥 Tải text_data/{split} (HF có thể giới hạn tốc độ -> tự chờ rồi thử lại)...")
    snapshot_download(REPO_ID, repo_type="dataset", allow_patterns=[f"text_data/{split}/*"],
                      local_dir=text_root, max_workers=4)


def download_talk_wav(split, tid, tmp_dir, local_root=None):
    if local_root:
        p = os.path.join(local_root, "audio_data", split, "wav", f"{tid}.wav")
        return p if os.path.exists(p) else None
    from huggingface_hub import hf_hub_download
    for attempt in range(5):
        try:
            return hf_hub_download(REPO_ID, repo_type="dataset", filename=f"audio_data/{split}/wav/{tid}.wav",
                                   local_dir=tmp_dir)
        except Exception as e:
            wait = 30 * (attempt + 1)
            print(f"   ⚠️ Lỗi tải {tid}.wav ({e}) -> thử lại sau {wait}s")
            time.sleep(wait)
    return None


def main():
    ap = argparse.ArgumentParser(description="Chuẩn bị dữ liệu PhoST (cắt câu + manifest) cho S2ST Vi->En")
    ap.add_argument("--split", default="train", choices=["train", "dev", "test"])
    ap.add_argument("--hours", type=float, default=50.0, help="Số giờ audio tiếng Anh cần lấy (sau lọc)")
    ap.add_argument("--out-dir", default="data/phost")
    ap.add_argument("--text-root", default="phost_probe", help="Nơi chứa/tải text_data (tái dùng nếu đã tải)")
    ap.add_argument("--local-root", default=None, help="Thư mục PhoST đã tải sẵn (có audio_data/ text_data/) — bỏ qua HF")
    ap.add_argument("--min-sec", type=float, default=1.0)
    ap.add_argument("--max-sec", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-talk-wav", action="store_true", help="Giữ lại wav gốc của bài nói (tốn đĩa)")
    ap.add_argument("--drive", default=None, help="VD gdrive:datasets/phost/ — đẩy kết quả lên Drive sau mỗi 20 bài + khi xong")
    args = ap.parse_args()

    text_root = args.local_root or args.text_root
    split_dir = os.path.join(text_root, "text_data", args.split)
    if not args.local_root and not os.path.isdir(split_dir):
        download_text(args.split, text_root)
    elif not args.local_root:
        n_have = len(os.listdir(split_dir))
        print(f"ℹ️ Đã có {n_have} bài trong {split_dir} (nếu thiếu, xóa thư mục để tải lại)")
    talks = sorted(os.listdir(split_dir))
    print(f"📚 {len(talks)} bài nói trong split '{args.split}'")

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = os.path.join(args.out_dir, f"manifest_{args.split}.jsonl")
    done = set()
    have_sec = 0.0
    if os.path.exists(manifest):
        with open(manifest, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done.add(r["talk_id"])
                have_sec += r["en_dur"]
        print(f"⚡ Tiếp tục: đã có {len(done)} bài, {have_sec/3600:.2f} giờ trong {manifest}")

    stats = {k: 0 for k in ["talk_missing_text", "talk_misaligned", "drop_duration", "drop_text", "drop_ratio",
                            "talk_no_audio"]}
    rng = random.Random(args.seed)
    order = talks[:]
    rng.shuffle(order)                      # thứ tự chọn bài cố định theo seed -> tái lập được
    target = args.hours * 3600.0
    tmp_dir = os.path.join(args.out_dir, "_tmp_talk_wav")
    n_new, t0 = 0, time.time()
    with open(manifest, "a", encoding="utf-8") as mf:
        for tid in order:
            if have_sec >= target:
                break
            if tid in done:
                continue
            sents = load_talk(text_root, args.split, tid, args.min_sec, args.max_sec, stats)
            if not sents:
                continue
            wav = download_talk_wav(args.split, tid, tmp_dir, args.local_root)
            if not wav:
                stats["talk_no_audio"] += 1
                continue
            rows = cut_talk(wav, sents, os.path.join(args.out_dir, "wav", args.split, tid), tid, args.split)
            for r in rows:
                mf.write(json.dumps(r, ensure_ascii=False) + "\n")
            mf.flush()
            have_sec += sum(r["en_dur"] for r in rows)
            n_new += 1
            if not args.local_root and not args.keep_talk_wav:
                try:
                    os.remove(wav)
                except OSError:
                    pass
            print(f"   ✓ [{n_new}] bài {tid}: {len(rows)} câu | tổng {have_sec/3600:.2f}/{args.hours} giờ "
                  f"| {(time.time()-t0)/max(1,n_new):.1f}s/bài")
            if args.drive and n_new % 20 == 0:
                subprocess.call(["rclone", "copy", args.out_dir, args.drive, "--exclude", "_tmp_talk_wav/**"])

    if os.path.isdir(tmp_dir) and not args.keep_talk_wav:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    n_rows = sum(1 for _ in open(manifest, encoding="utf-8"))
    print("\n" + "=" * 70)
    print(f"✅ XONG: {manifest} | {n_rows} câu | {have_sec/3600:.2f} giờ audio tiếng Anh (16kHz)")
    print(f"   Bỏ: thời lượng {stats['drop_duration']} | text {stats['drop_text']} | tỉ lệ Vi/En {stats['drop_ratio']} "
          f"| bài lệch dòng {stats['talk_misaligned']} | bài thiếu audio {stats['talk_no_audio']}")
    print("=" * 70)
    if args.drive:
        print(f"☁️ Đẩy lên {args.drive} ...")
        subprocess.call(["rclone", "copy", args.out_dir, args.drive, "--exclude", "_tmp_talk_wav/**", "-P"])


if __name__ == "__main__":
    main()
