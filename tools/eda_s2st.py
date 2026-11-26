#!/usr/bin/env python3
"""EDA dữ liệu FLEURS Việt -> Anh cho khóa luận S2ST (xuất ảnh PNG + bảng tóm tắt).

Chạy trên Molab:  bash run.sh eda            (đầy đủ, có căn chỉnh wav2vec2 ~150 bản ghi)
                  bash run.sh eda --quick    (chỉ thống kê từ metadata, không cần GPU / wav2vec2)
Kết quả: outputs/eda/*.png, outputs/eda/summary.md, outputs/eda/summary.json
Tự kiểm tra (không cần mạng):  python tools/eda_s2st.py --selftest

Các hình:
  fig01  thời lượng audio Việt / Anh theo split
  fig02  thời lượng Anh so với Việt cùng câu, tỉ lệ En/Vi
  fig03  số bản ghi (người đọc) mỗi câu
  fig04  độ dài câu Anh (ký tự, token NLLB)
  fig05  tốc độ nói và độ chênh thời lượng giữa các người đọc cùng câu
  fig06  khoảng lặng đầu / cuối câu (forced alignment wav2vec2)
  fig07  số frame Mel mỗi ký tự theo loại ký tự
  fig08  tốc độ nói khi bỏ khoảng lặng
  fig09  ví dụ: cùng một câu, hai người đọc, ranh giới từ theo căn chỉnh
"""
import argparse
import io
import json
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

MEL_FPS = 24000.0 / 256.0          # frame Mel Vocos / giây (khớp mel_lengths trong train_utils)
VOWELS = set("aeiou")


# ------------------------------------------------------------------------------
# Tiện ích
# ------------------------------------------------------------------------------
def unit_ids_from_text(text):
    """Chuỗi đơn vị ký tự như lúc train: [LẶNG] + ký tự (gộp dấu cách) + [LẶNG]."""
    from train_utils import EN_CTC_CHARS, normalize_en
    cidx = {c: i + 1 for i, c in enumerate(EN_CTC_CHARS)}
    ids = []
    for ch in normalize_en(text):
        if ch in cidx and not (ch == " " and (not ids or ids[-1] == cidx[" "])):
            ids.append(cidx[ch])
    return [0] + (ids or [cidx[" "]]) + [0]


def char_class(ch):
    if ch == " ":
        return "dấu cách"
    if ch in VOWELS:
        return "nguyên âm"
    if ch.isalpha():
        return "phụ âm"
    return "số / dấu '"


def decode_audio(audio):
    """dict audio của datasets (decode=False) -> numpy float32 16 kHz."""
    import soundfile as sf
    if isinstance(audio, dict):
        if audio.get("bytes"):
            arr, sr = sf.read(io.BytesIO(audio["bytes"]))
        elif audio.get("array") is not None:
            arr, sr = np.asarray(audio["array"]), audio.get("sampling_rate", 16000)
        else:
            arr, sr = sf.read(audio["path"])
    else:
        arr, sr = sf.read(audio)
    if arr.ndim > 1:
        arr = arr.mean(-1)
    if sr != 16000:
        n = int(round(len(arr) * 16000 / sr))
        arr = np.interp(np.linspace(0, len(arr) - 1, n), np.arange(len(arr)), arr)
    return arr.astype(np.float32)


def words(text):
    from train_utils import normalize_en
    return normalize_en(text).split()


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


# ------------------------------------------------------------------------------
# Thu thập dữ liệu
# ------------------------------------------------------------------------------
def collect(splits, max_per_split, keep_audio_n):
    """-> records (list dict) + audio_keep: {(id, idx): audio dict} cho bản ghi Anh train (để căn chỉnh)."""
    from datasets import load_dataset, Audio
    recs, keep = [], []
    for lang in ("vi_vn", "en_us"):
        for split in splits:
            print(f"📡 Đọc FLEURS {lang} / {split} (streaming)...", flush=True)
            ds = load_dataset("google/fleurs", lang, split=split, streaming=True)
            try:
                ds = ds.cast_column("audio", Audio(decode=False))
            except Exception:
                pass
            n = 0
            for item in ds:
                ns = item.get("num_samples")
                if not ns:
                    try:
                        ns = len(decode_audio(item["audio"]))
                    except Exception:
                        continue
                rec = {"lang": lang, "split": split, "id": item.get("id"), "dur": float(ns) / 16000.0,
                       "text": item.get("transcription", "") or "", "gender": item.get("gender")}
                recs.append(rec)
                if lang == "en_us" and split == "train" and len(keep) < keep_audio_n:
                    keep.append((rec, item["audio"]))
                n += 1
                if max_per_split and n >= max_per_split:
                    break
            print(f"   → {n} bản ghi", flush=True)
    return recs, keep


# ------------------------------------------------------------------------------
# Thống kê từ metadata (không cần GPU)
# ------------------------------------------------------------------------------
def meta_stats(recs, splits, nllb_tok=None):
    S = {}
    by = lambda lang, split: [r for r in recs if r["lang"] == lang and r["split"] == split]
    for lang in ("vi_vn", "en_us"):
        for split in splits:
            rs = by(lang, split)
            if not rs:
                continue
            d = np.array([r["dur"] for r in rs])
            ids = {r["id"] for r in rs}
            S[f"{lang}/{split}"] = {"bản ghi": len(rs), "câu (id)": len(ids), "giờ": float(d.sum() / 3600),
                                    "thời lượng tb (s)": float(d.mean()), "p5 (s)": pct(d, 5), "p95 (s)": pct(d, 95),
                                    "max (s)": float(d.max()),
                                    "% > 12 s (bị cắt khi train)": float((d > 12.0).mean() * 100),
                                    "bản ghi / câu (tb)": len(rs) / max(1, len(ids))}
    # cặp Vi-En theo id (trung bình các bản ghi)
    pairs = {}
    for split in splits:
        mv, me = {}, {}
        for r in by("vi_vn", split):
            mv.setdefault(r["id"], []).append(r["dur"])
        for r in by("en_us", split):
            me.setdefault(r["id"], []).append(r["dur"])
        common = sorted(set(mv) & set(me))
        pairs[split] = [(float(np.mean(mv[i])), float(np.mean(me[i]))) for i in common]
        if common:
            ratio = np.array([e / v for v, e in pairs[split]])
            S[f"cặp/{split}"] = {"câu có cả Vi và En": len(common), "tỉ lệ En/Vi tb": float(ratio.mean()),
                                 "p5": pct(ratio, 5), "p95": pct(ratio, 95)}
    # chồng lấn id giữa các split (phải = 0)
    for lang in ("vi_vn", "en_us"):
        sets = {sp: {r["id"] for r in by(lang, sp)} for sp in splits}
        for i, a in enumerate(splits):
            for b in splits[i + 1:]:
                S.setdefault("chồng lấn id", {})[f"{lang} {a}∩{b}"] = len(sets[a] & sets[b])
    # OOV: từ của val/test không có trong train (Anh)
    vocab = {w for r in by("en_us", "train") for w in words(r["text"])}
    for sp in splits:
        if sp == "train" or not vocab:
            continue
        toks = [w for r in by("en_us", sp) for w in words(r["text"])]
        if toks:
            S.setdefault("OOV tiếng Anh so với train", {})[sp] = {
                "% token": float(np.mean([w not in vocab for w in toks]) * 100),
                "% từ khác nhau": float(np.mean([w not in vocab for w in set(toks)]) * 100)}
    # tốc độ nói & chênh lệch giữa người đọc cùng câu (Anh, train)
    en_tr = by("en_us", "train")
    rate = np.array([len(" ".join(words(r["text"]))) / max(r["dur"], 1e-3) for r in en_tr]) if en_tr else np.array([])
    grp = {}
    for r in en_tr:
        grp.setdefault(r["id"], []).append(r["dur"])
    spread = np.array([max(v) / min(v) for v in grp.values() if len(v) >= 2 and min(v) > 0])
    S["tốc độ nói (Anh, train)"] = {"ký tự/giây tb (cả khoảng lặng)": float(rate.mean()) if len(rate) else float("nan"),
                                     "p5": pct(rate, 5), "p95": pct(rate, 95),
                                     "câu có ≥2 người đọc": int(len(spread)),
                                     "dài nhất / ngắn nhất cùng câu (trung vị)": pct(spread, 50),
                                     "p90": pct(spread, 90)}
    g = [r["gender"] for r in recs if r["gender"] is not None]
    if g:
        vals, cnt = np.unique(np.array(g, dtype=str), return_counts=True)
        S["giới tính (mã FLEURS)"] = {str(v): int(c) for v, c in zip(vals, cnt)}
    ntok = None
    if nllb_tok is not None and en_tr:
        ntok = np.array([len(nllb_tok(r["text"])["input_ids"]) for r in en_tr])
        nch = np.array([len(" ".join(words(r["text"]))) for r in en_tr])
        S["độ dài câu Anh (train)"] = {"ký tự tb": float(nch.mean()), "token NLLB tb": float(ntok.mean()),
                                       "ký tự / token": float(nch.sum() / max(1, ntok.sum()))}
    return S, pairs, rate, spread, ntok, grp


def plot_meta(out, recs, splits, pairs, rate, spread, ntok, grp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = []
    # fig01
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, lang, name in zip(axes, ("vi_vn", "en_us"), ("Tiếng Việt", "Tiếng Anh")):
        for sp in splits:
            d = [r["dur"] for r in recs if r["lang"] == lang and r["split"] == sp]
            if d:
                ax.hist(d, bins=40, range=(0, 30), alpha=0.5, density=True, label=f"{sp} ({len(d)})")
        ax.axvline(12, color="k", ls="--", lw=1)
        ax.text(12.3, ax.get_ylim()[1] * 0.9, "cắt 12 s", fontsize=8)
        ax.set_title(f"Thời lượng audio — {name}")
        ax.set_xlabel("giây")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("mật độ")
    files.append(_save(fig, out, "fig01_durations.png"))
    # fig02
    pr = pairs.get("train") or next((v for v in pairs.values() if v), [])
    if pr:
        v, e = np.array(pr).T
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].scatter(v, e, s=6, alpha=0.4)
        m = max(v.max(), e.max())
        axes[0].plot([0, m], [0, m], "k--", lw=1)
        axes[0].set_xlabel("thời lượng Việt (s)")
        axes[0].set_ylabel("thời lượng Anh (s)")
        axes[0].set_title("Cùng câu: Anh so với Việt (trung bình người đọc)")
        r_ = e / v
        axes[1].hist(r_, bins=50, range=(0.3, 2.0))
        axes[1].axvline(np.median(r_), color="r", lw=1)
        axes[1].set_title(f"Tỉ lệ En/Vi (trung vị {np.median(r_):.2f})")
        axes[1].set_xlabel("En / Vi")
        files.append(_save(fig, out, "fig02_pair_ratio.png"))
    # fig03
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for lang, name in (("vi_vn", "Việt"), ("en_us", "Anh")):
        cnt = {}
        for r in recs:
            if r["lang"] == lang and r["split"] == "train":
                cnt[r["id"]] = cnt.get(r["id"], 0) + 1
        if cnt:
            vals = np.array(list(cnt.values()))
            ks = np.arange(1, vals.max() + 1)
            ax.bar(ks + (-0.2 if lang == "vi_vn" else 0.2), [(vals == k).sum() for k in ks], width=0.4, label=name)
    ax.set_xlabel("số bản ghi (người đọc) của một câu")
    ax.set_ylabel("số câu")
    ax.set_title("Số bản ghi mỗi câu (train)")
    ax.legend()
    files.append(_save(fig, out, "fig03_recordings_per_sentence.png"))
    # fig04
    en_tr = [r for r in recs if r["lang"] == "en_us" and r["split"] == "train"]
    if en_tr:
        nch = [len(" ".join(words(r["text"]))) for r in en_tr]
        fig, axes = plt.subplots(1, 2 if ntok is not None else 1, figsize=(11 if ntok is not None else 6, 3.8))
        axes = np.atleast_1d(axes)
        axes[0].hist(nch, bins=40)
        axes[0].set_title(f"Độ dài câu Anh (ký tự), tb {np.mean(nch):.0f}")
        axes[0].set_xlabel("số ký tự (= số đơn vị của Bridge, chưa tính 2 đơn vị lặng)")
        if ntok is not None:
            axes[1].hist(ntok, bins=40, color="tab:orange")
            axes[1].set_title(f"Độ dài câu Anh (token NLLB), tb {np.mean(ntok):.0f}")
            axes[1].set_xlabel("số token")
        files.append(_save(fig, out, "fig04_text_length.png"))
    # fig05
    if len(rate):
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        axes[0].hist(rate, bins=40)
        axes[0].set_title(f"Tốc độ nói (ký tự/giây, tính cả lặng), tb {rate.mean():.1f}")
        axes[0].set_xlabel("ký tự / giây")
        if len(spread):
            axes[1].hist(spread, bins=40, range=(1, 3), color="tab:green")
            axes[1].axvline(np.median(spread), color="r", lw=1)
            axes[1].set_title(f"Cùng câu: bản dài nhất / ngắn nhất (trung vị {np.median(spread):.2f}×)")
            axes[1].set_xlabel("tỉ lệ thời lượng giữa các người đọc")
        files.append(_save(fig, out, "fig05_speaking_rate.png"))
    return files


def _save(fig, out, name):
    p = os.path.join(out, name)
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"   🖼️  {p}", flush=True)
    return p


# ------------------------------------------------------------------------------
# Căn chỉnh wav2vec2 (khoảng lặng, frame/ký tự)
# ------------------------------------------------------------------------------
def align_stats(keep, aligner):
    from train_utils import EN_CTC_CHARS, mel_lengths
    rows, per_class, ex = [], {}, {}
    for i, (rec, audio) in enumerate(keep):
        try:
            w = decode_audio(audio)
        except Exception as e:
            print(f"   [bỏ qua {i}] {e}")
            continue
        import torch
        T = mel_lengths([len(w)])[0]
        ids = unit_ids_from_text(rec["text"])
        d = np.asarray(aligner.durations(torch.from_numpy(w), ids, T), dtype=np.float64)
        core = d[1:-1]
        speech_sec = core.sum() / MEL_FPS
        rows.append({"lead": d[0] / MEL_FPS, "trail": d[-1] / MEL_FPS, "share": (d[0] + d[-1]) / max(1.0, d.sum()),
                     "rate_all": len(core) / max(1e-3, d.sum() / MEL_FPS), "rate_speech": len(core) / max(1e-3, speech_sec),
                     "one_frame": float((core <= 1).mean()), "core": core})
        for k, u in enumerate(ids[1:-1]):
            per_class.setdefault(char_class(EN_CTC_CHARS[u - 1]), []).append(core[k])
        ex.setdefault(rec["id"], []).append((rec, w, ids, d))
        if (i + 1) % 25 == 0:
            print(f"   căn chỉnh {i + 1}/{len(keep)}", flush=True)
    return rows, per_class, ex


def plot_align(out, rows, per_class, ex):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from train_utils import EN_CTC_CHARS
    files = []
    lead = np.array([r["lead"] for r in rows])
    trail = np.array([r["trail"] for r in rows])
    share = np.array([r["share"] for r in rows]) * 100
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].hist(lead, bins=30, alpha=0.6, label=f"đầu câu (tb {lead.mean():.2f} s)")
    axes[0].hist(trail, bins=30, alpha=0.6, label=f"cuối câu (tb {trail.mean():.2f} s)")
    axes[0].set_xlabel("giây")
    axes[0].set_title("Khoảng lặng đầu / cuối câu (wav2vec2)")
    axes[0].legend(fontsize=8)
    axes[1].hist(share, bins=30, color="tab:purple")
    axes[1].axvline(np.median(share), color="r", lw=1)
    axes[1].set_title(f"% thời lượng là lặng đầu+cuối (trung vị {np.median(share):.0f}%)")
    axes[1].set_xlabel("%")
    files.append(_save(fig, out, "fig06_silence.png"))

    allc = np.concatenate([r["core"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].hist(allc, bins=np.arange(0, 41), color="tab:blue")
    axes[0].set_yscale("log")
    axes[0].set_title(f"Frame Mel mỗi ký tự (tb {allc.mean():.1f}; ≤1 frame {np.mean(allc <= 1) * 100:.1f}%)")
    axes[0].set_xlabel("frame (1 frame ≈ 10,7 ms)")
    names = [k for k in ("nguyên âm", "phụ âm", "dấu cách", "số / dấu '") if k in per_class]
    axes[1].bar(names, [np.mean(per_class[k]) for k in names],
                yerr=[np.std(per_class[k]) for k in names], capsize=4, color="tab:cyan")
    axes[1].set_title("Thời lượng trung bình theo loại ký tự")
    axes[1].set_ylabel("frame")
    files.append(_save(fig, out, "fig07_frames_per_char.png"))

    ra = np.array([r["rate_all"] for r in rows])
    rs = np.array([r["rate_speech"] for r in rows])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist(ra, bins=30, alpha=0.6, label=f"tính cả lặng (tb {ra.mean():.1f})")
    ax.hist(rs, bins=30, alpha=0.6, label=f"bỏ lặng đầu/cuối (tb {rs.mean():.1f}, CV {rs.std() / rs.mean():.2f})")
    ax.set_xlabel("ký tự / giây")
    ax.set_title("Tốc độ nói theo bản ghi")
    ax.legend(fontsize=8)
    files.append(_save(fig, out, "fig08_rate_no_silence.png"))

    two = next((v for v in ex.values() if len(v) >= 2), None) or (next(iter(ex.values()), None))
    if two:
        two = two[:2]
        fig, axes = plt.subplots(len(two), 1, figsize=(12, 3.0 * len(two)), squeeze=False)
        tmax = max(len(w) for _, w, _, _ in two) / 16000.0
        for ax, (rec, w, ids, d) in zip(axes[:, 0], two):
            spec = _logspec(w)
            ax.imshow(spec, origin="lower", aspect="auto", extent=[0, len(w) / 16000.0, 0, 8], cmap="magma")
            b = np.concatenate([[0], np.cumsum(d)]) / MEL_FPS
            ax.axvspan(0, b[1], color="white", alpha=0.25)
            ax.axvspan(b[-2], b[-1], color="white", alpha=0.25)
            word_start = True
            for k, u in enumerate(ids):
                ch = EN_CTC_CHARS[u - 1] if u > 0 else ""
                if u > 0 and ch == " ":
                    ax.axvline(b[k], color="cyan", lw=0.8)
                    word_start = True
                elif u > 0 and word_start:
                    j = k
                    wd = ""
                    while j < len(ids) - 1 and ids[j] > 0 and EN_CTC_CHARS[ids[j] - 1] != " ":
                        wd += EN_CTC_CHARS[ids[j] - 1]
                        j += 1
                    ax.text(b[k], 7.3, wd, color="w", fontsize=7)
                    word_start = False
            ax.set_xlim(0, tmax)
            ax.set_ylabel("kHz")
            ax.set_title(f"\"{rec['text'][:80]}\" — {len(w) / 16000:.2f} s, lặng {d[0] / MEL_FPS:.2f} s + {d[-1] / MEL_FPS:.2f} s",
                         fontsize=9)
        axes[-1, 0].set_xlabel("giây (vùng sáng = đơn vị lặng; vạch xanh = ranh giới từ theo wav2vec2)")
        files.append(_save(fig, out, "fig09_alignment_example.png"))
    return files


def _logspec(w, n_fft=512, hop=160):
    if len(w) < n_fft:
        w = np.pad(w, (0, n_fft - len(w)))
    win = np.hanning(n_fft)
    frames = np.lib.stride_tricks.sliding_window_view(w, n_fft)[::hop] * win
    mag = np.abs(np.fft.rfft(frames, axis=-1)).T
    return np.log(mag + 1e-5)


# ------------------------------------------------------------------------------
def write_summary(out, S, A):
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": S, "align": A}, f, ensure_ascii=False, indent=1)
    L = ["# EDA FLEURS Việt → Anh", ""]
    for k, v in {**S, **({"căn chỉnh wav2vec2 (Anh, train)": A} if A else {})}.items():
        L.append(f"## {k}")
        L.append("")
        for kk, vv in v.items():
            if isinstance(vv, dict):
                vv = ", ".join(f"{a}: {b:.2f}" if isinstance(b, float) else f"{a}: {b}" for a, b in vv.items())
            elif isinstance(vv, float):
                vv = f"{vv:.3f}"
            L.append(f"- {kk}: {vv}")
        L.append("")
    p = os.path.join(out, "summary.md")
    open(p, "w", encoding="utf-8").write("\n".join(L))
    print(f"   📝 {p}")


def align_summary(rows, per_class):
    if not rows:
        return {}
    allc = np.concatenate([r["core"] for r in rows])
    rs = np.array([r["rate_speech"] for r in rows])
    return {"số bản ghi": len(rows),
            "lặng đầu tb (s)": float(np.mean([r["lead"] for r in rows])),
            "lặng cuối tb (s)": float(np.mean([r["trail"] for r in rows])),
            "% thời lượng là lặng đầu+cuối (trung vị)": float(np.median([r["share"] for r in rows]) * 100),
            "frame/ký tự tb": float(allc.mean()), "% ký tự ≤ 1 frame": float(np.mean(allc <= 1) * 100),
            "ký tự/giây khi bỏ lặng (tb)": float(rs.mean()), "hệ số biến thiên tốc độ": float(rs.std() / rs.mean()),
            **{f"frame tb — {k}": float(np.mean(v)) for k, v in per_class.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/eda")
    ap.add_argument("--splits", default="train,validation,test")
    ap.add_argument("--max-per-split", type=int, default=0, help="0 = toàn bộ")
    ap.add_argument("--align-n", type=int, default=150, help="số bản ghi Anh train để căn chỉnh (0 = bỏ qua)")
    ap.add_argument("--quick", action="store_true", help="bỏ phần căn chỉnh wav2vec2")
    ap.add_argument("--selftest", action="store_true", help="dữ liệu giả, không cần mạng")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    splits = [s for s in a.splits.split(",") if s]
    align_n = 0 if a.quick else a.align_n

    if a.selftest:
        recs, keep, aligner, tok = _fake_data(splits)
    else:
        recs, keep = collect(splits, a.max_per_split, align_n)
        tok = None
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="eng_Latn")
        except Exception as e:
            print(f"ℹ️ Không nạp được tokenizer NLLB ({e}) — bỏ thống kê token.")
        aligner = None
        if align_n:
            import torch
            from train_utils import CTCCharAligner
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"⏳ Nạp wav2vec2-base-960h ({dev})...", flush=True)
            aligner = CTCCharAligner("facebook/wav2vec2-base-960h", device=dev)

    print("📊 Thống kê metadata...", flush=True)
    S, pairs, rate, spread, ntok, grp = meta_stats(recs, splits, tok)
    files = plot_meta(a.out, recs, splits, pairs, rate, spread, ntok, grp)
    A = {}
    if aligner is not None and keep:
        # ưu tiên giữ đủ các bản ghi của cùng một câu để có hình fig09
        print(f"📐 Căn chỉnh {len(keep)} bản ghi Anh (train)...", flush=True)
        rows, per_class, ex = align_stats(keep, aligner)
        if rows:
            files += plot_align(a.out, rows, per_class, ex)
            A = align_summary(rows, per_class)
    write_summary(a.out, S, A)
    print(f"✓ Xong: {len(files)} hình trong {a.out}/")


def _fake_data(splits):
    rng = np.random.default_rng(0)
    sents = ["the quick brown fox jumps over the lazy dog", "satellite phones are not a replacement",
             "some animals like elephants come close to cars", "it was 1945 when he joined them"]
    recs, keep = [], []
    for lang in ("vi_vn", "en_us"):
        for sp in splits:
            for i in range(30):
                for _ in range(1 + i % 3):
                    recs.append({"lang": lang, "split": sp, "id": f"{sp}{i}", "dur": float(rng.uniform(3, 15)),
                                 "text": sents[i % 4], "gender": int(rng.integers(0, 2))})
    for r in [r for r in recs if r["lang"] == "en_us" and r["split"] == "train"][:12]:
        n = int(r["dur"] * 16000)
        keep.append((r, {"array": (0.1 * rng.standard_normal(n)).astype(np.float32), "sampling_rate": 16000}))

    class FakeAligner:
        def durations(self, w, ids, T):
            d = rng.integers(2, 12, len(ids)).astype(float)
            d[0], d[-1] = rng.integers(10, 60), rng.integers(10, 60)
            d = np.floor(d * T / d.sum())
            d[-1] += T - d.sum()
            return d

    class FakeTok:
        def __call__(self, t):
            return {"input_ids": list(range(len(t.split()) + 2))}
    return recs, keep, FakeAligner(), FakeTok()


if __name__ == "__main__":
    main()
