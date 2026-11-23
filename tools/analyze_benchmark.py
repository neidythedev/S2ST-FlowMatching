#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHÂN TÍCH KẾT QUẢ BENCHMARK S2ST (gom từ nhiều notebook chạy độc lập)

Đọc mọi */metrics.json trong các thư mục truyền vào (VD: outputs/benchmark_all tải từ Drive,
outputs/benchmark trên máy), rồi xuất:
  - Bảng tổng hợp từng biến thể: S2TT-BLEU, ASR-BLEU (corpus SacreBLEU + khoảng tin cậy 95%
    bootstrap), điểm mất do TTS, SECS, RTF, số câu không dịch được.
  - So sánh ghép cặp (paired bootstrap trên CÙNG các câu):
      1. Model nhỏ vs lớn (whisper-small vs whisper-large-v3, PhoWhisper-small vs PhoWhisper-large)
      2. 2-step tốt nhất vs 3-step tốt nhất (mỗi engine TTS)
      3. XTTS-v2 vs F5-TTS (cùng biến thể S2TT)
  - File: analysis_report.md, analysis_summary.csv, analysis.json

Cách dùng:
  python tools/analyze_benchmark.py outputs/benchmark_all outputs/benchmark --out outputs/benchmark_all
"""
import argparse
import csv
import glob
import json
import math
import os
import random
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ------------------------------------------------------------------------------
# Chuẩn hóa văn bản — PHẢI GIỐNG normalize_en_text trong eval_benchmark.py
# ------------------------------------------------------------------------------
def normalize_en_text(text):
    t = str(text or "").lower()
    t = t.replace("’", "'")
    t = re.sub(r"[^\w\s']", " ", t)
    t = re.sub(r"(?<!\w)'|'(?!\w)", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# ------------------------------------------------------------------------------
# Corpus BLEU nhanh qua thống kê đủ (sufficient statistics) từng câu.
# Ưu tiên sacrebleu (khớp số liệu của eval_benchmark.py); fallback cài đặt BLEU-4 thuần Python.
# ------------------------------------------------------------------------------
class BleuScorer:
    def __init__(self):
        self.backend = "python"
        self._bleu = None
        try:
            from sacrebleu.metrics import BLEU
            self._bleu = BLEU()
            # Kiểm tra API nội bộ dùng cho bootstrap
            st = self._bleu._extract_corpus_statistics(["a b c d"], [["a b c d"]])
            self._bleu._compute_score_from_stats(st[0])
            self.backend = "sacrebleu"
        except Exception:
            self._bleu = None

    def sentence_stats(self, hyps, refs):
        """Trả về list thống kê của từng câu (list số nguyên cùng độ dài)."""
        hyps = [normalize_en_text(h) for h in hyps]
        refs = [normalize_en_text(r) for r in refs]
        if self._bleu is not None:
            return [list(map(int, s)) for s in self._bleu._extract_corpus_statistics(hyps, [refs])]
        stats = []
        for h, r in zip(hyps, refs):
            ht, rt = h.split(), r.split()
            row = [len(ht), len(rt)]
            for n in range(1, 5):
                hc, rc = {}, {}
                for i in range(len(ht) - n + 1):
                    g = tuple(ht[i:i + n]); hc[g] = hc.get(g, 0) + 1
                for i in range(len(rt) - n + 1):
                    g = tuple(rt[i:i + n]); rc[g] = rc.get(g, 0) + 1
                match = sum(min(c, rc.get(g, 0)) for g, c in hc.items())
                row += [match, max(0, len(ht) - n + 1)]
            stats.append(row)
        return stats

    def score(self, stats_rows):
        if not stats_rows:
            return float("nan")
        total = [sum(col) for col in zip(*stats_rows)]
        if self._bleu is not None:
            return float(self._bleu._compute_score_from_stats(total).score)
        hyp_len, ref_len = total[0], total[1]
        if hyp_len == 0:
            return 0.0
        log_p, smooth = 0.0, 1.0
        for n in range(4):
            m, t = total[2 + 2 * n], total[3 + 2 * n]
            if t == 0:
                return 0.0
            if m == 0:
                smooth *= 2.0
                p = 1.0 / (smooth * t)
            else:
                p = m / t
            log_p += math.log(p) / 4.0
        bp = 1.0 if hyp_len > ref_len else math.exp(1.0 - ref_len / max(1, hyp_len))
        return 100.0 * bp * math.exp(log_p)


def bootstrap_ci(values_fn, n, n_boot, seed=1234):
    """CI 95% bằng bootstrap: values_fn(indices) -> giá trị thống kê."""
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        v = values_fn(idx)
        if v == v:
            vals.append(v)
    if not vals:
        return float("nan"), float("nan"), []
    vals.sort()
    lo = vals[int(0.025 * (len(vals) - 1))]
    hi = vals[int(0.975 * (len(vals) - 1))]
    return lo, hi, vals


def mean_std(xs):
    xs = [x for x in xs if x is not None and x == x]
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return m, sd


def fmt(x, nd=2):
    return "nan" if (x is None or x != x) else f"{x:.{nd}f}"


# ------------------------------------------------------------------------------
# Nạp dữ liệu
# ------------------------------------------------------------------------------
def load_runs(dirs):
    """Đọc mọi metrics.json; khử trùng lặp theo case_tag (ưu tiên nhiều mẫu hơn, rồi file mới hơn)."""
    runs = {}
    for d in dirs:
        for path in glob.glob(os.path.join(d, "**", "metrics.json"), recursive=True):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    m = json.load(f)
            except Exception as e:
                print(f"⚠️ Bỏ qua {path}: {e}")
                continue
            if not m.get("samples"):
                continue
            tag = m.get("case_tag") or os.path.basename(os.path.dirname(path))
            # Thư mục 'case_6' / 'case_7' chỉ là bản sao đại diện cho gallery -> bỏ để tránh đếm trùng
            if re.fullmatch(r"case_[67]", os.path.basename(os.path.dirname(path))):
                continue
            m["_path"] = path
            m["_mtime"] = os.path.getmtime(path)
            old = runs.get(tag)
            if old is None or (len(m["samples"]), m["_mtime"]) > (len(old["samples"]), old["_mtime"]):
                runs[tag] = m
    return runs


def describe(m):
    """Trích engine / mode / model của 1 lần chạy (kể cả Case 0, 4, 5 không có các trường này)."""
    engine = m.get("tts_engine")
    mode = m.get("s2tt_mode")
    model = m.get("s2tt_model")
    cid = m.get("case_id")
    if engine is None:
        engine = {0: "mms-tts", 5: "seamless"}.get(cid, "proposed")
        mode = {0: "3step", 5: "direct"}.get(cid, "direct")
        model = {0: "vinai/PhoWhisper-small"}.get(cid, m.get("case_name", ""))
    short = str(model).split("/")[-1]
    return engine, mode, short


def sample_hyp_s2tt(s):
    return s.get("translated_en", s.get("mt_translated_en"))


# ------------------------------------------------------------------------------
# Phân tích
# ------------------------------------------------------------------------------
def analyze(runs, n_boot):
    scorer = BleuScorer()
    print(f"ℹ️ BLEU backend: {scorer.backend} | bootstrap: {n_boot} lần")

    # Kiểm tra tất cả biến thể dùng CÙNG các câu test (để so sánh ghép cặp là hợp lệ)
    ref_by_id = {}
    mismatches = []
    for tag, m in runs.items():
        for s in m["samples"]:
            sid = s.get("sample_id")
            ref = normalize_en_text(s.get("ref_en_text", ""))
            if sid in ref_by_id and ref_by_id[sid][0] != ref:
                mismatches.append((tag, sid, ref_by_id[sid][1]))
            ref_by_id.setdefault(sid, (ref, tag))

    rows = []
    cache = {}
    for tag, m in sorted(runs.items()):
        engine, mode, short = describe(m)
        samples = sorted(m["samples"], key=lambda s: s.get("sample_id", 0))
        refs = [s.get("ref_en_text", "") for s in samples]
        asr_h = [s.get("asr_transcribed_en", "") or "" for s in samples]
        s2tt_h = [sample_hyp_s2tt(s) for s in samples]
        has_s2tt = all(h is not None for h in s2tt_h)
        n = len(samples)

        asr_stats = scorer.sentence_stats(asr_h, refs)
        s2tt_stats = scorer.sentence_stats([h or "" for h in s2tt_h], refs) if has_s2tt else None
        asr_bleu = scorer.score(asr_stats)
        s2tt_bleu = scorer.score(s2tt_stats) if has_s2tt else float("nan")
        asr_lo, asr_hi, _ = bootstrap_ci(lambda idx: scorer.score([asr_stats[i] for i in idx]), n, n_boot)
        if has_s2tt:
            s_lo, s_hi, _ = bootstrap_ci(lambda idx: scorer.score([s2tt_stats[i] for i in idx]), n, n_boot)
        else:
            s_lo = s_hi = float("nan")
        secs_m, secs_sd = mean_std([s.get("secs_timbre") for s in samples])
        rtf_m, rtf_sd = mean_std([s.get("rtf") for s in samples])
        dur_m, _ = mean_std([s.get("gen_duration") for s in samples])
        n_vi = sum(1 for s in samples if s.get("s2tt_not_translated"))

        row = {
            "tag": tag, "engine": engine, "mode": mode, "model": short, "n": n,
            "s2tt_bleu": s2tt_bleu, "s2tt_lo": s_lo, "s2tt_hi": s_hi,
            "asr_bleu": asr_bleu, "asr_lo": asr_lo, "asr_hi": asr_hi,
            "tts_loss": (s2tt_bleu - asr_bleu) if has_s2tt else float("nan"),
            "secs_mean": secs_m, "secs_std": secs_sd,
            "rtf_mean": rtf_m, "rtf_std": rtf_sd,
            "dur_mean": dur_m, "not_translated": n_vi,
        }
        rows.append(row)
        cache[tag] = {
            "ids": [s.get("sample_id") for s in samples],
            "asr_stats": asr_stats, "s2tt_stats": s2tt_stats,
            "secs": [s.get("secs_timbre") for s in samples],
            "rtf": [s.get("rtf") for s in samples],
        }

    # ---------------- So sánh ghép cặp ----------------
    def paired(tag_a, tag_b, label):
        A, B = cache[tag_a], cache[tag_b]
        common = [sid for sid in A["ids"] if sid in set(B["ids"])]
        if len(common) < 2:
            return None
        ia = [A["ids"].index(sid) for sid in common]
        ib = [B["ids"].index(sid) for sid in common]
        out = {"label": label, "a": tag_a, "b": tag_b, "n": len(common)}

        def bleu_delta(key):
            if A[key] is None or B[key] is None:
                return None
            sa = [A[key][i] for i in ia]
            sb = [B[key][i] for i in ib]
            d = scorer.score(sb) - scorer.score(sa)
            lo, hi, vals = bootstrap_ci(lambda idx: scorer.score([sb[i] for i in idx]) - scorer.score([sa[i] for i in idx]),
                                        len(common), n_boot)
            p_le0 = sum(1 for v in vals if v <= 0) / max(1, len(vals))
            return {"delta": d, "lo": lo, "hi": hi, "p_b_not_better": p_le0}

        def mean_delta(key):
            da = [A[key][i] for i in ia]
            db = [B[key][i] for i in ib]
            diffs = [y - x for x, y in zip(da, db) if x is not None and y is not None and x == x and y == y]
            if len(diffs) < 2:
                return None
            d = sum(diffs) / len(diffs)
            lo, hi, _ = bootstrap_ci(lambda idx: sum(diffs[i] for i in idx) / len(idx), len(diffs), n_boot)
            return {"delta": d, "lo": lo, "hi": hi}

        out["asr_bleu"] = bleu_delta("asr_stats")
        out["s2tt_bleu"] = bleu_delta("s2tt_stats")
        out["secs"] = mean_delta("secs")
        out["rtf"] = mean_delta("rtf")
        return out

    by_key = {(r["engine"], r["mode"], r["model"]): r["tag"] for r in rows}
    comparisons = []
    size_pairs = [("whisper-small", "whisper-large-v3"), ("PhoWhisper-small", "PhoWhisper-large")]
    for eng in sorted({r["engine"] for r in rows}):
        for mode in ["2step", "3step"]:
            for small, large in size_pairs:
                ta, tb = by_key.get((eng, mode, small)), by_key.get((eng, mode, large))
                if ta and tb:
                    c = paired(ta, tb, f"[Kích thước model] {eng} {mode}: {small} → {large}")
                    if c:
                        comparisons.append(c)
        best = {}
        for mode in ["2step", "3step"]:
            cands = [r for r in rows if r["engine"] == eng and r["mode"] == mode and r["asr_bleu"] == r["asr_bleu"]]
            if cands:
                best[mode] = max(cands, key=lambda r: r["asr_bleu"])
        if "2step" in best and "3step" in best:
            c = paired(best["2step"]["tag"], best["3step"]["tag"],
                       f"[2-step vs 3-step] {eng}: {best['2step']['model']} (2-step tốt nhất) → {best['3step']['model']} (3-step tốt nhất)")
            if c:
                comparisons.append(c)
    for mode in ["2step", "3step"]:
        for model in sorted({r["model"] for r in rows}):
            ta, tb = by_key.get(("xtts", mode, model)), by_key.get(("f5tts", mode, model))
            if ta and tb:
                c = paired(ta, tb, f"[Engine TTS] {mode} {model}: XTTS-v2 → F5-TTS")
                if c:
                    comparisons.append(c)

    return rows, comparisons, mismatches, scorer.backend


# ------------------------------------------------------------------------------
# Xuất báo cáo
# ------------------------------------------------------------------------------
def write_report(rows, comparisons, mismatches, backend, out_dir, n_boot):
    os.makedirs(out_dir, exist_ok=True)
    L = []
    L.append("# 📊 PHÂN TÍCH BENCHMARK S2ST VI → EN (FLEURS test)\n")
    ns = sorted({r["n"] for r in rows})
    bleu_name = "corpus SacreBLEU" if backend == "sacrebleu" else "corpus BLEU-4 tự cài (chưa có sacrebleu)"
    L.append(f"- Số biến thể: **{len(rows)}** | Số mẫu mỗi biến thể: {ns} | BLEU: {bleu_name}, "
             f"chữ thường + bỏ dấu câu | CI 95%: bootstrap {n_boot} lần\n")
    L.append("- **S2TT-BLEU** = văn bản dịch (trước TTS) vs Target Ref. **ASR-BLEU** = Whisper-En nghe lại audio sinh ra vs Target Ref. "
             "**Mất do TTS** = S2TT-BLEU − ASR-BLEU.\n")
    if mismatches:
        L.append(f"- ⚠️ **{len(mismatches)} mẫu có câu tham chiếu KHÁC nhau giữa các biến thể** → so sánh ghép cặp có thể không hợp lệ "
                 f"(VD: {mismatches[0][0]} mẫu {mismatches[0][1]} khác {mismatches[0][2]}).\n")
    else:
        L.append("- ✓ Mọi biến thể dùng cùng các câu test (kiểm tra theo sample_id + Target Ref) → so sánh ghép cặp hợp lệ.\n")
    if ns and min(ns) < 50:
        L.append(f"- ⚠️ Chỉ {min(ns)} mẫu/biến thể → khoảng tin cậy rất rộng; chỉ nên coi là thử nghiệm sơ bộ. "
                 "Kết luận trong khóa luận nên dùng ≥ 100 mẫu.\n")

    L.append("\n## 1. Bảng tổng hợp\n")
    L.append("| Engine | Chế độ | Model S2TT/ASR | n | S2TT-BLEU ↑ [CI 95%] | ASR-BLEU ↑ [CI 95%] | Mất do TTS | SECS ↑ (mean±sd) | RTF ↓ (mean±sd) | Câu không dịch |")
    L.append("| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    order = {"2step": 0, "3step": 1, "direct": 2}
    for r in sorted(rows, key=lambda r: (r["engine"], order.get(r["mode"], 9), r["model"])):
        L.append(f"| {r['engine']} | {r['mode']} | {r['model']} | {r['n']} | "
                 f"{fmt(r['s2tt_bleu'])} [{fmt(r['s2tt_lo'])}, {fmt(r['s2tt_hi'])}] | "
                 f"**{fmt(r['asr_bleu'])}** [{fmt(r['asr_lo'])}, {fmt(r['asr_hi'])}] | {fmt(r['tts_loss'])} | "
                 f"{fmt(r['secs_mean'], 3)} ± {fmt(r['secs_std'], 3)} | {fmt(r['rtf_mean'], 3)} ± {fmt(r['rtf_std'], 3)} | "
                 f"{r['not_translated']}/{r['n']} |")

    L.append("\n## 2. So sánh ghép cặp (B − A trên cùng các câu)\n")
    L.append("Δ > 0 nghĩa là B tốt hơn A (riêng RTF: Δ < 0 là B nhanh hơn). **Có ý nghĩa** khi CI 95% không chứa 0.\n")
    if not comparisons:
        L.append("_Chưa đủ biến thể để so sánh._\n")
    for c in comparisons:
        L.append(f"\n### {c['label']}  (n = {c['n']})\n")
        L.append("| Chỉ số | Δ (B − A) | CI 95% | Kết luận |")
        L.append("| :--- | :---: | :---: | :--- |")
        for key, name, higher_better in [("s2tt_bleu", "S2TT-BLEU", True), ("asr_bleu", "ASR-BLEU", True),
                                         ("secs", "SECS", True), ("rtf", "RTF", False)]:
            v = c.get(key)
            if not v:
                continue
            sig = v["lo"] > 0 or v["hi"] < 0
            if not sig:
                verdict = "Không khác biệt có ý nghĩa"
            else:
                better = (v["delta"] > 0) == higher_better
                verdict = "B tốt hơn ✅" if better else "B kém hơn ❌"
            L.append(f"| {name} | {v['delta']:+.3f} | [{v['lo']:+.3f}, {v['hi']:+.3f}] | {verdict} |")

    L.append("\n## 3. Nhận xét tự động\n")
    notes = []
    for r in rows:
        if r["n"] and r["not_translated"] >= 0.5 * r["n"]:
            notes.append(f"- **{r['engine']} / {r['mode']} / {r['model']}**: {r['not_translated']}/{r['n']} câu đầu ra vẫn là tiếng Việt "
                         "→ model **không có khả năng dịch** (task='translate'); BLEU thấp do đó, không phải do TTS.")
        if r["tts_loss"] == r["tts_loss"] and r["tts_loss"] > 10:
            notes.append(f"- **{r['engine']} / {r['mode']} / {r['model']}**: TTS làm mất {r['tts_loss']:.1f} điểm BLEU "
                         "→ nên nghe lại audio (đọc sai từ, cắt cụt, lặp).")
    valid = [r for r in rows if r["asr_bleu"] == r["asr_bleu"]]
    if valid:
        b = max(valid, key=lambda r: r["asr_bleu"])
        notes.append(f"- ASR-BLEU cao nhất: **{b['engine']} / {b['mode']} / {b['model']}** = {b['asr_bleu']:.2f}.")
        s = max((r for r in valid if r["secs_mean"] == r["secs_mean"]), key=lambda r: r["secs_mean"], default=None)
        if s:
            notes.append(f"- Giữ giọng tốt nhất (SECS): **{s['engine']} / {s['mode']} / {s['model']}** = {s['secs_mean']:.3f}.")
        f = min((r for r in valid if r["rtf_mean"] == r["rtf_mean"]), key=lambda r: r["rtf_mean"], default=None)
        if f:
            notes.append(f"- Nhanh nhất (RTF): **{f['engine']} / {f['mode']} / {f['model']}** = {f['rtf_mean']:.3f}.")
    L.extend(notes or ["_Không có nhận xét._"])
    L.append("\n> Lưu ý: Mel L1 không được đưa vào phân tích vì vô nghĩa với hệ cascaded "
             "(khác người nói, khác độ dài, không căn chỉnh thời gian).\n")

    report = "\n".join(L)
    with open(os.path.join(out_dir, "analysis_report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    keys = ["tag", "engine", "mode", "model", "n", "s2tt_bleu", "s2tt_lo", "s2tt_hi", "asr_bleu", "asr_lo", "asr_hi",
            "tts_loss", "secs_mean", "secs_std", "rtf_mean", "rtf_std", "dur_mean", "not_translated"]
    with open(os.path.join(out_dir, "analysis_summary.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(r[k], 4) if isinstance(r[k], float) else r[k]) for k in keys})
    with open(os.path.join(out_dir, "analysis.json"), "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "comparisons": comparisons, "mismatches": mismatches}, f,
                  ensure_ascii=False, indent=2, default=str)
    return report


def main():
    ap = argparse.ArgumentParser(description="Phân tích kết quả benchmark S2ST từ nhiều lần chạy độc lập")
    ap.add_argument("dirs", nargs="+", help="Các thư mục chứa */metrics.json")
    ap.add_argument("--out", default="outputs/benchmark_all", help="Thư mục xuất báo cáo")
    ap.add_argument("--n-boot", type=int, default=1000, help="Số lần bootstrap (mặc định 1000)")
    args = ap.parse_args()

    dirs = [d for d in args.dirs if os.path.isdir(d)]
    runs = load_runs(dirs)
    if not runs:
        print(f"❌ Không tìm thấy metrics.json nào trong: {args.dirs}")
        sys.exit(1)
    print(f"📂 Tìm thấy {len(runs)} biến thể: " + ", ".join(sorted(runs)))
    rows, comparisons, mismatches, backend = analyze(runs, args.n_boot)
    report = write_report(rows, comparisons, mismatches, backend, args.out, args.n_boot)
    print("\n" + report)
    print(f"\n📁 Đã lưu: {os.path.join(args.out, 'analysis_report.md')} | analysis_summary.csv | analysis.json")


if __name__ == "__main__":
    main()
