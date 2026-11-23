#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CHUẨN HÓA VĂN BẢN TIẾNG VIỆT CHO TTS (giữ dấu câu để TTS ngắt nghỉ tự nhiên)
  - Số nguyên -> chữ (tới hàng tỷ): 1967 -> "một nghìn chín trăm sáu mươi bảy"
  - Số có dấu phân cách nghìn: 1.000 / 1,000 -> "một nghìn"
  - Số thập phân: 3,5 -> "ba phẩy năm"
  - Ký hiệu: % -> "phần trăm", $ -> "đô la", & -> "và", + -> "cộng"
Ngoài ra có các hàm làm sạch transcript TED (MuST-C/PhoST):
  - bỏ tên người nói ở đầu câu ("Pat Mitchell: ..."), gộp gạch nối bị tách ("Al - Sakkaf" -> "Al-Sakkaf"),
    bỏ chú thích âm thanh "(Laughter)", "(Vỗ tay)", ký hiệu nhạc "♫".
"""
import re

_UNITS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]


def _read_two_digits(n, full=False):
    """Đọc 0..99. full=True: đang ở sau hàng trăm (cần 'lẻ' cho 1..9)."""
    if n < 10:
        return ("lẻ " + _UNITS[n]) if full and n > 0 else _UNITS[n]
    if n < 20:
        return "mười" if n == 10 else "mười " + ("lăm" if n % 10 == 5 else _UNITS[n % 10])
    tens, rem = divmod(n, 10)
    s = _UNITS[tens] + " mươi"
    if rem == 0:
        return s
    if rem == 1:
        return s + " mốt"
    if rem == 4:
        return s + " tư"
    if rem == 5:
        return s + " lăm"
    return s + " " + _UNITS[rem]


def _read_three_digits(n, full=False):
    """Đọc 0..999. full=True: nhóm không đứng đầu (phải đọc cả 'không trăm')."""
    hundreds, rem = divmod(n, 100)
    if hundreds == 0 and not full:
        return _read_two_digits(rem)
    s = _UNITS[hundreds] + " trăm"
    if rem == 0:
        return s
    return s + " " + _read_two_digits(rem, full=True)


def int_to_vi(n):
    """Số nguyên (kể cả rất lớn) -> chữ tiếng Việt."""
    n = int(n)
    if n < 0:
        return "âm " + int_to_vi(-n)
    if n == 0:
        return "không"
    groups = []
    while n > 0:
        n, g = divmod(n, 1000)
        groups.append(g)
    scales = ["", "nghìn", "triệu", "tỷ"]
    words = []
    for i in range(len(groups) - 1, -1, -1):
        g = groups[i]
        if g == 0:
            if i % 3 == 0 and i > 0 and words:     # nhóm tỷ trống vẫn cần chữ "tỷ"
                words.append("tỷ")
            continue
        is_first = (i == len(groups) - 1)
        words.append(_read_three_digits(g, full=not is_first))
        scale = scales[i % 3] if i % 3 else ("tỷ" if i >= 3 else "")
        if scale:
            words.append(scale)
    return re.sub(r"\s+", " ", " ".join(words)).strip()


def _digits_to_vi(s):
    return " ".join(_UNITS[int(c)] for c in s)


def normalize_vi_for_tts(text):
    """Chuẩn hóa câu tiếng Việt để đưa vào TTS. GIỮ dấu câu và chữ hoa."""
    t = str(text or "")
    t = t.replace("’", "'").replace("“", '"').replace("”", '"')
    # Ký hiệu đặc biệt
    t = re.sub(r"(\d)\s*%", r"\1 phần trăm", t)
    t = re.sub(r"\$\s*(\d[\d.,]*)", r"\1 đô la", t)
    t = t.replace("&", " và ").replace(" + ", " cộng ")
    # Số có phân cách nghìn: 1.000.000 hoặc 1,000,000
    t = re.sub(r"\b\d{1,3}(?:[.,]\d{3})+\b", lambda m: int_to_vi(re.sub(r"[.,]", "", m.group(0))), t)
    # Số thập phân: 3,5 hoặc 3.5
    t = re.sub(r"\b(\d+)[.,](\d+)\b", lambda m: f"{int_to_vi(m.group(1))} phẩy {_digits_to_vi(m.group(2)) if m.group(2).startswith('0') else int_to_vi(m.group(2))}", t)
    # Số nguyên còn lại
    t = re.sub(r"\b\d+\b", lambda m: int_to_vi(m.group(0)), t)
    t = re.sub(r"\s+([,.!?;:])", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()


# ------------------------------------------------------------------------------
# Làm sạch transcript TED (MuST-C / PhoST)
# ------------------------------------------------------------------------------
_SPEAKER_RE = re.compile(r"^\s*((?:[A-ZÀ-Ỹ][\w'.\-]*)(?:\s+(?:[A-ZÀ-Ỹ][\w'.\-]*|de|da|van|von|al|bin|el)){0,4})\s*:\s+")


def join_split_hyphens(text):
    """'Al - Sakkaf' -> 'Al-Sakkaf', 'five - game' -> 'five-game' (MuST-C tách gạch nối thành token riêng)."""
    return re.sub(r"(\w)\s+-\s+(\w)", r"\1-\2", str(text or ""))


def strip_sound_tags(text):
    """Bỏ chú thích âm thanh: (Laughter), (Vỗ tay), [Music]..."""
    t = re.sub(r"\([^)]*\)", " ", str(text or ""))
    t = re.sub(r"\[[^\]]*\]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def clean_ted_pair(en, vi):
    """Làm sạch một cặp câu TED. Trả về (en, vi) hoặc None nếu cặp không dùng được."""
    if "♫" in (en or "") or "♪" in (en or ""):
        return None   # đoạn hát / nhạc
    en = strip_sound_tags(join_split_hyphens(en))
    vi = strip_sound_tags(join_split_hyphens(vi))
    m = _SPEAKER_RE.match(en)
    if m:
        # Tên người nói không được đọc trong audio -> bỏ ở cả 2 phía (tên thường giữ nguyên khi dịch)
        en = en[m.end():]
        mv = _SPEAKER_RE.match(vi)
        if mv:
            vi = vi[mv.end():]
    en, vi = en.strip(), vi.strip()
    if len(en.split()) < 2 or len(vi.split()) < 2:
        return None
    return en, vi


if __name__ == "__main__":
    for x in ["1967", "105", "2021", "1.000.000", "3,5%", "15", "21", "1001", "1000005", "$20"]:
        print(f"{x:>10} -> {normalize_vi_for_tts(x)}")
    print(clean_ted_pair("Nadia Al - Sakkaf: Well, I'm glad to be here. (Laughter)",
                         "Nadia Al - Sakkaf: Tôi rất sung sướng được ở đây. (Tiếng cười)"))
