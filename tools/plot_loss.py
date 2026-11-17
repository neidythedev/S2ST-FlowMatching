#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trích xuất loss từ log file và trực quan hóa:
1. In biểu đồ ASCII trực tiếp trên Terminal.
2. Xuất đồ thị vector SVG (mở được ngay trên trình duyệt).
3. Xuất file ảnh PNG (nếu có matplotlib).
"""

import os
import sys
import re
from pathlib import Path

def parse_losses(log_file):
    if not os.path.exists(log_file):
        return [], []
    
    train_losses = []
    val_losses = []

    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # Pattern 1: [E00|Step 010/500] Train Loss: 0.8124
            m1 = re.search(r"Train Loss:\s*([0-9.]+)", line)
            if m1:
                try:
                    train_losses.append(float(m1.group(1)))
                except ValueError:
                    pass

            # Pattern 2: [Overfit Step 050/500] CFM Loss: 0.8421
            m2 = re.search(r"(?:CFM|Overfit|Bridge)\s*Loss:\s*([0-9.]+)", line)
            if m2 and not m1:
                try:
                    train_losses.append(float(m2.group(1)))
                except ValueError:
                    pass

            # Pattern 3: Val Loss: 0.7231
            m3 = re.search(r"Val Loss:\s*([0-9.]+)", line)
            if m3:
                try:
                    val_losses.append(float(m3.group(1)))
                except ValueError:
                    pass

    return train_losses, val_losses

def print_ascii_chart(train_losses, title="TRAIN LOSS PROGRESSION"):
    if not train_losses:
        print("⚠️ Chưa có dữ liệu loss trong file log.")
        return

    print("\n" + "=" * 65)
    print(f"📊 {title} (Tổng số điểm log: {len(train_losses)})")
    print("=" * 65)

    # Lấy mẫu tối đa 20 điểm phân bố đều nếu log quá dài
    if len(train_losses) > 20:
        step = len(train_losses) / 20.0
        sampled = [train_losses[int(i * step)] for i in range(20)]
        sampled.append(train_losses[-1])
    else:
        sampled = train_losses

    min_val = min(sampled)
    max_val = max(sampled)
    span = max(1e-6, max_val - min_val)

    for idx, val in enumerate(sampled):
        # Tính độ dài thanh bar (tối đa 35 ký tự)
        bar_len = int(((val - min_val) / span) * 35)
        bar = "█" * max(1, bar_len)
        print(f" Pt {idx+1:02d} | {val:7.4f} | {bar}")

    first_val = train_losses[0]
    last_val = train_losses[-1]
    pct_change = ((last_val - first_val) / max(1e-6, first_val)) * 100.0
    status = "GIẢM TỐT ✅" if pct_change < 0 else "TĂNG/DAO ĐỘNG ⚠️"
    print("-" * 65)
    print(f"👉 Điểm đầu: {first_val:.4f} | Điểm cuối: {last_val:.4f} ({pct_change:+.1f}%) -> {status}")
    print("=" * 65 + "\n")

def export_svg_chart(train_losses, val_losses, out_svg, exp_name):
    if not train_losses:
        return
    
    width = 700
    height = 360
    pad_left = 60
    pad_bottom = 50
    pad_top = 40
    pad_right = 30
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    all_vals = train_losses + val_losses
    min_v = min(all_vals) * 0.95
    max_v = max(all_vals) * 1.05
    val_range = max(1e-6, max_v - min_v)

    def to_coords(data):
        coords = []
        n = len(data)
        for i, v in enumerate(data):
            x = pad_left + (i / max(1, n - 1)) * plot_w
            y = pad_top + (1.0 - (v - min_v) / val_range) * plot_h
            coords.append(f"{x:.1f},{y:.1f}")
        return " ".join(coords)

    train_pts = to_coords(train_losses)
    val_pts = to_coords(val_losses) if val_losses else ""

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="background:#1e1e2e; font-family:sans-serif;">
  <rect width="{width}" height="{height}" fill="#181825" rx="10"/>
  <text x="{width/2}" y="25" fill="#cdd6f4" font-size="16" font-weight="bold" text-anchor="middle">Loss Curve: {exp_name}</text>
  
  <!-- Trục tọa độ -->
  <line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{height - pad_bottom}" stroke="#45475a" stroke-width="1.5"/>
  <line x1="{pad_left}" y1="{height - pad_bottom}" x2="{width - pad_right}" y2="{height - pad_bottom}" stroke="#45475a" stroke-width="1.5"/>
  
  <!-- Nhãn Min / Max trục Y -->
  <text x="{pad_left - 10}" y="{pad_top + 10}" fill="#a6adc8" font-size="11" text-anchor="end">{max_v:.3f}</text>
  <text x="{pad_left - 10}" y="{height - pad_bottom}" fill="#a6adc8" font-size="11" text-anchor="end">{min_v:.3f}</text>
  
  <!-- Đường Train Loss -->
  <polyline fill="none" stroke="#89b4fa" stroke-width="2.5" points="{train_pts}"/>
  
  <!-- Chú thích Legend -->
  <circle cx="{width - 160}" cy="22" r="5" fill="#89b4fa"/>
  <text x="{width - 150}" y="26" fill="#89b4fa" font-size="12">Train Loss</text>
"""
    if val_pts:
        svg += f"""
  <!-- Đường Val Loss -->
  <polyline fill="none" stroke="#a6e3a1" stroke-width="2.5" stroke-dasharray="4" points="{val_pts}"/>
  <circle cx="{width - 80}" cy="22" r="5" fill="#a6e3a1"/>
  <text x="{width - 70}" y="26" fill="#a6e3a1" font-size="12">Val Loss</text>
"""

    svg += "</svg>"

    os.makedirs(os.path.dirname(out_svg), exist_ok=True)
    with open(out_svg, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"📈 Đã xuất đồ thị vector SVG xem ngay trên trình duyệt tại: {out_svg}")

def export_png_if_possible(train_losses, val_losses, out_png, exp_name):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(9, 4.5))
        plt.plot(train_losses, label="Train Loss", color="#1f77b4", linewidth=1.8)
        if val_losses:
            val_x = [int(i * (len(train_losses) / len(val_losses))) for i in range(len(val_losses))]
            plt.plot(val_x, val_losses, label="Val Loss", color="#2ca02c", linewidth=2.0, linestyle="--")

        plt.title(f"Training Loss Curve: {exp_name}", fontsize=13, fontweight="bold")
        plt.xlabel("Step Index", fontsize=11)
        plt.ylabel("Loss", fontsize=11)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"🖼️ Đã xuất file ảnh PNG tại: {out_png}")
    except Exception:
        # Bỏ qua nếu môi trường không có matplotlib
        pass

def main():
    exp_name = sys.argv[1] if len(sys.argv) > 1 else "cfm_en"
    log_file = f"logs/{exp_name}.log"
    out_svg = f"logs/{exp_name}_loss.svg"
    out_png = f"logs/{exp_name}_loss.png"

    if not os.path.exists(log_file):
        print(f"⚠️ Không tìm thấy file log: {log_file}")
        print("   Hãy đảm bảo tiến trình đã được khởi chạy hoặc có file log trong thư mục logs/.")
        sys.exit(1)

    train_losses, val_losses = parse_losses(log_file)
    print_ascii_chart(train_losses, title=f"LOSS PROGRESSION: {exp_name}")
    export_svg_chart(train_losses, val_losses, out_svg, exp_name)
    export_png_if_possible(train_losses, val_losses, out_png, exp_name)

if __name__ == "__main__":
    main()
