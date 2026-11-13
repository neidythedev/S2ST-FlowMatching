#!/usr/bin/env python3
# ==============================================================================
# TỰ ĐỘNG CẤU HÌNH RCLONE TỪ FILE .ENV TRÊN MOLAB
# ==============================================================================
import json
import os
import subprocess
import sys
from pathlib import Path

def load_env(env_path=".env"):
    """Đọc file .env thuần Python không phụ thuộc thư viện bên ngoài."""
    env_vars = {}
    p = Path(env_path)
    if not p.exists():
        return env_vars
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip("'\"")
            env_vars[k] = v
            os.environ[k] = v
    return env_vars

def setup_rclone():
    env = load_env()
    client_id = env.get("GDRIVE_CLIENT_ID") or os.environ.get("GDRIVE_CLIENT_ID")
    client_secret = env.get("GDRIVE_CLIENT_SECRET") or os.environ.get("GDRIVE_CLIENT_SECRET")
    token_str = env.get("GDRIVE_TOKEN") or os.environ.get("GDRIVE_TOKEN")

    if not client_id or not client_secret or not token_str:
        print("[LỖI] Thiếu thông tin trong .env! Cần có GDRIVE_CLIENT_ID, GDRIVE_CLIENT_SECRET, GDRIVE_TOKEN.")
        sys.exit(1)

    try:
        # Kiểm tra token_str có đúng định dạng JSON không
        token_json = json.loads(token_str)
        cleaned_token = json.dumps(token_json)
    except Exception as e:
        print(f"[LỖI] GDRIVE_TOKEN không phải chuỗi JSON hợp lệ: {e}")
        sys.exit(1)

    conf_dir = Path.home() / ".config" / "rclone"
    conf_dir.mkdir(parents=True, exist_ok=True)
    conf_file = conf_dir / "rclone.conf"

    conf_content = f"""[gdrive]
type = drive
client_id = {client_id}
client_secret = {client_secret}
scope = drive
token = {cleaned_token}
"""
    conf_file.write_text(conf_content, encoding="utf-8")
    print(f"✓ Đã ghi cấu hình Rclone vào: {conf_file}")

    # Kiểm tra kết nối
    res = subprocess.run(["rclone", "lsd", "gdrive:"], capture_output=True, text=True)
    if res.returncode == 0:
        print("🎉 KẾT NỐI GOOGLE DRIVE THÀNH CÔNG! Danh sách thư mục:")
        print(res.stdout)
    else:
        print("❌ Kết nối thất bại:")
        print(res.stderr)

if __name__ == "__main__":
    setup_rclone()
