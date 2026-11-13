#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tự động cấu hình Rclone Google Drive từ file .env hoặc KLTN.env sang ~/.config/rclone/rclone.conf
"""
import os
import sys
from pathlib import Path

def main():
    conf_dir = Path.home() / ".config" / "rclone"
    conf_dir.mkdir(parents=True, exist_ok=True)
    conf_file = conf_dir / "rclone.conf"

    # 1. Tìm file nguồn: ưu tiên .env, rồi KLTN.env
    env_file = None
    for fname in [".env", "KLTN.env"]:
        if os.path.exists(fname):
            env_file = fname
            break

    if not env_file:
        print("[setup_rclone] Không tìm thấy .env hoặc KLTN.env")
        return 1

    with open(env_file, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # Trường hợp 1: File đã là định dạng INI của rclone ([gdrive] ...)
    if "[gdrive]" in content:
        lines = content.splitlines()
        rclone_lines = []
        capture = False
        for line in lines:
            if line.strip() == "[gdrive]":
                capture = True
            if capture:
                if line.strip().startswith("[") and line.strip() != "[gdrive]":
                    break
                rclone_lines.append(line)
        
        with open(conf_file, "w", encoding="utf-8") as f:
            f.write("\n".join(rclone_lines) + "\n")
        print(f"✓ Đã cấu hình rclone thành công vào {conf_file} (từ định dạng INI)!")
        return 0

    # Trường hợp 2: File là biến môi trường KEY=VALUE
    env_vars = {}
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            env_vars[k.strip()] = v.strip().strip('"').strip("'")

    client_id = env_vars.get("RCLONE_CONFIG_GDRIVE_CLIENT_ID") or env_vars.get("CLIENT_ID", "")
    client_secret = env_vars.get("RCLONE_CONFIG_GDRIVE_CLIENT_SECRET") or env_vars.get("CLIENT_SECRET", "")
    scope = env_vars.get("RCLONE_CONFIG_GDRIVE_SCOPE", "drive")
    token = env_vars.get("RCLONE_CONFIG_GDRIVE_TOKEN") or env_vars.get("TOKEN", "")

    if client_id and token:
        ini_content = f"""[gdrive]
type = drive
client_id = {client_id}
client_secret = {client_secret}
scope = {scope}
token = {token}
"""
        with open(conf_file, "w", encoding="utf-8") as f:
            f.write(ini_content)
        print(f"✓ Đã cấu hình rclone thành công vào {conf_file} (từ biến môi trường)!")
        return 0

    print("[setup_rclone] Không trích xuất được thông tin cấu hình Google Drive hợp lệ.")
    return 1

if __name__ == "__main__":
    sys.exit(main())
