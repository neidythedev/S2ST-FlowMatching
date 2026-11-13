#!/usr/bin/env bash
# Stage 1: Acoustic Sanity Check Runbook (SKILL.md compliant)
set -eo pipefail

export LC_ALL=C
export PYTHONUNBUFFERED=1

# Auto-detect Molab Python interpreter
if [ -f "/tmp/uv-venv/bin/python" ]; then
    PY="/tmp/uv-venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
else
    PY="python"
fi

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${WORKDIR}"

echo "=================================================================="
echo "🎯 MOLAB TERMINAL RUNNER: GIAI ĐOẠN 1 (ACOUSTIC SANITY CHECK)"
echo "=================================================================="
echo "Interpreter: ${PY}"
echo "Working Dir: ${WORKDIR}"

# Check GPU & PyTorch
${PY} -c '
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
'

# Ensure required packages exist
${PY} -c '
import sys, subprocess
def check_or_install(pkg):
    try:
        __import__(pkg)
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
check_or_install("soundfile")
check_or_install("vocos")
'

# Run Stage 1 Acoustic Sanity Smoke Test
echo -e "\n🚀 Đang chạy smoke test kiểm tra tái tạo âm thanh Vocos + 100-Mel..."
${PY} smoke_test_stage1_vocos.py --device auto --output-dir ./stage1_artifacts

echo -e "\n✅ HOÀN TẤT GIAI ĐOẠN 1! Báo cáo và file âm thanh lưu tại: ${WORKDIR}/stage1_artifacts"

