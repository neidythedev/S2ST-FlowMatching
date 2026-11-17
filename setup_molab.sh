#!/usr/bin/env bash
# ==============================================================================
# S2ST MOLAB SETUP AUTOMATOR: THIẾT LẬP TOÀN DIỆN MÔI TRƯỜNG CHỈ VỚI 1 LỆNH
# ==============================================================================
set -e

echo "🚀 BẮT ĐẦU THIẾT LẬP MÔI TRƯỜNG MOLAB CHO S2ST VI -> EN..."

# 1. Cài đặt Rclone nếu chưa có
if ! command -v rclone &> /dev/null; then
    echo "📦 Đang cài đặt rclone..."
    curl -fsSL https://rclone.org/install.sh | bash || true
fi

# 2. Cấu hình Rclone tự động từ KLTN.env hoặc .env
if [ -f "tools/setup_rclone.py" ]; then
    python tools/setup_rclone.py || true
fi

# 3. Gỡ bỏ torchaudio C++ bị lỗi ABI trên Python 3.13
echo "🔧 Đang dọn dẹp thư viện torchaudio..."
pip uninstall torchaudio -y >/dev/null 2>&1 || true

# 4. Cài đặt các gói phụ thuộc cần thiết
echo "📦 Đang cài đặt các thư viện Python..."
pip install "datasets<4.0.0" soundfile vocos sentencepiece sacremoses sacrebleu transformers -q

# 5. Kiểm tra kết nối Google Drive
echo "📡 Đang kiểm tra kết nối Google Drive..."
if rclone lsd gdrive: >/dev/null 2>&1; then
    echo "✓ Kết nối Google Drive THÀNH CÔNG!"
else
    echo "⚠️ Chưa kết nối được Google Drive. Hãy kiểm tra lại file KLTN.env hoặc rclone.conf."
fi

# 6. Tạo sẵn các thư mục cần thiết
mkdir -p logs checkpoints outputs/sanity outputs/benchmark

echo "=========================================================="
echo "✅ THIẾT LẬP HOÀN TẤT 100%! BẠN CÓ THỂ BẮT ĐẦU NGAY:"
echo "  bash run.sh sanity            -> Chạy EDA & Sanity Check (Vocos + Whisper)"
echo "  bash run.sh overfit cfm       -> Overfit 500 bước 16 câu tiếng Anh cho CFM"
echo "  bash run.sh start cfm_en      -> Bắt đầu train Stage 1 CFM Tiếng Anh"
echo "  bash run.sh eval 0            -> Đo Baseline Cascaded"
echo "=========================================================="
