#!/usr/bin/env bash
# ==============================================================================
# SCRIPT CHẠY HUẤN LUYỆN CHẠY NGẦM BỀN BỈ TRÊN TERMINAL MOLAB
# Tuân thủ nghiêm ngặt quy chuẩn SKILL2.md:
# Tách tiến trình khỏi shell bằng nohup, lưu PID, không làm nghẽn UI Marimo.
# ==============================================================================
set -euo pipefail

MODE="${1:-train}"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
PID_FILE="$LOG_DIR/train_bridge.pid"

if [ "$MODE" = "train" ]; then
    echo "=================================================================="
    echo "🚀 ĐANG KHỞI CHẠY HUẤN LUYỆN TONE-AWARE BRIDGE TRONG TIẾN TRÌNH NGẦM..."
    echo "=================================================================="
    
    nohup python -u train_bridge.py \
        --config configs/stage2_bridge.yaml \
        > "$LOG_DIR/train.log" 2>&1 &
    
    echo $! > "$PID_FILE"
    echo "✓ Huấn luyện đã khởi chạy với PID $(cat "$PID_FILE")."
    echo "👉 Xem log thời gian thực: tail -f $LOG_DIR/train.log"

elif [ "$MODE" = "resume" ]; then
    echo "=================================================================="
    echo "🔄 ĐANG KHÔI PHỤC HUẤN LUYỆN TỪ CHECKPOINT MODEL-LAST.PTH..."
    echo "=================================================================="
    
    nohup python -u train_bridge.py \
        --config configs/stage2_bridge.yaml \
        --resume \
        > "$LOG_DIR/train.log" 2>&1 &
    
    echo $! > "$PID_FILE"
    echo "✓ Đã resume huấn luyện với PID $(cat "$PID_FILE")."
    echo "👉 Xem log thời gian thực: tail -f $LOG_DIR/train.log"

elif [ "$MODE" = "stop" ]; then
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        echo "🛑 Đang dừng tiến trình huấn luyện PID $PID..."
        kill -15 "$PID" 2>/dev/null || true
        rm -f "$PID_FILE"
        echo "✓ Đã dừng tiến trình an toàn."
    else
        echo "⚠️ Không tìm thấy file PID ($PID_FILE). Tiến trình có thể đã dừng trước đó."
    fi

else
    echo "Cú pháp không hợp lệ: $MODE"
    echo "Cách dùng: bash run_bridge_molab.sh [train | resume | stop]"
    exit 1
fi
