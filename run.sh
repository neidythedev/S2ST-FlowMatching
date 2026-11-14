#!/usr/bin/env bash
# ==============================================================================
# S2ST MASTER RUNNER: TỰ ĐỘNG HÓA TOÀN BỘ CHỈ VỚI 1 LỆNH
# Cú pháp: bash run.sh [start | logs | stop | status | sync] [tên_exp]
# ==============================================================================
set -euo pipefail

ACTION="${1:-help}"
EXP_NAME="${2:-bridge_envi}"
LOG_DIR="logs"
mkdir -p "$LOG_DIR" "checkpoints"
PID_FILE="$LOG_DIR/${EXP_NAME}.pid"
LOG_FILE="$LOG_DIR/${EXP_NAME}.log"

case "$ACTION" in
    start)
        # 0. Gỡ bỏ torchaudio C++ bị lỗi ABI trên Python 3.13
        pip uninstall torchaudio -y >/dev/null 2>&1 || true

        # 1. Tự động kiểm tra và cấu hình rclone từ .env nếu chưa có
        if [ -f "tools/setup_rclone.py" ] && [ -f ".env" ] && [ ! -f "$HOME/.config/rclone/rclone.conf" ]; then
            python tools/setup_rclone.py >/dev/null 2>&1 || true
        fi
        
        # 2. Dừng tiến trình cũ nếu đang chạy
        if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            echo "⚠️ Thử nghiệm '$EXP_NAME' đang chạy (PID $(cat "$PID_FILE")). Đang dừng để restart..."
            kill -15 "$(cat "$PID_FILE")" 2>/dev/null || true
            sleep 1
        fi

        # 3. Kiểm tra checkpoint cũ để tự động nối tiếp (--resume)
        EXTRA_ARGS=""
        if [[ "${*:-}" == *"--fresh"* ]]; then
            echo "🆕 Bắt đầu huấn luyện MỚI HOÀN TOÀN từ đầu (--fresh), bỏ qua checkpoint cũ!"
            rm -f "checkpoints/${EXP_NAME}-last.pth"
            EXTRA_ARGS=""
        elif [ -f "checkpoints/${EXP_NAME}-last.pth" ] || [ -f "checkpoints/model-last.pth" ]; then
            echo "⚡ Phát hiện Checkpoint lưu trước đó -> Tự động nối tiếp huấn luyện (--resume)!"
            EXTRA_ARGS="--resume"
        fi

        # 4. Khởi chạy ngầm độc lập (detached background)
        if [[ "$EXP_NAME" == *"cfm"* ]]; then
            SCRIPT="train_cfm.py"
            CONFIG="configs/stage1_cfm.yaml"
        else
            SCRIPT="train_bridge.py"
            CONFIG="configs/stage2_bridge.yaml"
        fi

        echo "🚀 Đang khởi chạy '$EXP_NAME' (Script: $SCRIPT, Config: $CONFIG)..."
        nohup python -u "$SCRIPT" \
            --config "$CONFIG" \
            --exp-name "$EXP_NAME" \
            $EXTRA_ARGS \
            > "$LOG_FILE" 2>&1 &
        
        echo $! > "$PID_FILE"
        echo "✓ Đã khởi chạy thành công với PID $(cat "$PID_FILE")!"
        echo "👉 Xem log ngay bằng lệnh: bash run.sh logs $EXP_NAME"
        ;;

    logs)
        if [ -f "$LOG_FILE" ]; then
            tail -f "$LOG_FILE"
        else
            echo "⚠️ Chưa có file log: $LOG_FILE. Bạn đã chạy 'bash run.sh start $EXP_NAME' chưa?"
        fi
        ;;

    stop)
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            echo "🛑 Đang dừng '$EXP_NAME' (PID $PID)..."
            kill -15 "$PID" 2>/dev/null || true
            rm -f "$PID_FILE"
            echo "✓ Đã dừng thành công."
        else
            echo "⚠️ Không tìm thấy tiến trình nào đang chạy cho '$EXP_NAME'."
        fi
        ;;

    status)
        if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            echo "🟢 '$EXP_NAME' ĐANG CHẠY BÌNH THƯỜNG (PID: $(cat "$PID_FILE"))."
        else
            echo "⚪ '$EXP_NAME' HIỆN TẠI KHÔNG CHẠY."
        fi
        ;;

    sync)
        echo "☁️ Đang đồng bộ Checkpoints, Logs & Âm thanh outputs lên Google Drive..."
        rclone copy checkpoints/ "gdrive:runs/${EXP_NAME}/checkpoints/" -P || true
        rclone copy "$LOG_DIR" "gdrive:runs/${EXP_NAME}/logs/" -P || true
        if [ -d "outputs" ]; then
            rclone copy outputs/ "gdrive:runs/${EXP_NAME}/outputs/" -P || true
        fi
        echo "✓ Đồng bộ hoàn tất lên Google Drive!"
        ;;

    restore)
        echo "📥 Đang tải Checkpoints từ Google Drive về máy Molab..."
        if [ -f "tools/setup_rclone.py" ] && [ -f ".env" ] && [ ! -f "$HOME/.config/rclone/rclone.conf" ]; then
            python tools/setup_rclone.py >/dev/null 2>&1 || true
        fi
        rclone copy "gdrive:runs/${EXP_NAME}/checkpoints/" checkpoints/ -P || true
        echo "✓ Khôi phục Checkpoints thành công từ Google Drive!"
        ;;

    sanity)
        echo "🧪 Đang chạy bài kiểm tra Tiền Huấn luyện (Sanity Overfit Check 50 steps trên 4 mẫu)..."
        pip uninstall torchaudio -y >/dev/null 2>&1 || true
        pip install "datasets<4.0.0" soundfile vocos sentencepiece sacremoses -q 2>/dev/null || true
        python train_bridge.py --sanity-check
        ;;

    eval)
        CASE_CHOICE="${2:-all}"
        echo "📊 Đang chuẩn bị môi trường & chạy Benchmark thực nghiệm cho Case: $CASE_CHOICE..."
        # Tự động gỡ bỏ bản torchaudio C++ bị vỡ symbol ABI trên Python 3.13
        pip uninstall torchaudio -y >/dev/null 2>&1 || true
        pip install "datasets<4.0.0" soundfile vocos sentencepiece sacremoses sacrebleu -q 2>/dev/null || true
        python eval_benchmark.py --case "$CASE_CHOICE" --num-samples 10
        ;;

    *)
        echo "=========================================================="
        echo "💡 CÚ PHÁP CHUẨN ĐÚNG CÁC LỆNH:"
        echo "  bash run.sh start [tên_exp]      -> Chạy huấn luyện ngầm"
        echo "  bash run.sh logs [tên_exp]       -> Xem log trực tiếp"
        echo "  bash run.sh eval [0|1|2|3|4|all] -> Chạy benchmark đo metric (0: Pretrained Baseline)"
        echo "  bash run.sh stop [tên_exp]       -> Dừng huấn luyện"
        echo "  bash run.sh sync [tên_exp]       -> Đẩy checkpoint lên Drive"
        echo "  bash run.sh restore [tên_exp]    -> Tải checkpoint từ Drive về"
        echo "=========================================================="
        exit 1
        ;;
esac

