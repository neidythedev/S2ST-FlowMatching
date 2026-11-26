#!/usr/bin/env bash
# ==============================================================================
# S2ST MASTER RUNNER: TỰ ĐỘNG HÓA TOÀN BỘ CHỈ VỚI 1 LỆNH
# Cú pháp: bash run.sh [start | logs | stop | status | sync] [tên_exp]
# ==============================================================================
set -euo pipefail

ACTION="${1:-help}"
EXP_NAME="${2:-bridge_envi}"
LOG_DIR="logs"
mkdir -p "$LOG_DIR" "checkpoints" "outputs/sanity" "outputs/benchmark"
PID_FILE="$LOG_DIR/${EXP_NAME}.pid"
LOG_FILE="$LOG_DIR/${EXP_NAME}.log"

# Tự động liên kết thư mục dự án với mọi vị trí hiển thị của Marimo File Explorer trên Molab.
# Dùng đường dẫn THẬT (pwd -P): nếu dùng $PWD khi đang đứng trong chính đường link
# /marimo/S2ST-FlowMatching, 'ln -sfn' sẽ ghi đè link thành link TRỎ VÀO CHÍNH NÓ
# ("Too many levels of symbolic links").
REPO_REAL="$(pwd -P)"
safe_link() {
    local dest="$1/S2ST-FlowMatching"
    [ -d "$1" ] || return 0
    [ "$1" = "$REPO_REAL" ] && return 0
    # Đích là thư mục thật (không phải link) -> có thể chính là repo / dữ liệu khác: KHÔNG đụng vào
    if [ -e "$dest" ] && [ ! -L "$dest" ]; then return 0; fi
    # Link đã trỏ đúng repo -> giữ nguyên
    if [ -L "$dest" ] && [ "$(readlink -f "$dest" 2>/dev/null)" = "$REPO_REAL" ]; then return 0; fi
    ln -sfn "$REPO_REAL" "$dest" 2>/dev/null || true
}
for target_dir in "/marimo" "/home/marimo" "/workspace"; do
    safe_link "$target_dir"
done
NOTEBOOK_PY=$(find /marimo /home/marimo /root /workspace -maxdepth 2 -name "notebook.py" 2>/dev/null | head -n 1 || true)
# Chỉ tạo link khi thật sự tìm thấy notebook.py (nếu không, dirname "" = "." sẽ tạo bản sao repo lồng trong repo)
if [ -n "$NOTEBOOK_PY" ]; then
    NOTEBOOK_DIR=$(dirname "$NOTEBOOK_PY")
    [ "$NOTEBOOK_DIR" != "." ] && safe_link "$NOTEBOOK_DIR"
fi

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
            if [ -f "configs/stage1_cfm_en.yaml" ]; then
                CONFIG="configs/stage1_cfm_en.yaml"
            else
                CONFIG="configs/stage1_cfm.yaml"
            fi
        else
            SCRIPT="train_bridge.py"
            CONFIG="configs/stage2_bridge.yaml"
            # Tên thí nghiệm có "sem" hoặc "mas" -> bản hai lượt (nguồn ngữ nghĩa NLLB + MAS)
            if [[ "$EXP_NAME" == *"sem"* || "$EXP_NAME" == *"mas"* ]]; then CONFIG="configs/stage2_bridge_sem.yaml"; fi
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

    upload)
        echo "📤 Đang đẩy toàn bộ outputs (audio, HTML, metrics) lên GitHub..."
        # Đếm số file sẽ upload
        WAV_COUNT=$(find outputs/ -name "*.wav" 2>/dev/null | wc -l)
        HTML_COUNT=$(find outputs/ -name "*.html" 2>/dev/null | wc -l)
        JSON_COUNT=$(find outputs/ -name "*.json" 2>/dev/null | wc -l)
        MD_COUNT=$(find outputs/ -name "*.md" 2>/dev/null | wc -l)
        echo "   📊 Tìm thấy: ${WAV_COUNT} file .wav | ${HTML_COUNT} file .html | ${JSON_COUNT} file .json | ${MD_COUNT} file .md"

        # Git add outputs
        git add outputs/ -f
        CHANGED=$(git diff --cached --stat 2>/dev/null | tail -1)
        if [ -z "$CHANGED" ]; then
            echo "ℹ️ Không có file mới nào cần đẩy lên. Outputs đã được cập nhật trên GitHub rồi!"
            exit 0
        fi

        echo "   📦 Thay đổi: $CHANGED"
        TIMESTAMP=$(date +"%Y-%m-%d %H:%M")
        git commit -m "📤 Upload outputs: ${WAV_COUNT} wavs, ${HTML_COUNT} html, ${JSON_COUNT} json (${TIMESTAMP})"
        git push origin s2st-vi-en
        echo ""
        echo "✅ Đẩy outputs lên GitHub thành công!"
        echo "👉 Tải về tại: https://github.com/neidythedev/S2ST-FlowMatching/tree/s2st-vi-en/outputs"
        ;;

    sanity)
        echo "🧪 Đang chạy trọn bộ EDA & Sanity Check Suite (Vocos Round-trip + Whisper Vi health)..."
        pip uninstall torchaudio -y >/dev/null 2>&1 || true
        pip install "datasets<4.0.0" soundfile vocos sentencepiece sacremoses -q 2>/dev/null || true
        python sanity_and_eda_suite.py --all
        ;;

    overfit)
        TARGET="${2:-bridge}"
        pip uninstall torchaudio -y >/dev/null 2>&1 || true
        pip install "datasets<4.0.0" soundfile vocos sentencepiece sacremoses -q 2>/dev/null || true
        if [[ "$TARGET" == *"cfm"* ]]; then
            echo "🧪 Đang chạy Sanity Check #3: Overfit 500 bước trên 16 câu Tiếng Anh cho Stage 1 CFM..."
            python -u train_cfm.py --config configs/stage1_cfm_en.yaml --overfit-16
        else
            echo "🧪 Đang chạy Sanity Check #3: Overfit 500 bước trên 16 cặp câu Vi-En cho Stage 2 Bridge..."
            # Tham số thứ 3 trở đi chuyển thẳng cho train_bridge.py, VD:
            #   bash run.sh overfit bridge --overfit-steps 3000 --overfit-lr 1e-3
            python -u train_bridge.py --overfit-16 "${@:3}"
        fi
        ;;

    overfit-diag)
        # Chẩn đoán vì sao Bridge không học thuộc được c* khi overfit 16 câu. Mỗi biến thể chạy riêng, cùng 16 câu:
        #   A  = chỉ L1(c*), không dropout, đầu vào Vi        -> Bridge có học thuộc c* không (không bị CFM kéo lệch)?
        #   B  = như A nhưng đầu vào = PhoWhisper(audio Anh)   -> ánh xạ gần đồng nhất; học không được = lỗi kiến trúc
        #   C  = đủ loss, không dropout, đầu vào Vi           -> so với A: CFM loss có kéo Bridge lệch khỏi c* không?
        # Cách dùng: bash run.sh overfit-diag [bước=2000] [lr=2e-4] [biến_thể=A,B,C]
        D_STEPS="${2:-2000}"
        D_LR="${3:-2e-4}"
        D_VARS="${4:-A,B,C}"
        pip uninstall torchaudio -y >/dev/null 2>&1 || true
        pip install "datasets<4.0.0" soundfile vocos sentencepiece sacremoses -q 2>/dev/null || true
        mkdir -p "$LOG_DIR"
        IFS=',' read -ra VLIST <<< "$D_VARS"
        for V in "${VLIST[@]}"; do
            case "$V" in
                A) FLAGS="--overfit-cond-only --overfit-no-dropout" ;;
                B) FLAGS="--overfit-cond-only --overfit-no-dropout --overfit-oracle-src" ;;
                C) FLAGS="--overfit-no-dropout" ;;
                *) echo "❌ Biến thể không hợp lệ: $V (chỉ A, B, C)"; continue ;;
            esac
            echo "================ 🔬 Chẩn đoán $V: $FLAGS ($D_STEPS bước, LR $D_LR) ================"
            python -u train_bridge.py --overfit-16 --overfit-steps "$D_STEPS" --overfit-lr "$D_LR" \
                --overfit-tag "diag$V" $FLAGS 2>&1 | tee "$LOG_DIR/overfit_diag_$V.log"
        done
        echo ""
        echo "=================== 🏁 TÓM TẮT CHẨN ĐOÁN ==================="
        grep -h "KẾT LUẬN\|CFM loss: với\|L1(Bridge, c\*)" "$LOG_DIR"/overfit_diag_*.log
        ;;

    eval)
        CASE_CHOICE="${2:-sota}"
        S2TT_MODE="${3:-both}"
        NUM_SAMPLES="${4:-10}"
        MT_MODEL="${5:-auto}"
        # Tham số thứ 6 trở đi được chuyển thẳng cho eval_benchmark.py, VD:
        #   bash run.sh eval sota both 10 auto --s2tt-models openai/whisper-large-v3 --vi-asr-models vinai/PhoWhisper-large
        echo "📊 Đang chuẩn bị môi trường & chạy Benchmark thực nghiệm..."
        echo "   • Kịch bản: $CASE_CHOICE | Chế độ S2TT: $S2TT_MODE | Số mẫu: $NUM_SAMPLES | MT Model: $MT_MODEL"

        # 0. Gỡ bỏ torchaudio C++ bị vỡ symbol ABI trên Python 3.13
        pip uninstall torchaudio -y >/dev/null 2>&1 || true

        # 1. Cài đặt các thư viện nền tảng
        echo "📦 [Deps 1/4] Cài đặt thư viện nền tảng (datasets, soundfile, vocos, sacrebleu)..."
        pip install "datasets<4.0.0" soundfile vocos sacrebleu -q 2>/dev/null || true

        # 2. Nâng cấp sentencepiece + protobuf (sửa lỗi VinAI-Translate parse sentencepiece.bpe.model)
        echo "📦 [Deps 2/4] Nâng cấp sentencepiece & protobuf (fix VinAI-Translate)..."
        pip install --upgrade sentencepiece protobuf sacremoses -q 2>/dev/null || true
        # Xóa HF cache VinAI bị corrupted nếu có (sẽ tự tải lại bản mới)
        VINAI_CACHE="$HOME/.cache/huggingface/hub/models--vinai--vinai-translate-vi2en-v2"
        if [ -d "$VINAI_CACHE" ]; then
            # Kiểm tra xem sentencepiece.bpe.model có bị lỗi không
            if python -c "
import sentencepiece as spm
import glob, sys
bpe_files = glob.glob('$VINAI_CACHE/**/sentencepiece.bpe.model', recursive=True)
if bpe_files:
    sp = spm.SentencePieceProcessor()
    sp.Load(bpe_files[0])
    sys.exit(0)
sys.exit(1)
" 2>/dev/null; then
                echo "✓ VinAI-Translate cache hợp lệ."
            else
                echo "⚠️ VinAI-Translate cache bị lỗi, đang xóa để tải lại..."
                rm -rf "$VINAI_CACHE"
            fi
        fi

        # 3. Cài đặt TTS engines (XTTS-v2 & F5-TTS) nếu cần
        if [[ "$CASE_CHOICE" == "sota" || "$CASE_CHOICE" == "both_tts" || "$CASE_CHOICE" == "all" || "$CASE_CHOICE" == "xtts" || "$CASE_CHOICE" == "6" ]]; then
            echo "📦 [Deps 3/4] Cài đặt XTTS-v2 (coqui-tts + torchcodec)..."
            export COQUI_TOS_AGREED=1  # Bỏ qua prompt license interactive
            pip install torchcodec -q 2>/dev/null || true
            pip install coqui-tts -q 2>/dev/null || true
        fi
        if [[ "$CASE_CHOICE" == "sota" || "$CASE_CHOICE" == "both_tts" || "$CASE_CHOICE" == "all" || "$CASE_CHOICE" == "f5" || "$CASE_CHOICE" == "f5tts" || "$CASE_CHOICE" == "7" ]]; then
            echo "📦 [Deps 4/4] Cài đặt F5-TTS (f5-tts + google-cloud-storage)..."
            pip install google-cloud-storage -q 2>/dev/null || true
            pip install f5-tts -q 2>/dev/null || true
            # Gỡ flash-attn SAU khi cài f5-tts (vì f5-tts re-install nó nhưng bị broken trên Molab)
            pip uninstall flash-attn -y 2>/dev/null || true
        fi

        echo "✓ Cài đặt dependencies hoàn tất! Bắt đầu chạy benchmark..."
        export COQUI_TOS_AGREED=1
        python -u eval_benchmark.py --case "$CASE_CHOICE" --s2tt-mode "$S2TT_MODE" --num-samples "$NUM_SAMPLES" --mt-model "$MT_MODEL" "${@:6}" 2>&1 | tee -a "$LOG_DIR/eval.log"
        ;;

    encoders)
        SAMPLES="${2:-30}"
        echo "🔬 Đang chạy Benchmark so sánh 3 Speech Encoders trên FLEURS, VietMed, VIVOS & Common Voice (Tổng cộng: $SAMPLES mẫu trên nhiều batch)..."
        pip uninstall torchaudio -y >/dev/null 2>&1 || true
        pip install "datasets<4.0.0" soundfile librosa transformers -q 2>/dev/null || true
        python tools/benchmark_encoders.py --num-samples "$SAMPLES" --batch-size 10
        ;;

    compare-encoder)
        # EXP02: so sánh encoder PHÍA ĐÍCH cho Stage 1 (decoder cố định theo config), cùng dữ liệu/seed/số bước
        # Cú pháp: bash run.sh compare-encoder [số_bước] [danh_sách_encoder]
        STEPS="${2:-3000}"
        ENCODERS="${3:-vinai/PhoWhisper-small,openai/whisper-small}"
        pip uninstall torchaudio -y >/dev/null 2>&1 || true
        pip install "datasets<4.0.0" soundfile vocos sentencepiece -q 2>/dev/null || true
        echo "⚖️ So sánh encoder phía đích: $ENCODERS | $STEPS bước/biến thể"
        python -u train_cfm.py --config configs/stage1_cfm_en.yaml --exp-name encoder_compare             --compare-encoders "$ENCODERS" --compare-steps "$STEPS" 2>&1 | tee "$LOG_DIR/encoder_compare.log"
        if [ -d "outputs/sanity/encoder_compare" ]; then
            rclone copy outputs/sanity/encoder_compare "gdrive:runs/encoder_compare/" -P || echo "⚠️ Không đẩy được lên Drive"
        fi
        ;;

    exp03)
        # EXP03: lưới 2×2 dữ liệu (fleurs | fleurs+phost) × encoder đích (PhoWhisper | whisper-small), cùng seed/số bước
        # Cú pháp: bash run.sh exp03 [số_bước] [encoders] [nguồn_dữ_liệu]
        STEPS="${2:-3000}"
        ENCODERS="${3:-vinai/PhoWhisper-small,openai/whisper-small}"
        DATAS="${4:-fleurs,fleurs+phost}"
        MANIFEST="data/phost/manifest_train.jsonl"
        if [ ! -f "$MANIFEST" ]; then
            echo "📥 Chưa có $MANIFEST trên máy -> tải từ gdrive:datasets/phost/ ..."
            rclone copy gdrive:datasets/phost/ data/phost/ --exclude "_tmp_talk_wav/**" -P || true
        fi
        if [ ! -f "$MANIFEST" ]; then
            echo "❌ Vẫn chưa có $MANIFEST. Chạy trước: bash run.sh prepare-phost train 50"
            exit 1
        fi
        echo "ℹ️ PhoST: $(wc -l < "$MANIFEST") câu trong $MANIFEST"
        pip uninstall torchaudio -y >/dev/null 2>&1 || true
        pip install "datasets<4.0.0" soundfile vocos sentencepiece -q 2>/dev/null || true
        echo "⚖️ EXP03: dữ liệu [$DATAS] × encoder [$ENCODERS] | $STEPS bước/biến thể"
        python -u train_cfm.py --config configs/stage1_cfm_en.yaml --exp-name exp03             --compare-data "$DATAS" --compare-encoders "$ENCODERS" --compare-steps "$STEPS" 2>&1 | tee "$LOG_DIR/exp03.log"
        if [ -d "outputs/sanity/exp03_data_encoder" ]; then
            rclone copy outputs/sanity/exp03_data_encoder "gdrive:runs/exp03_data_encoder/" -P || echo "⚠️ Không đẩy được lên Drive"
        fi
        ;;

    prepare-phost)
        # Cắt câu PhoST (MuST-C) -> 16kHz + manifest. Cần: hf auth login + đã chấp nhận điều khoản vinai/PhoST
        # Cú pháp: bash run.sh prepare-phost [train|dev|test] [số_giờ]
        P_SPLIT="${2:-train}"
        P_HOURS="${3:-50}"
        pip install soundfile huggingface_hub -q 2>/dev/null || true
        echo "📦 Chuẩn bị PhoST split=$P_SPLIT, mục tiêu $P_HOURS giờ -> data/phost (đẩy lên gdrive:datasets/phost/)"
        echo "⚠️ PhoST cấm phân phối lại: dữ liệu chỉ lưu local + Drive riêng, KHÔNG 'bash run.sh upload'."
        python -u tools/prepare_phost.py --split "$P_SPLIT" --hours "$P_HOURS" --out-dir data/phost             --drive gdrive:datasets/phost/ 2>&1 | tee "$LOG_DIR/prepare_phost_${P_SPLIT}.log"
        ;;

    compare-decoder)
        # So sánh kiến trúc decoder Stage 1 một cách công bằng (cùng dữ liệu, thứ tự, seed, số bước)
        # Cú pháp: bash run.sh compare-decoder [số_bước] [danh_sách_biến_thể]
        STEPS="${2:-3000}"
        VARIANTS="${3:-v1,v2-small,v2,v2+ln}"
        pip uninstall torchaudio -y >/dev/null 2>&1 || true
        pip install "datasets<4.0.0" soundfile vocos sentencepiece -q 2>/dev/null || true
        echo "⚖️ So sánh decoder: $VARIANTS | $STEPS bước/biến thể"
        python -u train_cfm.py --config configs/stage1_cfm_en.yaml --exp-name decoder_compare             --compare-decoders "$VARIANTS" --compare-steps "$STEPS" 2>&1 | tee "$LOG_DIR/decoder_compare.log"
        if [ -d "outputs/sanity/decoder_compare" ]; then
            rclone copy outputs/sanity/decoder_compare "gdrive:runs/decoder_compare/" -P || echo "⚠️ Không đẩy được lên Drive"
        fi
        ;;

    rclone-token)
        # ------------------------------------------------------------------
        # Ghi token Google Drive MỚI vào rclone.conf (không cần trả lời y/n, không bị lỗi dán nhiều dòng)
        # Cú pháp: bash run.sh rclone-token 'eyJ0b2tlbi...'   (chuỗi in ra bởi 'rclone authorize' trên Windows)
        # ------------------------------------------------------------------
        TOKEN_B64="${2:-}"
        if [[ -z "$TOKEN_B64" ]]; then
            echo "❌ Thiếu token. Cú pháp: bash run.sh rclone-token 'eyJ0b2tlbi...'"
            exit 1
        fi
        CONF="$(rclone config file 2>/dev/null | tail -n 1)"
        [ -f "$CONF" ] || CONF="/home/marimo/.config/rclone/rclone.conf"
        if [ ! -f "$CONF" ] || ! grep -q '^\[gdrive\]' "$CONF"; then
            echo "❌ Chưa có cấu hình [gdrive] tại $CONF -> hãy dán khối lệnh trong MOLAB_1CLICK_SETUP.md trước."
            exit 1
        fi
        python3 - "$TOKEN_B64" "$CONF" <<'PYEOF'
import base64, json, re, sys
raw, conf = sys.argv[1].strip(), sys.argv[2]
if raw.startswith("eyJjbGllbnRf"):
    sys.exit("❌ Đây là chuỗi ĐẦU VÀO (client_id). Hãy dán chuỗi KẾT QUẢ bắt đầu bằng eyJ0b2tlbi...")
s = raw.replace("-", "+").replace("_", "/")
s += "=" * (-len(s) % 4)
try:
    tok = json.loads(base64.b64decode(s))["token"]
    json.loads(tok)
except Exception as e:
    sys.exit(f"❌ Chuỗi token không hợp lệ ({e}). Copy đủ phần giữa '--->' và '<---End paste'.")
c = open(conf).read()
c, n = re.subn(r"(?m)^token = .*$", lambda m: "token = " + tok, c)
if n == 0:
    c = c.replace("[gdrive]\n", "[gdrive]\ntoken = " + tok + "\n", 1)
open(conf, "w").write(c)
print("✓ Đã ghi token mới vào", conf)
PYEOF
        for other in /home/marimo/.config/rclone/rclone.conf /root/.config/rclone/rclone.conf; do
            if [ "$other" != "$CONF" ]; then
                mkdir -p "$(dirname "$other")" && cp "$CONF" "$other"
            fi
        done
        echo "📡 Kiểm tra kết nối..."
        if rclone lsd gdrive:runs; then
            echo "✓ Kết nối Google Drive THÀNH CÔNG!"
            echo ""
            bash "$0" rclone-export
        else
            echo "❌ Vẫn lỗi -> token đã bị thu hồi, chạy lại 'rclone authorize' trên Windows để lấy token mới."
            exit 1
        fi
        ;;

    rclone-export)
        # In dòng RCLONE_B64 mới để dán đè vào MOLAB_1CLICK_SETUP.md (máy Windows) cho lần reset sau
        CONF="$(rclone config file 2>/dev/null | tail -n 1)"
        [ -f "$CONF" ] || CONF="/home/marimo/.config/rclone/rclone.conf"
        echo "📋 Copy NGUYÊN dòng dưới, dán ĐÈ dòng RCLONE_B64=... trong MOLAB_1CLICK_SETUP.md (máy Windows):"
        echo ""
        echo "RCLONE_B64=\"$(base64 -w0 "$CONF")\""
        echo ""
        ;;

    bench)
        # ------------------------------------------------------------------
        # Chạy ĐÚNG 1 biến thể benchmark (để chạy song song trên nhiều notebook)
        # Cú pháp: bash run.sh bench <xtts|f5> <2step|3step> <model_hf_id> [số_mẫu]
        # Kết quả tự đẩy lên gdrive:runs/benchmark/<tag>/ để gom lại bằng 'bash run.sh analyze'
        # ------------------------------------------------------------------
        ENGINE="${2:-}"
        MODE="${3:-}"
        MODEL_ID="${4:-}"
        NUM_SAMPLES="${5:-10}"
        # SeamlessM4T-v2 (direct S2ST, không có biến thể): bash run.sh bench seamless [số_mẫu]
        if [[ "$ENGINE" == "seamless" ]]; then
            NUM_SAMPLES="${3:-10}"
            TAG="case_5_seamless_direct_seamless-m4t-v2-large"
            rm -f "outputs/benchmark/case_5/metrics.json"
            echo "🧪 Benchmark SeamlessM4T-v2-large (Direct S2ST, $NUM_SAMPLES mẫu)"
            bash "$0" eval 5 both "$NUM_SAMPLES" auto 2>&1 | tee "$LOG_DIR/bench_${TAG}.log" || true
            if [ ! -f "outputs/benchmark/case_5/metrics.json" ]; then
                echo "❌ Seamless lỗi — KHÔNG đẩy lên Drive. Xem log: $LOG_DIR/bench_${TAG}.log"
                exit 1
            fi
            cp "$LOG_DIR/bench_${TAG}.log" "outputs/benchmark/case_5/bench.log"
            rclone copy "outputs/benchmark/case_5" "gdrive:runs/benchmark/${TAG}/" -P \
                && echo "✓ Đã đẩy $TAG lên Drive." || { echo "❌ rclone lỗi"; exit 1; }
            exit 0
        fi
        if [[ -z "$ENGINE" || -z "$MODE" || -z "$MODEL_ID" ]]; then
            echo "❌ Thiếu tham số. Cú pháp: bash run.sh bench <xtts|f5> <2step|3step> <model_hf_id> [số_mẫu]"
            echo "   VD: bash run.sh bench xtts 3step vinai/PhoWhisper-large 10"
            exit 1
        fi
        case "$ENGINE" in
            xtts) CASE_NUM=6; TTS_TAG="xtts" ;;
            f5|f5tts) CASE_NUM=7; TTS_TAG="f5tts" ;;
            *) echo "❌ ENGINE phải là xtts hoặc f5"; exit 1 ;;
        esac
        case "$MODE" in
            2step) MODEL_FLAG="--s2tt-models" ;;
            3step) MODEL_FLAG="--vi-asr-models" ;;
            *) echo "❌ MODE phải là 2step hoặc 3step"; exit 1 ;;
        esac
        SHORT_NAME="${MODEL_ID##*/}"
        TAG="case_${CASE_NUM}_${TTS_TAG}_${MODE}_${SHORT_NAME}"
        echo "🧪 Benchmark 1 biến thể: $TAG ($NUM_SAMPLES mẫu)"
        # Xóa metrics cũ để nếu lần chạy này lỗi thì không đẩy nhầm kết quả cũ lên Drive
        rm -f "outputs/benchmark/${TAG}/metrics.json"

        # Chạy qua nhánh eval (cài deps + gọi eval_benchmark.py); log riêng cho từng biến thể
        bash "$0" eval "$ENGINE" "$MODE" "$NUM_SAMPLES" auto "$MODEL_FLAG" "$MODEL_ID" 2>&1 | tee "$LOG_DIR/bench_${TAG}.log" || true

        RESULT_DIR="outputs/benchmark/${TAG}"
        if [ ! -f "$RESULT_DIR/metrics.json" ]; then
            echo "❌ Không thấy $RESULT_DIR/metrics.json — benchmark lỗi, KHÔNG đẩy lên Drive. Xem log: $LOG_DIR/bench_${TAG}.log"
            exit 1
        fi
        cp "$LOG_DIR/bench_${TAG}.log" "$RESULT_DIR/bench.log"
        echo "☁️ Đang đẩy kết quả lên gdrive:runs/benchmark/${TAG}/ ..."
        if rclone copy "$RESULT_DIR" "gdrive:runs/benchmark/${TAG}/" -P; then
            echo "✓ Đã đẩy $TAG lên Drive. Khi đủ các biến thể, chạy: bash run.sh analyze"
        else
            echo "❌ rclone lỗi — kết quả vẫn nằm ở $RESULT_DIR (kiểm tra 'rclone lsd gdrive:')"
            exit 1
        fi
        ;;

    bench-list)
        # In ra toàn bộ lệnh của bộ benchmark đầy đủ (12 biến thể) để copy sang từng notebook
        N="${2:-10}"
        echo "📋 13 lệnh benchmark độc lập ($N mẫu mỗi lệnh) — mỗi lệnh chạy trên 1 notebook:"
        echo "  bash run.sh bench seamless $N"
        for ENG in xtts f5; do
            for M in openai/whisper-small openai/whisper-large-v3 vinai/PhoWhisper-small vinai/PhoWhisper-large; do
                echo "  bash run.sh bench $ENG 2step $M $N"
            done
            for M in vinai/PhoWhisper-small vinai/PhoWhisper-large; do
                echo "  bash run.sh bench $ENG 3step $M $N"
            done
        done
        echo "👉 Xong hết thì chạy: bash run.sh analyze"
        ;;

    analyze)
        # Gom metrics.json của mọi biến thể từ Drive về và phân tích
        ANALYSIS_DIR="outputs/benchmark_all"
        mkdir -p "$ANALYSIS_DIR"
        echo "📥 Đang tải metrics.json + bench.log từ gdrive:runs/benchmark/ ..."
        rclone copy "gdrive:runs/benchmark/" "$ANALYSIS_DIR/" --include "*/metrics.json" --include "*/bench.log" -P \
            || echo "⚠️ Không tải được từ Drive — phân tích các kết quả đang có sẵn trong $ANALYSIS_DIR và outputs/benchmark"
        python tools/analyze_benchmark.py "$ANALYSIS_DIR" outputs/benchmark --out "$ANALYSIS_DIR"
        ;;

    plot)
        echo "📊 Đang vẽ đồ thị tiến trình loss cho thử nghiệm '$EXP_NAME'..."
        python tools/plot_loss.py "$EXP_NAME"
        ;;

    *)
        echo "=========================================================="
        echo "💡 CÚ PHÁP CHUẨN ĐÚNG CÁC LỆNH (S2ST VI -> EN):"
        echo "  bash run.sh sanity               -> Chạy trọn bộ EDA & Sanity Check (Vocos + Whisper)"
        echo "  bash run.sh overfit [cfm|bridge] -> Overfit 500 bước trên 16 câu & xuất audio nghe thử"
        echo "  bash run.sh compare-decoder [bước] [biến_thể] -> So sánh kiến trúc decoder (mặc định 3000 bước, v1,v2-small,v2,v2+ln)"
        echo "  bash run.sh prepare-phost [split] [giờ] -> Cắt câu PhoST + manifest (mặc định train, 50 giờ)"
        echo "  bash run.sh compare-encoder [bước] [encoders] -> EXP02: so encoder phía đích (PhoWhisper vs whisper-small)"
        echo "  bash run.sh exp03 [bước]          -> EXP03: lưới dữ liệu (FLEURS | +PhoST) × encoder đích"
        echo "  bash run.sh start cfm_en         -> Huấn luyện Stage 1 OT-CFM Tiếng Anh"
        echo "  bash run.sh start bridge_vien    -> Huấn luyện Stage 2 CrossAttentionBridge Vi -> En"
        echo "  bash run.sh logs [tên_exp]       -> Xem log trực tiếp"
        echo "  bash run.sh plot [tên_exp]       -> Vẽ biểu đồ ASCII và xuất file ảnh SVG/PNG đồ thị loss"
        echo "  bash run.sh eval sota both 10    -> Benchmark cả F5-TTS & XTTS-v2 ở 2 chế độ (2-step & 3-step)"
        echo "  bash run.sh eval xtts [both|2step|3step] -> Benchmark riêng XTTS-v2 Voice Cloning"
        echo "  bash run.sh eval f5 [both|2step|3step]   -> Benchmark riêng F5-TTS Flow-Matching"
        echo "  bash run.sh eval all             -> Benchmark toàn bộ hệ sinh thái (Case 0, 6, 7, 1, 2, 3, 4)"
        echo "  bash run.sh bench-list [N]       -> In 13 lệnh benchmark độc lập (mỗi notebook chạy 1 lệnh)"
        echo "  bash run.sh bench seamless [N]   -> Chạy SeamlessM4T-v2 (Direct S2ST) + tự đẩy lên Drive"
        echo "  bash run.sh bench <xtts|f5> <2step|3step> <model> [N] -> Chạy 1 biến thể + tự đẩy lên Drive"
        echo "  bash run.sh analyze              -> Gom kết quả mọi biến thể từ Drive + xuất báo cáo phân tích"
        echo "  bash run.sh stop [tên_exp]       -> Dừng huấn luyện"
        echo "  bash run.sh sync [tên_exp]       -> Đẩy checkpoint lên Drive"
        echo "  bash run.sh restore [tên_exp]    -> Tải checkpoint từ Drive về"
        echo "  bash run.sh upload               -> Đẩy outputs (audio, HTML) lên GitHub để tải về"
        echo "  bash run.sh rclone-token 'eyJ0b2tlbi...' -> Ghi token Drive mới (sau khi 'rclone authorize' trên Windows)"
        echo "  bash run.sh rclone-export        -> In dòng RCLONE_B64 mới để cập nhật MOLAB_1CLICK_SETUP.md"
        echo "=========================================================="
        exit 1
        ;;
esac

