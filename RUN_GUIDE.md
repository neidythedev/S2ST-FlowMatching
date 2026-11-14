# 🚀 CẨM NANG TOÀN TẬP LỆNH CHẠY S2ST (EN-VI) TRÊN MOLAB

Tài liệu tổng hợp toàn bộ các nhánh Git (Branches), quy trình thiết lập môi trường, các lệnh thực thi huấn luyện và đánh giá thực nghiệm cho Khóa luận tốt nghiệp.

---

## 📌 1. CÁC NHÁNH GIT (GIT BRANCHES) VÀ VAI TRÒ

| Tên nhánh (Branch) | Mục đích & Vai trò |
| :--- | :--- |
| **`main`** | **Nhánh chính thức (Production)**: Chứa toàn bộ mã nguồn ổn định nhất, đồng bộ đầy đủ các tính năng. |
| **`ablation-study`** | **Nghiên cứu Thực nghiệm**: Nhánh chuyên dụng chạy Benchmark 5 Cases (Case 0 Cascaded Baseline, Case 1-4 Ablation), đo RTF, Mel L1, SECS, ASR-BLEU và sinh Web Audio Gallery. |
| **`train-cfm`** | **Huấn luyện Stage 1**: Nhánh tập trung tối ưu bộ giải mã âm học Optimal Transport Conditional Flow Matching (OT-CFM) trên âm thanh tiếng Việt. |

> 💡 **Ghi chú**: Hiện tại cả 3 nhánh đã được đồng bộ chuẩn hóa (`git pull` trên bất kỳ nhánh nào cũng có đầy đủ tính năng mới nhất).

---

## 🔑 2. THIẾT LẬP RCLONE GOOGLE DRIVE TRÊN CONTAINER MỚI

Khi khởi động một phiên làm việc mới trên Molab, cấu hình Rclone sẽ bị reset. Dán lệnh sau vào terminal để kết nối Google Drive:

```bash
mkdir -p ~/.config/rclone
cat << 'EOF' > ~/.config/rclone/rclone.conf
[gdrive]
type = drive
client_id = <CLIENT_ID_CỦA_BẠN>
client_secret = <CLIENT_SECRET_CỦA_BẠN>
scope = drive
token = <TOKEN_JSON_CỦA_BẠN>
EOF
```

Kiểm tra kết nối thành công:
```bash
rclone lsd gdrive:
```

---

## 📥 3. QUẢN LÝ CHECKPOINT (KÉO VỀ & ĐỒNG BỘ LÊN DRIVE)

### A. Kéo Checkpoints từ Google Drive về máy Molab:
```bash
mkdir -p checkpoints
rclone copy "gdrive:runs/bridge/checkpoints/" checkpoints/ -P
rclone copy "gdrive:runs/cfm/checkpoints/" checkpoints/ -P

# Kiểm tra xem đã có đủ file chưa:
ls -lh checkpoints/
# Kỳ vọng thấy: bridge-best.pth và cfm-best.pth
```

### B. Tự động đồng bộ Checkpoints & Logs lên Google Drive:
```bash
# Đồng bộ tiến trình của bridge:
bash run.sh sync bridge

# Hoặc đồng bộ tiến trình của cfm:
bash run.sh sync cfm
```

---

## 📊 4. CHẠY BENCHMARK ĐÁNH GIÁ (EVALUATION & METRICS)

Lệnh chạy đánh giá thực nghiệm chuẩn hóa trên tập **Google FLEURS Test Set**:

### Chạy toàn bộ 5 kịch bản Ablation (Khuyến nghị cho KLTN):
```bash
git pull
bash run.sh eval all
```

### Chạy riêng từng kịch bản cụ thể:
* **Case 0 (Pretrained Cascaded Baseline - Whisper + OpusMT + MMS-TTS)**:
  ```bash
  python eval_benchmark.py --case 0 --num-samples 10
  ```
* **Case 1 (Zero-Baseline: Chưa train Bridge, Chưa train CFM)**:
  ```bash
  python eval_benchmark.py --case 1 --num-samples 10
  ```
* **Case 2 (Acoustic Only: Chưa train Bridge, Đã train CFM)**:
  ```bash
  python eval_benchmark.py --case 2 --num-samples 10
  ```
* **Case 3 (Semantic Only: Đã train Bridge, Chưa train CFM)**:
  ```bash
  python eval_benchmark.py --case 3 --num-samples 10
  ```
* **Case 4 (Full Proposed System: Hệ thống đề xuất hoàn chỉnh)**:
  ```bash
  python eval_benchmark.py --case 4 --num-samples 10
  ```

---

## 🏋️ 5. HUẤN LUYỆN MÔ HÌNH (TRAINING WORKFLOW)

Hệ thống hỗ trợ chạy ngầm hoàn toàn độc lập (không sợ mất kết nối mạng hay tắt tab trình duyệt):

| Tác vụ | Lệnh thực thi | Mô tả |
| :--- | :--- | :--- |
| **Kiểm tra tiền huấn luyện (Sanity Check)** | `bash run.sh sanity` | Chạy overfit 50 steps trên 4 mẫu cố định để kiểm chứng luồng gradient thông suốt. |
| **Khởi chạy ngầm Stage 2 (Bridge)** | `bash run.sh start bridge` | Tự động nạp Pretrained CFM, đóng băng CFM, và huấn luyện sạch ToneAwareBridge từ đầu. |
| **Khởi chạy ngầm Stage 1 (CFM)** | `bash run.sh start cfm` | Tự động chạy `train_cfm.py` với cấu hình `configs/stage1_cfm.yaml`. |
| **Xem log trực tiếp** | `bash run.sh logs bridge` | Theo dõi Loss, Epoch, Step thời gian thực (Ấn `Ctrl+C` để thoát xem). |
| **Kiểm tra trạng thái** | `bash run.sh status bridge` | Báo tiến trình còn chạy (PID) hay đã dừng. |
| **Dừng an toàn** | `bash run.sh stop bridge` | Gửi tín hiệu dừng graceful, lưu checkpoint cuối. |

---

## 🌐 6. XEM BÁO CÁO & NGHE THỬ AUDIO MẪU

Sau khi chạy xong lệnh `eval`, kết quả sẽ nằm tại thư mục `outputs/benchmark/`:
1. **File Báo cáo Markdown**: `outputs/benchmark/ablation_summary.md` (chứa bảng tổng hợp số liệu RTF, Mel L1, SECS, ASR-BLEU sẵn sàng copy vào Luận văn).
2. **File Web Audio Gallery**: `outputs/benchmark/index.html` (chứa trình nghe trực tiếp so sánh 4 cột âm thanh: Audio En $\leftrightarrow$ Audio Cascaded $\leftrightarrow$ Audio FlowMatching $\leftrightarrow$ Audio Vi chuẩn).

