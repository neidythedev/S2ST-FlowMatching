# 📌 CHEATSHEET LỆNH THAO TÁC S2ST (VIỆT ➔ ANH) TRÊN MOLAB GPU
> **Cẩm nang tra cứu nhanh các lệnh thực thi từ A đến Z cho dự án S2ST-FlowMatching (Branch: `s2st-vi-en`)**

---

## ⚡ 1. KHỞI ĐỘNG PHIÊN MỚI TRÊN MOLAB (1 LỆNH COPY & DÁN DUY NHẤT)
Mỗi khi khởi động lại session Molab, bạn có thể copy đoạn script 1-chạm (đã điền sẵn token Google Drive trong file nội bộ `MOLAB_1CLICK_SETUP.md`) dán thẳng vào terminal Molab.

Hoặc nếu thao tác thủ công, chạy cụm lệnh sau:

```bash
# 1. Clone hoặc chuyển sang nhánh s2st-vi-en mới nhất
REPO_DIR="/marimo/S2ST-FlowMatching"
[ -d "$REPO_DIR" ] || REPO_DIR="$HOME/S2ST-FlowMatching"
if [ ! -d "$REPO_DIR" ]; then git clone -b s2st-vi-en https://github.com/neidythedev/S2ST-FlowMatching.git "$REPO_DIR"; fi
cd "$REPO_DIR"
git fetch origin && git checkout s2st-vi-en && git pull origin s2st-vi-en

# 2. Cấu hình tự động môi trường (Rclone, Google Drive, PyTorch 3.13 mock, dependencies)
bash setup_molab.sh
```
*(Nếu cần paste cấu hình Rclone trực tiếp, xem nội dung file bí mật `MOLAB_1CLICK_SETUP.md` trên máy cá nhân).*

---

## 🧪 2. EDA & SANITY CHECK (KIỂM TRA SINH TỬ TRƯỚC KHI TRAIN)

| Mục tiêu kiểm tra | Lệnh thực thi | Ý nghĩa & Tiêu chuẩn đạt |
| :--- | :--- | :--- |
| **Trọn bộ EDA + Vocos + Whisper** | `bash run.sh sanity` | Kiểm tra độ dài En/Vi, Vocos tái tạo âm thanh trong vắt (Mel L1 < 0.35) và Whisper nghe tiếng Việt WER < 20%. |
| **Overfit 16 câu Stage 1 CFM** | `bash run.sh overfit cfm` | Cho OT-CFM học thuộc lòng 16 câu tiếng Anh (500 bước), xuất file `.wav` tại `outputs/sanity/cfm_en_overfit16_sample0.wav`. |
| **Overfit 16 câu Stage 2 Bridge** | `bash run.sh overfit bridge` | Cho Bridge học đảo ngữ pháp 16 cặp câu Vi-En (500 bước), xuất file `.wav` tại `outputs/sanity/bridge_vi_en_overfit16_sample0.wav`. |

---

## 🚀 3. HUẤN LUYỆN MÔ HÌNH (TRAINING WORKFLOW)

### Giai đoạn 1: Huấn luyện OT-CFM Tiếng Anh (`cfm_en`)
```bash
# Khởi chạy huấn luyện ngầm Stage 1 (Acoustic Teacher Tiếng Anh)
bash run.sh start cfm_en

# Bắt đầu train mới hoàn toàn (bỏ qua checkpoint cũ nếu có)
bash run.sh start cfm_en --fresh

# Xem log trực tiếp
bash run.sh logs cfm_en

# Kiểm tra trạng thái đang chạy hay đã xong
bash run.sh status cfm_en

# Dừng huấn luyện khi cần
bash run.sh stop cfm_en
```

### Giai đoạn 2: Huấn luyện CrossAttentionBridge Vi ➔ En (`bridge_vien`)
```bash
# Khởi chạy huấn luyện ngầm Stage 2 (Cầu nối dịch thuật Vi -> En)
bash run.sh start bridge_vien

# Bắt đầu train mới hoàn toàn từ đầu
bash run.sh start bridge_vien --fresh

# Xem log trực tiếp
bash run.sh logs bridge_vien

# Kiểm tra trạng thái
bash run.sh status bridge_vien

# Dừng huấn luyện
bash run.sh stop bridge_vien
```

---

## ☁️ 4. ĐỒNG BỘ DỮ LIỆU GOOGLE DRIVE (RCLONE CLOUD SYNC)

```bash
# Đẩy toàn bộ Checkpoints, Logs và Outputs lên Google Drive (rất nên chạy sau khi train xong)
bash run.sh sync cfm_en
bash run.sh sync bridge_vien

# Khôi phục Checkpoints từ Google Drive về máy Molab (khi bật session mới)
bash run.sh restore cfm_en
bash run.sh restore bridge_vien
```

---

## 📊 5. BENCHMARK ĐO METRIC & XUẤT AUDIO GALLERY

| Kịch bản đánh giá | Lệnh thực thi | Mục đích đối sánh |
| :--- | :--- | :--- |
| **Case 0: Cascaded Baseline** | `bash run.sh eval 0` | Chạy hệ thống ghép kênh: Whisper Vi $\rightarrow$ Opus-MT $\rightarrow$ MMS English TTS. |
| **Case 5: Meta SeamlessM4T-v2** | `bash run.sh eval 5` | Chạy mô hình SOTA Direct S2ST toàn cầu của Meta (`vie -> eng`). |
| **Case 4: Full Proposed System** | `bash run.sh eval 4` | Đánh giá hệ thống đề xuất hoàn chỉnh (Bridge + CFM). |
| **Đánh giá toàn bộ 5 Cases** | `bash run.sh eval all` | Chạy lần lượt từ Case 0 đến Case 4, xuất bảng Markdown so sánh các chỉ số. |

> 🎧 **Xem và nghe thử kết quả audio trực quan**:
> Sau khi chạy `eval`, mở file **`outputs/benchmark/index.html`** trên trình duyệt để nghe đối chiếu 4 cột âm thanh:
> `🇻🇳 Giọng Việt nguồn` ➔ `⚙️ Cascaded Baseline` ➔ `✨ Đề xuất (Vi-En S2ST)` ➔ `🎯 Giọng Anh chuẩn FLEURS`.

---

## 🛠️ 6. XỬ LÝ LỖI NHANH (TROUBLESHOOTING)

* **Lỗi `torchaudio undefined symbol`**:
  ```bash
  pip uninstall torchaudio -y
  ```
  *(Các file script của chúng ta đã có sẵn shim in-memory mock nên hoàn toàn không cần cài lại torchaudio).*
* **Lỗi `Không khớp danh sách tham số optimizer`**:
  - Dùng cờ `--fresh` khi start: `bash run.sh start bridge_vien --fresh` (script sẽ tự xóa checkpoint cũ và train từ đầu trơn tru).
* **Kiểm tra GPU đang chạy**:
  ```bash
  nvidia-smi
  ```
