# KẾ HOẠCH TÁC CHIẾN NGHIÊN CỨU & VIẾT BÀI BÁO KHOA HỌC (8 TUẦN)
> **Đề tài**: *Tone-Aware Expressive English-to-Vietnamese Speech-to-Speech Translation via Continuous Flow Matching*  
> **Tác giả / Nhóm nghiên cứu**: Sinh viên & Nhóm nghiên cứu MoLab  
> **Hạ tầng sử dụng**: Laptop cá nhân (RTX 4050 6GB) & Server MoLab (NVIDIA RTX PRO 6000 Blackwell Server Edition 48GB)

---

## 1. TỔNG QUAN LỘ TRÌNH 8 TUẦN (MASTER ROADMAP)

```
[Tuần 1] THIẾT LẬP DỮ LIỆU & NẠP STREAMING
   └── Kiểm tra FLEURS (en_us -> vi_vn), xây dựng hàm trích xuất Mel & Speaker embedding.

[Tuần 2] XÂY DỰNG CÁC BASELINES ĐỐI CHỨNG
   └── Chạy đo hệ thống Cascade SOTA (Whisper + NLLB + VITS) và Untrained Model.

[Tuần 3 - 4] HUẤN LUYỆN CHÍNH THỨC TRÊN MOLAB RTX 6000
   └── Huấn luyện 30 - 40 Epochs, lưu Best Checkpoints, chạy Ablation Studies.

[Tuần 5] ĐO ĐẠC TOÀN DIỆN 4 TRỤC METRICS
   └── ASR-BLEU, ASR-WER, UTMOS (tự nhiên), SIM-O (giữ giọng), F0-PCC (ngữ điệu), Tone-Acc.

[Tuần 6 - 7] VIẾT BẢN THẢO BÀI BÁO BẰNG LATEX (OVERLEAF)
   └── Hoàn thiện 6 - 8 trang chuẩn IEEE / Springer / Interspeech template.

[Tuần 8] PHẢN BIỆN NỘI BỘ, SỬA ĐỔI & NỘP BÀI (SUBMISSION)
   └── Nộp vào hội nghị RIVF / O-COCOSDA / KSE / INTERSPEECH.
```

---

## 2. KẾ HOẠCH CHI TIẾT TỪNG GIAI ĐOẠN

### 🎯 GIAI ĐOẠN 1: DỮ LIỆU & TIỀN XỬ LÝ (TUẦN 1)
- **Mục tiêu**: Chuẩn bị luồng dữ liệu song ngữ Anh - Việt, đảm bảo nạp streaming mượt mà trên MoLab.
- **Công việc chi tiết**:
  1. *Ngày 1 - 2*: Kết nối tập dữ liệu **Google FLEURS (`google/fleurs`)**:
     - Tải tập Train (~2,500 câu), Validation (~400 câu), Test (~800 câu) cho 2 ngôn ngữ `en_us` và `vi_vn`.
  2. *Ngày 3 - 4*: Chuẩn hóa pipeline trích xuất:
     - Audio tiếng Anh: Resample về 16kHz đưa qua `Whisper-large-v3` (hoặc `small`).
     - Audio tiếng Việt: Trích xuất Log Mel-spectrogram 80 kênh (hop_length=256, win_length=1024).
     - Giọng nói người nói: Trích xuất vector 192 chiều bằng `speechbrain/spkrec-ecapa-voxceleb`.
  3. *Ngày 5 - 7*: Kiểm tra Dataloader trên MoLab RTX 6000. Đo tốc độ nạp dữ liệu theo batch (16 - 32 samples).

---

### 🎯 GIAI ĐOẠN 2: THI CÔNG HỆ THỐNG BASELINE ĐỐI CHỨNG (TUẦN 2)
- **Mục tiêu**: Thu thập số liệu của các phương pháp hiện hành để đưa vào bảng so sánh trong bài báo.
- **Công việc chi tiết**:
  1. *Ngày 8 - 10*: Xây dựng và đo đạc **Cascade SOTA Baseline**:
     - Ghép nối: `Whisper-large-v3 (ASR)` $\rightarrow$ `facebook/nllb-200-distilled-600M (MT)` $\rightarrow$ `VITS Tiếng Việt (TTS)`.
     - Chạy toàn bộ 800 câu tập Test.
     - Thu thập: File âm thanh kết quả, đo thời gian tính toán trung bình (RTF) và ASR-BLEU.
  2. *Ngày 11 - 12*: Đo đạc **Mốc Untrained Model** (Trọng số ngẫu nhiên ban đầu):
     - Xác nhận ASR-BLEU = 0.0, UTMOS < 1.5, SIM-O < 0.1 (chứng minh tính hợp lệ của việc huấn luyện).
  3. *Ngày 13 - 14*: Tổng hợp số liệu ban đầu vào Bảng thực nghiệm nháp.

---

### 🎯 GIAI ĐOẠN 3: HUẤN LUYỆN TRÊN MOLAB VỚI GPU RTX 6000 (TUẦN 3 - 4)
- **Mục tiêu**: Huấn luyện mô hình đạt độ hội tụ cao, Mel sinh ra sắc nét.
- **Cấu hình Huấn luyện**:
  - Model: `EnViS2STSystem` (ToneAwareBridge 6 layers + FlowMatchingTargetDecoder).
  - Kiểu dữ liệu: `bfloat16` trên GPU RTX PRO 6000 Blackwell Server Edition.
  - Batch size: 16 hoặc 32.
  - Optimizer: AdamW (`lr=2e-4`, `weight_decay=1e-2`, $\beta_1=0.9, \beta_2=0.98$).
  - Số epochs: 30 - 40 epochs (ước tính mất khoảng **18 - 30 giờ GPU** chạy liên tục).
- **Công việc chi tiết**:
  1. *Tuần 3*: Chạy huấn luyện chính thức, theo dõi đồ thị hàm mất mát ($\mathcal{L}_{\text{CFM}}$ giảm từ $\sim 2.3$ xuống $< 0.35$). Lưu file `s2st_en_vi_best.pt`.
  2. *Tuần 4*: Huấn luyện các biến thể cho phần **Ablation Study**:
     - *Biến thể 1 (w/o Tone Head)*: Tắt Tone Classifier, chỉ train CFM thuần túy.
     - *Biến thể 2 (w/o Downsample)*: Giữ nguyên tốc độ frame 50Hz để so sánh tốc độ và bộ nhớ.

---

### 🎯 GIAI ĐOẠN 4: ĐO ĐẠC 4 TRỤC METRICS & CHỌN MẪU ÂM THANH (TUẦN 5)
- **Mục tiêu**: Có đầy đủ số liệu định lượng và định tính cho bài báo.
- **Công việc chi tiết**:
  1. *Ngày 29 - 31*: Chạy Inference toàn bộ 800 câu tập Test với bộ giải **Euler ODE 25 bước**.
  2. *Ngày 32 - 33*: Tính toán tự động 4 nhóm chỉ số:
     - **Trục 1 (Dịch thuật)**: Dùng PhoWhisper nhận dạng audio sinh ra $\rightarrow$ tính điểm **ASR-BLEU** và **ASR-WER**.
     - **Trục 2 (Tự nhiên)**: Tính điểm **UTMOS** (Universal TTS MOS) tự động cho 800 câu.
     - **Trục 3 (Giữ giọng & Ngữ điệu)**:
       - Tính khoảng cách cosin **SIM-O** (WavLM / ECAPA) giữa giọng gốc và giọng dịch.
       - Tính tương quan cao độ **F0-PCC** (Pearson Correlation của đường cong $F_0$).
       - Đo tỷ lệ phần trăm phát âm chuẩn dấu **Tone-Accuracy (%)**.
     - **Trục 4 (Hiệu năng)**: Đo chính xác hệ số thời gian thực **RTF** ($T_{\text{calc}} / T_{\text{audio}}$).
  3. *Ngày 34 - 35*: Chọn ra 5 cặp câu audio mẫu ấn tượng nhất để upload lên GitHub Pages (làm Demo Web đính kèm bài báo).

---

### 🎯 GIAI ĐOẠN 5: VIẾT BẢN THẢO BÀI BÁO (TUẦN 6 - 7)
- **Mục tiêu**: Hoàn thành bài báo 6 - 8 trang bằng **LaTeX (Overleaf)** theo chuẩn IEEE / Springer / Interspeech.
- **Bố cục các mục trong bài báo**:
  1. **Abstract (150 - 200 từ)**: Nêu bài toán, điểm nghẽn của Cascade, giải pháp Tone-CFM và các con số kết quả ấn tượng.
  2. **Section 1: Introduction (1.5 trang)**: Đặt vấn đề, phân tích xung đột ngôn ngữ Non-tonal $\rightarrow$ Tonal, liệt kê 3 đóng góp (*Our Contributions*).
  3. **Section 2: Related Work (1 trang)**: Trích dẫn tổng quan từ tài liệu `S2ST_CONTINUOUS_FLOW_MATCHING_SURVEY.md`.
  4. **Section 3: Proposed Architecture (2 trang)**: Trình bày chi tiết công thức toán học của Optimal Transport CFM ($u_t = x_1 - x_0$), hàm mất mát đa nhiệm Tone Loss, và bộ giải Euler ODE.
  5. **Section 4: Experiments & Setup (1 trang)**: Dataset FLEURS, hạ tầng MoLab RTX 6000, các siêu tham số.
  6. **Section 5: Results & Discussion (1.5 trang)**: Bảng so sánh tổng hợp, bảng bóc tách Ablation Study, biểu đồ phổ Mel và đường cong cao độ $F_0$.
  7. **Section 6: Conclusion (0.5 trang)**: Tóm tắt kết quả và định hướng phát triển tương lai.

---

### 🎯 GIAI ĐOẠN 6: PHẢN BIỆN, SỬA ĐỔI & NỘP BÀI (TUẦN 8)
- **Mục tiêu**: Hoàn thiện bản thảo và nộp bài (Paper Submission).
- **Công việc chi tiết**:
  1. *Ngày 50 - 52*: Gửi bản thảo cho Giảng viên hướng dẫn / Thầy cô trong lab đọc phản biện và góp ý.
  2. *Ngày 53 - 54*: Rà soát ngữ pháp học thuật tiếng Anh, kiểm tra chuẩn danh mục tài liệu tham khảo BibTeX.
  3. *Ngày 55 - 56*: Nộp bài lên hệ thống nộp bài trực tuyến (CMT / OpenReview / EasyChair) của hội nghị mục tiêu.

---

## 3. CÁC HỘI NGHỊ MỤC TIÊU PHÙ HỢP

1. **RIVF** (Hội nghị Quốc tế IEEE về Điện toán và Truyền thông tại Việt Nam) - Kỷ yếu IEEE Xplore.
2. **KSE** (Hội nghị Quốc tế về Kỹ thuật Tri thức và Hệ thống) - Kỷ yếu Springer ISI/Scopus.
3. **O-COCOSDA** (Hội nghị Quốc tế về Xử lý Tiếng nói và Ngôn ngữ Châu Á).
4. **INTERSPEECH / ICASSP** (Hội nghị đỉnh cao thế giới về Xử lý Tiếng nói).

---
*File được lưu tại: `d:\Github\ai biet\S2ST\MASTER_RESEARCH_AND_PAPER_EXECUTION_PLAN.md`.*

