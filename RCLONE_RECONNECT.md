# 🔑 KẾT NỐI RCLONE (GOOGLE DRIVE) TRÊN MOLAB

> File này không chứa token/secret. Cấu hình thật nằm trong `MOLAB_1CLICK_SETUP.md` (máy Windows, đã .gitignore).
> Refresh token của OAuth app ở chế độ *Testing* chỉ sống **~7 ngày** → mỗi tuần làm Bảng 2 một lần.

---

## 📅 BẢNG 1 — TRONG TUẦN (token còn sống, chỉ reset session / đổi GPU)

| Bước | Làm ở đâu | Lệnh / Thao tác | Kết quả đúng |
| :-: | :--- | :--- | :--- |
| 1 | Molab | Dán **khối lệnh trong `MOLAB_1CLICK_SETUP.md`** | Tạo `rclone.conf`, clone/pull repo, cài thư viện |
| 2 | Molab | `rclone lsd gdrive:runs` | Liệt kê `bridge`, `cfm`, `benchmark`... |
| 3 | — | ✅ Xong. Báo `invalid_grant` → token đã chết → sang **Bảng 2** | |

> Bước 1 PHẢI làm trước mọi lệnh rclone. Chưa dán → lỗi `didn't find section in config file` (máy chưa có cấu hình).

---

## 🔄 BẢNG 2 — TUẦN MỚI (token hết hạn, báo `invalid_grant`)

| Bước | Làm ở đâu | Lệnh / Thao tác | Ghi chú |
| :-: | :--- | :--- | :--- |
| 1 | Molab | Dán khối lệnh `MOLAB_1CLICK_SETUP.md` (nếu chưa) | Cần có `[gdrive]` trong cấu hình |
| 2 | **Windows** (PowerShell / tab Terminal của app) | Chạy lệnh `rclone authorize "drive" "eyJjbGllbnRf..."` — lấy **nguyên dòng** ở mục *"Lệnh authorize cố định"* cuối `MOLAB_1CLICK_SETUP.md` | Trình duyệt mở → đăng nhập Google → **Allow** (báo "chưa xác minh" → *Advanced* → *Go to ... (unsafe)*) |
| 3 | Windows | Copy chuỗi giữa `--->` và `<---End paste` | Bắt đầu bằng **`eyJ0b2tlbi`** |
| 4 | Molab | `bash run.sh rclone-token 'eyJ0b2tlbi...'` | Tự ghi token, chép sang cả `/home/marimo` và `/root`, kiểm tra kết nối, **in dòng `RCLONE_B64=...` mới** |
| 5 | **Windows** | Mở `MOLAB_1CLICK_SETUP.md` → dán **đè** dòng `RCLONE_B64="..."` bằng dòng vừa in ở bước 4 | ⚠️ Bỏ bước này → mọi session sau trong tuần lại báo `invalid_grant` |

- Lệnh ở bước 2 **không đổi** giữa các tuần (chỉ chứa client_id/secret của app) → không cần chạy `rclone config reconnect` trên Molab nữa.
- Cần in lại dòng `RCLONE_B64` bất cứ lúc nào: `bash run.sh rclone-export`.
- Cách thủ công (nếu muốn): `rclone config reconnect gdrive:` → `y` → **`n`** → dán `eyJ0b2tlbi...` → `n`, rồi
  `cp /home/marimo/.config/rclone/rclone.conf /root/.config/rclone/rclone.conf` (**chiều marimo → root**).

---

## 🩺 LỖI HAY GẶP

| Triệu chứng | Nguyên nhân | Cách xử lý |
| :--- | :--- | :--- |
| `didn't find section in config file` | Chưa dán khối setup → chưa có `rclone.conf` | Bảng 1, bước 1 |
| `invalid_grant` ngay khi mở session mới | `RCLONE_B64` trong `MOLAB_1CLICK_SETUP.md` còn token cũ | Bảng 2 (nhớ bước 5) |
| `backend doesn't support reconnect` | Gõ thiếu dấu `:` (`gdrive` thay vì `gdrive:`), hoặc chạy `reconnect` trên Windows | `gdrive:` có dấu hai chấm, chạy trên Molab |
| Treo ở `Waiting for code...` | Chọn `y` ở *Use auto config?* | Ctrl+C, dùng Bảng 2 (hoặc chọn **n**) |
| `Couldn't decode response` / `Đây là chuỗi ĐẦU VÀO` | Dán chuỗi `eyJjbGllbnRf...` (đầu vào) thay vì kết quả | Dán chuỗi `eyJ0b2tlbi...` từ Windows |
| `bind: Only one usage of each socket address` | Chạy 2 lệnh `rclone authorize` cùng lúc (cổng 53682) | Đóng cửa sổ cũ, chỉ chạy 1 lệnh |
| `invalid_grant` ngay sau khi vừa reconnect | Đã `cp` ngược chiều root → marimo | Chạy lại `bash run.sh rclone-token '...'` với token vừa lấy |
| Dán lệnh nhiều dòng bị cắt / lẫn chữ (`EOFnt(...`) | Terminal Molab cắt khi dán đoạn dài nhiều dòng | Dùng lệnh 1 dòng (`bash run.sh rclone-token ...`) |
| `run.sh restore/sync` báo "thành công" nhưng không có file | `run.sh` nuốt lỗi rclone (`|| true`) | Kiểm tra bằng `ls -lh checkpoints/` hoặc `rclone ls gdrive:...` |

**Vì sao phải làm mỗi tuần:** OAuth app (project `KLTN`) đang ở chế độ *Testing* → refresh token hết hạn sau 7 ngày.
Chuyển sang *In production* (Google Auth Platform → **Audience** → *Publish app*) đòi hỏi khai báo **domain** ở mục
*Branding* → **quyết định (24/11/2026): giữ Testing, gia hạn token mỗi tuần theo Bảng 2** (~2 phút/lần).

---

## 📁 Dữ liệu trên Drive
- `gdrive:runs/cfm/`, `gdrive:runs/bridge/` = checkpoint **CŨ hướng En→Vi** (13/11) — không dùng cho Vi→En.
- `gdrive:runs/benchmark/<tag>/` = kết quả `bash run.sh bench ...`; gom bằng `bash run.sh analyze`.
