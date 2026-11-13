import os
import time
import shutil
import threading
from pathlib import Path

class DriveBackup:
    """
    Worker tự động sao lưu checkpoint và logs từ Molab lên Google Drive.
    Sử dụng cơ chế 2-slot luân phiên (slot-0, slot-1) để chống hỏng file khi upload dở dang.
    """
    def __init__(self, fs, work, remote_root, interval=120):
        self.fs = fs
        self.work = Path(work)
        self.remote_root = remote_root.strip("/")
        self.interval = interval
        self._running = False
        self._thread = None
        self._last_mtime = 0

    def sync_once(self, force=False):
        ckpt_dir = self.work / "checkpoints"
        ckpt_path = ckpt_dir / "model-last.pth"
        if not ckpt_path.exists():
            # Thử tìm các file .pth khác trong checkpoints/ hoặc stage1_artifacts/
            artifacts_dir = self.work / "stage1_artifacts"
            if artifacts_dir.exists():
                candidates = list(artifacts_dir.glob("*.pth")) + list(artifacts_dir.glob("*.pt"))
                if candidates:
                    ckpt_path = candidates[0]
                else:
                    return False
            else:
                return False

        mtime = ckpt_path.stat().st_mtime
        if not force and mtime <= self._last_mtime:
            return False

        slot = "slot-0" if int(time.time() // self.interval) % 2 == 0 else "slot-1"
        target_dir = f"{self.remote_root}/{slot}"
        
        print(f"[DriveBackup] Đang đồng bộ checkpoint sang Drive: {target_dir}...")
        self.fs.makedirs(target_dir, exist_ok=True)
        self.fs.put(str(ckpt_path), f"{target_dir}/{ckpt_path.name}")
        
        # Đồng bộ thêm file log nếu có
        for log_candidate in [self.work / "logs" / "train.log", self.work / "stage1_artifacts" / "sanity_report.json"]:
            if log_candidate.exists():
                try:
                    self.fs.put(str(log_candidate), f"{target_dir}/{log_candidate.name}")
                except Exception as e:
                    print(f"[DriveBackup] Lỗi đồng bộ log phụ: {e}")

        self._last_mtime = mtime
        print(f"✓ [DriveBackup] Hoàn tất sao lưu sang {slot}!")
        return True

    def restore_latest(self):
        """Khôi phục checkpoint từ Google Drive về máy local của Molab."""
        for slot in ["slot-0", "slot-1"]:
            remote_slot = f"{self.remote_root}/{slot}"
            try:
                files = self.fs.ls(remote_slot) if hasattr(self.fs, "ls") else []
            except Exception:
                files = []

            for f in files:
                fname = os.path.basename(f)
                if fname.endswith(".pth") or fname.endswith(".pt"):
                    dest_dir = self.work / "checkpoints"
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    print(f"[DriveBackup] Đang tải {fname} từ {f} về máy...")
                    self.fs.get(f, str(dest_dir / fname))
                    print(f"✓ [DriveBackup] Đã khôi phục checkpoint {fname} thành công!")
                    return True
        print("[DriveBackup] Không tìm thấy bản checkpoint nào trên Drive.")
        return False

    def _loop(self):
        while self._running:
            try:
                self.sync_once()
            except Exception as e:
                print(f"[DriveBackup Cảnh báo] {e}")
            time.sleep(self.interval)

    def start(self):
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            print("[DriveBackup] Tiến trình backup ngầm đã BẮT ĐẦU.")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            print("[DriveBackup] Tiến trình backup ngầm đã DỪNG.")

