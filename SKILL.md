---
name: molab-drive-experiment-handoff
description: Package, run, monitor, back up, and resume reproducible ML experiments on Molab with Google Drive storage. Use for source ZIP handover, terminal runbooks, marimo Drive cells, CUDA setup failures, and checkpoint recovery; not for unrelated local training or changing a model architecture without a request.
---

# Molab + Google Drive: chuyển giao và vận hành thực nghiệm

Tài liệu tự chứa cho người chạy và coding agent. Gom quy trình đã dùng trong dự án
DDVC: source local → ZIP có provenance → upload Drive → Molab → smoke → train/log →
backup Drive → restore/resume → lấy kết quả. Chỉ những phần liên quan được tổng hợp;
không yêu cầu các skill về ảnh, slide hoặc ứng dụng khác.

## 1. Phạm vi và nguyên tắc cần giữ

- Ưu tiên **terminal CMD/Bash trên Molab**. Cell marimo chỉ dùng để kết nối Drive và
  giữ backup worker; không biến toàn bộ training thành notebook dài.
- Molab local storage/runtime có thể mất. Source ZIP, config, dependency manifest,
  checkpoint, log, prediction và metric phải có cách lưu bền vững.
- Dùng **môi trường Molab hiện có**, kiểm tra interpreter thực tế; không mặc định tạo
  `.venv` mới, không đặt venv trên Google Drive, không tự cài đè GPU Torch/torchvision.
  Session mới vẫn có thể phải dựng lại packages/toolchain từ manifest.
- Google Drive qua `gdrive_fsspec` là remote filesystem, **không phải POSIX mount**.
  `/STaRC`, `/Drive`, `/content/drive` không tự trỏ tới Drive chỉ vì đặt tên như vậy.
- Stage data/checkpoint xuống local để compute. Upload/download bằng `fs.put/get/open`.
  Git clone/pull tại local; xuất snapshot lên Drive. ZIP source không cần chứa `.git`.
- Khi tái hiện baseline, pin source commit và effective config. Phân biệt **runtime
  compatibility fix** với thay model/loss/precision/batch/LR/ranking recipe.
- Không tự thêm cải tiến vào baseline để “cho chạy được”. Không đổi nhiều yếu tố rồi
  quy toàn bộ chênh lệch metric cho một module.
- Workdir, data, bank, checkpoint và output phải cấu hình được bằng CLI/env. Các path
  `/marimo/...` dưới đây là **ví dụ gói hiện tại**, không phải quy tắc cho mọi dự án.
- Không yêu cầu root. Dùng launcher tự tìm workdir và interpreter; lệnh đưa cho người
  dùng nên có path đầy đủ, tránh phụ thuộc `$PY`, `$WORK`, `$RUN` đã mất ở terminal mới.
- Không xoá/giải nén đè source hoặc run đang có checkpoint. Run mới dùng ID/folder riêng.
- Không gọi import-only là smoke hoàn chỉnh. Cần CUDA operator, actual model forward/
  backward, generation, validation, lưu checkpoint; resume smoke khi chạy tiếp.
- Checkpoint phải chứa model, optimizer, scheduler, epoch/iteration, RNG, config và
  trạng thái hoàn tất epoch/đang chờ validation. Không hứa resume chính xác giữa mini-batch.
- Giữ native environment và runtime-specific patches minh bạch. Một CPU test pass
  không phải bằng chứng full GPU training trên Molab đã pass.

Nguồn chính thức: [Molab remote storage](https://molab.marimo.io/blog/drive-storage),
[marimo reactivity](https://docs.marimo.io/guides/reactivity/),
[marimo package management](https://docs.marimo.io/guides/package_management/installing_packages/).
Đọc lại tài liệu/API hiện hành khi runtime hoặc generated connection code thay đổi.

## 2. Chốt một bản chạy trước khi đóng gói

Ghi lại: repo/branch/commit, dataset/split, selected IDs, feature encoder, caption-bank
nguồn train, pretrained components, seed, epochs, optimizer/LR/milestones, batch thật,
accumulation, precision/TF32, query count, caption cost chunk, loss và selection flags.

**Snapshot DDVC hiện tại để làm ví dụ, không áp đặt cho dự án khác:**

| Mục | Giá trị |
|---|---|
| Repo | https://github.com/thloc1212/DDVC |
| Branch | `exp/gt-aware-dedup-softnms` |
| Commit của ZIP bên dưới | `aab51205d7c319f737a36632294757007cc9109b` |
| Mode người dùng đã chọn | `dedup_softnms_merge_consecutive` |
| Data | Full feature-matched: 1237 train / 436 validation |
| Epoch / seed | 15 / 777 |
| Batch train/eval / accumulation | 1 / 1 / 1 |
| Optimizer / LR / weight decay | Adam / `5e-5` / `1e-4` |
| LR milestones / gamma | `[8, 11]` / `0.5` |
| Precision / TF32 | FP32 / off |
| Queries / caption chunk / workers | 50 / 16 / 0 |
| Caption decoder | GPT-2 2 layer, khởi tạo từ đầu |
| Dedup | Mean BCE trên eligible unmatched queries; coef 0.1 từ epoch 1, epoch 0 coef 0 |
| Soft-NMS | Temporal Gaussian, sigma 0.5; không có gradient |
| Overlap fusion | tIoU ≥0.7, caption token Jaccard ≥0.9 |
| Consecutive merge | Cùng caption chuẩn hoá có thứ tự; gap/overlap nhỏ ≤2s, tối đa 3 selected events, span ≤60s |
| Best checkpoint | METEOR + SODA_c sau selection/merge của run |

Raw video không cần cho recipe này. `clipvitl14.pth` ~872 MiB là **video features**,
không phải weights. CLIP ViT-L/14 weights được dùng để encode caption train tạo bank.
Bank đúng split có thể reuse; không dùng validation captions làm train retrieval bank.
Full feature-matched không có nghĩa đủ 1333/457 official IDs: một số video thiếu feature.

Tài liệu chi tiết trong source: `docs/CONSECUTIVE_MERGE_MOLAB.md` và
`docs/DEDUP_SOFTNMS_MOLAB.md`. Nếu source/config đổi, đọc lại các file và tạo ZIP mới;
không tiếp tục dùng checksum/manifest cũ như thể cùng một bản.

## 3. Thứ tự local: source → folder chuyển giao → ZIP

### 3.1 Kiểm tra source trên máy local

PowerShell, từ folder cha chứa repo `DDVC`:

```powershell
git -C DDVC status --short
git -C DDVC branch --show-current
git -C DDVC rev-parse HEAD
git -C DDVC diff --check
```

Commit các file cần chuyển giao trước khi export; `git archive` chỉ chứa tracked files
ở commit được chọn, không tự lấy thay đổi chưa commit/untracked. Không reset hoặc stash
code của người dùng chỉ để làm sạch status. Review danh sách file; không đưa credentials,
`.env`, `.venv`, build/cache, data lớn, bank hoặc checkpoint vào ZIP source.

Test theo thay đổi thực tế. Với nhánh DDVC này: config/recipe guards, dedup gradient và
normalization, selection/merge, eval-save roundtrip, checkpoint/resume và CUDA smoke.
Lưu output test local; tách rõ test CPU với test GPU Molab chưa thực hiện.

### 3.2 Folder ZIP nên có

```text
DDVC_Dedup_SoftNMS_Consecutive/
  source/                         source tại một commit xác định + licenses
    tools/dedup_molab.sh
    tools/molab_pipeline.py
    tools/molab-requirements.txt
    tools/molab_cuda130.py
    tools/drive_backup.py
    tools/kaggle_recipe.py
    smoke_test_*.py
    docs/
  BUNDLE_MANIFEST.json             commit/branch/recipe/runtime expectations
  SKILL.md                        tài liệu chuyển giao này
  SHA256SUMS.txt                  checksum các file trong gói
  VALIDATION_LOCAL.txt            kết quả các test đã thực sự chạy
```

`BUNDLE_MANIFEST.json` nên ghi source commit, repo/branch, data/feature references,
default flags, checkpoint format, runbook và giới hạn validation. Giữ nguyên license
của code thích nghi từ repo khác. Cung cấp companion file `.zip.sha256`.

Windows: dùng `.gitattributes` với `*.sh text eol=lf`. **Kiểm tra bytes của `.sh` trong
ZIP**, không chỉ chạy `bash -n`: CRLF từng gây `set: pipefail: invalid option name`.
Sau đóng gói chạy ZIP CRC test và đối chiếu manifest/source revision. Không ghi đè ZIP
đã phát link; dùng tên version mới nếu thay code.

Ví dụ export source mới bằng PowerShell, dùng tên mới để bảo vệ ZIP đã gửi:

```powershell
if (Test-Path DDVC_Consecutive_rebuild.zip) { throw 'ZIP đã tồn tại; chọn tên mới' }
git -C DDVC archive --format=zip --prefix=DDVC_Dedup_SoftNMS_Consecutive/source/ --output=../DDVC_Consecutive_rebuild.zip HEAD
```

Lệnh trên mới xuất `source/`: cần thêm manifest, tài liệu, test log và checksums như cây
folder đã mô tả trước khi upload. Có thể dùng helper đóng gói được kiểm chứng của task;
không khẳng định `git archive` một mình đã tạo đủ mọi artifact chuyển giao.

### 3.3 Upload và giữ provenance trên Drive

Folder code dài hạn của người dùng:
[1JtVmVYfZYlxbcR96pKMhWqxODi2WkJZI](https://drive.google.com/drive/folders/1JtVmVYfZYlxbcR96pKMhWqxODi2WkJZI).

Upload ZIP, `.zip.sha256` và `SKILL.md` vào folder này bằng Drive UI hoặc filesystem
đã được kết nối. Ghi lại **file ID của ZIP**, không nhầm với folder ID. Không tự đổi
quyền chia sẻ sang public để giải quyết lỗi download; ưu tiên kết nối Drive được cấp quyền.

ZIP hiện tại:
[1DY7uWvgyp8SO3fugF-pNYkDsjHZGuhpJ](https://drive.google.com/file/d/1DY7uWvgyp8SO3fugF-pNYkDsjHZGuhpJ/view).

```text
Filename: DDVC_Dedup_SoftNMS_Consecutive_Molab.zip
SHA256: 094be0716ca8bc1c703fb74835aa520a58ab668db60edcabdbaac0b720c9bdfd
```

Source ZIP và backup run là hai thứ riêng: code giữ ở folder code, artifacts dưới
`runs/<experiment>/slot-*`. Không giả định backup helper sẽ backup toàn bộ source.

## 4. Molab: nhận code và kiểm tra môi trường

### 4.1 Runtime

Chọn GPU + Internet. Kiểm tra interpreter, Torch/CUDA/GPU thực tế trước khi quyết định
stack; đường dẫn native từng dùng là `/tmp/uv-venv/bin/python` nhưng có thể đổi theo runtime.
Không dùng Python 3.10 venv chỉ để né cảnh báo project `requires-python >=3.12` nếu user
đã chọn native environment. Không tự upgrade Transformers legacy/vendor code.

Ví dụ terminal cho môi trường đã xác nhận của dự án:

```bash
/tmp/uv-venv/bin/python -c 'import sys,torch; print(sys.executable); print(torch.__version__,torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO_GPU")'
```

RTX lớn không có nghĩa tự tăng batch: caption head hiện yêu cầu batch thật 1.
Accumulation 8 giảm optimizer updates/epoch khoảng 8 lần, là thay đổi recipe.
BF16/TF32 cũng là thay đổi numerical training, không gọi là runtime fix thuần tuý.

### 4.2 Tải ZIP bằng terminal

```bash
uv pip install --python /tmp/uv-venv/bin/python gdown
/tmp/uv-venv/bin/python -m gdown '1DY7uWvgyp8SO3fugF-pNYkDsjHZGuhpJ' -O /marimo/DDVC_Dedup_SoftNMS_Consecutive_Molab.zip
```

Không dùng `--fuzzy` với gdown version không hỗ trợ; truyền file ID trực tiếp.
Nếu file/folder private trả 401, dùng `fs.get` sau khi kết nối như mục 7:

```python
fs.get('DDVC_Dedup_SoftNMS_Consecutive_Molab.zip', '/marimo/DDVC_Dedup_SoftNMS_Consecutive_Molab.zip')
```

Path remote này chỉ đúng khi `fs` root là folder chứa ZIP và filename thật sự khớp.
Dùng sidebar generated snippet hoặc `fs.find('')` có lọc để tìm đúng file; không đoán path.

### 4.3 Kiểm tra và giải nén, terminal

Copy nguyên block, bao gồm dòng `PY` cuối; không gõ Python trực tiếp vào Bash:

```bash
/tmp/uv-venv/bin/python - <<'PY'
import hashlib, json, zipfile
from pathlib import Path
archive = Path('/marimo/DDVC_Dedup_SoftNMS_Consecutive_Molab.zip')
work = Path('/marimo/DDVC_Dedup_SoftNMS_Consecutive')
expected = '094be0716ca8bc1c703fb74835aa520a58ab668db60edcabdbaac0b720c9bdfd'
assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected, 'ZIP không khớp'
if work.exists():
    raise FileExistsError(f'{work} đã tồn tại; không giải nén đè source/run')
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None, 'ZIP CRC lỗi'
    for name in z.namelist():
        target = (work.parent / name).resolve()
        if not target.is_relative_to(work.resolve()):
            raise ValueError(f'Unsafe ZIP path: {name}')
        if name.endswith('.sh'):
            assert b'\r\n' not in z.read(name), f'CRLF shell script: {name}'
    z.extractall(work.parent)
manifest = json.loads((work / 'BUNDLE_MANIFEST.json').read_text())
assert manifest['source_commit'] == 'aab51205d7c319f737a36632294757007cc9109b'
print('SOURCE_READY:', work)
PY
```

Nếu workdir đã tồn tại, kiểm tra manifest/run để reuse hoặc chọn folder mới; không xoá
checkpoint để lệnh giải nén chạy được. Code được sửa local trên máy tác giả không tự
cập nhật Molab: cần ZIP mới, Git update phù hợp hoặc patch nhỏ có backup/provenance.

## 5. Setup → prepare-smoke → smoke → train-smoke

Mỗi lệnh chạy riêng, kiểm tra thành công rồi mới chạy lệnh tiếp theo.

```bash
LC_ALL=C bash /marimo/DDVC_Dedup_SoftNMS_Consecutive/source/tools/dedup_molab.sh setup full dedup_softnms_merge_consecutive
```

Setup dùng native packages, compatibility patches, Java cho METEOR, compile và CUDA
operator test. Ghi package versions/runtime ở `provenance/`. Không thay Torch để sửa
lỗi compiler. Shell dùng `pipefail` để lỗi trong pipeline không bị `tee` che mất.

```bash
LC_ALL=C bash /marimo/DDVC_Dedup_SoftNMS_Consecutive/source/tools/dedup_molab.sh prepare-smoke full dedup_softnms_merge_consecutive
```

Đợi `CONFIG_READY`. Data download và CLIP caption-bank encoding thuộc prepare,
không phải train. `BANK_PROGRESS ... /9586` ở full là số caption, không phải epoch.

Kiểm tra file trước smoke:

```bash
ls -lh /marimo/DDVC_Dedup_SoftNMS_Consecutive/prepared/yc2_smoke_dedup_softnms_merge_consecutive_e2_b1/config.yml
```

```bash
LC_ALL=C bash /marimo/DDVC_Dedup_SoftNMS_Consecutive/source/tools/dedup_molab.sh smoke full dedup_softnms_merge_consecutive
```

Cần `REAL_MODEL_FORWARD_BACKWARD_PASS` và `GENERATION_POSTPROCESS_PASS`. Khi resume,
cần thêm checkpoint load pass. Smoke benchmark của helper lặp một video, không optimizer
update, không đại diện thời gian full epoch. Muốn kiểm tra optimizer/validation/save thật:

```bash
LC_ALL=C bash /marimo/DDVC_Dedup_SoftNMS_Consecutive/source/tools/dedup_molab.sh train-smoke full dedup_softnms_merge_consecutive
```

Train-smoke hai epoch nhỏ; dedup weight epoch 0 là 0, epoch 1 là 0.1. Không tự tạo
`cuda_ready`/`model_smoke_*_ready` để bỏ qua lỗi; markers phải do test pass tạo.

## 6. Prepare full: kiểm tra config, data và run ID

```bash
LC_ALL=C bash /marimo/DDVC_Dedup_SoftNMS_Consecutive/source/tools/dedup_molab.sh prepare-run full dedup_softnms_merge_consecutive
```

Đợi `KAGGLE_RECIPE_OK`, xác nhận 1237/436 IDs, 15 epochs, LR/batch/selection đúng.
Config/bank/refs nằm dưới `prepared/yc2_full_dedup_softnms_merge_consecutive_e15_b1/`.
Không lấy nhầm config smoke, subset hoặc run cũ. Bank phải từ cùng selected training
annotations; vocabulary và data signature được giữ để kiểm tra resume.

Bật backup bên dưới **trước** train dài. Một checkpoint smoke có thể đã được backup;
nó không thay thế việc full checkpoint được lưu sau này.

## 7. Google Drive: hai cell, không trùng global

Mỗi tên global chỉ được định nghĩa ở một cell trong marimo, bao gồm imports, `fs`,
workdir và backup handle. Nếu đã có `fs`, dùng lại hoặc sửa cell định nghĩa hiện tại;
không thêm một cell khác cùng gán `fs`. Temporary names dùng prefix `_`.

### Cell kết nối — chỉ tạo khi chưa có kết nối phù hợp

```python
import marimo as mo
from gdrive_fsspec import GoogleDriveFileSystem

fs = GoogleDriveFileSystem(
    root_file_id='1JtVmVYfZYlxbcR96pKMhWqxODi2WkJZI',
    use_listings_cache=False,
    skip_instance_cache=True,
    auth_kwargs={'use_local_webserver': False},
)
mo.output.clear_console()
```

Hoàn tất auth bằng generated Molab flow; không đưa token/cookies/.env vào source ZIP.
Kiểm tra root thực tế từ sidebar/generated snippet nếu API/runtime thay đổi.

### Cell backup

```python
import sys as _sys
from pathlib import Path as _Path

_consecutive_work = _Path('/marimo/DDVC_Dedup_SoftNMS_Consecutive')
_sys.path.insert(0, str(_consecutive_work / 'source' / 'tools'))
from drive_backup import DriveBackup as _DriveBackup

consecutive_backup = _DriveBackup(
    fs,
    work=_consecutive_work,
    remote_root='runs/ddvc_dedup_softnms_consecutive_e15',
    interval=120,
)
consecutive_backup.start()
```

Giữ notebook/kernel còn sống; `nohup` của train không duy trì backup worker nếu kernel
chết. Một worker cho một remote root. Trước khi tạo lại handle, gọi `.stop()` trên
handle cũ; không để nhiều thread ghi cùng slot.

### Backup hoạt động thế nào

```text
Drive folder code /
  source ZIP + checksums + SKILL.md
  runs/ddvc_dedup_softnms_consecutive_e15/
    slot-0/
      runs/...
      prepared/...
      logs/...
      provenance/...
      READY.json
    slot-1/
      ...
```

- Hai slot luân phiên; `READY.json` ghi **sau cùng**. Upload hỏng không biến slot đang
  ghi thành backup hoàn tất; slot hoàn tất trước vẫn giữ được.
- Worker poll 120 giây, nhưng chỉ sync khi `model-last.pth` thay đổi hoặc force.
  Không có checkpoint thì `sync_once()` có thể trả False; không coi đó là lỗi.
- Checkpoint được pin/copy ổn định trước network upload; training lưu bằng atomic replace.
- Lưu runs/prepared/logs/provenance, gồm model-best/last, bank, refs, config, metrics.
- Bỏ downloadable video features, HF cache/toolchain, build artifacts, `.pid`, `.tmp`,
  source-backup/tensorboard folders. Source ZIP phải được giữ riêng trên Drive.
- `drive_sync.log` là log controller local và bị helper loại khỏi backup để restore
  không ghi đè file controller đang mở. Training logs vẫn được backup.
- 120 giây là thời gian poll, không bảo đảm upload xong trong 120 giây. Checkpoint
  ~GB và hai slot có thể tốn nhiều dung lượng/network; phải xem `BACKUP_COMPLETE`.

```bash
tail -n 40 -f /marimo/DDVC_Dedup_SoftNMS_Consecutive/logs/drive_sync.log
```

Không gọi mọi snapshot là “đã lưu Drive” chỉ vì `.start()` đã chạy. Cần log complete
và READY marker; khi debug dùng `fs.find`/`fs.open` đúng remote root.

## 8. Train nền, xem log và kiểm tra process

```bash
LC_ALL=C bash /marimo/DDVC_Dedup_SoftNMS_Consecutive/source/tools/dedup_molab.sh train full dedup_softnms_merge_consecutive
```

Launcher dùng unbuffered Python, `nohup`, log và PID file. Có PID không chứng minh
process còn sống: kiểm tra status/log ngay sau launch.

```bash
tail -n 80 -f /marimo/DDVC_Dedup_SoftNMS_Consecutive/logs/yc2_full_dedup_softnms_merge_consecutive_e15_b1.log
```

```bash
bash /marimo/DDVC_Dedup_SoftNMS_Consecutive/source/tools/dedup_molab.sh status full dedup_softnms_merge_consecutive
```

Theo dõi `EPOCH_START`, `PROGRESS`, optimizer_steps, LR, loss, validation progress,
`EPOCH_COMPLETE`. `diag_dedup_queries=0` có thể chỉ là chưa có negative đủ điều kiện;
không tăng loss mù để buộc nó khác 0. Mean mask/gate trên mọi query không đo trực tiếp
phần trăm semantic information bị mất.

Ctrl+C ở `tail -f` chỉ thoát xem log. `tail` chờ mà không có dòng mới không tự chứng
minh train treo. Xem process/GPU và phase thực tế trước khi kill hoặc build lại.
Nếu cần dừng, dùng đúng PID của run đã kiểm tra, không `pkill python` toàn runtime.
Nếu bị ngắt giữa epoch, resume từ checkpoint boundary đã lưu; có thể phải chạy lại
phần epoch chưa checkpoint. `nohup` không chống runtime bị Molab thu hồi.

## 9. Kết quả, metric và sync cuối

```bash
bash /marimo/DDVC_Dedup_SoftNMS_Consecutive/source/tools/dedup_molab.sh report full dedup_softnms_merge_consecutive
```

Run folder:

```text
/marimo/DDVC_Dedup_SoftNMS_Consecutive/runs/yc2_full_dedup_softnms_merge_consecutive_e15_b1/
  model-last.pth
  model-best.pth
  info.json
  train.log
  performance.jsonl
  prediction/
    num436_epochN.raw.json         mọi candidate
    num436_epochN.json             đúng selected/merged output đã chấm, có metrics
```

- `info.json → best` ghi checkpoint tốt nhất theo tiêu chí của run; metric đầy đủ đọc
  ở `history.val_result_history[str(best_epoch)].eval_score`. Epoch là 0-based.
- CIDEr, METEOR, BLEU-4, SODA_c trong JSON thường ở thang 0–1; bảng báo cáo dự án nhân100.
  Không nhân thêm lần nữa nếu file đã được chuyển thang.
- Best-by-MetaScore có thể khác epoch max-CIDEr. Muốn lưu cả hai, triển khai tiêu chí
  rõ từ đầu, không sửa lịch sử best cũ bằng metric hậu xử lý offline.
- Prediction raw 50 proposals và selected output là hai loại khác nhau. Visualization,
  re-evaluation và metric phải cùng file selection. Không ghi raw đè selected sau chấm.
- Nếu chỉ chạy Soft-NMS/merge offline, không có checkpoint mới. File prediction mới
  phải bỏ metric cũ và được chấm lại; tên file nên mô tả method/sigma/merge.
- Đây là YouCook2 **validation**, không gọi test. Test annotations không có đầy đủ
  reference captions công khai như train/val. Không claim SOTA từ khác protocol/seed.
- `performance.jsonl` có thể chỉ đo phần training, không gồm validation/setup;
  peak VRAM của train không đồng nghĩa peak toàn pipeline.

Sau khi train xong, cell:

```python
consecutive_backup.sync_once(force=True)
```

Đợi `BACKUP_COMPLETE` của bản cuối. Khi yêu cầu phân tích, ưu tiên info/config/train.log/
performance, final + raw prediction của best/last, run manifest và val.json. Không cần
tải checkpoint GB chỉ để đọc metric. Kiểm tra ZIP đủ epoch, CRC và IDs trước kết luận.

## 10. Mất session: restore rồi mới bật backup

1. Tải đúng source ZIP/commit vào workdir mới, dựng native dependencies/CUDA bằng setup.
2. Kết nối Drive bằng cell `fs` như mục7.
3. Tạo `consecutive_backup` như cell trên **nhưng bỏ dòng `.start()`**.
4. Restore; không start worker trước, tránh upload run mới đè lịch sử cần khôi phục.

```python
consecutive_backup.restore_latest()
```

Helper chọn READY mới nhất, kiểm tra file size, tải qua `.download` rồi replace,
remap workdir trong config. Model-last format2 là nguồn model/optimizer/scheduler/RNG/
history nhất quán; không ghép model-last của run này với info của run khác.

5. Tải/stage lại assets không nằm trong backup và kiểm tra checkpoint:

```bash
bash /marimo/DDVC_Dedup_SoftNMS_Consecutive/source/tools/dedup_molab.sh restage full dedup_softnms_merge_consecutive
bash /marimo/DDVC_Dedup_SoftNMS_Consecutive/source/tools/dedup_molab.sh smoke-resume full dedup_softnms_merge_consecutive
```

6. Đợi checkpoint load + model smoke pass; bật worker trong cell:

```python
consecutive_backup.start()
```

7. Tiếp tục đúng run:

```bash
bash /marimo/DDVC_Dedup_SoftNMS_Consecutive/source/tools/dedup_molab.sh resume full dedup_softnms_merge_consecutive
```

Không prepare đè config restore. Không đổi epoch budget, loss, batch/accumulation,
precision hoặc selection/merge khi exact-resume; nếu muốn đổi, tạo experiment mới
và phân biệt fine-tune với resume. Giữ phase validation_pending/epoch_complete để
không lặp optimizer update hoặc bỏ mất validation khi session gãy.

## 11. Các lỗi đã gặp: xác định đúng tầng lỗi

| Dấu hiệu | Kiểm tra / xử lý |
|---|---|
| `bash: : command not found`, path `/logs/.log` | Biến shell đã rỗng; dùng launcher/path đầy đủ, không dựa terminal trước |
| `set: pipefail: invalid option name` | Shell script có CRLF; kiểm tra bytes, sửa LF tại source/package; không hạ shell thành `sh` để né |
| gdown `unrecognized arguments: --fuzzy` | Dùng file ID trực tiếp, kiểm tra help/version thực tế |
| Drive 401/không list folder | Dùng authenticated Molab `fs` hoặc phiên browser được cấp quyền; không copy cookie/token hay tự public folder |
| `This cell redefines variables...` | Sửa cell gốc; một global một cell, dùng `_` locals; không thêm bản `fs`/root/handle thứ hai |
| `DRIVE_RUN`/`DRIVE_ARCHIVE` chưa định nghĩa | Bảo đảm dependency/config cell tồn tại; tránh global rải rác hoặc reliance vào cell order |
| `config.yml` không tồn tại khi smoke | Chạy **prepare-smoke đúng experiment** và đợi CONFIG_READY; đọc prepare_smoke.log, không tải ZIP/build CUDA lại vô cớ |
| `No module named kaggle_recipe` | Nạp sibling helper bằng đường dẫn tuyệt đối; không phụ thuộc sys.path sau imports. Bản current đã sửa |
| SimpleNamespace thiếu `contrastive_loss_coef` | Smoke phải dùng parser defaults rồi YAML overrides như train, không namespace từ YAML thiếu field |
| Thiếu para/grounding JSON của ActivityNet khi YC2 | Tạo refs YC2 đúng schema, tắt evaluator không liên quan; không tạo file rỗng để giả metric |
| Vocab size assertion | Đối chiếu vocabulary thực với config và selected train annotations |
| `device=1, num_gpus=1` | Đặt CUDA_VISIBLE_DEVICES trước khởi chạy process; dùng logical cuda:0; không đổi sau Torch init |
| CLIP thiếu `pkg_resources` lúc build | Legacy build needs setuptools phù hợp; bootstrap setuptools<81 và build isolation policy rõ, không cài package giả tên pkg_resources |
| uv không tìm CUDA package ở first index | Xem index/pin thực; dùng index chính thức rõ ràng. Không tự nâng bản legacy; chỉ dùng unsafe-best-match nếu các index đã xác định cùng được tin cậy |
| `cython_sources` lúc build | Xác định package và wheel/Python compatibility; pin build deps hoặc wheel đã kiểm chứng, không quy mọi lỗi cho model |
| Missing `nv/target` | Thiếu CCCL headers, không phải Java/locale. Kiểm tra CUDA_HOME, nvcc/header versions và actual header path |
| `MultiScaleDeformableAttention` không import / `MSDA` undefined | Xem build log nguyên nhân đầu, `.so`, PYTHONPATH và ABI. Không che import bằng except-pass hoặc giả ready marker |
| Build CUDA dừng, `tail -f` cứ chờ | Kiểm tra exit code/process; grep dòng fatal/error đầu thay vì chỉ 80 dòng wrapper traceback cuối |
| Torch cu130 nhưng nvcc/header13.3 | Dùng private compiler/header assets cu130 của helper nếu đúng runtime; giữ Torch, không đoán path `include/cccl` chưa kiểm tra |
| OOM GPT-2 CE | Xem batch/chunk/sequence/query. Chunk all-pair caption cost thay vì bỏ query-caption matches; không gọi GPU lớn là lý do tăng batch bất chấp head batch1 |
| Checkpoint pickle `eval_metrics.<locals>.<lambda>` | Chuyển metadata/defaultdict thành plain types; giữ nguyên tensor model/optimizer; save atomic và cleanup failed temp |
| `cfgs` thiếu khi backup source | Đọc effective config, chỉ copy folder thực có; không bắt buộc source archive phải chứa author folder |
| `info.json` chưa có | Có thể validation chưa hoàn tất; xem train log và checkpoint phase, không bịa metric từ loss |
| Locale/SyntaxWarning `\s` | Thường là warning, không phải nguyên nhân traceback. Tìm exception thực sự trước khi sửa |

CUDA ghi nhận trong dự án: native Torch từng là `2.11.0+cu130`, hệ thống có nvcc13.3.
`tools/molab_cuda130.py` chỉ hỗ trợ runtime Linux x86_64 cu130 đã quan sát, stage 5 NVIDIA
wheels pin riêng (~105MB), **không phải venv**, không thay native Torch. Không áp dụng
mù cho runtime khác. `nvidia-smi` CUDA là khả năng driver, không phải nvcc đang compile.

Lệnh đọc lỗi build đầu tiên, terminal Molab:

```bash
grep -n -B 4 -A 10 -E 'fatal error:|error:|nvcc fatal|Killed|FAILED:' /marimo/DDVC_Dedup_SoftNMS_Consecutive/logs/build_cuda.log | head -n 120
```

## 12. Khi giao lại cho người khác hoặc coding agent

Giao file này cùng ZIP/manifest/checksums và nêu chính xác repo/branch/commit, mode,
workdir, Drive code folder ID, artifact remote root, data references và trạng thái run.
Không đưa auth secrets. Agent đọc actual source/manifest trước, không tái tạo code từ
nhớ hội thoại, không giả định patch local đã vào ZIP được upload trước đó.

Trước khi báo hoàn tất, xác nhận tuần tự: source đúng → native runtime/CUDA pass →
config đúng → actual smoke pass → train/validation/save pass → train log đang cập nhật →
Drive BACKUP_COMPLETE/READY → restore smoke có đường chạy → metric đúng selected file.
Nếu chưa thực hiện bước nào thì nói rõ chưa kiểm chứng bước đó.

File có YAML frontmatter để có thể dùng như một Codex skill. Muốn cài discoverable,
đặt vào folder `molab-drive-experiment-handoff/SKILL.md` trong skill directory của máy
đích. Việc xuất một file này **không tự cài** hoặc thay thế skill đang có.
