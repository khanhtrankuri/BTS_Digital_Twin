# BTS Digital Twin — AbsGS v11

Pipeline 3D Gaussian Splatting dành cho bộ dữ liệu BTS Digital Twin. Phiên bản
phát hành hiện tại sử dụng true absolute screen-space gradient của AbsGS để
densify các vùng chi tiết, sau đó khóa topology trước opacity reset và tiếp tục
tối ưu geometry/SH đến iteration 15.000.

Repository chỉ chứa source code, CUDA extension, cấu hình và test. Dataset,
checkpoint, ảnh render và ZIP submission không được đưa lên Git.

## Kết quả lựa chọn model

Kết quả dưới đây được đo trên position-extrapolation holdout HCM0674; RGB của
private test không được dùng để chọn cấu hình.

| Model | Iteration | Holdout score |
|---|---:|---:|
| Standard 3DGS | 7.000 | 69.9796 |
| AbsGS, khóa topology trước opacity reset | 7.000 | 72.4042 |
| Standard 3DGS | 15.000 | 72.6666 |
| **AbsGS v11** | **15.000** | **73.1173** |
| AbsGS v11 | 30.000 | 73.1166 |

Cấu hình 15k thắng 15k standard 3DGS `+0.4507` điểm và không có lợi ích khi
train tiếp đến 30k. Chỉ số holdout của model được chọn:

- PSNR: `21.388983`
- SSIM: `0.823026`
- LPIPS: `0.110172`

Đây không phải điểm private leaderboard và không phải cam kết đạt 80 điểm.

## Yêu cầu cho RTX 4090 24 GB

Thiết lập đã được chuẩn hóa cho Windows:

- NVIDIA RTX 4090 24 GB.
- Driver NVIDIA hỗ trợ CUDA 12.8.
- Windows 10/11 64-bit.
- Visual Studio 2022 Build Tools với workload **Desktop development with C++**.
- Conda/Miniconda.
- Khoảng 25 GB trống cho environment, dataset chuẩn bị và source.
- Dung lượng bổ sung cho checkpoint/output tùy số scene.

CUDA extension đã được vendored trong `submodules/`, vì vậy không cần chạy
`git submodule update`.

## Cài đặt

```powershell
git clone https://github.com/khanhtrankuri/BTS_Digital_Twin.git
cd BTS_Digital_Twin

conda env create -f environment.yml
conda activate BTS

$env:CUDA_HOME = Join-Path $env:CONDA_PREFIX 'Library'

python -m pip install --no-build-isolation `
  submodules/diff-gaussian-rasterization `
  submodules/simple-knn `
  submodules/fused-ssim
```

Kiểm tra CUDA và true AbsGS rasterizer:

```powershell
python -c "import torch; from diff_gaussian_rasterization import GaussianRasterizer; print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory // 2**30, 'GB')"
```

Chạy unit test:

```powershell
python -m pytest -q
```

## Chạy tự động bằng `run.sh`

Trên Ubuntu, Linux hoặc WSL, script sau tự kiểm tra CUDA extension, chuẩn bị
năm scene HCM nếu cần, train đủ bảy scene, render và tạo ZIP:

```bash
conda activate BTS
chmod +x run.sh
./run.sh /data/Val_Race
```

Ví dụ dataset nằm trên ổ D khi dùng WSL:

```bash
./run.sh /mnt/d/Val_Race
```

Các output mặc định:

```text
data/bts_v11_prepared/
output/bts_v11_4090/
submission_bts_v11/
submission_bts_v11.zip
submission_bts_v11.manifest.json
```

Có thể thay đường dẫn hoặc profile bằng biến môi trường:

```bash
DATA_ROOT=/data/Val_Race \
MODEL_ROOT=/workspace/models/bts_v11 \
ZIP_PATH=/workspace/submission_bts_v11.zip \
GPU_PROFILE=rtx4090_24gb \
./run.sh
```

Xem toàn bộ tùy chọn:

```bash
./run.sh --help
```

## Cấu trúc dataset

Ví dụ đặt dữ liệu tại `D:\Val_Race`:

```text
D:\Val_Race\
├── bonsai\
├── chair\
├── HCM0421\
├── HCM0539\
├── HCM0540\
├── HCM0644\
└── HCM0674\
```

Mỗi scene phải chứa ảnh train, COLMAP sparse model và danh sách private test
pose theo định dạng của bộ dữ liệu. `bonsai` và `chair` được dùng trực tiếp.
Năm scene HCM dùng camera `SIMPLE_RADIAL`, do đó phải được undistort về camera
pinhole trước khi train.

## Chuẩn bị năm scene HCM

Lệnh sau giữ nguyên dataset gốc và ghi dữ liệu đã chuẩn bị vào thư mục
`data/bts_v11_prepared`:

```powershell
$dataRoot = 'D:\Val_Race'
$preparedRoot = 'data\bts_v11_prepared'
$hcmScenes = 'HCM0421', 'HCM0539', 'HCM0540', 'HCM0644', 'HCM0674'

foreach ($scene in $hcmScenes) {
  python tools\prepare_undistorted_scene.py `
    --source (Join-Path $dataRoot $scene) `
    --output (Join-Path $preparedRoot $scene) `
    --copy_sparse
}
```

Công cụ cố ý không ghi đè thư mục output không rỗng. Nếu cần chuẩn bị lại,
hãy xóa đúng scene tương ứng trong `data/bts_v11_prepared` trước khi chạy.

## Train, render và tạo ZIP bằng RTX 4090

Lệnh dưới đây train tuần tự đủ bảy scene, render private poses, redistort năm
scene HCM về lưới ảnh gốc, kiểm tra toàn bộ JPEG/CRC và tạo ZIP:

```powershell
$python = Join-Path $env:CONDA_PREFIX 'python.exe'

& $python tools\train_render_submission_v11.py `
  --python $python `
  --data_root 'D:\Val_Race' `
  --prepared_root 'data\bts_v11_prepared' `
  --model_root 'output\bts_v11_4090' `
  --render_root 'submission_bts_v11' `
  --zip_path 'submission_bts_v11.zip' `
  --gpu_profile rtx4090_24gb `
  --quiet
```

Profile `rtx4090_24gb`:

- Chạy mỗi scene trong một process liên tục đến iteration 15.000.
- Lưu checkpoint/point cloud tại iteration 7.000 và 15.000.
- Không restart tại iteration 2.000 như profile 8 GB.
- Dùng resolution 2 đã thắng ablation; ảnh private test vẫn render đúng kích
  thước yêu cầu.
- True AbsGS densification từ iteration 500.
- Khóa topology tại iteration 2.500, trước opacity reset iteration 3.000.
- Tối đa 2,5 triệu Gaussian và antialiasing bật.

24 GB VRAM được dùng để tránh allocator pressure và tăng throughput, không tự
ý thay đổi resolution hoặc Gaussian cap chưa qua validation.

Pipeline có thể resume. Chạy lại đúng lệnh trên sẽ bỏ qua model đã có
`point_cloud/iteration_15000/point_cloud.ply` và resume scene chưa hoàn tất từ
checkpoint mới nhất.

## Train một scene

Ví dụ train riêng HCM0674:

```powershell
python train.py `
  --config configs\bts_v11\absgrad_early_stop_15k.yaml `
  -s data\bts_v11_prepared\HCM0674 `
  -m output\bts_v11_4090\HCM0674 `
  --iterations 15000 `
  --eval `
  --disable_viewer `
  --test_iterations -1 `
  --save_iterations 7000 15000 `
  --checkpoint_iterations 7000 15000 `
  --quiet
```

Resume từ checkpoint:

```powershell
python train.py `
  --config configs\bts_v11\absgrad_early_stop_15k.yaml `
  -s data\bts_v11_prepared\HCM0674 `
  -m output\bts_v11_4090\HCM0674 `
  --iterations 15000 `
  --start_checkpoint output\bts_v11_4090\HCM0674\chkpnt7000.pth `
  --save_iterations 15000 `
  --checkpoint_iterations 15000 `
  --eval --disable_viewer --test_iterations -1 --quiet
```

## Output

Sau khi pipeline hoàn tất:

```text
output/bts_v11_4090/<scene>/
├── chkpnt7000.pth
├── chkpnt15000.pth
└── point_cloud/iteration_15000/point_cloud.ply

submission_bts_v11/<scene>/<test image>.JPG
submission_bts_v11.zip
submission_bts_v11.manifest.json
```

Validation cuối yêu cầu:

- Tổng cộng 386 ảnh.
- `bonsai`: 28 ảnh, 1920×1080.
- `chair`: 58 ảnh, 720×1280.
- Mỗi HCM scene: 60 ảnh, 1320×989.
- Tất cả ảnh là RGB JPEG decode được.
- ZIP không có lỗi CRC.

## Cấu hình chính

- `configs/bts_v11/absgrad_probe.yaml`: true AbsGS và giới hạn Gaussian.
- `configs/bts_v11/absgrad_early_stop.yaml`: khóa topology tại 2.500.
- `configs/bts_v11/absgrad_early_stop_15k.yaml`: model phát hành.
- `tools/train_render_submission_v11.py`: train/render/package tự động.
- `tools/prepare_undistorted_scene.py`: undistort/redistort camera HCM.

## Dọn output

`output/`, `submission*/`, `evaluation/`, `data/`, checkpoint, ZIP, manifest,
binary CUDA và cache đều đã nằm trong `.gitignore`. Không commit các artifact
này lên GitHub.

## Nguồn kỹ thuật

Pipeline được phát triển từ implementation 3D Gaussian Splatting của GraphDeco
và ý tưởng absolute-gradient densification của AbsGS:

- 3D Gaussian Splatting: <https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/>
- AbsGS: <https://ty424.github.io/AbsGS.github.io/>

Xem `LICENSE.md` và license trong từng thư mục CUDA extension trước khi phân
phối lại.
