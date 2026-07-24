# BTS v12: metric-aligned full-resolution refinement

## Kết luận chẩn đoán

Leaderboard hiện tại:

| Chỉ số | Giá trị |
|---|---:|
| Score | 65.0192 |
| PSNR | 22.665939 |
| SSIM | 0.743145 |
| LPIPS | 0.271867 |

Score khớp với công thức:

```text
100 * (0.40 * (1 - LPIPS) + 0.30 * SSIM + 0.30 * PSNR / 50)
```

Ba thành phần đóng góp lần lượt là 29.1253, 22.2944 và 13.5996 điểm.
Vì vậy, nút thắt lớn nhất là LPIPS và SSIM, không phải chỉ riêng PSNR.

v11 đạt holdout nội bộ 73.1173 nhưng leaderboard chỉ đạt 65.0192. Điều tra
pipeline cho thấy:

- v11 train và validate với `resolution: 2`;
- ảnh validation nội bộ vì thế chỉ bằng nửa chiều rộng và chiều cao;
- submission lại render ở kích thước native;
- holdout position-extrapolation của riêng HCM0674 khó theo một hướng khác với
  phân bố pose private test, vốn chủ yếu là các frame xen kẽ gần camera train.

Do đó 73.1173 không phải ước lượng đáng tin cho leaderboard. Việc train v11 từ
15k lên 30k gần như không đổi score nội bộ, nên chỉ tăng iteration không giải
quyết sai lệch này.

## v11 có phải kiến trúc hai stage không?

Không theo nghĩa hai model độc lập. v11 là **một training loop**:

1. AbsGS densification xây topology đến iteration 2.500.
2. Topology được khóa và cùng model đó tiếp tục tối ưu geometry/SH đến 15.000.

Đây là hai pha tối ưu của một model, không có model stage 1 sinh dữ liệu cho
model stage 2. v12 vẫn giữ nguyên nguyên tắc một model, nhưng dùng ba chế độ loss
và learning-rate rõ ràng để tránh geometry dao động khi chuyển lên full-resolution.

## Kiến trúc v12 đã triển khai

### A0 — control trung thực

`configs/bts_v12/control_v11.yaml` giữ nguyên đường tối ưu v11, nhưng validation
luôn được nạp ở native resolution. Đây là mốc so sánh đúng, không dùng lại số
73.1173 ở half-resolution.

### A1 — khớp train/test resolution

`configs/bts_v12/fullres.yaml` dùng progressive resolution:

```text
0–2.499       0.50x
2.500–5.999   0.75x
6.000–20.000  1.00x
```

Pha đầu rẻ để xây topology; phần lớn refinement diễn ra đúng lưới pixel của
submission. Đây là thay đổi có ưu tiên cao nhất.

### A2 — quality, nhánh thử nghiệm đã ablate

`configs/bts_v12/quality.yaml` bổ sung:

- loss schedule L1/MSE/DSSIM theo ba pha;
- SSIM đa tỉ lệ: native và ảnh giảm 2 lần;
- LPIPS trên crop native-resolution, bắt đầu muộn sau khi topology ổn định;
- crop đầu tiên tập trung vào vùng residual lớn nhất, crop còn lại lấy ngẫu
  nhiên để tránh chỉ học một vùng;
- giảm learning rate geometry/opacity ở pha cuối, vẫn cho SH/features thích nghi.

LPIPS không còn resize toàn frame về một ảnh nhỏ. Cách mới giữ lại chi tiết tần
số cao mà leaderboard đang phạt mạnh.

`configs/bts_v12/fullres_8gb.yaml` và `fullres_hcm_8gb.yaml` là các biến thể
đã đo footprint cho RTX 4060 8 GB:

- giới hạn 450.000 Gaussian cho bonsai/chair và 350.000 cho HCM;
- tối đa 20.000 Gaussian mới mỗi lần densify, 15.000 trên HCM;
- lên full-resolution muộn hơn;
- cache ảnh trên CPU để dành VRAM cho Gaussian và native raster.

Ablation thực tế trên fold 0 đã loại cấu hình `quality_8gb`: trên bonsai,
`fullres_8gb` đạt 71,5014 so với 70,6161 của perceptual-loss candidate và
70,7386 của v11 control. Trên HCM0674, full-resolution đạt 75,1704 so với
72,4361 của control cùng cap.

### A3 — multi-view, chỉ bật sau ablation

`configs/bts_v12/quality_multiview.yaml` thêm regularization RGB giữa các view
lân cận với trọng số thấp. Đây là nhánh thử nghiệm, không phải default, vì
regularization sai có thể làm mờ vùng occlusion và làm LPIPS xấu hơn.

## Validation không dùng private RGB

Tạo ba fold temporal-matched trên cả bảy scene:

```powershell
$python = 'C:\Users\Lenovo\anaconda3\envs\BTS\python.exe'

& $python tools\prepare_v12_validation.py `
  --data_root 'C:\Users\Lenovo\Documents\Val_Race' `
  --prepared_root 'data\bts_v5_prepared' `
  --output_root 'data\bts_v12_validation' `
  --folds 3
```

Script chỉ đọc RGB train. Danh sách private test chỉ được dùng để lấy **số
lượng pose**, từ đó đặt tỉ lệ holdout tương ứng cho từng scene. Không đọc private
test RGB.

Chạy ablation nhỏ trước:

```powershell
& $python tools\run_v12_ablation.py `
  --data_root 'C:\Users\Lenovo\Documents\Val_Race' `
  --prepared_root 'data\bts_v5_prepared' `
  --validation_root 'data\bts_v12_validation' `
  --experiments control_v11 fullres_8gb quality_8gb `
  --scenes bonsai chair HCM0674 `
  --folds 0 `
  --seeds 0
```

Sau đó xác nhận trên cả bảy scene và nhiều fold:

```powershell
& $python tools\run_v12_ablation.py `
  --data_root 'C:\Users\Lenovo\Documents\Val_Race' `
  --prepared_root 'data\bts_v5_prepared' `
  --validation_root 'data\bts_v12_validation' `
  --experiments control_v11 fullres_8gb quality_8gb `
  --folds 0 1 2 `
  --seeds 0
```

Kết quả tổng hợp được ghi vào
`output/bts_v12_ablation/ablation_summary.json`.

## Luật chọn model

Không chọn theo một scene hay một metric đơn lẻ:

1. `fullres` phải thắng `control_v11` về score trung bình.
2. `quality` phải thắng `fullres` trên ít nhất 2/3 fold.
3. Không scene nào được giảm quá 0,5 điểm nếu tổng thể tăng.
4. Ưu tiên giảm LPIPS và tăng SSIM; không chấp nhận PSNR tăng nhưng score tổng
   giảm.
5. Chỉ thử `multiview` sau khi A2 đã thắng.

Mốc nghiên cứu thực tế đầu tiên là vượt 70 điểm. Một tổ hợp minh họa gần mốc này
là PSNR 23,9, SSIM 0,79 và LPIPS 0,20, tương ứng khoảng 70,04 điểm. Đây là mục
tiêu, không phải kết quả đã đo.

## Train và đóng gói candidate

Ablation hiện tại không cho `quality_8gb` vượt qua gate. Candidate cuối vì thế
dùng `fullres_8gb` cho bonsai/chair và `fullres_hcm_8gb` cho năm scene HCM:

```powershell
& $python tools\train_render_submission_v12.py `
  --python $python `
  --data_root 'C:\Users\Lenovo\Documents\Val_Race' `
  --prepared_root 'data\bts_v5_prepared' `
  --model_root 'output\bts_v12_fullres' `
  --render_root 'submission_bts_v12' `
  --zip_path 'submission_bts_v12.zip' `
  --gpu_profile memory_safe_8gb `
  --quiet
```

Runner tự resume checkpoint, render native resolution, redistort năm scene HCM,
kiểm tra đủ 386 ảnh/CRC rồi mới tạo ZIP và manifest.

### Kết quả lượt train cuối

Lượt chạy `memory_safe_8gb` đã hoàn tất cả bảy scene ở iteration 20.000. Model
bonsai kết thúc với 293.198 Gaussian, chair với 387.367 Gaussian; năm scene HCM
đều chạm cap 350.000 Gaussian. Submission có đúng 386 JPEG:

| Scene | Số ảnh | Kích thước |
|---|---:|---:|
| bonsai | 28 | 1920 × 1080 |
| chair | 58 | 720 × 1280 |
| HCM0421 | 60 | 1320 × 989 |
| HCM0539 | 60 | 1320 × 989 |
| HCM0540 | 60 | 1320 × 989 |
| HCM0644 | 60 | 1320 × 989 |
| HCM0674 | 60 | 1320 × 989 |

Artifact cuối là `submission_bts_v12.zip`, 148.346.332 byte, SHA-256
`4b95f0be7c1cccc44904dc8ab9ab96d0f073a03f9790b3cd1a0b4228a07dbdd8`.
Đây là candidate đã train/render/kiểm định, chưa phải điểm private leaderboard;
chỉ lượt chấm chính thức mới xác nhận mức tăng so với 65,0192.

## Cơ sở nghiên cứu

- Mip-Splatting: mismatch sampling rate giữa train và render gây aliasing;
  3D smoothing và mip filtering giải quyết vấn đề theo hướng có nguyên lý.
  <https://arxiv.org/abs/2311.16493>
- Pixel-GS: gradient và densification nên phụ thuộc số pixel mà Gaussian tác
  động, đặc biệt cho chi tiết nhỏ và vùng thưa.
  <https://arxiv.org/abs/2403.15530>
- WildGaussians: appearance variation và transient occluder cần được mô hình hóa
  riêng trong dữ liệu in-the-wild.
  <https://arxiv.org/abs/2407.08447>
- Scaffold-GS: cấu trúc anchor và thuộc tính Gaussian theo view cải thiện khả
  năng thích nghi theo góc nhìn.
  <https://arxiv.org/abs/2312.00109>

Candidate v12 hiện áp dụng trực tiếp kết luận ít rủi ro nhất: khớp resolution
train/render và validation đúng phân bố. Metric-aligned perceptual loss đã được
triển khai nhưng giữ ở nhánh thử nghiệm vì chưa qua ablation gate. Pixel-aware
densification, mip filtering và appearance embedding là các nhánh kiến trúc
tiếp theo.
