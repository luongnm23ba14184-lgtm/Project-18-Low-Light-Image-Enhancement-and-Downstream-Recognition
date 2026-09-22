# Kế Hoạch Triển Khai Mã Nguồn Toàn Diện Project 18 (End-to-End Pipeline)

Kế hoạch này hiện thực hóa toàn bộ mã nguồn của đề tài **"Project 18: Low-Light Image Enhancement and Downstream Recognition"**, bám sát 100% **Sơ đồ Pipeline Tổng Thể (Mục 6)** và **Cấu trúc Thư mục & Kịch bản `run.py` (Mục 11)** trong [README.md](file:///home/minhluong/Documents/Project18/README.md).

Đề tài tập trung chuyên sâu vào **bản chất thuật toán toán học** của các mô hình (không chỉ gọi thư viện qua loa), bao gồm:
1. Tự cài đặt kiến trúc mạng **DCE-Net** và **Zero-DCE++** với phép biến đổi đường cong ánh sáng lặp bậc cao ($LE_n$).
2. Tự cài đặt từ công thức toán học **4 hàm loss tự giám sát phi tham chiếu**: $\mathcal{L}_{spa}, \mathcal{L}_{exp}, \mathcal{L}_{col}, \mathcal{L}_{tv\_A}$.
3. Thuật toán DIP kinh điển: **CIE LAB + CLAHE + Bilateral Filter** làm mốc đối chứng.
4. Hệ thống đánh giá toàn diện: **No-Reference IQA (NIQE, BRISQUE)** cho GĐ1 và **mAP@0.5, mAP@0.5:0.95** cho GĐ2.
5. Kịch bản tự động hóa **`run.py` qua 6 phase** kết nối liền mạch từ dữ liệu gốc đến biểu đồ so sánh.

---

## User Review Required

> [!IMPORTANT]
> **Chuẩn hóa cấu trúc dữ liệu `Dataset/`:**
> Hiện tại dữ liệu đang nằm trực tiếp ở `Dataset/train`, `Dataset/valid`, `Dataset/test`. Để tuân thủ đúng sơ đồ Mục 11, chúng ta sẽ thiết lập:
> * `Dataset/exdark_yolo_dark/`: Chứa tập ảnh tối gốc (dùng symlink hoặc thư mục chuẩn để tiết kiệm ổ đĩa và tránh copy dư thừa).
> * `Dataset/exdark_yolo_zerodce/`: Thư mục chứa tập ảnh sau khi tăng cường qua Zero-DCE (kèm nhãn giữ nguyên 100%).
> * Sửa tên thư mục hiện có `Resultes/` thành `Results/` (gồm `Results/weights/` và `Results/figures/`).

> [!NOTE]
> **Môi trường chạy:**
> Máy cục bộ sử dụng môi trường Python `/home/minhluong/my_env` (PyTorch 2.14, Ultralytics, OpenCV). Code sẽ được thiết kế linh hoạt hỗ trợ cả chạy cục bộ (CPU/GPU) và có thể copy chạy mượt mà trên Google Colab / Kaggle T4 GPU.

---

## Proposed Changes

### Khối 1: Chuẩn hóa Môi trường & Thư mục Dự án

#### [NEW] [requirements.txt](file:///home/minhluong/Documents/Project18/requirements.txt)
Khai báo đầy đủ các thư viện phụ thuộc của đề tài:
* `torch>=2.0.0`, `torchvision`
* `ultralytics>=8.0.0`
* `opencv-python`, `pillow`
* `numpy`, `scipy`, `matplotlib`
* `pyyaml`, `tqdm`

#### [MODIFY] Đồng bộ thư mục `Results/` và `Dataset/`
* Đổi tên `Resultes/` $\rightarrow$ `Results/`
* Tạo cấu trúc con: `Results/weights/`, `Results/figures/`
* Tạo cấu trúc thư mục dữ liệu: `Dataset/exdark_yolo_dark/` và `Dataset/exdark_yolo_zerodce/`

---

### Khối 2: Xây dựng Bộ Module Thuật Toán Cốt Lõi (`src/`)

```
src/
├── __init__.py
├── model_zerodce.py      # DCE-Net & Zero-DCE++ & LE-Curve
├── loss_zerodce.py       # 4 Non-Reference Loss Functions
├── preprocess_dip.py     # CLAHE + Bilateral Filter (DIP Baseline)
├── metrics.py            # NIQE, BRISQUE, FPS, mAP parsers
└── visualize.py          # So sánh 4 khung hình & Biểu đồ mAP
```

#### [NEW] [src/__init__.py](file:///home/minhluong/Documents/Project18/src/__init__.py)
* Định nghĩa package, export các class và hàm chính.

#### [NEW] [src/model_zerodce.py](file:///home/minhluong/Documents/Project18/src/model_zerodce.py)
Hiện thực hóa từ bản chất toán học:
* **`DCENet` (PyTorch `nn.Module`):**
  * 7 tầng tích chập đối xứng ($32$ channels, kernel $3 \times 3$, ReLU).
  * Skip connections bảo toàn đặc trưng không gian đa tỉ lệ: tầng 1 nối tầng 6, tầng 2 nối tầng 5, tầng 3 nối tầng 4.
  * Tầng ra: Conv $3 \times 3$ với hàm kích hoạt Tanh, sinh tensor kích thước $(B, 24, H, W)$ tương ứng với 8 bước lặp $\times$ 3 kênh RGB.
* **`ZeroDCEpp` (PyTorch `nn.Module`):**
  * Sử dụng Depthwise Separable Convolution (Depthwise $3 \times 3$ + Pointwise $1 \times 1$) giúp rút gọn tham số xuống $\approx 10.000$ params, siêu nhẹ và thời gian thực.
* **`enhance_image(x, A)`:**
  * Hiện thực hóa công thức lặp bậc 2 $LE_n$:
    $$LE_n(x) = LE_{n-1}(x) + \mathcal{A}_n(x) \cdot LE_{n-1}(x) \cdot (1 - LE_{n-1}(x))$$
  * Xử lý thuần túy trên Tensor PyTorch, hỗ trợ autograd.

#### [NEW] [src/loss_zerodce.py](file:///home/minhluong/Documents/Project18/src/loss_zerodce.py)
Tự lập trình 4 hàm mất mát không tham chiếu từ công thức giải tích:
* **`L_spa` (Spatial Consistency Loss):**
  * Dùng tích chập với các kernel vi phân 4 hướng (trên, dưới, trái, phải) trên cả ảnh gốc $I$ và ảnh kết quả $Y$. Phạt sai khác giữa biến thiên cục bộ để bảo toàn biên nét.
* **`L_exp` (Exposure Control Loss):**
  * Sử dụng `F.avg_pool2d(kernel_size=16, stride=16)` tính mức sáng trung bình các khối $16 \times 16$, phạt độ lệch tuyệt đối so với mức phơi sáng tối ưu $E = 0.6$.
* **`L_color` (Color Constancy Loss):**
  * Tính giá trị trung bình từng kênh $R, G, B$ trên toàn ảnh, áp dụng giả thuyết Gray-World để triệt tiêu sai lệch màu: $(R-G)^2 + (R-B)^2 + (G-B)^2$.
* **`L_TV_A` (Illumination Smoothness Loss):**
  * Tính Total Variation trên gradient ngang và dọc của các bản đồ tham số $\mathcal{A}$ để đảm bảo độ mượt mà, chống sọc vằn.
* **`ZeroDCELoss`:**
  * Lớp tổng hợp tính $\mathcal{L}_{total} = \mathcal{L}_{spa} + \mathcal{L}_{exp} + 5.0 \cdot \mathcal{L}_{col} + 200.0 \cdot \mathcal{L}_{tv\_A}$.

#### [NEW] [src/preprocess_dip.py](file:///home/minhluong/Documents/Project18/src/preprocess_dip.py)
* **`enhance_clahe_bilateral(image_bgr, clip_limit=2.0, tile_grid=(8,8))`:**
  * Chuyển đổi $BGR \rightarrow CIE\text{-}LAB$.
  * Tách kênh $L$ (Luminance) và thực hiện CLAHE (cân bằng lược đồ mức xám thích nghi có giới hạn tương phản).
  * Lọc biên mịn bằng Bilateral Filter (`cv2.bilateralFilter`) nhằm triệt tiêu nhiễu hạt nhưng giữ nét biên vật thể.
  * Chuyển ngược về $RGB$.
  * Hỗ trợ xử lý đơn ảnh và xử lý theo thư mục (batch processing).

#### [NEW] [src/metrics.py](file:///home/minhluong/Documents/Project18/src/metrics.py)
* **Đánh giá ảnh GĐ1:**
  * Tính điểm **NIQE** (Naturalness Image Quality Evaluator) $\downarrow$.
  * Tính điểm **BRISQUE** $\downarrow$.
  * Đo tốc độ suy luận **FPS** và độ trễ **Latency (ms)**.
* **Đánh giá nhận diện GĐ2:**
  * Parse và tính toán các chỉ số: $Precision$, $Recall$, $mAP@0.5$, $mAP@0.5:0.95$ từ model YOLOv8.

#### [NEW] [src/visualize.py](file:///home/minhluong/Documents/Project18/src/visualize.py)
* **`plot_side_by_side_comparison(...)`:**
  * Xuất ảnh đối chứng 4 khung hình ghép: `[Ảnh tối gốc] | [Ảnh CLAHE] | [Ảnh Zero-DCE] | [Dự đoán Bounding Box YOLOv8]`.
* **`plot_metrics_comparison(...)`:**
  * Vẽ biểu đồ cột trực quan hóa $mAP@0.5$ và $mAP@0.5:0.95$ qua 4 kịch bản đối chứng, tự động lưu vào `Results/figures/`.

---

### Khối 3: Kịch bản Master `run.py` Tự Động Hóa 6 Phase

#### [NEW] [run.py](file:///home/minhluong/Documents/Project18/run.py)
Script điều phối toàn bộ pipeline thực nghiệm tự động theo đúng Mục 11.2 README:
* **Phase 0: Setup:** Kiểm tra GPU CUDA / CPU, tạo cấu trúc thư mục `Results/weights`, `Results/figures`.
* **Phase 1: Data Verification:** Kiểm tra tính toàn vẹn của dataset ExDark, ánh xạ đường dẫn `data.yaml`.
* **Phase 2: Giai đoạn 1 - Image Enhancement:**
  * Khởi tạo mạng Zero-DCE / Zero-DCE++, huấn luyện hoặc nạp checkpoint.
  * Tăng cường sáng hàng loạt tập ảnh ExDark sang `Dataset/exdark_yolo_zerodce/`.
  * Tính toán chỉ số NIQE, BRISQUE và FPS của Zero-DCE vs CLAHE.
* **Phase 3: Giai đoạn 2 - Object Detection 4 Kịch bản:**
  * Kịch bản 1: Huấn luyện / Đánh giá YOLOv8 trên ảnh tối gốc $\rightarrow$ $mAP_{dark}$.
  * Kịch bản 2: Đánh giá Cascaded ảnh CLAHE trên model tối $\rightarrow$ $mAP_{clahe}$.
  * Kịch bản 3: Đánh giá Cascaded ảnh Zero-DCE trên model tối $\rightarrow$ $mAP_{cascaded}$.
  * Kịch bản 4: Huấn luyện YOLOv8 trên ảnh Zero-DCE (`hsv_v=0.1, close_mosaic=10`) $\rightarrow$ $mAP_{retrained}$.
* **Phase 4: Tổng hợp Báo cáo:** Xuất file `Results/comparisons_table.csv` và biểu đồ đối chứng.
* **Phase 5: Demo Thời gian thực:** Hàm suy luận trọn gói end-to-end: `Input tối ➔ Zero-DCE ➔ YOLOv8 ➔ Xuất ảnh bounding box`.

---

## Verification Plan

### Kiểm thử Tự động (Unit Tests & Smoke Tests)
1. **Kiểm tra Mạng Zero-DCE & Zero-DCE++:**
   * Tạo dummy tensor $(1, 3, 256, 256)$ chạy qua `DCENet` và `ZeroDCEpp`.
   * Kiểm tra kích thước tensor đầu ra $\mathcal{A}$ $(1, 24, 256, 256)$ và ảnh $Y$ sau phép toán $LE_n$ có nằm trong đoạn $[0, 1]$ không.
2. **Kiểm tra 4 Hàm Loss:**
   * Truyền tensor dummy qua `L_spa`, `L_exp`, `L_color`, `L_TV_A` và kiểm tra gradient lan truyền ngược (`loss.backward()`) hoạt động ổn định, không bị `NaN` hoặc `Inf`.
3. **Kiểm tra Module DIP CLAHE:**
   * Chạy hàm `enhance_clahe_bilateral` trên 1 ảnh mẫu từ `Dataset/train/images`, kiểm tra ảnh đầu ra sắc nét, không bị biến dạng kích thước hay vỡ kênh màu.
4. **Kiểm tra YOLOv8 Pipeline:**
   * Nạp `ultralytics.YOLO` và chạy inference thử 1 ảnh tối để kiểm tra cấu hình `data.yaml` nhận diện đủ 12 classes.

### Kiểm thử Tích hợp Thực nghiệm
* Chạy thử `run.py --phase 0` và `run.py --phase 1` để kiểm tra luồng tự động hóa hoạt động trơn tru.
