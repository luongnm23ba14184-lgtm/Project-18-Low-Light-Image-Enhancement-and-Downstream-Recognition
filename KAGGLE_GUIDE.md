# HƯỚNG DẪN CHI TIẾT HUẤN LUYỆN MODEL TRÊN KAGGLE (GPU T4 MIỄN PHÍ)
## Đề tài: Low-Light Image Enhancement and Downstream Recognition (Project 18)

> **Mục tiêu:** Tận dụng **GPU NVIDIA T4 (hoặc P100) miễn phí** của Kaggle để huấn luyện toàn diện mô hình Zero-DCE và 4 kịch bản YOLOv8 trong khoảng **30 – 45 phút** (thay vì chạy 15 – 20 tiếng trên CPU máy tính cá nhân).

---

## 1. Chuẩn Bị & Khởi Tạo Notebook Trên Kaggle

1. Truy cập [Kaggle.com](https://www.kaggle.com/) và đăng nhập tài khoản.
2. Bấm vào nút **`Create`** (góc trên bên trái) $\rightarrow$ Chọn **`New Notebook`**.
3. Đổi tên Notebook thành: `Project18_LowLight_YOLOv8`.

---

## 2. Cấu Hình Cực Kỳ Quan Trọng (BẮT BUỘC)

Ở thanh công cụ bên phải màn hình (**Notebook Options / Settings**), bạn cần cài đặt chính xác 2 thông số sau:

```
┌─────────────────────────────────────────────────────────────┐
│ ⚙️ NOTEBOOK SETTINGS                                         │
│                                                             │
│ 1. Accelerator:  [ GPU T4 x2 ]  (hoặc GPU P100)  ◄── BẬT    │
│ 2. Internet:     [ Internet on ]                 ◄── BẬT    │
│ 3. Environment:  [ Always use latest environment ]          │
└─────────────────────────────────────────────────────────────┘
```

> [!CAUTION]
> * **Nếu không bật `GPU`:** Code sẽ chạy bằng CPU và mất nhiều tiếng.
> * **Nếu không bật `Internet on`:** Kaggle sẽ chặn mạng, bạn sẽ không thể `git clone`, không cài được `pip install`, và không tải được trọng số mô hình!

---

## 3. Các Ô Lệnh (Code Cells) Thực Thi Từng Bước

Sau khi cấu hình xong, bạn lần lượt tạo các ô Code Cell trong Notebook và bấm nút **Run (`Ctrl + Enter`)**:

### Cell 1: Kiểm tra GPU của Kaggle
Xác nhận xem máy chủ ảo đã nhận diện GPU NVIDIA hay chưa:
```python
# Kiểm tra thông số GPU
!nvidia-smi

import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name:    {torch.cuda.get_device_name(0)}")
```

---

### Cell 2: Clone Mã Nguồn Dự Án Từ GitHub
Chuyển vào thư mục làm việc `/kaggle/working/` và tải toàn bộ source code của nhóm về:
```python
import os
import shutil

# Chuyển vào thư mục làm việc có quyền ghi
%cd /kaggle/working

# Xóa nếu đã tồn tại thư mục cũ (để cập nhật mới nhất)
if os.path.exists("Project18"):
    shutil.rmtree("Project18")

# Clone dự án từ GitHub
!git clone https://github.com/luongnm23ba14184-lgtm/Project-18-Low-Light-Image-Enhancement-and-Downstream-Recognition.git Project18

# Chuyển vào thư mục dự án
%cd /kaggle/working/Project18
!ls -la
```

---

### Cell 3: Cài Đặt Thư Viện Phụ Thuộc
Cài đặt Ultralytics YOLOv8 và các thư viện hỗ trợ:
```python
# Cài đặt thư viện theo requirements.txt
!pip install -q ultralytics opencv-python matplotlib scipy pandas tqdm pyyaml
print("✅ Cài đặt thư viện hoàn tất!")
```

---

### Cell 4: Kiểm Tra Môi Trường & Toàn Vẹn Dữ Liệu (Phase 0 & 1)
Kiểm tra xem 7,345 ảnh và 7,345 file nhãn của ExDark đã sẵn sàng:
```python
# Chạy Phase 0 (Setup) và Phase 1 (Data Verification)
!python run.py --phase 0,1
```
*Kết quả mong đợi:* Hệ thống báo `🚀 Using GPU: Tesla T4`, nhận diện đủ 12 classes và đếm đủ `Train: 5142`, `Valid: 1469`, `Test: 734`.

---

### Cell 5: Huấn Luyện Giai Đoạn 1 - Zero-DCE (Phase 2)
Huấn luyện mạng nơ-ron tự giám sát DCE-Net, đo điểm NIQE/BRISQUE và tăng cường sáng toàn bộ dataset:
```python
# Chạy Phase 2: Train Zero-DCE 10 epochs và tạo dataset ảnh sáng
!python run.py --phase 2 --epochs_dce 10
```
*Thời gian chạy:* ~5 – 8 phút trên GPU T4.  
*Kết quả:* Checkpoint `zerodce_best.pth` được lưu vào `Results/weights/` và toàn bộ ảnh sáng lưu vào `Dataset/exdark_yolo_zerodce/`.

---

### Cell 6: Huấn Luyện Giai Đoạn 2 - 4 Kịch Bản YOLOv8 (Phase 3 & 4)
Huấn luyện mô hình YOLOv8 trên ảnh tối và ảnh sáng, đánh giá 4 kịch bản đối chứng và xuất báo cáo:
```python
# Chạy Phase 3 (Train YOLOv8 40 epochs) và Phase 4 (Tổng hợp kết quả + Biểu đồ)
!python run.py --phase 3,4 --epochs_yolo 40 --batch_size 16
```
*Thời gian chạy:* ~25 – 35 phút trên GPU T4.  
*Kết quả:* Tự động sinh ra:
* `Results/weights/yolov8n_dark_best.pt`
* `Results/weights/yolov8n_zerodce_best.pt`
* `Results/comparisons_table.csv`
* `Results/figures/map_comparison.png`
* `Results/figures/enhancement_comparison.png`

> 💡 **Mẹo chạy trọn gói từ A đến Z:** Bạn cũng có thể gộp Cell 4, 5, 6 thành một lệnh duy nhất:
> ```bash
> !python run.py --phase all --epochs_dce 10 --epochs_yolo 40 --batch_size 16
> ```

---

### Cell 7: Xem Bảng Kết Quả & Biểu Đồ Trực Tiếp Trên Kaggle
Hiển thị trực quan bảng số liệu và hình ảnh đối chứng ngay trong cell:
```python
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# 1. Hiển thị Bảng kết quả định lượng
csv_path = "Results/comparisons_table.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print("📊 BẢNG TỔNG HỢP KẾT QUẢ THỰC NGHIỆM ĐỐI CHỨNG:")
    display(df)

# 2. Hiển thị Biểu đồ so sánh mAP
chart_path = "Results/figures/map_comparison.png"
if os.path.exists(chart_path):
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(cv2.imread(chart_path), cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Biểu đồ So sánh mAP 4 Kịch bản", fontsize=14, fontweight="bold")
    plt.show()

# 3. Hiển thị Ảnh so sánh 4 khung hình song song
qual_path = "Results/figures/enhancement_comparison.png"
if os.path.exists(qual_path):
    plt.figure(figsize=(18, 5))
    plt.imshow(cv2.cvtColor(cv2.imread(qual_path), cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Ảnh Đối chứng Side-by-Side (Ảnh tối | CLAHE | Zero-DCE | YOLOv8)", fontsize=14, fontweight="bold")
    plt.show()
```

---

### Cell 8: Tải Toàn Bộ File Trọng Số & Kết Quả Về Máy Tính
Nén thư mục `Results/` thành 1 file ZIP và tạo link tải 1-click về máy tính:
```python
import shutil
from IPython.display import FileLink

# Nén toàn bộ thư mục Results
archive_name = "/kaggle/working/Project18_Results"
shutil.make_archive(archive_name, 'zip', "/kaggle/working/Project18/Results")

print("🎉 Đã nén xong kết quả!")
print("👉 Bấm vào đường link dưới đây để tải về máy tính:")
FileLink(r'Project18_Results.zip')
```

---

## 4. Mẹo Nâng Cao: Chạy Ẩn Ngầm (Không Cần Bật Màn Hình Chờ Đợi)

Nếu không muốn ngồi canh máy tính trong lúc train:
1. Bấm vào nút **`Save Version`** (ở góc trên bên phải màn hình).
2. Chọn **`Save & Run All (Commit)`** $\rightarrow$ Bấm **`Save`**.
3. **Tắt máy tính hoặc đóng tab trình duyệt đi ngủ!**
4. Máy chủ đám mây của Kaggle sẽ tự động chạy toàn bộ code từ Cell 1 đến Cell 8. 
5. Sáng hôm sau, mở lại link Notebook đó, vào tab **`Output`** ở góc phải là bạn có thể tải ngay file `Project18_Results.zip` chứa đầy đủ mô hình đã train và biểu đồ hoàn chỉnh.

---

## 5. Xử Lý Các Lỗi Thường Gặp (Troubleshooting)

| Lỗi gặp phải | Nguyên nhân | Cách khắc phục |
| :--- | :--- | :--- |
| `CUDA out of memory` | Bộ nhớ VRAM bị tràn do batch size quá lớn | Giảm `--batch_size 16` xuống `--batch_size 8` trong lệnh chạy. |
| `Failed to clone repository` | Quên chưa bật Internet trên Kaggle | Vào cột phải (Settings) $\rightarrow$ Gạt **Internet** sang **On**. |
| `No such file or directory: data.yaml` | Chạy lệnh sai thư mục | Đảm bảo đã chạy `%cd /kaggle/working/Project18` trước khi chạy `run.py`. |
| `Your quota has been exceeded` | Hết 30 tiếng GPU tuần này của Kaggle | Đổi Accelerator từ `GPU T4 x2` sang `GPU P100` hoặc dùng tài khoản Kaggle của thành viên khác trong nhóm. |
