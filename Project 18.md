# Phân Tích Đề Tài: Low-Light Image Enhancement and Downstream Recognition

---

## 1. Tổng quan Đề tài

* **Tên đề tài:** *Low-Light Image Enhancement and Downstream Recognition* (Tăng cường chất lượng ảnh thiếu sáng và bài toán nhận diện hạ nguồn).
* **Mục tiêu chính trong mô tả:**
  1. **Enhance images captured under low-light conditions using deep-learning models:** Làm sáng, khử nhiễu, cân bằng màu sắc cho các bức ảnh chụp trong điều kiện ánh sáng yếu/ban đêm bằng Deep Learning.
  2. **Investigate whether improved visual quality also leads to better performance on downstream recognition tasks:** Đánh giá xem việc ảnh trông "đẹp hơn với mắt người" (visual quality) có thực sự giúp các mô hình AI nhận diện (Object Detection, Classification, Segmentation, Face Recognition, v.v.) hoạt động chính xác hơn hay không.

---

## 2. Đề tài này thuộc dạng bài toán nào trong Computer Vision?

Đây là một đề tài dạng **Pipeline kết hợp hai giai đoạn (Two-Stage / Multi-task)**, không thuần túy là một tác vụ đơn lẻ:

### Giai đoạn 1: Low-Level Vision (Xử lý ảnh cơ bản)
* **Dạng bài:** **Image Restoration / Image Enhancement / Image-to-Image Translation**.
* **Đặc điểm:** Input là ảnh thiếu sáng (kém tương phản, nhiều noise), Output là ảnh đã được tăng cường sáng và làm rõ nét.
* **Mục tiêu:** Tối ưu các chỉ số cảm quan hình ảnh (PSNR, SSIM, LPIPS, NIQE).

### Giai đoạn 2: High-Level Vision (Thị giác máy tính ngữ nghĩa / Hạ nguồn - Downstream Tasks)
* **Dạng bài:** Các tác vụ phân tích cấp cao do bạn tự chọn thử nghiệm, tiêu biểu gồm:
  * **Object Detection** (Phát hiện vật thể ban đêm: xe cộ, người đi bộ, biển báo).
  * **Image Classification** (Phân loại bối cảnh/đối tượng).
  * **Semantic / Instance Segmentation** (Phân vùng ảnh ban đêm cho xe tự hành/camera an ninh).
  * **Face Detection / Recognition** (Nhận diện khuôn mặt thiếu sáng).
* **Mục tiêu:** Tối ưu độ chính xác nhận diện ($mAP$, Accuracy, $mIoU$, $F_1$-score).

> **Cốt lõi khoa học của đề tài:**  
> Trong thực tế, nhiều thuật toán làm sáng ảnh tối ưu cho mắt người (nhìn thuận mắt) nhưng lại tạo ra các artifacts (nhiễu vi mô, mờ biên) khiến mạng High-Level (như YOLO, Mask R-CNN) nhận diện kém hơn cả ảnh gốc chưa xử lý. Nghiên cứu này nhằm chứng minh, so sánh và tìm ra giải pháp tối ưu cho nghịch lý này.

---

## 3. Kiến trúc & Hướng tiếp cận (Pipeline tổng thể)

Có 3 mô hình triển khai chính:

```
[Ảnh tối (Low-Light Image)]
          │
          ▼
┌───────────────────────────────────────────────┐
│ Giai đoạn 1: Image Enhancement (DIP Pipeline) │ (Không gian màu LAB -> CLAHE -> Bilateral Filter)
└───────────────────────────────────────────────┘
          │
          ▼ [Ảnh sau khi làm sáng & khử nhiễu biên]
┌───────────────────────────────────────────────┐
│ Giai đoạn 2: Downstream Model                 │ (YOLOv8 Detection)
└───────────────────────────────────────────────┘
          │
          ▼
   [Kết quả dự đoán] (Bounding boxes + Class labels)
```

### Các hướng kết hợp (Paradigms):
1. **Direct Baseline:** Huấn luyện và đánh giá trực tiếp YOLOv8 trên ảnh tối gốc $\rightarrow$ $mAP_{dark}$.
2. **Cascaded Inference (Không retrain):** Cho ảnh tối qua pipeline CLAHE + Bilateral rồi đưa thẳng vào mô hình YOLOv8 Baseline $\rightarrow$ $mAP_{clahe\_cascaded}$.
3. **Retrained Pipeline:** Huấn luyện lại mô hình YOLOv8 trực tiếp trên tập ảnh đã được tiền xử lý bởi CLAHE + Bilateral Filter $\rightarrow$ $mAP_{clahe\_retrained}$.

---

## 4. Lựa chọn Phương pháp & Công nghệ đề xuất

### 4.1. Các kỹ thuật Enhancement (Khâu 1) - Xử lý ảnh Kinh điển (DIP)
| Phân loại | Kỹ thuật tiêu biểu | Ưu / Nhược điểm |
| :--- | :--- | :--- |
| **Point Processing & Color Space** | **CIE LAB / HSV Conversion** | Tách riêng kênh độ sáng $L$, giữ nguyên thông tin màu $A, B$ để chống méo màu tuyệt đối. |
| **Histogram Processing** | **Global Histogram Equalization (HE)** | Cân bằng toàn cục cơ bản trong giáo trình; nhược điểm: dễ làm cháy sáng các bóng đèn và khuếch đại nhiễu. |
| **Adaptive Histogram (Đề xuất)** | **CLAHE (Contrast Limited)** | Cân bằng theo từng ô cục bộ ($8 \times 8$) kết hợp cắt ngọn trần tương phản (`clipLimit=2.0`); làm sáng tự nhiên, không khuếch đại nhiễu. |
| **Spatial Filtering** | **Bilateral Filter (Lọc song phương)** | Làm mịn các hạt nhiễu cảm biến lộ ra sau khi tăng sáng nhưng **bảo toàn nguyên vẹn độ sắc nét của đường biên**, hỗ trợ đắc lực cho YOLOv8. |

### 4.2. Các mô hình Downstream Task (Khâu 2)
* **Object Detection (Mô hình cốt lõi):**
  * `YOLOv8` (Ultralytics: phiên bản `yolov8n` hoặc `yolov8s`): Trọng số nhẹ, huấn luyện nhanh trên Google Colab T4, trích xuất đặc trưng đa tầng (PAN-FPN) và đầu dự đoán anchor-free.

### 4.3. Các Tinh Chỉnh Khớp Nối GĐ1 và GĐ2 (Co-Design Tuning)
1. **Khống chế tham số GĐ1:** Giới hạn `clipLimit = 2.0` (tránh cháy sáng đèn xe/đèn đường) và lọc Bilateral nhẹ ($d=5, \sigma=35$) để không làm mờ các vật thể nhỏ (`Bottle`, `Cup`, chân ghế `Chair`).
2. **Cấu hình lại Data Augmentation YOLOv8:** Giảm `hsv_v = 0.1` (tránh model tự động làm tối ảnh) và tắt mosaic ở 10 epochs cuối (`close_mosaic = 10`).
3. **Thực hiện Ablation Study:** Khảo sát định lượng tác động của từng siêu tham số lên chỉ số $mAP@0.5$.

---

## 5. Bộ dữ liệu (Datasets) phù hợp

* **ExDark (Exclusively Dark Dataset):** Bộ dữ liệu tiêu chuẩn lý tưởng nhất vì chứa **7.363 ảnh thiếu sáng tự nhiên được gán nhãn sẵn Bounding Box cho 12 lớp vật thể** (`Bicycle, Bus, Car, Cat, Dog, People...`).

---

## 6. Kế hoạch triển khai thực nghiệm chi tiết (Step-by-Step)

### Bước 1: Chuẩn bị Dữ liệu & Xây dựng Baseline (Tuần 1)
1. Tải tập dữ liệu **ExDark** (định dạng YOLO chia sẵn Train/Val/Test).
2. Huấn luyện `yolov8n.pt` trực tiếp trên ảnh tối gốc $\rightarrow$ Ghi nhận $mAP_{dark}$ làm mốc cơ sở.

### Bước 2: Triển khai Pipeline Xử Lý Ảnh DIP (Tuần 2)
1. Cài đặt hàm `enhance_clahe_bilateral()` bằng OpenCV:
   * Chuyển $BGR \rightarrow LAB$.
   * Áp dụng `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))` trên kênh $L$.
   * Lọc khử nhiễu giữ biên bằng `cv2.bilateralFilter()`.
   * Gộp kênh và chuyển về $BGR$.
2. Đo lường chất lượng ảnh sau khi làm sáng bằng các chỉ số không tham chiếu: **NIQE** và **BRISQUE**.
3. Batch convert toàn bộ dataset ExDark sang thư mục `dataset/exdark_clahe/`.

### Bước 3: Thử nghiệm Kết hợp & Đánh giá Đối chứng (Cuối Tuần 2 - Nửa Tuần 3)
1. **Cascaded Testing:** Đưa ảnh sau khi làm sáng qua YOLOv8 Dark Baseline $\rightarrow$ Đo $mAP_{cascaded}$.
2. **Retrain Testing:** Huấn luyện model YOLOv8 mới trên tập ảnh đã làm sáng bởi CLAHE $\rightarrow$ Đo $mAP_{retrained}$.
3. So sánh: Tiền xử lý ảnh kinh điển (DIP) có giúp tăng $mAP$ không? Mức tăng là bao nhiêu %?

---

## 7. Các tiêu chí đánh giá (Evaluation Metrics)

* **Về chất lượng ảnh (Enhancement):** $NIQE \downarrow$, $BRISQUE \downarrow$ (No-reference quality score).
* **Về tác vụ nhận diện (Downstream Task):** $Precision$, $Recall$, $mAP_{50}$, $mAP_{50:95}$.
* **Về hiệu năng:** FPS (Frames per second), Latency của toàn bộ pipeline (ms).