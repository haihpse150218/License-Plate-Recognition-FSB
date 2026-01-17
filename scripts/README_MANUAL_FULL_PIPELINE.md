# Hướng Dẫn Sử Dụng Manual Full Pipeline

## Tổng Quan

Script `manual_full_pipline.py` là hệ thống nhận diện biển số xe tự động sử dụng:
- **YOLO11**: Phát hiện vị trí biển số trong ảnh
- **TrOCR (Transformer-based OCR)**: Đọc ký tự từ biển số đã cắt
- **Hỗ trợ**: Biển số 1 dòng và 2 dòng

## Tính Năng

- ✅ Phát hiện tự động vị trí biển số trong ảnh
- ✅ Hỗ trợ cả biển số 1 dòng và 2 dòng
- ✅ OCR trực tiếp trên ảnh gốc (không qua preprocessing)
- ✅ Tự động cắt biển số 2 dòng thành 2 phần riêng biệt
- ✅ Làm sạch và chuẩn hóa text kết quả
- ✅ Lưu kết quả chi tiết (ảnh, text, thông tin)

## Yêu Cầu Hệ Thống

### Phần Mềm
- Python 3.10 trở lên
- Windows 10/11 (hoặc Linux/MacOS)

### Thư Viện Python
```bash
pip install ultralytics opencv-python numpy pillow torch torchvision transformers
```

### Model YOLO11
- File model: `runs/detect/latest_best.pt` (hoặc đường dẫn khác)
- Model phải được train trước với dữ liệu biển số xe

### Model TrOCR
- Tự động download từ Hugging Face lần đầu chạy
- Model: `microsoft/trocr-base-printed`
- Kích thước: ~500MB
- Cần kết nối internet lần đầu

## Cài Đặt

### Bước 1: Clone hoặc tải project
```bash
cd D:\MSE\Session_1\Basic_Python_For_Beginners\final_project\License-Plate-Recognition-FSB
```

### Bước 2: Cài đặt thư viện
```bash
pip install -r requirements.txt
```

Hoặc cài đặt thủ công:
```bash
pip install ultralytics opencv-python numpy pillow torch torchvision transformers
```

### Bước 3: Kiểm tra model YOLO11
Đảm bảo có file model tại:
- `runs/detect/latest_best.pt` (khuyến nghị)
- Hoặc `runs/detect/yolo11-license-plate_20260113_225923/weights/best.pt`

## Cách Sử Dụng

### Cách 1: Chạy trực tiếp (tự động tìm ảnh test)

```bash
cd scripts
python manual_full_pipline.py
```

Script sẽ tự động:
1. Tìm thư mục `data/test/images` hoặc `data/images`
2. Tìm model YOLO11 tự động
3. Xử lý tất cả ảnh trong thư mục

### Cách 2: Chạy với đường dẫn cụ thể

Sửa file `manual_full_pipline.py` ở phần `__main__`:

```python
if __name__ == "__main__":
    # Chỉ định đường dẫn ảnh
    test_folder = "D:/path/to/your/images"
    
    # Hoặc chỉ định 1 ảnh
    # test_folder = "D:/path/to/image.jpg"
    
    # Chỉ định model (nếu cần)
    # model_path = "runs/detect/your_model.pt"
    
    full_pipeline(test_folder)
```

### Cách 3: Sử dụng như module

```python
from scripts.manual_full_pipline import full_pipeline

# Xử lý 1 thư mục
full_pipeline("path/to/images")

# Xử lý 1 ảnh
full_pipeline("path/to/image.jpg")

# Chỉ định model
full_pipeline("path/to/images", model_path="runs/detect/model.pt")
```

## Cấu Trúc Kết Quả

Sau khi chạy, kết quả được lưu trong thư mục `results_final/`:

```
results_final/
├── ten_anh_1/
│   ├── plate_1/
│   │   ├── original_crop.jpg          # Ảnh biển số đã cắt
│   │   ├── top_line.jpg               # Dòng trên (nếu là biển số 2 dòng)
│   │   ├── bottom_line.jpg            # Dòng dưới (nếu là biển số 2 dòng)
│   │   └── result.txt                 # Thông tin kết quả OCR
│   ├── plate_2/                       # Biển số thứ 2 (nếu có)
│   └── result_with_boxes.jpg          # Ảnh gốc có vẽ bounding box
├── ten_anh_2/
│   └── ...
└── ...
```

### File result.txt

Mỗi biển số có file `result.txt` chứa:
```
Biển số: 30A12345
Confidence: 0.9000
Loại: 1 dòng
Xử lý: Ảnh gốc (không qua preprocessing)
Kích thước: 200x50
Tọa độ: (100, 200) -> (300, 250)
```

## Quy Trình Xử Lý

### 1. Phát hiện biển số (YOLO11)
- Đọc ảnh đầu vào
- Sử dụng YOLO11 để detect vị trí biển số
- Confidence threshold: 0.4 (có thể điều chỉnh)

### 2. Crop biển số
- Cắt vùng biển số từ ảnh gốc
- Lưu ảnh crop vào `original_crop.jpg`

### 3. Phân loại biển số
- Kiểm tra xem là biển số 1 dòng hay 2 dòng
- Phương pháp:
  - Tính horizontal projection
  - Tìm các vùng text liên tục
  - Kiểm tra khoảng cách giữa các vùng
  - Kiểm tra aspect ratio (tỷ lệ rộng/cao)

### 4. Xử lý theo loại

#### Biển số 1 dòng:
- OCR trực tiếp trên ảnh gốc
- Làm sạch text

#### Biển số 2 dòng:
- Cắt ảnh thành 2 phần (dòng trên và dòng dưới)
- OCR từng dòng riêng biệt
- Kết hợp kết quả

### 5. Làm sạch text
- Chuyển thành chữ hoa
- Loại bỏ ký tự đặc biệt
- Sửa lỗi nhận nhầm phổ biến:
  - O → 0
  - I → 1
  - Z → 2
  - S → 5
  - B → 8

### 6. Lưu kết quả
- Lưu ảnh crop
- Lưu ảnh 2 dòng (nếu có)
- Lưu thông tin vào file text
- Vẽ bounding box lên ảnh gốc

## Tùy Chỉnh

### Thay đổi confidence threshold YOLO11

Trong hàm `full_pipeline()`, tìm dòng:
```python
results = model(img_path, conf=0.4, save=False, verbose=False)
```

Thay đổi `conf=0.4` thành giá trị khác (0.0 - 1.0):
- Giá trị cao hơn: Chỉ detect biển số có confidence cao (ít false positive)
- Giá trị thấp hơn: Detect nhiều hơn (có thể có false positive)

### Thay đổi model TrOCR

Trong hàm `load_trocr_model()`, thay đổi `model_name`:
```python
load_trocr_model(model_name="microsoft/trocr-small-printed")  # Model nhỏ hơn, nhanh hơn
```

Các model có sẵn:
- `microsoft/trocr-base-printed` (mặc định, khuyến nghị)
- `microsoft/trocr-small-printed` (nhỏ hơn, nhanh hơn)
- `microsoft/trocr-base-handwritten` (cho chữ viết tay)

### Sử dụng CPU thay vì GPU

Trong hàm `load_trocr_model()`, thay đổi `device_name`:
```python
load_trocr_model(device_name="cpu")
```

### Thay đổi thư mục output

Trong hàm `full_pipeline()`, tìm dòng:
```python
output_dir = os.path.join(project_root, "results_final")
```

Thay đổi `"results_final"` thành tên thư mục khác.

## Xử Lý Lỗi

### Lỗi 1: "ModuleNotFoundError: No module named 'ultralytics'"

**Nguyên nhân**: Chưa cài đặt thư viện

**Giải pháp**:
```bash
pip install ultralytics opencv-python numpy pillow torch torchvision transformers
```

### Lỗi 2: "Không tìm thấy model!"

**Nguyên nhân**: Không tìm thấy file model YOLO11

**Giải pháp**:
1. Kiểm tra file model có tồn tại không
2. Chỉ định đường dẫn model cụ thể:
```python
full_pipeline("path/to/images", model_path="runs/detect/your_model.pt")
```

### Lỗi 3: "Lỗi load TrOCR model"

**Nguyên nhân**: 
- Chưa cài đặt transformers
- Không có kết nối internet (lần đầu download model)
- GPU không đủ bộ nhớ

**Giải pháp**:
1. Cài đặt lại:
```bash
pip install transformers torch torchvision
```

2. Sử dụng CPU:
```python
load_trocr_model(device_name="cpu")
```

3. Kiểm tra kết nối internet (lần đầu cần download model)

### Lỗi 4: "CUDA out of memory"

**Nguyên nhân**: GPU không đủ bộ nhớ

**Giải pháp**:
- Sử dụng CPU:
```python
load_trocr_model(device_name="cpu")
```

- Hoặc sử dụng model nhỏ hơn:
```python
load_trocr_model(model_name="microsoft/trocr-small-printed")
```

### Lỗi 5: "Không đọc được ảnh!"

**Nguyên nhân**: 
- File ảnh bị hỏng
- Định dạng không hỗ trợ
- Đường dẫn sai

**Giải pháp**:
- Kiểm tra file ảnh có tồn tại không
- Đảm bảo định dạng là JPG, JPEG hoặc PNG
- Kiểm tra đường dẫn có đúng không

## Ví Dụ Sử Dụng

### Ví dụ 1: Xử lý 1 thư mục ảnh

```python
from scripts.manual_full_pipline import full_pipeline

# Xử lý tất cả ảnh trong thư mục
full_pipeline("data/test/images")
```

### Ví dụ 2: Xử lý 1 ảnh

```python
from scripts.manual_full_pipline import full_pipeline

# Xử lý 1 ảnh cụ thể
full_pipeline("data/test/images/image1.jpg")
```

### Ví dụ 3: Chỉ định model

```python
from scripts.manual_full_pipline import full_pipeline

# Chỉ định model cụ thể
full_pipeline("data/test/images", model_path="runs/detect/my_model.pt")
```

## Hiệu Suất

### Tốc độ xử lý
- **GPU (CUDA)**: ~0.5-1 giây/ảnh
- **CPU**: ~2-5 giây/ảnh

### Độ chính xác
- **YOLO11 Detection**: ~95-98% (tùy thuộc vào model đã train)
- **TrOCR OCR**: ~90-95% (tùy thuộc vào chất lượng ảnh)

### Tối ưu hóa
- Sử dụng GPU để tăng tốc độ
- Batch processing nhiều ảnh cùng lúc (có thể thêm vào code)
- Resize ảnh nhỏ hơn nếu không cần độ phân giải cao

## Lưu Ý

1. **Lần đầu chạy**: TrOCR sẽ tự động download model (~500MB), cần kết nối internet
2. **Model YOLO11**: Phải được train trước với dữ liệu biển số xe
3. **Chất lượng ảnh**: Ảnh rõ nét sẽ cho kết quả tốt hơn
4. **Biển số 2 dòng**: Cần cắt chính xác để OCR tốt
5. **Ký tự đặc biệt**: Một số ký tự có thể bị nhận nhầm (O/0, I/1, Z/2, S/5, B/8)

## Tài Liệu Tham Khảo

- **YOLO11**: https://docs.ultralytics.com/
- **TrOCR**: https://huggingface.co/microsoft/trocr-base-printed
- **OpenCV**: https://docs.opencv.org/
- **Transformers**: https://huggingface.co/docs/transformers/

## Hỗ Trợ

Nếu gặp vấn đề, vui lòng:
1. Kiểm tra lại các bước cài đặt
2. Xem phần "Xử Lý Lỗi" ở trên
3. Kiểm tra log để xem lỗi cụ thể
4. Đảm bảo đã cài đặt đầy đủ thư viện

## Phiên Bản

- **Version**: 1.0
- **Ngày cập nhật**: 2025-01-13
- **Tác giả**: License Plate Recognition Team
