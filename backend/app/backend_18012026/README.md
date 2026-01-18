# Backend API - Nhận diện biển số xe Việt Nam

Module backend cung cấp API REST để nhận diện biển số xe Việt Nam sử dụng YOLO11 (phát hiện) và TrOCR (OCR).

## 📁 Cấu trúc thư mục

```
backend_18012026/
├── api_license_plate.py      # FastAPI server chính
├── manual_full_pipline.py    # Module xử lý pipeline OCR
└── README.md                 # File hướng dẫn này
```

## 🎯 Chức năng

- **Phát hiện biển số**: Sử dụng YOLO11 để phát hiện vị trí biển số trong ảnh
- **Nhận diện ký tự**: Sử dụng TrOCR để đọc text từ biển số
- **Hỗ trợ 2 loại biển số**:
  - Biển số 1 dòng (biển số dài)
  - Biển số 2 dòng (biển số vuông)
- **API REST**: Cung cấp các endpoint để tích hợp với frontend

## 📋 Yêu cầu hệ thống

### Python
- Python 3.8 trở lên

### Dependencies
Các thư viện cần thiết (xem `requirements.txt` ở project root):
- `fastapi`
- `uvicorn`
- `opencv-python` (cv2)
- `numpy`
- `pillow` (PIL)
- `ultralytics` (YOLO11)
- `transformers` (TrOCR)
- `torch` (PyTorch)
- `pydantic`

### Model files
- YOLO11 model: Phải có file `.pt` trong thư mục `runs/detect/` ở project root
  - `runs/detect/latest_best.pt` hoặc
  - `runs/detect/yolo11-license-plate_20260113_225923/weights/best.pt`
- TrOCR model: Tự động download từ HuggingFace khi chạy lần đầu (`microsoft/trocr-base-printed`)

## 🚀 Cài đặt

1. **Cài đặt dependencies** (từ project root):
```bash
pip install -r requirements.txt
```

2. **Kiểm tra model YOLO**:
Đảm bảo file model YOLO tồn tại tại một trong các đường dẫn:
- `runs/detect/latest_best.pt`
- `runs/detect/yolo11-license-plate_20260113_225923/weights/best.pt`

3. **Cấu trúc project**:
```
License-Plate-Recognition-FSB/
├── backend/
│   └── app/
│       └── backend_18012026/    # Thư mục này
├── runs/
│   └── detect/                  # Chứa YOLO model
└── ...
```

## 💻 Cách sử dụng

### 1. Chạy API Server

#### Cách 1: Chạy trực tiếp file Python
```bash
cd backend/app/backend_18012026
python api_license_plate.py
```

#### Cách 2: Sử dụng uvicorn
```bash
cd backend/app
uvicorn backend_18012026.api_license_plate:app --host 0.0.0.0 --port 8000 --reload
```

Server sẽ chạy tại: `http://localhost:8000`

### 2. Sử dụng API

#### Health Check
```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "yolo_loaded": true,
  "trocr_loaded": true
}
```

#### Xem tất cả endpoints
```bash
curl http://localhost:8000/
```

Hoặc mở trình duyệt: `http://localhost:8000/docs` (Swagger UI)

## 📡 API Endpoints

### 1. `GET /`
Thông tin về API và các endpoints có sẵn.

### 2. `GET /health`
Kiểm tra trạng thái server và models.

**Response:**
```json
{
  "status": "healthy",
  "yolo_loaded": true,
  "trocr_loaded": true
}
```

### 3. `POST /detect`
Detect biển số từ file ảnh upload.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Parameters:
  - `file`: File ảnh (jpg, png, jpeg)
  - `conf`: Confidence threshold (optional, default: 0.4)

**Example (cURL):**
```bash
curl -X POST "http://localhost:8000/detect?conf=0.4" \
  -F "file=@path/to/image.jpg"
```

**Response:**
```json
{
  "success": true,
  "message": "Detect thành công 1 biển số",
  "plates": [
    {
      "plate_text": "30A12345",
      "confidence": 0.95,
      "plate_type": "1-line",
      "bbox": {
        "x1": 100,
        "y1": 200,
        "x2": 300,
        "y2": 250
      },
      "detection_confidence": 0.92
    }
  ],
  "image_size": {
    "width": 1920,
    "height": 1080
  }
}
```

### 4. `POST /detect/base64`
Detect biển số từ ảnh dạng base64.

**Request:**
- Method: `POST`
- Content-Type: `application/json`
- Body:
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "conf": 0.4
}
```

**Example (cURL):**
```bash
curl -X POST "http://localhost:8000/detect/base64" \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/jpeg;base64,...",
    "conf": 0.4
  }'
```

**Response:** Giống như endpoint `/detect`

### 5. `POST /ocr`
Endpoint tương thích với frontend Streamlit.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Parameters:
  - `file`: File ảnh
  - `conf`: Confidence threshold (optional, default: 0.4)

**Response:**
```json
{
  "status": "success",
  "detections": [
    {
      "plate": "30A12345",
      "confidence": 0.95,
      "bbox": [100, 200, 300, 250]
    }
  ],
  "processed_image_url": "data:image/jpeg;base64,...",
  "type": "1-line"
}
```

## 📝 Ví dụ sử dụng

### Python (requests)

```python
import requests

# Upload file ảnh
url = "http://localhost:8000/detect"
files = {"file": open("image.jpg", "rb")}
params = {"conf": 0.4}

response = requests.post(url, files=files, params=params)
result = response.json()

print(f"Tìm thấy {len(result['plates'])} biển số:")
for plate in result['plates']:
    print(f"  - {plate['plate_text']} (confidence: {plate['confidence']:.2f})")
```

### JavaScript (fetch)

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/detect?conf=0.4', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => {
  console.log('Kết quả:', data);
  data.plates.forEach(plate => {
    console.log(`Biển số: ${plate.plate_text}`);
  });
});
```

## 🔧 Module `manual_full_pipline.py`

Module này chứa các hàm xử lý pipeline OCR:

### Các hàm chính:

1. **`load_trocr_model(model_name, device_name)`**
   - Load TrOCR model từ HuggingFace
   - Tự động chọn GPU nếu có

2. **`detect_two_line_plate(cropped)`**
   - Phát hiện xem biển số là 1 dòng hay 2 dòng
   - Trả về `True` nếu là 2 dòng

3. **`process_two_line_plate(cropped, save_dir)`**
   - Xử lý biển số 2 dòng
   - Cắt thành 2 dòng và OCR riêng biệt
   - Trả về: `(plate_text, confidence, version)`

4. **`process_single_line_plate(cropped, save_dir)`**
   - Xử lý biển số 1 dòng
   - OCR trực tiếp trên ảnh
   - Trả về: `(plate_text, confidence, version)`

5. **`full_pipeline(source_path, model_path)`**
   - Pipeline đầy đủ để xử lý nhiều ảnh
   - Có thể chạy độc lập từ command line

### Sử dụng module độc lập:

```python
from manual_full_pipline import (
    load_trocr_model,
    detect_two_line_plate,
    process_two_line_plate,
    process_single_line_plate
)

# Load model
load_trocr_model()

# Xử lý ảnh
import cv2
image = cv2.imread("plate.jpg")
is_two_line = detect_two_line_plate(image)

if is_two_line:
    text, conf, _ = process_two_line_plate(image)
else:
    text, conf, _ = process_single_line_plate(image)

print(f"Biển số: {text}, Confidence: {conf}")
```

## ⚠️ Troubleshooting

### Lỗi: "Không tìm thấy YOLO model"
- **Nguyên nhân**: File model không tồn tại tại các đường dẫn mặc định
- **Giải pháp**: 
  - Kiểm tra file model tại `runs/detect/`
  - Hoặc chỉnh sửa `possible_paths` trong hàm `init_models()`

### Lỗi: "Không thể load TrOCR model"
- **Nguyên nhân**: Chưa cài đặt `transformers` hoặc `torch`
- **Giải pháp**: 
  ```bash
  pip install transformers torch torchvision
  ```

### Lỗi: Import module không được
- **Nguyên nhân**: Đường dẫn import không đúng
- **Giải pháp**: Đảm bảo chạy từ đúng thư mục hoặc cài đặt package

### Model load chậm
- **Lần đầu chạy**: TrOCR sẽ download model từ HuggingFace (có thể mất vài phút)
- **Các lần sau**: Model đã cache, sẽ nhanh hơn

### Kết quả OCR không chính xác
- Thử điều chỉnh `conf` threshold (giảm xuống 0.3 nếu không detect được)
- Kiểm tra chất lượng ảnh đầu vào
- Đảm bảo biển số rõ ràng, không bị mờ hoặc che khuất

## 📊 Cấu trúc Response

### PlateResult
```python
{
  "plate_text": str,           # Text biển số (ví dụ: "30A12345")
  "confidence": float,          # Confidence của OCR (0-1)
  "plate_type": str,           # "1-line" hoặc "2-line"
  "bbox": {                    # Bounding box
    "x1": int,
    "y1": int,
    "x2": int,
    "y2": int
  },
  "detection_confidence": float # Confidence của YOLO detection (0-1)
}
```

## 🔗 Liên kết

- **Frontend**: Xem thư mục `frontend/` để tích hợp
- **Documentation**: Swagger UI tại `http://localhost:8000/docs`
- **Alternative Docs**: ReDoc tại `http://localhost:8000/redoc`

## 📄 License

Xem file LICENSE ở project root.

---

**Lưu ý**: Đảm bảo đã train và có model YOLO11 trước khi sử dụng API. Model TrOCR sẽ tự động download từ HuggingFace khi chạy lần đầu.
