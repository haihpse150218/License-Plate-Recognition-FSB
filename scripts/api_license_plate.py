

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import os
import base64
import io
from PIL import Image
from typing import List, Optional
import uvicorn

# Import các function từ manual_full_pipline.py
from manual_full_pipline import (
    load_trocr_model,
    detect_two_line_plate,
    process_two_line_plate,
    process_single_line_plate
)
from ultralytics import YOLO

app = FastAPI(
    title="License Plate Recognition API",
    description="API nhận diện biển số xe Việt Nam sử dụng YOLO11 và TrOCR",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


yolo_model = None
trocr_loaded = False
project_root = None


class PlateResult(BaseModel):

    plate_text: str
    confidence: float
    plate_type: str  
    bbox: dict  
    detection_confidence: float  


class DetectionResponse(BaseModel):

    success: bool
    message: str
    plates: List[PlateResult]
    image_size: dict  # {"width": int, "height": int}


def init_models():

    global yolo_model, trocr_loaded, project_root
    
    # Tìm project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Tìm YOLO model
    possible_paths = [
        os.path.join(project_root, "runs", "detect", "latest_best.pt"),
        os.path.join(project_root, "runs", "detect", "yolo11-license-plate_20260113_225923", "weights", "best.pt"),
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        raise FileNotFoundError("Không tìm thấy YOLO model! Vui lòng kiểm tra đường dẫn.")
    
    # Load YOLO model
    print(f"Đang load YOLO model từ: {model_path}")
    yolo_model = YOLO(model_path)
    print("YOLO model đã load")
    
    # Load TrOCR model
    print("Đang khởi tạo TrOCR...")
    if load_trocr_model(model_name="microsoft/trocr-base-printed", device_name="auto"):
        trocr_loaded = True
        print("TrOCR model đã load")
    else:
        raise RuntimeError("Không thể load TrOCR model")


def image_from_bytes(image_bytes: bytes) -> np.ndarray:

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Không thể decode ảnh")
    return img


def image_from_base64(base64_string: str) -> np.ndarray:

    # Loại bỏ prefix nếu có (data:image/jpeg;base64,...)
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    image_bytes = base64.b64decode(base64_string)
    return image_from_bytes(image_bytes)


def detect_plates(image: np.ndarray, conf_threshold: float = 0.4) -> List[PlateResult]:

    global yolo_model, trocr_loaded
    
    if yolo_model is None:
        raise RuntimeError("YOLO model chưa được khởi tạo")
    
    if not trocr_loaded:
        raise RuntimeError("TrOCR model chưa được khởi tạo")
    
    h_img, w_img = image.shape[:2]
    results = []
    
    # Detect biển số bằng YOLO
    yolo_results = yolo_model(image, conf=conf_threshold, save=False, verbose=False)
    
    # Xử lý từng biển số được detect
    for r in yolo_results:
        boxes = r.boxes
        for box in boxes:
            # Lấy tọa độ bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detection_conf = float(box.conf[0].item())
            
            # Bảo vệ tọa độ
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            
            # Kiểm tra kích thước hợp lệ
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue
            
            # Crop biển số
            cropped = image[y1:y2, x1:x2]
            
            # Phân loại: 1 dòng hay 2 dòng
            is_two_line = detect_two_line_plate(cropped)
            plate_type = "2-line" if is_two_line else "1-line"
            
            # OCR biển số
            if is_two_line:
                plate_text, ocr_confidence, _ = process_two_line_plate(cropped, save_dir=None)
            else:
                plate_text, ocr_confidence, _ = process_single_line_plate(cropped, save_dir=None)
            
            # Tạo kết quả
            result = PlateResult(
                plate_text=plate_text,
                confidence=ocr_confidence,
                plate_type=plate_type,
                bbox={"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                detection_confidence=detection_conf
            )
            results.append(result)
    
    return results


@app.on_event("startup")
async def startup_event():

    try:
        init_models()
        print("API đã sẵn sàng!")
    except Exception as e:
        print(f"Lỗi khởi tạo: {e}")
        raise


@app.get("/")
async def root():

    return {
        "status": "running",
        "message": "License Plate Recognition API",
        "endpoints": {
            "POST /detect": "Detect biển số từ ảnh (file upload)",
            "POST /detect/base64": "Detect biển số từ ảnh (base64)",
            "GET /health": "Health check"
        }
    }


@app.get("/health")
async def health():
    """Health check với thông tin models"""
    global yolo_model, trocr_loaded
    
    return {
        "status": "healthy",
        "yolo_loaded": yolo_model is not None,
        "trocr_loaded": trocr_loaded
    }


@app.post("/detect", response_model=DetectionResponse)
async def detect_from_file(
    file: UploadFile = File(..., description="Ảnh cần detect biển số"),
    conf: float = 0.4
):


    try:
        # Đọc file
        image_bytes = await file.read()
        image = image_from_bytes(image_bytes)
        
        # Detect biển số
        plates = detect_plates(image, conf_threshold=conf)
        
        h, w = image.shape[:2]
        
        return DetectionResponse(
            success=True,
            message=f"Detect thành công {len(plates)} biển số",
            plates=plates,
            image_size={"width": w, "height": h}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")


class Base64Request(BaseModel):

    image: str 
    conf: Optional[float] = 0.4  


@app.post("/detect/base64", response_model=DetectionResponse)
async def detect_from_base64(request: Base64Request):

    try:
        # Decode base64
        image = image_from_base64(request.image)
        
        # Detect biển số
        plates = detect_plates(image, conf_threshold=request.conf)
        
        h, w = image.shape[:2]
        
        return DetectionResponse(
            success=True,
            message=f"Detect thành công {len(plates)} biển số",
            plates=plates,
            image_size={"width": w, "height": h}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")


if __name__ == "__main__":
    # Chạy server
    # uvicorn api_license_plate:app --host 0.0.0.0 --port 8000 --reload
    uvicorn.run(
        "api_license_plate:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
