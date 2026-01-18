from pydantic import BaseModel
import cv2
import numpy as np
import os
import base64
import io
from PIL import Image
from typing import List, Optional
import uvicorn
import uuid
import config
import re
import json
# Import các function từ manual_full_pipline.py
from manual_full_pipline import (
    load_trocr_model,
    detect_two_line_plate,
    process_two_line_plate,
    process_single_line_plate
)

from ultralytics import YOLO



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
    model_path = config.MODEL_PATH   
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


def detect_plates(image: np.ndarray, conf_threshold: float = 0.4, isDebug=False) -> List[PlateResult]:

    global yolo_model, trocr_loaded
    img_name = f"temp_{uuid.uuid4().hex}.jpg"
    if yolo_model is None:
        raise RuntimeError("YOLO model chưa được khởi tạo")
    
    if not trocr_loaded:
        raise RuntimeError("TrOCR model chưa được khởi tạo")
    
    h_img, w_img = image.shape[:2]
    detections = []
    
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
                
            #Lưu crop nếu cần debug
            safe_plate = re.sub(r'[<>:"/\\|?*]', '_', plate_text)
            crop_name = f"{img_name}_plate_{safe_plate}.jpg"
            crop_path = os.path.join("crop_images", crop_name)
            detections.append(
                {
                    "plate": plate_text,
                    "confidence": detection_conf,
                    "bbox": [x1, y1, x2, y2],
                    "crop_path": crop_path if isDebug else None,
                })
    #Lưu crop nếu cần debug
    print("detections:", detections)
    if isDebug:
        os.makedirs("crop_images", exist_ok=True)
        cv2.imwrite(crop_path, cropped)
        
    #Lưu result nếu cần debug
    if isDebug:
        debug_img = image.copy()
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(debug_img, f"{plate_text} ({detection_conf:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        debug_path = os.path.join("results", f"processed_{img_name}")
        os.makedirs("results", exist_ok=True)
        cv2.imwrite(debug_path, debug_img)
    # Phân loại xe
    type_vehicle = detect_plate_type(plate_text)
    print("===================================")
    res = {
        "processed_image": debug_path if isDebug else None,
        "detections": detections,
        "type": type_vehicle
    }
    print(json.dumps(res, ensure_ascii=False, indent=2))
        
    return res

def detect_plate_type(plate: str) -> str:
    if re.match(config.REGEX_OTO, plate):
        return "Ôtô"
    elif re.match(config.REGEX_XEMAY, plate):
        return "Xe máy"
    else:
        return "Không xác định"
    
