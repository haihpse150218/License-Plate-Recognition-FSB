

from ultralytics import YOLO
import cv2
import easyocr
import os
import glob
import re
import uuid  # Để tạo tên file unique
import config
from pathlib import Path
print("import OCR")

def full_pipeline(image_bytes: bytes, model_path=None):
    """
    Nhận bytes ảnh từ upload → xử lý → trả về dict kết quả
    """
    if model_path is None:
        model_path = (
            Path(config.BASE_DIR)
            / "runs/detect/latest_best.pt"
        )
        model_path = model_path.resolve()
    # Tạo tên file tạm unique
    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
    temp_path = os.path.join("uploads", temp_filename)
    os.makedirs("uploads", exist_ok=True)

    # Lưu tạm ảnh upload
    with open(temp_path, "wb") as f:
        f.write(image_bytes)

    # Load model & OCR
    model = YOLO(model_path)
    reader = easyocr.Reader(['en'], gpu=False)

    # Đọc ảnh từ file tạm
    img = cv2.imread(temp_path)
    if img is None:
        os.remove(temp_path)
        return {"error": "Không đọc được ảnh"}

    h_img, w_img, _ = img.shape
    img_name = os.path.splitext(temp_filename)[0]

    results = model(temp_path, conf=0.1, save=False, verbose=False)

    detections = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)

            if x2 - x1 < 5 or y2 - y1 < 5:
                continue

            cropped = img[y1:y2, x1:x2]

            ocr_result = reader.readtext(cropped, detail=0, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-.')
            plate = ''.join(ocr_result).upper().replace(' ', '').replace('.', '')

            if not plate:
                plate = "Unknown"

            safe_plate = re.sub(r'[<>:"/\\|?*]', '_', plate)
            crop_name = f"{img_name}_plate_{safe_plate}.jpg"
            crop_path = os.path.join("crop_images", crop_name)
            os.makedirs("crop_images", exist_ok=True)
            cv2.imwrite(crop_path, cropped)

            # Vẽ lên ảnh gốc
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{plate} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            detections.append({
                "plate": plate,
                "confidence": float(conf),
                "bbox": [x1, y1, x2, y2],
                "crop_path": crop_path
            })

    # Lưu ảnh kết quả
    output_path = os.path.join("results", f"processed_{img_name}.jpg")
    os.makedirs("results", exist_ok=True)
    cv2.imwrite(output_path, img)

    # Xóa file tạm
    os.remove(temp_path)

    return {
        "processed_image": output_path,
        "detections": detections
    }