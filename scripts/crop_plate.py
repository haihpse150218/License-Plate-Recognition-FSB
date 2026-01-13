import cv2
from ultralytics import YOLO
import os

def crop_plates(model_path, source_img_path, output_dir="results_crop"):
    os.makedirs(output_dir, exist_ok=True)
    
    model = YOLO(model_path)
    results = model(source_img_path, conf=0.4, iou=0.5)  # điều chỉnh conf nếu cần
    
    img = cv2.imread(source_img_path)
    img_name = os.path.basename(source_img_path)
    
    for i, r in enumerate(results):
        boxes = r.boxes
        for j, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped = img[y1:y2, x1:x2]
            output_path = os.path.join(output_dir, f"plate_{img_name}_{j}.jpg")
            cv2.imwrite(output_path, cropped)
            print(f"Đã crop và lưu: {output_path}")

# Ví dụ sử dụng
if __name__ == "__main__":
    crop_plates(
        model_path="runs/detect/yolo11-license-plate/weights/best.pt",
        source_img_path="test_images/test_car.jpg"
    )