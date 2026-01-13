import easyocr
import cv2
import os

def ocr_plate(image_path):
    reader = easyocr.Reader(['en', 'vi'], gpu=False)  # gpu=True nếu có GPU
    img = cv2.imread(image_path)
    result = reader.readtext(img, detail=0, paragraph=False)
    
    plate_text = ''.join(result).upper().replace(' ', '')
    print(f"Ảnh: {os.path.basename(image_path)} → Biển số dự đoán: {plate_text}")
    return plate_text

# Ví dụ
if __name__ == "__main__":
    ocr_plate("results_crop/plate_test_car_0.jpg")