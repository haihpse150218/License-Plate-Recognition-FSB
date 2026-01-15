from ultralytics import YOLO
import cv2
import easyocr
import os
import glob
import re  # Thêm thư viện để xử lý tên file

def full_pipeline(source_path, model_path="runs/detect/latest_best.pt"):
    """
    Lưu ý: model_path mặc định trỏ vào 'latest_best.pt' 
    (file tự động copy từ bước train_yolo11.py để luôn dùng bản mới nhất)
    """
    
    # 1. Load model & OCR (Cấu hình tối ưu cho biển số)
    print(f"Đang load model từ: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Lỗi load model: {e}. Hãy kiểm tra lại đường dẫn!")
        return

    # Chỉ dùng 'en' và gpu=True nếu máy có card rời NVIDIA
    reader = easyocr.Reader(['en'], gpu=False) 

    # 2. Lấy danh sách ảnh (Logic của bạn rất tốt)
    if os.path.isfile(source_path):
        image_paths = [source_path]
    elif os.path.isdir(source_path):
        # Case-insensitive glob patterns
        image_paths = glob.glob(os.path.join(source_path, "*.[jJ][pP][gG]")) + \
                      glob.glob(os.path.join(source_path, "*.[jJ][pP][eE][gG]")) + \
                      glob.glob(os.path.join(source_path, "*.[pP][nN][gG]"))
        if not image_paths:
            print("Không tìm thấy file ảnh nào trong thư mục!")
            return
    else:
        print("Đường dẫn không hợp lệ!")
        return

    # Tạo thư mục output
    output_dir = "results_final"
    crop_dir = "crop_images"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(crop_dir, exist_ok=True)

    # 3. Xử lý từng ảnh
    for img_path in image_paths:
        print(f"-> Đang xử lý: {os.path.basename(img_path)}")
        
        # Detect
        # verbose=False để đỡ spam log terminal
        results = model(img_path, conf=0.4, save=False, verbose=False) 
        
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        h_img, w_img, _ = img.shape # Lấy kích thước ảnh gốc
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        
        has_detection = False

        for r in results:
            boxes = r.boxes
            for box in boxes:
                has_detection = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()

                # --- BẢO VỆ TỌA ĐỘ (Tránh lỗi crash do tọa độ âm) ---
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_img, x2), min(h_img, y2)
                
                # Nếu box quá nhỏ (do lỗi detect), bỏ qua để tránh lỗi crop
                if x2 - x1 < 5 or y2 - y1 < 5:
                    continue

                cropped = img[y1:y2, x1:x2]
                
                # --- OCR TỐI ƯU ---
                # allowlist: Chỉ cho phép số và chữ, loại bỏ ký tự lạ
                ocr_result = reader.readtext(cropped, detail=0, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-.')
                plate = ''.join(ocr_result).upper().replace(' ', '').replace('.', '')
                
                if not plate: 
                    plate = "Unknown"

                # --- VẼ & LƯU ---
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{plate} ({conf:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # --- XỬ LÝ TÊN FILE AN TOÀN ---
                # Loại bỏ ký tự cấm trong tên file Windows (/:*?"<>|)
                safe_plate_name = re.sub(r'[<>:"/\\|?*]', '_', plate)
                
                crop_name = f"{img_name}_plate_{safe_plate_name}.jpg"
                cv2.imwrite(os.path.join(crop_dir, crop_name), cropped)

        # Chỉ lưu ảnh kết quả nếu có phát hiện biển số
        if has_detection:
            output_path = os.path.join(output_dir, f"processed_{os.path.basename(img_path)}")
            cv2.imwrite(output_path, img)
            print(f"   Đã lưu kết quả tại: {output_path}")
        else:
            print("   Không tìm thấy biển số nào.")

if __name__ == "__main__":
    # Test
    # Đảm bảo đường dẫn này tồn tại
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    test_folder = os.path.join(project_root, "data", "test", "images")
    full_pipeline(test_folder)