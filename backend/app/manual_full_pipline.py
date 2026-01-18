"""
Hệ thống nhận diện biển số xe sử dụng YOLO11 và TrOCR
- YOLO11: Phát hiện vị trí biển số trong ảnh
- TrOCR: Đọc ký tự từ biển số đã cắt
- Hỗ trợ cả biển số 1 dòng và 2 dòng
"""

from ultralytics import YOLO
import cv2
import os
import glob
import re
import numpy as np
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Biến toàn cục cho TrOCR model
trocr_processor = None
trocr_model = None
device = None

# Hàm tiền xử lí nhưng mà chưa được áp dụng vì OCR Tro xử lí quá tốt hơn so với các OCR
# Chỉ có 1 số case về độ sáng làm lêch kết quả như B -> 8, Z -> 2 
# Những trường hợp này nên xử lí hâụ kì theo luật VN ổn hơn 
def preprocess_plate_image(cropped):

    # Chuyển sang grayscale nếu ảnh là màu
    if len(cropped.shape) == 3:
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    else:
        gray = cropped.copy()
    
    # Resize ảnh lên để OCR hoạt động tốt hơn
    height, width = gray.shape
    if height < 60:
        # Ảnh quá nhỏ: scale lên để đạt tối thiểu 60px chiều cao
        scale = 60 / height
        new_width = int(width * scale)
        gray = cv2.resize(gray, (new_width, 60), interpolation=cv2.INTER_CUBIC)
    else:
        # Ảnh lớn hơn: tăng gấp 2 lần
        gray = cv2.resize(gray, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
    
    # Bước 1: Khử nhiễu bằng Non-local Means Denoising
    # Loại bỏ các điểm ảnh ngẫu nhiên, giữ lại chi tiết quan trọng
    denoised = cv2.fastNlMeansDenoising(gray, h=15, templateWindowSize=7, searchWindowSize=21)
    
    # Bước 2: Tăng tương phản bằng CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # Chia ảnh thành các ô nhỏ (8x8) và tăng tương phản cục bộ
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # Bước 3: Threshold Otsu - tự động tìm ngưỡng tối ưu
    # Chuyển ảnh xám thành ảnh nhị phân (đen/trắng)
    _, binary1 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary1


# Hàm load model TrOCR
def load_trocr_model(model_name="microsoft/trocr-base-printed", device_name="auto"):
    global trocr_processor, trocr_model, device
    
    try:
        # Tự động chọn device: GPU nếu có, CPU nếu không
        if device_name == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_name)
        
        # Load processor (xử lý ảnh đầu vào) và model (nhận diện text)
        trocr_processor = TrOCRProcessor.from_pretrained(model_name)
        trocr_model = VisionEncoderDecoderModel.from_pretrained(model_name)
        trocr_model.to(device)
        trocr_model.eval()  # Chuyển sang chế độ inference (không train)
        
        return True
    except Exception as e:
        print(f"Lỗi load TrOCR model: {e}")
        print(f"Cài đặt: pip install transformers torch torchvision")
        return False


# Hàm OCR trên ảnh để đọc text  
def trocr_ocr(image):
    global trocr_processor, trocr_model, device
    
    if trocr_processor is None or trocr_model is None:
        return "", 0
    
    try:
        # Chuyển đổi format ảnh: OpenCV (BGR) -> PIL (RGB)
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                # Ảnh màu: BGR -> RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # Ảnh grayscale: GRAY -> RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            pil_image = Image.fromarray(image_rgb)
        else:
            pil_image = image
        
        # Preprocess ảnh và chuyển sang tensor
        pixel_values = trocr_processor(images=pil_image, return_tensors="pt").pixel_values.to(device)
        
        # Thực hiện OCR (không tính gradient)
        with torch.no_grad():
            generated_ids = trocr_model.generate(pixel_values)
            generated_text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Làm sạch text: chuyển thành chữ hoa
        text = generated_text.strip().upper()
        
        # Lọc chỉ giữ lại ký tự hợp lệ cho biển số Việt Nam
        valid_chars = set('0123456789ABCDEFGHKLMNPQRSTUVWXYZ.-')
        filtered_text = ''.join([c for c in text if c in valid_chars])
        
        if filtered_text:
            return filtered_text, 0.9
        else:
            return "", 0
            
    except Exception as e:
        return "", 0


# Hàm phát hiện xem biển số có phải 2 dòng hay không
# Này chat gpt => Đơn giản trong pipeline chỉ là check xem có phải biển số 2 dòng không 
# Có thể sử dụng phương pháp height/width > ngưỡng để phát hiện nhưng mà tỉ lệ này không phải là quy tắc cố định
def detect_two_line_plate(cropped):
    h, w = cropped.shape[:2]
    aspect_ratio = w / h if h > 0 else 0
    
    # Chuyển sang grayscale nếu cần
    if len(cropped.shape) == 3:
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    else:
        gray = cropped.copy()
    
    # Làm mịn ảnh để giảm nhiễu
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Tính horizontal projection: tổng giá trị pixel theo từng hàng
    # Hàng có text (chữ đen) sẽ có giá trị thấp, hàng trống (nền trắng) có giá trị cao
    horizontal_projection = np.sum(blurred, axis=1)
    
    # Tính ngưỡng động dựa trên mean và std
    mean_proj = np.mean(horizontal_projection)
    std_proj = np.std(horizontal_projection)
    
    # Vùng text (chữ đen) có giá trị thấp hơn ngưỡng
    threshold = mean_proj - 0.3 * std_proj
    text_regions = horizontal_projection < threshold
    
    # Đếm số vùng text liên tục
    regions = []
    in_region = False
    start = 0
    
    for i, is_text in enumerate(text_regions):
        if is_text and not in_region:
            # Bắt đầu vùng text mới
            start = i
            in_region = True
        elif not is_text and in_region:
            # Kết thúc vùng text
            # Chỉ thêm vùng nếu có chiều cao đủ lớn (ít nhất 8 pixel)
            if i - start >= 8:
                regions.append((start, i))
            in_region = False
    
    # Xử lý vùng text ở cuối ảnh
    if in_region:
        if len(text_regions) - start >= 8:
            regions.append((start, len(text_regions)))
    
    # Nếu có 2 vùng text rõ ràng, kiểm tra khoảng cách
    if len(regions) >= 2:
        gap = regions[1][0] - regions[0][1]
        total_height = h
        gap_ratio = gap / total_height if total_height > 0 else 0
        
        # Biển số 2 dòng có khoảng cách giữa 2 dòng khoảng 8-45% chiều cao
        if 0.08 <= gap_ratio <= 0.45:
            return True
    
    # Kiểm tra aspect ratio: biển số 2 dòng thường có tỷ lệ w/h < 2.5
    # Biển số 1 dòng: w/h thường > 2.5
    if aspect_ratio < 2.5 and h > 40:
        return True
    
    return False


# Hàm cắt ảnh biển số thành 2 dòng
# Generate bằng AI có thể dùng height/width > ngưỡng để cắt nhưng mà tỉ lệ này không phải là quy tắc cố định
# Hãy trust công thức height/width vì Yolo đã cắt gần chuẩn rồi theo hình chữ nhật =)))
def split_plate_into_lines(cropped):

    h, w = cropped.shape[:2]
    
    # Chuyển sang grayscale nếu ảnh là màu (quan trọng để tính horizontal projection đúng)
    if len(cropped.shape) == 3:
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    else:
        gray = cropped.copy()
    
    # Tìm kiếm vị trí cắt thích hợp
    # Tính horizontal projection để tìm vùng trống (nền trắng)
    # Vùng trống có giá trị cao, vùng có chữ có giá trị thấp
    # Lưu ý: phải dùng grayscale (2D) để np.sum trả về 1D array
    horizontal_projection = np.sum(gray, axis=1)
    
    # Đảm bảo horizontal_projection là 1D array (flatten nếu cần)
    horizontal_projection = horizontal_projection.flatten()
    
    # Tìm điểm chia tốt nhất trong khoảng 45-55% chiều cao
    # Điểm chia là vị trí có giá trị cao nhất (vùng trống giữa 2 dòng)
    search_start = int(h * 0.45)
    search_end = int(h * 0.55)
    
    max_val = -1
    best_split = h // 2
    
    for y in range(search_start, search_end):
        # Lấy giá trị scalar từ array
        proj_val = float(horizontal_projection[y])
        if proj_val > max_val:
            max_val = proj_val
            best_split = y
    
    split_y = best_split
    
    # Cắt thành 2 phần
    top_line = cropped[0:split_y, :]      # Dòng trên: từ đầu đến điểm chia
    bottom_line = cropped[split_y:h, :]  # Dòng dưới: từ điểm chia đến cuối
    
    return top_line, bottom_line


# Hậu kì xử lí text để đảm bảo độ chính xác với biển số xe Việt Nam
# Chưa xử lí kĩ 
def clean_plate_text(text, keep_special_chars=False):

    # Loại bỏ khoảng trắng và chuyển thành chữ hoa
    text = text.upper().replace(' ', '')
    
    # Loại bỏ ký tự đặc biệt nếu không cần giữ
    # Dấu / thường xuất hiện do cắt ảnh sai ở biển số 2 dòng
    if not keep_special_chars:
        text = text.replace('.', '').replace('-', '').replace('/', '')
    
    # Các lỗi nhận nhầm phổ biến của OCR
    # OCR thường nhầm chữ và số do hình dạng giống nhau
    replacements = {
        'O': '0',  # Chữ O thành số 0
        'I': '1',  # Chữ I thành số 1
        'Z': '2',  # Chữ Z thành số 2
        'S': '5',  # Chữ S thành số 5
        'B': '8',  # Chữ B thành số 8
    }
    
    # Chỉ thay thế trong phần số (sau 3 ký tự đầu)
    # 3 ký tự đầu là mã tỉnh/thành, cần giữ nguyên
    if len(text) > 3:
        prefix = text[:3]  # Giữ nguyên phần mã tỉnh/thành
        suffix = text[3:]  # Phần số
        for old, new in replacements.items():
            suffix = suffix.replace(old, new)
        text = prefix + suffix
    
    return text

# Gọi ocr detect text từ ảnh
def try_ocr_on_image(img_to_ocr, version_name="original"):
    try:
        # Resize ảnh nếu quá nhỏ (TrOCR hoạt động tốt với ảnh lớn hơn)
        h, w = img_to_ocr.shape[:2]
        if h < 32 or w < 100:
            # Ảnh quá nhỏ: scale lên để đạt tối thiểu
            scale = max(32/h, 100/w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            img_resized = cv2.resize(img_to_ocr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        else:
            # Ảnh lớn hơn: tăng gấp 2 lần để TrOCR dễ đọc hơn
            img_resized = cv2.resize(img_to_ocr, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        
        # Thực hiện OCR
        plate_text, avg_conf = trocr_ocr(img_resized)
        
        # Chỉ trả về nếu có ít nhất 3 ký tự
        if plate_text and len(plate_text) >= 3:
            return plate_text, avg_conf, version_name
        
    except Exception as e:
        pass
    
    return None, 0, version_name


# Hàm xử lí ảnh biển số 2 dòng 
# 1. Cắt ảnh thành 2 dòng (2 ảnh)
# 2. OCR từng dòng riêng biệt
# 3. Kết hợp kết quả
# 4. Làm sạch text
def process_two_line_plate(cropped, save_dir=None):

    
    # Cắt ảnh thành 2 dòng
    top_line, bottom_line = split_plate_into_lines(cropped)
    
    # Lưu ảnh 2 dòng nếu có thư mục (Debug)
    if save_dir:
        cv2.imwrite(os.path.join(save_dir, "top_line.jpg"), top_line)
        cv2.imwrite(os.path.join(save_dir, "bottom_line.jpg"), bottom_line)
    
    # Xử lí OCR dòng trên
    top_text, top_conf, _ = try_ocr_on_image(top_line, "top_line")
    
    # Nếu không detect được text dòng trên, trả về kết quả là empty string và confidence là 0
    if not top_text:
        top_text = ""
        top_conf = 0
    
    # Xử lí OCR dòng dưới
    bottom_text, bottom_conf, _ = try_ocr_on_image(bottom_line, "bottom_line")
    
    # Nếu không detect được text dòng dưới, trả về kết quả là empty string và confidence là 0
    if not bottom_text:
        bottom_text = ""
        bottom_conf = 0
    
    # Kết hợp kết quả
    if top_text and bottom_text:
        # Nếu cả 2 dòng đều có text: kết hợp lại
        combined_text = top_text + bottom_text
        avg_conf = (top_conf + bottom_conf) / 2
        plate = clean_plate_text(combined_text, keep_special_chars=False)
        best_confidence = avg_conf
        best_version = -1
    elif top_text:
        # Nếu chỉ có dòng trên
        plate = clean_plate_text(top_text, keep_special_chars=False)
        best_confidence = top_conf
        best_version = -1
    elif bottom_text:
        # Nếu chỉ có dòng dưới
        plate = clean_plate_text(bottom_text, keep_special_chars=False)
        best_confidence = bottom_conf
        best_version = -1
    else:
        # Nếu không có dòng nào
        plate = "Unknown"
        best_confidence = 0
        best_version = -1
    
    # Trả về kết quả biển số và confidence
    return plate, best_confidence, best_version


def process_single_line_plate(cropped, save_dir=None):
    
    plate = "Unknown"
    best_confidence = 0
    best_version = -1
    
    # Xử lí OCR trực tiếp trên ảnh gốc (không qua preprocessing)
    plate_text, conf, version_name = try_ocr_on_image(cropped, "original")
    if not plate_text:
        plate_text = ""
        conf = 0
    
    # Làm sạch text cuối cùng
    if plate_text and len(plate_text) >= 3:
        plate = clean_plate_text(plate_text, keep_special_chars=False)
    else:
        plate = "Unknown"
    
    # Trả về kết quả biển số và confidence
    return plate, conf, best_version

# Hàm xử lí toàn bộ pipeline
# 1. Load model YOLO11 và TrOCR
# 2. Đọc danh sách ảnh từ thư mục hoặc file
# 3. Với mỗi ảnh:
#    - Detect biển số bằng YOLO11
#    - Crop biển số
#    - Phân loại: 1 dòng hay 2 dòng
#    - OCR bằng TrOCR
#    - Lưu kết quả
def full_pipeline(source_path, model_path=None):
 
    # Tự động tìm project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Lên 1 cấp từ scripts
    
    # Xử lý đường dẫn model
    if model_path is None:
        # Tự động tìm model trong các đường dẫn phổ biến
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
            print("Không tìm thấy model! Vui lòng chỉ định đường dẫn model.")
            print("Thử các đường dẫn:")
            for path in possible_paths:
                print(f"  - {path}")
            return
    else:
        # Nếu đường dẫn tương đối, chuyển thành tuyệt đối từ project root
        if not os.path.isabs(model_path):
            model_path = os.path.join(project_root, model_path)
    
    # Bước 1: Load model YOLO11
    print(f"Đang load model từ: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Lỗi load model: {e}. Hãy kiểm tra lại đường dẫn!")
        return
    
    # Bước 2: Load TrOCR OCR
    print("Đang khởi tạo TrOCR...")
    if not load_trocr_model(model_name="microsoft/trocr-base-printed", device_name="auto"):
        print("Không thể load TrOCR model. Vui lòng kiểm tra:")
        return
    
    # Bước 3: Lấy danh sách ảnh
    if os.path.isfile(source_path):
        # Nếu là file: chỉ xử lý 1 ảnh
        image_paths = [source_path]
    elif os.path.isdir(source_path):
        # Nếu là thư mục: tìm tất cả ảnh (jpg, jpeg, png)
        image_paths = glob.glob(os.path.join(source_path, "*.[jJ][pP][gG]")) + \
                      glob.glob(os.path.join(source_path, "*.[jJ][pP][eE][gG]")) + \
                      glob.glob(os.path.join(source_path, "*.[pP][nN][gG]"))
        if not image_paths:
            print("Không tìm thấy file ảnh nào trong thư mục!")
            return
    else:
        print("Đường dẫn không hợp lệ!")
        return
    
    # Tạo thư mục output chính (tại project root)
    output_dir = os.path.join(project_root, "results_final")
    os.makedirs(output_dir, exist_ok=True)
    
    # Bước 4: Xử lý từng ảnh
    print(f"\nBắt đầu xử lý {len(image_paths)} ảnh...\n")
    
    for idx, img_path in enumerate(image_paths, 1):
        print(f"[{idx}/{len(image_paths)}] Đang xử lý: {os.path.basename(img_path)}")
        
        # Detect biển số bằng YOLO11 và lấy kết quả
        results = model(img_path, conf=0.4, save=False, verbose=False)
        
        # Đọc ảnh gốc và lấy kích thước
        img = cv2.imread(img_path)
        if img is None:
            print("Không đọc được ảnh!")
            continue
        
        h_img, w_img, _ = img.shape
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        safe_img_name = re.sub(r'[<>:"/\\|?*]', '_', img_name)  # Loại bỏ ký tự đặc biệt
        
        # Tạo folder riêng cho mỗi ảnh
        image_folder = os.path.join(output_dir, safe_img_name)
        os.makedirs(image_folder, exist_ok=True)
        print(f"Tạo folder: {image_folder}/")
        
        has_detection = False
        plate_count = 0
        
        # Xử lý từng biển số được detect
        for r in results:
            boxes = r.boxes
            for box in boxes:
                has_detection = True
                plate_count += 1
                
                # Lấy tọa độ bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                
                # Bảo vệ tọa độ (đảm bảo không vượt quá kích thước ảnh)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_img, x2), min(h_img, y2)
                
                # Kiểm tra kích thước hợp lệ
                if x2 - x1 < 10 or y2 - y1 < 10:
                    print("Bounding box quá nhỏ, bỏ qua")
                    continue
                
                # Crop biển số từ ảnh gốc
                cropped = img[y1:y2, x1:x2]
                
                # Tạo folder riêng cho mỗi biển số
                plate_folder = os.path.join(image_folder, f"plate_{plate_count}")
                os.makedirs(plate_folder, exist_ok=True)
                
                # Lưu ảnh crop gốc
                cv2.imwrite(os.path.join(plate_folder, "original_crop.jpg"), cropped)
                print(f"Đã lưu original_crop.jpg vào {plate_folder}/")
                
                # Kiểm tra xem biển số là 1 dòng hay 2 dòng
                h_crop, w_crop = cropped.shape[:2]
                print(f"Kích thước ảnh crop: {w_crop}x{h_crop}")
                
                is_two_line = detect_two_line_plate(cropped)
                
                # Xử lý biển số theo loại
                if is_two_line:
                    plate, best_confidence, best_version = process_two_line_plate(cropped, save_dir=plate_folder)
                else:
                    plate, best_confidence, best_version = process_single_line_plate(cropped, save_dir=plate_folder)
                
                # Hiển thị kết quả
                if is_two_line:
                    print(f"Biển số {plate_count} (2 dòng): {plate} (confidence: {best_confidence:.2f}, TrOCR)")
                else:
                    print(f"Biển số {plate_count} (1 dòng): {plate} (confidence: {best_confidence:.2f}, original)")
                
                # Lưu thông tin kết quả vào file text
                result_file = os.path.join(plate_folder, "result.txt")

                # Debug
                with open(result_file, 'w', encoding='utf-8') as f:
                    f.write(f"Biển số: {plate}\n")
                    f.write(f"Confidence: {best_confidence:.4f}\n")
                    f.write(f"Loại: {'2 dòng' if is_two_line else '1 dòng'}\n")
                    f.write(f"Xử lý: Ảnh gốc (không qua preprocessing)\n")
                    f.write(f"Kích thước: {w_crop}x{h_crop}\n")
                    f.write(f"Tọa độ: ({x1}, {y1}) -> ({x2}, {y2})\n")
                print(f"Đã lưu thông tin kết quả vào result.txt")
                
                # Vẽ bounding box và text lên ảnh gốc
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{plate} ({best_confidence:.2f})", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Lưu ảnh kết quả (có vẽ bounding box)
        if has_detection:
            output_path = os.path.join(image_folder, f"result_with_boxes.jpg")
            cv2.imwrite(output_path, img)
            print(f"Đã lưu ảnh kết quả tại: {output_path}\n")
        else:
            print("Không tìm thấy biển số nào.\n")
    
    # Thông báo hoàn thành
    print("=" * 60)
    print("Hoàn thành! Kiểm tra thư mục:")
    print(f"  - Kết quả: {output_dir}/")
    print(f"  - Mỗi ảnh có folder riêng với:")
    print(f"    + original_crop.jpg (ảnh crop gốc)")
    print(f"    + top_line.jpg, bottom_line.jpg (nếu là biển số 2 dòng)")
    print(f"    + result.txt (thông tin kết quả OCR)")
    print(f"    + result_with_boxes.jpg (ảnh gốc có vẽ bounding box)")


if __name__ == "__main__":

    # Tự động tìm project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Lên 1 cấp từ scripts
    
    # Tìm thư mục test images
    test_folder = os.path.join(project_root, "data", "test", "images")
    if not os.path.exists(test_folder):
        # Thử các đường dẫn khác
        test_folder = os.path.join(project_root, "data", "images")
        if not os.path.exists(test_folder):
            test_folder = None
    
    # Hoặc test với 1 ảnh cụ thể
    # test_folder = "path/to/your/image.jpg"
    # model_path = "path/to/your/model.pt"
    
    if test_folder:
        print(f"Thư mục test: {test_folder}")
        print(f"Project root: {project_root}")
        full_pipeline(test_folder)
    else:
        print("Không tìm thấy thư mục test images!")
        print(f"Project root: {project_root}")
        print("Sử dụng: full_pipeline('đường/dẫn/ảnh', 'đường/dẫn/model.pt')")
