from ultralytics import YOLO
import datetime
import os
import shutil  # Thêm thư viện này để copy file

# 1. Cấu hình tên & Load model
# Bắt đầu với nano để test nhanh (nhẹ). Sau đổi thành s/m/l nếu cần chính xác cao hơn.
model = YOLO("yolo11n.pt") 

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
train_name = f"yolo11-license-plate_{current_time}"

print(f"--- Bắt đầu train phiên bản: {train_name} ---")

# 2. Train model
results = model.train(
    data="D:/MSA-FPT/Final-project/dataset.yaml",  # Kiểm tra path yaml
    epochs=10,                  # Test thử với 10 epochs
    imgsz=640,
    batch=16,
    device='cpu',               # 'cpu' hoặc 0 (nếu có GPU)
    name=train_name,            # Tên folder động: runs/detect/yolo11-license-plate_2026...
    patience=20,
    save=True,
    plots=True
)

# Đường dẫn đến model vừa train xong (nằm trong thư mục có ngày giờ)
best_model_path = f"runs/detect/{train_name}/weights/best.pt"
print(f"Train xong! Model lưu tại: {best_model_path}")

# --- BƯỚC QUAN TRỌNG: Tự động cập nhật model cho Pipeline ---
# Copy file best.pt ra một file cố định tên là 'latest_best.pt'
# Giúp full_pipeline.py luôn tìm thấy model mới nhất mà không cần sửa code.
latest_model_path = "runs/detect/latest_best.pt"

if os.path.exists(best_model_path):
    shutil.copy(best_model_path, latest_model_path)
    print(f"-> Đã cập nhật model mới nhất sang: {latest_model_path}")
    print("-> Từ giờ full_pipeline.py sẽ tự động dùng model này.")
else:
    print("-> Lỗi: Không tìm thấy file best.pt để copy!")

# 3. Evaluate (Dùng luôn file latest vừa copy)
print("--- Đang đánh giá model trên tập Test ---")
# Load lại model từ file cố định
model = YOLO(latest_model_path)

metrics = model.val(
    data="D:/MSA-FPT/Final-project/dataset.yaml",
    split="test",       # Đảm bảo dataset.yaml có dòng 'test: ...'
    imgsz=640,
    batch=16,
    plots=True,
    conf=0.001,
    iou=0.7
)

# 4. In metrics
print("\n=== KẾT QUẢ ĐÁNH GIÁ ===")
print(f"mAP@50-95 (Chính xác TB): {metrics.box.map:.4f}")
print(f"mAP@50    (Độ nhạy):      {metrics.box.map50:.4f}")
print(f"Precision (Độ chính xác): {metrics.box.p.mean():.4f}")
print(f"Recall    (Độ bao phủ):   {metrics.box.r.mean():.4f}")