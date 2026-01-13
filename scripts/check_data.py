import os
from pathlib import Path
from collections import Counter

def check_dataset_classes(data_path):
    print(f"--- Đang kiểm tra dữ liệu tại: {data_path} ---")
    
    # Quét toàn bộ file .txt trong thư mục data (bao gồm cả train/val/test)
    labels_path = Path(data_path).rglob("*.txt")
    
    class_ids = []
    error_files = []
    total_files = 0
    
    for label_file in labels_path:
        if label_file.name == "classes.txt": continue 
        total_files += 1
        
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 0:
                    try:
                        cls_id = int(parts[0])
                        class_ids.append(cls_id)
                        # Nếu class_id khác 0 (biển số), ghi lại lỗi
                        if cls_id != 0:
                            error_files.append(str(label_file))
                    except ValueError:
                        pass

    counts = Counter(class_ids)
    print(f"Đã quét: {total_files} file labels.")
    print(f"Thống kê Class ID: {dict(counts)}")
    
    if len(counts) > 1 or (0 not in counts and len(counts) > 0):
        print("\n[CẢNH BÁO] Dữ liệu bị lẫn lộn class ID!")
        print(f"Có {len(error_files)} file chứa class lạ. Ví dụ:")
        for f in error_files[:5]:
            print(f" - {f}")
        print("-> Cần sửa lại toàn bộ về class 0 trước khi train.")
    else:
        print("\n[OK] Dữ liệu SẠCH. Toàn bộ là Class 0. Sẵn sàng train!")

if __name__ == "__main__":
    # Đường dẫn gốc tới folder data
    check_dataset_classes("D:/MSA-FPT/Final-project/data")