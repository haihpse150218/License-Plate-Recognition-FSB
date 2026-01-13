import shutil
import random
from pathlib import Path
import os

def split_dataset():
    # Cấu hình đường dẫn
    base_dir = Path("D:/MSA-FPT/Final-project/data")
    
    # Giả sử bạn gom hết ảnh vào thư mục 'all_images' và label vào 'all_labels' để chia lại
    # Nếu data của bạn đang lộn xộn, hãy gom về 2 folder này trước khi chạy script
    source_imgs = base_dir / "all_images"
    source_lbls = base_dir / "all_labels"

    if not source_imgs.exists():
        print(f"Không tìm thấy thư mục {source_imgs}. Hãy gom ảnh về đây nếu muốn chia lại.")
        return

    # Tạo cấu trúc train/val/test
    for split in ["train", "val", "test"]:
        (base_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (base_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # Lấy danh sách file
    img_files = list(source_imgs.glob("*.jpg")) + list(source_imgs.glob("*.png")) + list(source_imgs.glob("*.jpeg"))
    random.shuffle(img_files)
    
    total = len(img_files)
    train_end = int(0.7 * total) # 70%
    val_end = int(0.9 * total)   # 20% (từ 70-90)
    # Test: 10% còn lại

    print(f"Tổng: {total} ảnh. Train: {train_end}, Val: {val_end-train_end}, Test: {total-val_end}")

    for i, img_path in enumerate(img_files):
        if i < train_end: split = "train"
        elif i < val_end: split = "val"
        else: split = "test"
        
        lbl_name = img_path.stem + ".txt"
        lbl_path = source_lbls / lbl_name
        
        # Copy ảnh
        shutil.copy(img_path, base_dir / split / "images" / img_path.name)
        
        # Copy label
        if lbl_path.exists():
            shutil.copy(lbl_path, base_dir / split / "labels" / lbl_name)
            
    print("Đã chia dataset thành công!")

if __name__ == "__main__":
    split_dataset()