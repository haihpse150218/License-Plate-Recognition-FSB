import sys
import os
from pathlib import Path

def setup_config():
    # 1. Xác định đường dẫn file hiện tại
    # file_path = .../Final-project/frontend/app.py
    current_file = Path(__file__).resolve()
    
    # 2. Tìm thư mục gốc (Project Root)
    # .parents[0] = thư mục chứa file (frontend)
    # .parents[1] = thư mục cha của frontend (Final-project) -> ĐÂY LÀ ROOT
    base_dir = current_file.parents[1]
    
    # 3. Convert sang string và thêm vào sys.path
    base_dir_str = str(base_dir)
    
    if base_dir_str not in sys.path:
        sys.path.append(base_dir_str)
        print(f"[Info] Đã thêm Root Path vào hệ thống: {base_dir_str}")
    else:
        print(f"[Info] Root Path đã tồn tại: {base_dir_str}")

setup_config()
