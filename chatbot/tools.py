import requests
import json
from colorama import Fore

# Địa chỉ FastAPI Server của bạn (đảm bảo port 8000 đúng với server đang chạy)
API_URL = "http://localhost:8000/plates/"

def lookup_plate_api(plate_number):
    """
    Hàm này thực sự gửi request sang FastAPI Server.
    """
    print(f"{Fore.MAGENTA} >>> [Tool] Đang gọi API kiểm tra biển số: {plate_number}...{Fore.RESET}")
    
    try:
        # Gọi GET http://localhost:8000/plates/{plate_number}
        response = requests.get(f"{API_URL}{plate_number}")
        
        if response.status_code == 200:
            return json.dumps(response.json(), ensure_ascii=False)
        elif response.status_code == 404:
            return json.dumps({"error": "Không tìm thấy biển số này trong hệ thống."}, ensure_ascii=False)
        else:
            return json.dumps({"error": f"Lỗi server: {response.status_code}"}, ensure_ascii=False)
            
    except Exception as e:
        return json.dumps({"error": f"Không kết nối được API: {str(e)}"}, ensure_ascii=False)

# Định nghĩa Schema (Cấu trúc) để DeepSeek hiểu cách dùng
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "lookup_plate_api",
            "description": "Tra cứu thông tin chủ xe, lỗi vi phạm và tiền phạt dựa trên biển số xe.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plate_number": {
                        "type": "string",
                        "description": "Biển số xe cần tra cứu (Ví dụ: 30A-12345, 51G-99999)",
                    }
                },
                "required": ["plate_number"],
            },
        },
    }
]