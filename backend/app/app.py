import core_ocr
import config
from fastapi import FastAPI, UploadFile, File, HTTPException, Path
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, List
import os
import sqlite3
import json

# --- CẤU HÌNH ---
DB_FILE = "license_plates.db"
config.setup_config()

# Thêm description cho App để Swagger UI đẹp hơn
app = FastAPI(
    title="Hệ thống OCR & Tra Cứu Phạt Nguội",
    description="""
    API cung cấp 2 chức năng chính:
    1. **OCR**: Nhận diện biển số từ hình ảnh.
    2. **Tra cứu**: Lấy thông tin chủ xe và lỗi vi phạm từ Database.
    """,
    version="2.1.0",
    contact={
        "name": "Team MSA-FPT",
    }
)

# Tạo thư mục cần thiết
os.makedirs("results", exist_ok=True)
os.makedirs("crop_images", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

app.mount("/results", StaticFiles(directory="results"), name="results")
app.mount("/crops", StaticFiles(directory="crop_images"), name="crops")

# --- 1. DATA MODELS (SCHEMA CHUẨN CHO AI) ---
# Việc thêm Field(..., description="...") giúp AI hiểu ý nghĩa từng trường dữ liệu

class PlateInfo(BaseModel):
    plate_number: str = Field(
        ..., 
        description="Biển số xe chính xác (VD: 30A12345). Đây là khóa chính để định danh xe.", 
        example="30A12345"
    )
    vehicle_type: Optional[str] = Field(
        None, 
        description="Loại phương tiện (Car, Truck, Bike, Bus...)"
    )
    owner_name: Optional[str] = Field(
        None, 
        description="Họ và tên đầy đủ của chủ sở hữu phương tiện."
    )
    points: int = Field(
        0, 
        description="Số điểm giấy phép lái xe bị trừ (nếu có)."
    )
    fine_amount: int = Field(
        0, 
        description="Tổng số tiền phạt chưa thanh toán (đơn vị: VNĐ)."
    )
    detected_at: Optional[str] = Field(
        None, 
        description="Thời gian hệ thống camera phát hiện vi phạm (ISO 8601)."
    )
    confidence: float = Field(
        0.0, 
        description="Độ tin cậy của thuật toán AI khi nhận diện (0.0 - 1.0)."
    )

class OCRResponse(BaseModel):
    status: str = Field(..., description="Trạng thái xử lý (success/error)")
    processed_image_url: Optional[str] = Field(None, description="Đường dẫn ảnh đã vẽ bounding box")
    detections: List[dict] = Field(..., description="Danh sách các biển số nhận diện được trong ảnh")
    type: str = Field(..., description="loại xe: Ôtô, Xe máy, Không xác định")

# --- 2. DATABASE HELPER ---
def get_plate_from_db(plate_input: str):
    if not os.path.exists(DB_FILE):
        return None 

    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Tìm kiếm chính xác
        cursor.execute("SELECT * FROM plates WHERE plate_number = ?", (plate_input,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
    except Exception as e:
        print(f"Lỗi DB: {e}")
        return None

# --- 3. API ENDPOINTS ---

@app.get("/")
def root():
    return {"message": "Hệ thống đã sẵn sàng. Truy cập /docs để xem hướng dẫn sử dụng API."}

@app.post(
    "/ocr", 
    response_model=OCRResponse,
    summary="Nhận diện biển số từ ảnh (OCR)",
    description="Upload file ảnh (jpg/png), hệ thống sẽ trả về danh sách biển số xe nhận diện được kèm tọa độ.",
    tags=["Xử lý ảnh"]
)
async def ocr_image(
    file: UploadFile = File(..., description="File ảnh chứa phương tiện cần nhận diện"), 
    conf: float = 0.4
):
    try:
        print("=======Start ocr=========")
        image_bytes = await file.read()
        image = core_ocr.image_from_bytes(image_bytes)
        
        result = core_ocr.detect_plates(image, conf_threshold=conf, isDebug=True)
        
        if "error" in result:
            return JSONResponse(status_code=400, content={"error": result["error"]})
        
        res = {
            "status": "success",
            "processed_image_url": f"/results/{os.path.basename(result['processed_image'])}" if result["processed_image"] else None,
            "detections": result["detections"],
            "type": result["type"]
        }
        print("=======End ocr=========")
        print("Response:", json.dumps(res, ensure_ascii=False, indent=2))
        return res
    except Exception as e:
        print(e)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get(
    "/plates/{plate_number}",
    response_model=PlateInfo,
    summary="Tra cứu thông tin chủ xe & Phạt nguội",
    description="""
    API này dùng để tra cứu cơ sở dữ liệu dựa trên biển số xe.
    
    - **Input**: Biển số xe (có thể viết hoa hoặc thường, hệ thống tự chuẩn hóa).
    - **Output**: Trả về tên chủ xe, số tiền phạt, loại xe...
    - **Lỗi 404**: Nếu không tìm thấy biển số trong hệ thống.
    """,
    tags=["Tra cứu dữ liệu"]
)
async def lookup_plate(
    plate_number: str = Path(
        ..., 
        title="Biển số xe",
        description="Nhập biển số cần tra cứu. Ví dụ: '30A-12345' hoặc '51H-99999'. Hệ thống sẽ tự động viết hoa và xử lý.",
        example="30A-12345"
    )
):
    # Logic xử lý
    clean_plate = plate_number.strip().upper()
    
    data = get_plate_from_db(clean_plate)
    
    if not data:
        raise HTTPException(
            status_code=404, 
            detail=f"Không tìm thấy thông tin cho biển số '{clean_plate}' trong hệ thống dữ liệu."
        )
    
    return data

# --- 4. STARTUP ---
@app.on_event("startup")
async def startup_event():
    try:
        core_ocr.init_models()
        if not os.path.exists(DB_FILE):
            print(f"[WARNING] Không tìm thấy file database '{DB_FILE}'. API Tra cứu sẽ không hoạt động.")
        else:
            print(f"[INFO] Đã kết nối Database: {DB_FILE}")
        print("[INFO] API Server đã khởi động thành công!")
    except Exception as e:
        print(f"Lỗi khởi tạo: {e}")
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)