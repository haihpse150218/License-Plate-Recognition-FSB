
import core_ocr
import config
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os
from ocr import full_pipeline
config.setup_config()
app = FastAPI(title="License Plate OCR API")

# Tạo thư mục nếu chưa tồn tại (fix lỗi mount)
os.makedirs("results", exist_ok=True)
os.makedirs("crop_images", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# Mount thư mục static
app.mount("/results", StaticFiles(directory="results"), name="results")
app.mount("/crops", StaticFiles(directory="crop_images"), name="crops")



@app.post("/ocr")
async def ocr_image(file: UploadFile = File(...), conf: float = 0.4):
    """
    Upload ảnh → chạy pipeline → trả về JSON với plate và link ảnh kết quả
    """
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
        print(res)
        return res
    except Exception as e:
        print(e)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.on_event("startup")
async def startup_event():

    try:
        core_ocr.init_models()
        print("API đã sẵn sàng!")
    except Exception as e:
        print(f"Lỗi khởi tạo: {e}")
        raise

@app.get("/")
def root():
    return {"message": "License Plate OCR API is running. POST /ocr to upload image."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)