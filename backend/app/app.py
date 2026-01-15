
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
async def ocr_image(file: UploadFile = File(...)):
    """
    Upload ảnh → chạy pipeline → trả về JSON với plate và link ảnh kết quả
    """
    try:
        print("=======Start ocr=========")
        contents = await file.read()
        result = full_pipeline(contents)
        
        if "error" in result:
            return JSONResponse(status_code=400, content={"error": result["error"]})
        
        res = {
            "status": "success",
            "processed_image_url": f"/results/{os.path.basename(result['processed_image'])}",
            "detections": result["detections"],
            "type": result["type"]
        }
        print(res)
        return res
    except Exception as e:
        print(e)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
def root():
    return {"message": "License Plate OCR API is running. POST /ocr to upload image."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)