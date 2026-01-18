# frontend/app.py
# Streamlit frontend cho License Plate Recognition
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoHTMLAttributes
import cv2
import numpy as np
import requests
import av
from PIL import Image
import io
import time
import base64
import os
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

# ======================== CONFIGURATION ========================
API_URL = "http://127.0.0.1:8000/ocr"
BACKEND_HOST = "http://127.0.0.1:8000"
CAPTURED_DIR = "captured_photos"

# Cấu hình tối ưu cho performance
DEFAULT_PROCESS_INTERVAL = 5.0  # Giây giữa các request
DEFAULT_SKIP_FRAMES = 10  # Số frame bỏ qua
DEFAULT_IMAGE_WIDTH = 640  # Chiều rộng ảnh tối đa
DEFAULT_JPEG_QUALITY = 70  # Chất lượng JPEG (0-100)
API_TIMEOUT = 10  # Timeout cho API requests (giây)
HEALTH_CHECK_CACHE_TTL = 5  # Cache health check trong 5 giây

# ======================== DATA CLASSES ========================
@dataclass
class Detection:
    """Kết quả nhận diện một biển số"""
    plate: str
    confidence: float
    bbox: List[int]

@dataclass
class OCRResult:
    """Kết quả OCR từ API"""
    status: str
    detections: List[Detection]
    processed_image_url: Optional[str] = None
    plate_type: Optional[str] = None
    error_message: Optional[str] = None

def init_session_state():

    defaults = {
        'input_image': None,
        'show_camera_input': False,
        'captured_frame': None,
        'captured_result': None,
        'capture_requested': False,
        'last_detection': None,
        'frame_count': 0,
        'api_error_count': 0,
        'process_interval': DEFAULT_PROCESS_INTERVAL,
        'skip_frames': DEFAULT_SKIP_FRAMES,
        'health_check_time': 0,
        'health_check_result': None,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

@st.cache_data(ttl=HEALTH_CHECK_CACHE_TTL)
def check_api_health_cached() -> Tuple[bool, str]:
    """Check API health với cache để giảm số lần request"""
    try:
        response = requests.get(f"{BACKEND_HOST}/health", timeout=API_TIMEOUT)
        if response.status_code == 200:
            return True, "Backend API đang chạy"
        return False, f"Backend trả về mã lỗi: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Không kết nối được Backend API"
    except requests.exceptions.Timeout:
        return False, "Timeout khi kiểm tra API"
    except Exception as e:
        return False, f"Lỗi: {str(e)[:50]}"

def check_api_health() -> Tuple[bool, str]:

    current_time = time.time()

    if (current_time - st.session_state.health_check_time) > HEALTH_CHECK_CACHE_TTL:
        st.session_state.health_check_result = check_api_health_cached()
        st.session_state.health_check_time = current_time
    return st.session_state.health_check_result

def parse_ocr_response(data: Dict) -> OCRResult:
    """Parse API response thành OCRResult object"""
    detections = []
    if data.get("detections"):
        for det in data["detections"]:
            detections.append(Detection(
                plate=det.get("plate", "Unknown"),
                confidence=det.get("confidence", 0.0),
                bbox=det.get("bbox", [])
            ))
    
    return OCRResult(
        status=data.get("status", "error"),
        detections=detections,
        processed_image_url=data.get("processed_image_url"),
        plate_type=data.get("type"),
        error_message=data.get("message")
    )

def process_image_ocr(image_bytes: bytes, filename: str = "image.jpg") -> Optional[OCRResult]:
    """Gửi ảnh đến OCR API và trả về kết quả"""
    try:
        files = {"file": (filename, image_bytes, "image/jpeg")}
        response = requests.post(API_URL, files=files, timeout=API_TIMEOUT)
        
        if response.status_code == 200:
            return parse_ocr_response(response.json())
        else:
            return OCRResult(
                status="error",
                detections=[],
                error_message=f"API error: {response.status_code}"
            )
    except requests.exceptions.Timeout:
        return OCRResult(
            status="error",
            detections=[],
            error_message="Timeout - API không phản hồi kịp thời"
        )
    except requests.exceptions.ConnectionError:
        return OCRResult(
            status="error",
            detections=[],
            error_message="Không kết nối được với API"
        )
    except Exception as e:
        return OCRResult(
            status="error",
            detections=[],
            error_message=str(e)
        )

def display_processed_image(processed_image_url: str):
    """Hiển thị ảnh đã được xử lý từ API"""
    if not processed_image_url:
        return
        
    if processed_image_url.startswith("data:image"):
        base64_data = processed_image_url.split(",")[1]
        image_bytes = base64.b64decode(base64_data)
        st.image(image_bytes, caption="Ảnh đã detect & OCR", use_container_width=True)
    elif processed_image_url.startswith("http"):
        try:
            img_response = requests.get(processed_image_url, timeout=5)
            if img_response.status_code == 200:
                st.image(img_response.content, caption="Ảnh đã detect & OCR", use_container_width=True)
        except Exception:
            st.warning("Không thể tải ảnh từ URL")

def display_ocr_result(result: OCRResult):
    """Hiển thị kết quả OCR trong Streamlit"""
    if result.status == "success" and result.detections:
        st.success(f"✅ Xử lý thành công! Tìm thấy {len(result.detections)} biển số")
        
        # Hiển thị ảnh đã xử lý
        if result.processed_image_url:
            display_processed_image(result.processed_image_url)
        
        # Hiển thị chi tiết từng biển số
        st.subheader("Kết quả nhận diện")
        for idx, det in enumerate(result.detections, 1):
            with st.expander(f"Biển số #{idx}: {det.plate}", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Biển số", det.plate)
                    st.metric("Độ tin cậy", f"{det.confidence:.2%}")
                with col2:
                    if result.plate_type:
                        st.metric("Loại", result.plate_type)
                    st.write(f"**BBox:** {det.bbox}")
    
    elif result.status == "no_detection":
        st.warning("⚠️ Không tìm thấy biển số nào trong ảnh")
    else:
        st.error(f"❌ Lỗi: {result.error_message or 'Xử lý thất bại'}")

# ======================== PAGE CONFIG ========================
st.set_page_config(
    page_title="Biển Số Xe OCR",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize
init_session_state()
os.makedirs(CAPTURED_DIR, exist_ok=True)

# Header
st.title("🚗 Nhận diện biển số xe Việt Nam")
st.markdown("Chọn tab để upload ảnh hoặc dùng camera realtime.")

# API Health Check
api_healthy, api_message = check_api_health()
if api_healthy:
    st.success(f"✅ {api_message}")
else:
    st.error(f"❌ {api_message}")


tab1, tab2 = st.tabs(["📁 Upload Ảnh", "🎥 Camera Real-time"])

with tab1:
    st.subheader("Upload hoặc chụp ảnh xe")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "📁 Chọn ảnh từ máy tính",
            type=["jpg", "jpeg", "png"],
            help="Hỗ trợ định dạng JPG, JPEG, PNG"
        )
        
        if uploaded_file:
            st.session_state.input_image = uploaded_file.getvalue()
            st.session_state.show_camera_input = False
    
    with col2:
        if st.button("📸 Chụp ảnh từ camera", use_container_width=True):
            st.session_state.show_camera_input = not st.session_state.show_camera_input
            st.rerun()
    
    # Camera input
    if st.session_state.show_camera_input:
        camera_photo = st.camera_input("Chụp ảnh")
        if camera_photo:
            st.session_state.input_image = camera_photo.getvalue()
            st.session_state.show_camera_input = False
            st.rerun()
    
    # Display input image
    if st.session_state.input_image:
        st.markdown("### Ảnh đầu vào")
        try:
            image = Image.open(io.BytesIO(st.session_state.input_image))
            st.image(image, use_container_width=True)
        except Exception as e:
            st.error(f"Lỗi đọc ảnh: {str(e)}")
            st.session_state.input_image = None
        
        # Action buttons
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("🚀 Xử lý ảnh", type="primary", use_container_width=True):
                with st.spinner("Đang xử lý..."):
                    result = process_image_ocr(st.session_state.input_image)
                    if result:
                        display_ocr_result(result)
        
        with col_btn2:
            if st.button("❌ Xóa ảnh", use_container_width=True):
                st.session_state.input_image = None
                st.session_state.show_camera_input = False
                st.rerun()
    else:
        st.info("📌 Chưa có ảnh. Vui lòng upload hoặc chụp ảnh.")

with tab2:
    st.subheader("Camera Real-time OCR")
    
    # Configuration sliders
    col_config1, col_config2 = st.columns(2)
    with col_config1:
        process_interval = st.slider(
            "Tần suất xử lý (giây)",
            1.0, 10.0, DEFAULT_PROCESS_INTERVAL, 0.5,
            help="Khoảng thời gian giữa các lần gửi request (Mặc định: 5 giây - Tối ưu)"
        )
        st.session_state.process_interval = process_interval
    
    with col_config2:
        skip_frames = st.slider(
            "Bỏ qua frame",
            5, 20, DEFAULT_SKIP_FRAMES, 1,
            help="Chỉ xử lý mỗi N frame (Mặc định: 10 - Tối ưu)"
        )
        st.session_state.skip_frames = skip_frames
    
    # Video Processor Class - Tối ưu để giảm lag
    class VideoProcessor(VideoTransformerBase):
        def __init__(self):
            self.last_time = time.time()
            self.frame_counter = 0
            self.last_detection_time = 0
            self.current_detections = []
            # Cache config để giảm truy cập session state
            self.skip_frames = DEFAULT_SKIP_FRAMES
            self.process_interval = DEFAULT_PROCESS_INTERVAL
        
        def update_config(self):
            """Cập nhật config từ session state (gọi ít thường xuyên)"""
            self.skip_frames = st.session_state.get("skip_frames", DEFAULT_SKIP_FRAMES)
            self.process_interval = st.session_state.get("process_interval", DEFAULT_PROCESS_INTERVAL)
        
        def draw_detections(self, img, detections):
            """Vẽ bounding boxes và labels lên ảnh"""
            for x1, y1, x2, y2, plate, conf in detections:
                color = (0, 255, 0) if plate != "Unknown" else (0, 0, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                label = f"{plate} ({conf:.2f})"
                if len(label) > 30:
                    label = label[:30] + "..."
                
                # Background cho text để dễ đọc
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    img, (x1, y1 - text_height - 10),
                    (x1 + text_width, y1), color, -1
                )
                cv2.putText(
                    img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )
        
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            """Xử lý từng frame từ camera"""
            img = frame.to_ndarray(format="bgr24")
            current_time = time.time()
            self.frame_counter += 1
            
            # Cập nhật frame count (chỉ mỗi 10 frame để giảm overhead)
            if self.frame_counter % 10 == 0:
                st.session_state.frame_count = self.frame_counter
                # Cập nhật config mỗi 10 frame
                self.update_config()
            
            # Bỏ qua frame theo skip_frames
            if self.frame_counter % self.skip_frames != 0:
                if self.current_detections:
                    self.draw_detections(img, self.current_detections)
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            
            # Kiểm tra interval giữa các request
            if current_time - self.last_time < self.process_interval:
                if self.current_detections:
                    self.draw_detections(img, self.current_detections)
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            
            # Xử lý frame - gửi đến API
            try:
                # Resize ảnh để giảm kích thước
                height, width = img.shape[:2]
                scale = 1.0
                if width > DEFAULT_IMAGE_WIDTH:
                    scale = DEFAULT_IMAGE_WIDTH / width
                    new_size = (DEFAULT_IMAGE_WIDTH, int(height * scale))
                    img_resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
                else:
                    img_resized = img
                
                # Encode thành JPEG với chất lượng tối ưu
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, DEFAULT_JPEG_QUALITY]
                _, buffer = cv2.imencode('.jpg', img_resized, encode_params)
                
                # Gửi đến API
                files = {"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")}
                response = requests.post(API_URL, files=files, timeout=API_TIMEOUT)
                
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.api_error_count = 0
                    
                    if data.get("status") == "success" and data.get("detections"):
                        self.current_detections = []
                        
                        for det in data["detections"]:
                            bbox = det.get("bbox", [])
                            if len(bbox) >= 4:
                                # Scale lại tọa độ bbox về kích thước gốc
                                x1, y1, x2, y2 = [int(c / scale) for c in bbox[:4]]
                                
                                self.current_detections.append((
                                    x1, y1, x2, y2,
                                    det.get("plate", "Unknown"),
                                    det.get("confidence", 0.0)
                                ))
                        
                        self.last_detection_time = current_time
                        st.session_state.last_detection = data
                    else:
                        # Xóa detection cũ sau 2 giây nếu không có detection mới
                        if current_time - self.last_detection_time > 2.0:
                            self.current_detections = []
                    
                    # Lưu frame nếu được yêu cầu chụp
                    if st.session_state.capture_requested:
                        st.session_state.captured_frame = img.copy()
                        st.session_state.captured_result = data
                        st.session_state.capture_requested = False
                
                else:
                    st.session_state.api_error_count += 1
                    
            except requests.exceptions.Timeout:
                st.session_state.api_error_count += 1
            except requests.exceptions.ConnectionError:
                st.session_state.api_error_count += 1
            except Exception:
                st.session_state.api_error_count += 1
            
            self.last_time = current_time
            
            # Vẽ detections lên frame
            if self.current_detections:
                self.draw_detections(img, self.current_detections)
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    # Hiển thị trạng thái
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        st.metric("Frame đã xử lý", st.session_state.frame_count)
    with col_stat2:
        det_count = 0
        if st.session_state.last_detection:
            det_count = len(st.session_state.last_detection.get("detections", []))
        st.metric("Biển số hiện tại", det_count)
    with col_stat3:
        if st.session_state.api_error_count > 0:
            st.error(f"⚠️ Lỗi: {st.session_state.api_error_count}")
        else:
            st.success("✅ API OK")
    
    # WebRTC Configuration
    RTC_CONFIG = RTCConfiguration({
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]}
        ]
    })
    
    # WebRTC Streamer
    ctx = webrtc_streamer(
        key="real-time-ocr",
        video_transformer_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 1280},
                "height": {"ideal": 720},
                "facingMode": "user"
            },
            "audio": False
        },
        video_html_attrs=VideoHTMLAttributes(
            autoplay=True,
            controls=True,
            muted=True,
            style={"width": "100%"}
        ),
        async_processing=True
    )
    
    # Camera status và controls
    if ctx.state.playing:
        st.success("✅ Camera đang chạy")
        
        col_act1, col_act2 = st.columns(2)
        with col_act1:
            if st.button("📸 Chụp ảnh", use_container_width=True, type="primary"):
                st.session_state.capture_requested = True
                st.success("Đã yêu cầu chụp ảnh!")
        with col_act2:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.captured_frame = None
                st.session_state.captured_result = None
                st.session_state.frame_count = 0
                st.session_state.api_error_count = 0
                st.rerun()
    else:
        st.info("⏸️ Nhấn START để bắt đầu camera")
    
    # Hiển thị ảnh vừa chụp
    if st.session_state.captured_frame is not None:
        st.divider()
        st.subheader("📸 Ảnh vừa chụp")
        
        captured_rgb = cv2.cvtColor(st.session_state.captured_frame, cv2.COLOR_BGR2RGB)
        st.image(captured_rgb, use_container_width=True)
        
        if st.session_state.captured_result:
            result = parse_ocr_response(st.session_state.captured_result)
            display_ocr_result(result)
            
            # Nút tải ảnh
            _, buffer = cv2.imencode('.jpg', st.session_state.captured_frame)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            st.download_button(
                label="📥 Tải ảnh về",
                data=buffer.tobytes(),
                file_name=f"plate_{timestamp}.jpg",
                mime="image/jpeg",
                use_container_width=True
            )
    
    # Hướng dẫn sử dụng
    with st.expander("ℹ️ Hướng dẫn sử dụng"):
        st.markdown("""
        ### 🎯 Cách sử dụng:
        1. Nhấn **START** trên video player
        2. Cho phép quyền truy cập webcam
        3. Điều chỉnh tần suất xử lý nếu cần (mặc định đã tối ưu)
        4. Nhấn **Chụp ảnh** để lưu kết quả
        
        ### ⚡ Cấu hình tối ưu (đã áp dụng):
        - **Tần suất xử lý**: 5 giây/lần (giảm tải API, tiết kiệm băng thông)
        - **Bỏ qua frame**: 10 frame (giảm lag, tăng hiệu suất)
        - **Kích thước ảnh**: Tự động resize xuống 640px
        - **Chất lượng JPEG**: 70% (cân bằng chất lượng và kích thước)
        - **Timeout**: 10 giây (đủ thời gian cho API xử lý)
        
        ### 🔧 Khắc phục sự cố:
        - **Không thấy video**: Kiểm tra quyền webcam, thử refresh (F5)
        - **Vẫn lag**: Tăng tần suất lên 7-10 giây, tăng bỏ qua frame lên 15-20
        - **Lỗi API**: Đảm bảo backend đang chạy tại `http://127.0.0.1:8000`
        - **Video đen**: Kiểm tra kết nối webcam, thử tắt bật lại
        
        ### 💡 Tips:
        - Cấu hình mặc định (5s + 10 frames) cho kết quả tốt nhất
        - Bounding box sẽ hiển thị liên tục khi phát hiện biển số
        - Đảm bảo ánh sáng đủ và biển số rõ ràng trong khung hình
        - Giữ xe/biển số tương đối tĩnh khi chụp để độ chính xác cao
        
        ### 📊 Hiệu suất:
        - **CPU**: Thấp (~10-20%)
        - **RAM**: ~200-300 MB
        - **Network**: ~1-2 MB mỗi 5 giây
        - **Độ trễ**: < 100ms (render video)
        """)
