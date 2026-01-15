# frontend/app.py
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
import uuid

# Địa chỉ backend FastAPI (có thể đổi khi deploy)
API_URL = "http://127.0.0.1:8000/ocr"
BACKEND_HOST = "http://127.0.0.1:8000"  # Dùng biến này để xây dựng URL đầy đủ

st.set_page_config(page_title="Biển Số Xe OCR", layout="wide")
st.title("Nhận diện biển số xe Việt Nam")
st.markdown("Chọn tab để upload ảnh hoặc dùng camera realtime.")

tab1, tab2 = st.tabs(["📁 Upload Ảnh", "🎥 Camera Real-time"])

# ----------------------- TAB 1: Upload Ảnh -----------------------
# ----------------------- TAB 1: Upload + Camera -----------------------
# ----------------------- TAB 1: Upload + Camera -----------------------
# ----------------------- TAB 1: Upload + Camera -----------------------
with tab1:
    st.subheader("Upload hoặc chụp ảnh xe")

    # ===== INIT SESSION STATE =====
    if "input_image_bytes" not in st.session_state:
        st.session_state.input_image_bytes = None
        st.session_state.input_image_name = None
        st.session_state.input_image_type = None

    if "show_camera" not in st.session_state:
        st.session_state.show_camera = False

    if "action" not in st.session_state:
        st.session_state.action = None

    # ===== UPLOAD IMAGE =====
    uploaded_file = st.file_uploader(
        "📁 Upload ảnh (jpg/png)",
        type=["jpg", "jpeg", "png"],
        key="upload_image_tab1"
    )

    if uploaded_file is not None:
        st.session_state.input_image_bytes = uploaded_file.getvalue()
        st.session_state.input_image_name = uploaded_file.name
        st.session_state.input_image_type = uploaded_file.type

    # ===== CAMERA BUTTON =====
    if not st.session_state.show_camera:
        if st.button("📸 Bật camera", key="btn_show_camera"):
            st.session_state.action = "show_camera"
            st.rerun()
    else:
        camera_photo = st.camera_input(
            "Chụp ảnh",
            key="camera_input_tab1"
        )
        if camera_photo is not None:
            st.session_state.input_image_bytes = camera_photo.getvalue()
            st.session_state.input_image_name = "camera.jpg"
            st.session_state.input_image_type = "image/jpeg"

            st.session_state.show_camera = False
            st.rerun()

    # ===== HANDLE ACTIONS (ONE PLACE ONLY) =====
    if st.session_state.action == "show_camera":
        st.session_state.show_camera = True
        st.session_state.action = None

    elif st.session_state.action == "clear_image":
        st.session_state.input_image_bytes = None
        st.session_state.input_image_name = None
        st.session_state.input_image_type = None
        st.session_state.show_camera = False
        st.session_state.action = None

    # ===== PREVIEW (ONE IMAGE BOX) =====
    st.markdown("### Ảnh đầu vào")

    if st.session_state.input_image_bytes:
        image = Image.open(
            io.BytesIO(st.session_state.input_image_bytes)
        )
        st.image(
            image,
            caption="Ảnh đang xử lý",
            use_container_width=True
        )
    else:
        st.info("Chưa có ảnh. Vui lòng upload hoặc chụp ảnh.")

    # ===== ACTION BUTTONS =====
    col1, col2 = st.columns(2)

    with col1:
        process_clicked = st.button(
            "🚀 Xử lý ảnh",
            key="btn_process_image",
            disabled=st.session_state.input_image_bytes is None
        )

    with col2:
        if st.button(
            "❌ Xóa ảnh",
            key="btn_clear_image",
            disabled=st.session_state.input_image_bytes is None
        ):
            st.session_state.action = "clear_image"
            st.rerun()

    # ===== OCR PROCESS =====
    if process_clicked:
        with st.spinner("Đang detect và OCR..."):
            files = {
                "file": (
                    st.session_state.input_image_name,
                    st.session_state.input_image_bytes,
                    st.session_state.input_image_type
                )
            }

            response = requests.post(API_URL, files=files)

            if response.status_code == 200:
                data = response.json()

                if data.get("status") == "success":
                    st.success("Xử lý thành công!")

                    processed_url = f"{BACKEND_HOST}{data['processed_image_url']}"
                    processed_response = requests.get(processed_url)

                    if processed_response.status_code == 200:
                        st.image(
                            processed_response.content,
                            caption="Ảnh đã detect & OCR",
                            use_container_width=True
                        )

                    st.subheader("Kết quả nhận diện")
                    for det in data["detections"]:
                        st.write(f"**Biển số:** {det['plate']}")
                        st.write(f"**Độ tin cậy:** {det['confidence']:.2f}")
                        st.write(f"**BBox:** {det['bbox']}")
                        st.write(f"**type:** {det['type']}")
                else:
                    st.error("Backend xử lý thất bại")
            else:
                st.error(f"Lỗi backend: {response.status_code}")

# ----------------------- TAB 2: Camera Real-time -----------------------
with tab2:
    st.subheader("Camera Real-time OCR")

    # Thư mục lưu ảnh chụp
    captured_dir = "captured_photos"
    os.makedirs(captured_dir, exist_ok=True)

    # Khởi tạo session state
    if 'captured_frame' not in st.session_state:
        st.session_state.captured_frame = None
    if 'captured_result' not in st.session_state:
        st.session_state.captured_result = None
    if 'capture_requested' not in st.session_state:
        st.session_state.capture_requested = False

    class VideoProcessor(VideoTransformerBase):
        def __init__(self):
            self.last_time = time.time()

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")

            # Chuyển frame thành bytes
            _, buffer = cv2.imencode('.jpg', img)
            image_bytes = buffer.tobytes()

            try:
                files = {"file": ("frame.jpg", image_bytes, "image/jpeg")}
                response = requests.post(API_URL, files=files, timeout=3)

                if response.status_code == 200:
                    data = response.json()
                    if data["status"] == "success" and data["detections"]:
                        for det in data["detections"]:
                            plate = det["plate"]
                            conf = det["confidence"]
                            bbox = det["bbox"]
                            x1, y1, x2, y2 = bbox

                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(img, f"{plate} ({conf:.2f})", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                        # Lưu frame nếu nút chụp được nhấn
                        if st.session_state.capture_requested:
                            st.session_state.captured_frame = img.copy()
                            st.session_state.captured_result = data
                            st.session_state.capture_requested = False

            except Exception as e:
                cv2.putText(img, f"Error: {str(e)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Giới hạn FPS
            current_time = time.time()
            if current_time - self.last_time < 0.1:
                time.sleep(0.1 - (current_time - self.last_time))
            self.last_time = current_time

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    RTC_CONFIGURATION = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })

    # Streamer
    ctx = webrtc_streamer(
        key="real-time-ocr",
        video_transformer_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_html_attrs=VideoHTMLAttributes(muted=True, volume=0)
    )

    # Nút chụp ảnh
    if st.button("📸 Chụp ảnh"):
        st.session_state.capture_requested = True
        st.success("Đã chụp! Đang xử lý...")

    # Hiển thị ảnh vừa chụp (nếu có)
    if st.session_state.captured_frame is not None:
        st.subheader("Ảnh vừa chụp")
        captured_rgb = cv2.cvtColor(st.session_state.captured_frame, cv2.COLOR_BGR2RGB)
        st.image(captured_rgb, channels="RGB", use_container_width=True)

        # Hiển thị kết quả OCR
        if st.session_state.captured_result and st.session_state.captured_result["detections"]:
            st.subheader("Kết quả nhận diện")
            for det in st.session_state.captured_result["detections"]:
                st.write(f"**Biển số:** {det['plate']}")
                st.write(f"**Độ tin cậy:** {det['confidence']:.2f}")

            # Nút tải ảnh
            _, buffer = cv2.imencode('.jpg', st.session_state.captured_frame)
            b64 = base64.b64encode(buffer).decode()
            href = f'<a href="data:image/jpeg;base64,{b64}" download="captured_plate.jpg">Tải ảnh về</a>'
            st.markdown(href, unsafe_allow_html=True)

        # Lưu ảnh vào thư mục
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(captured_dir, f"capture_{timestamp}.jpg")
        cv2.imwrite(save_path, st.session_state.captured_frame)
        st.info(f"Ảnh đã được lưu tại: {save_path}")

        # Nút xóa ảnh chụp
        if st.button("Xóa ảnh vừa chụp"):
            st.session_state.captured_frame = None
            st.session_state.captured_result = None
            st.rerun()

    st.info("Nếu camera không mở: Cho phép quyền truy cập webcam trong trình duyệt. Backend phải chạy tại port 8000.")