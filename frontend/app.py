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
with tab1:
    st.subheader("Upload ảnh xe")
    uploaded_file = st.file_uploader("Chọn file ảnh (jpg/png)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh gốc", use_container_width=True)

        if st.button("Xử lý ảnh"):
            with st.spinner("Đang detect và OCR..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(API_URL, files=files)

                if response.status_code == 200:
                    data = response.json()
                    if data["status"] == "success":
                        st.success("Xử lý thành công!")

                        # Xây dựng URL đầy đủ cho ảnh processed
                        processed_relative = data["processed_image_url"]
                        processed_url = f"{BACKEND_HOST}{processed_relative}"
                        processed_response = requests.get(processed_url)
                        if processed_response.status_code == 200:
                            st.image(processed_response.content, caption="Ảnh đã detect & OCR", use_container_width=True)

                        st.subheader("Kết quả nhận diện")
                        for det in data["detections"]:
                            st.write(f"**Biển số:** {det['plate']}")
                            st.write(f"**Độ tin cậy:** {det['confidence']:.2f}")
                            st.write(f"**Vị trí bbox:** {det['bbox']}")

                            if "crop_path" in det:
                                # Xây dựng URL đầy đủ cho crop
                                crop_relative = det["crop_path"].replace("crop_images", "/crops")
                                crop_url = f"{BACKEND_HOST}{crop_relative}"
                                crop_response = requests.get(crop_url)
                                if crop_response.status_code == 200:
                                    st.image(crop_response.content, caption=f"Crop biển số: {det['plate']}", width=300)
                    else:
                        st.error("Lỗi từ backend: " + str(data))
                else:
                    st.error(f"Lỗi kết nối backend: {response.status_code} - {response.text}")

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