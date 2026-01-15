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
API_URL = "http://127.0.0.1:8000/ocr"
BACKEND_HOST = "http://127.0.0.1:8000"

st.set_page_config(page_title="Biển Số Xe OCR", layout="wide")
st.title("Nhận diện biển số xe Việt Nam")
st.markdown("Chọn tab để upload ảnh hoặc dùng camera realtime.")

tab1, tab2 = st.tabs(["📁 Upload Ảnh", "🎥 Camera Real-time"])

# ----------------------- TAB 1: Upload Ảnh -----------------------
with tab1:
    st.subheader("📸 Chọn nguồn ảnh & Xem trước")

    # Chia giao diện thành 2 cột để dễ nhìn
    col_upload, col_cam = st.columns(2)

    # --- CỘT 1: UPLOAD ẢNH ---
    with col_upload:
        st.markdown("### 1. Upload từ máy")
        uploaded_file = st.file_uploader("Chọn ảnh (jpg, png)", type=["jpg", "jpeg", "png"])
        
        # [QUAN TRỌNG] Hiển thị ảnh upload NGAY LẬP TỨC nếu có file
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Ảnh bạn vừa upload", use_container_width=True)
            st.success("✅ Đã tải ảnh lên!")

    # --- CỘT 2: CAMERA ---
    with col_cam:
        st.markdown("### 2. Chụp từ Camera")
        
        # Logic bật/tắt camera
        if "show_camera" not in st.session_state:
            st.session_state.show_camera = False

        def toggle_camera():
            st.session_state.show_camera = not st.session_state.show_camera

        # Nút bật tắt
        btn_text = "❌ Đóng Camera" if st.session_state.show_camera else "📷 Mở Camera"
        st.button(btn_text, on_click=toggle_camera, key="cam_toggle_btn")

        camera_file = None
        if st.session_state.show_camera:
            # Khung chụp ảnh
            camera_file = st.camera_input("Hãy canh chỉnh biển số vào giữa", label_visibility="visible")

            # [QUAN TRỌNG] Hiển thị ảnh chụp NGAY LẬP TỨC nếu vừa chụp xong
            if camera_file is not None:
                st.image(camera_file, caption="Ảnh vừa chụp xong", use_container_width=True)
                st.success("✅ Đã chụp xong!")

    # --- PHẦN XỬ LÝ (NẰM DƯỚI CẢ 2 CỘT) ---
    st.divider()
    
    # Xác định file nào sẽ được gửi đi xử lý
    # Ưu tiên ảnh Camera nếu có, nếu không thì lấy ảnh Upload
    if camera_file is not None:
        final_file = camera_file
        source_name = "Camera"
    else:
        final_file = uploaded_file
        source_name = "Upload"

    # Chỉ hiện nút bấm khi đã có ít nhất 1 file
    if final_file is not None:
        st.markdown(f"**Đang chọn ảnh từ nguồn:** `{source_name}`")
        
        if st.button("🚀 Gửi đi nhận diện biển số", type="primary", use_container_width=True):
            with st.spinner("Đang kết nối tới AI Server..."):
                try:
                    # Reset con trỏ file về đầu (Bắt buộc)
                    final_file.seek(0)
                    
                    # Gửi file lên Backend
                    files = {"file": (final_file.name, final_file.getvalue(), "image/jpeg")}
                    response = requests.post(API_URL, files=files)

                    if response.status_code == 200:
                        data = response.json()
                        if data["status"] == "success":
                            st.balloons() # Hiệu ứng chúc mừng
                            
                            # Hiển thị kết quả chi tiết
                            st.subheader("🔎 Kết quả nhận diện")
                            
                            # Cột trái: Ảnh gốc đã vẽ khung
                            res_col1, res_col2 = st.columns([2, 1])
                            with res_col1:
                                processed_url = f"{BACKEND_HOST}{data['processed_image_url']}"
                                st.image(processed_url, caption="Vị trí biển số", use_container_width=True)
                            
                            # Cột phải: Các biển số cắt rời (Crops)
                            with res_col2:
                                st.write("Biển số đọc được:")
                                if data["detections"]:
                                    for det in data["detections"]:
                                        st.image(f"{BACKEND_HOST}{det['crop_url']}", width=150)
                                        st.info(f"Biển: **{det['plate']}**\n\nĐộ tin cậy: {det['confidence']:.2f}")
                                else:
                                    st.warning("Không tìm thấy biển số nào.")
                        else:
                            st.error(f"Lỗi Backend: {data.get('error')}")
                    else:
                        st.error(f"Lỗi kết nối: {response.status_code}")
                except Exception as e:
                    st.error(f"Có lỗi xảy ra: {e}")
    else:
        st.info("👈 Vui lòng Upload ảnh hoặc Chụp ảnh để bắt đầu.")
        
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