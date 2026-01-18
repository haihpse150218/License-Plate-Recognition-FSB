import config as fe_config_init
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
import json

from chatbot.ollama_client import AIClient
from chatbot import tools


# Địa chỉ backend FastAPI (có thể đổi khi deploy)
API_URL = "http://127.0.0.1:8000/ocr"
BACKEND_HOST = "http://127.0.0.1:8000"  # Dùng biến này để xây dựng URL đầy đủ

st.set_page_config(page_title="Biển Số Xe OCR", layout="wide")
st.title("Nhận diện biển số xe Việt Nam")
st.markdown("Chọn tab để upload ảnh hoặc dùng camera realtime.")

tab1, tab2 = st.tabs(["📁 Upload Ảnh", "Trợ lý Giao thông Thông minh (AI Agent)"])

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

            try:
                response = requests.post(API_URL, files=files)

                if response.status_code == 200:
                    data = response.json()
                    
                    if data.get("status") == "success":
                        st.success("Xử lý thành công!")
                        
                        # Hiển thị ảnh kết quả từ Backend
                        processed_url = f"{BACKEND_HOST}{data['processed_image_url']}"
                        processed_response = requests.get(processed_url)
                        if processed_response.status_code == 200:
                            st.image(
                                processed_response.content,
                                caption="Ảnh đã detect & OCR",
                                use_container_width=True
                            )

                        st.subheader("Kết quả chi tiết")
                        st.write(f"**Loại phương tiện:** {data.get('type', 'Unknown')}")

                        # --- VÒNG LẶP XỬ LÝ TỪNG BIỂN SỐ ---
                        for i, det in enumerate(data["detections"]):
                            plate_number = det['plate']
                            confidence = det['confidence']
                            
                            # Tạo Expand cho gọn gàng
                            with st.expander(f"🚗 Biển số #{i+1}: {plate_number}", expanded=True):
                                col_info, col_ai = st.columns([1, 2])
                                
                                # Cột 1: Thông tin kỹ thuật OCR
                                with col_info:
                                    st.markdown("#### Thông số OCR")
                                    st.write(f"**Biển số:** `{plate_number}`")
                                    st.write(f"**Độ tin cậy:** `{confidence:.2f}`")
                                    st.write(f"**Vị trí:** `{det['bbox']}`")

                                # Cột 2: Chatbot Phân tích (MỚI THÊM)
                                with col_ai:
                                    st.markdown("#### 🤖 Trợ lý AI Phân tích")
                                    
                                    # Logic gọi Bot ngay tại đây
                                    with st.spinner("AI đang tra cứu dữ liệu..."):
                                        # Bước 1: Gọi hàm Python tra cứu database (Nhanh hơn gọi qua tool calling)
                                        # Chúng ta lấy dữ liệu thô (JSON) trước
                                        db_result_json = tools.lookup_plate_api(plate_number)
                                        
                                        # Bước 2: Nhờ AI đọc và diễn giải JSON đó
                                        bot = AIClient()
                                        
                                        # Prompt hướng dẫn AI đọc dữ liệu
                                        prompt_for_ai = f"""
                                        Tôi vừa nhận diện được biển số: {plate_number}.
                                        Dưới đây là dữ liệu tra cứu từ Database về biển số này:
                                        {db_result_json}
                                        
                                        Nhiệm vụ:
                                        Hãy đóng vai Trợ lý Giao thông, đọc dữ liệu trên và viết một báo cáo ngắn gọn, lịch sự cho người dùng (bằng tiếng Việt).
                                        Giải thích rõ các trường: Chủ xe (owner_name), Phạt nguội (fine_amount), Điểm trừ (points).
                                        Nếu không tìm thấy dữ liệu (lỗi 404), hãy báo là xe chưa có dữ liệu.
                                        """
                                        
                                        # Gọi Chatbot (Chế độ chat thường, không cần tool vì ta đã tra cứu giúp nó rồi)
                                        try:
                                            # Gửi 1 message duy nhất để lấy phản hồi
                                            ai_response = bot.chat_with_tools(
                                                [{"role": "user", "content": prompt_for_ai}]
                                            )
                                            # Hiển thị lời nói của AI
                                            st.info(ai_response.content)
                                            
                                        except Exception as e:
                                            st.error(f"Lỗi AI: {e}")

                    else:
                        st.error(f"Backend xử lý thất bại: {data.get('error')}")
                else:
                    st.error(f"Lỗi kết nối API: {response.status_code}")
            
            except Exception as e:
                st.error(f"Có lỗi xảy ra: {e}")

# ----------------------- TAB 2: Camera Real-time -----------------------
# with tab2:
#     st.subheader("Camera Real-time OCR")

#     # Thư mục lưu ảnh chụp
#     captured_dir = "captured_photos"
#     os.makedirs(captured_dir, exist_ok=True)

#     # Khởi tạo session state
#     if 'captured_frame' not in st.session_state:
#         st.session_state.captured_frame = None
#     if 'captured_result' not in st.session_state:
#         st.session_state.captured_result = None
#     if 'capture_requested' not in st.session_state:
#         st.session_state.capture_requested = False

#     class VideoProcessor(VideoTransformerBase):
#         def __init__(self):
#             self.last_time = time.time()

#         def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
#             img = frame.to_ndarray(format="bgr24")

#             # Chuyển frame thành bytes
#             _, buffer = cv2.imencode('.jpg', img)
#             image_bytes = buffer.tobytes()

#             try:
#                 files = {"file": ("frame.jpg", image_bytes, "image/jpeg")}
#                 response = requests.post(API_URL, files=files, timeout=3)

#                 if response.status_code == 200:
#                     data = response.json()
#                     if data["status"] == "success" and data["detections"]:
#                         for det in data["detections"]:
#                             plate = det["plate"]
#                             conf = det["confidence"]
#                             bbox = det["bbox"]
#                             x1, y1, x2, y2 = bbox

#                             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                             cv2.putText(img, f"{plate} ({conf:.2f})", (x1, y1 - 10),
#                                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#                         # Lưu frame nếu nút chụp được nhấn
#                         if st.session_state.capture_requested:
#                             st.session_state.captured_frame = img.copy()
#                             st.session_state.captured_result = data
#                             st.session_state.capture_requested = False

#             except Exception as e:
#                 cv2.putText(img, f"Error: {str(e)}", (10, 30),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#             # Giới hạn FPS
#             current_time = time.time()
#             if current_time - self.last_time < 0.1:
#                 time.sleep(0.1 - (current_time - self.last_time))
#             self.last_time = current_time

#             return av.VideoFrame.from_ndarray(img, format="bgr24")

#     RTC_CONFIGURATION = RTCConfiguration({
#         "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
#     })

#     # Streamer
#     ctx = webrtc_streamer(
#         key="real-time-ocr",
#         video_transformer_factory=VideoProcessor,
#         rtc_configuration=RTC_CONFIGURATION,
#         media_stream_constraints={"video": True, "audio": False},
#         video_html_attrs=VideoHTMLAttributes(muted=True, volume=0)
#     )

#     # Nút chụp ảnh
#     if st.button("📸 Chụp ảnh"):
#         st.session_state.capture_requested = True
#         st.success("Đã chụp! Đang xử lý...")

#     # Hiển thị ảnh vừa chụp (nếu có)
#     if st.session_state.captured_frame is not None:
#         st.subheader("Ảnh vừa chụp")
#         captured_rgb = cv2.cvtColor(st.session_state.captured_frame, cv2.COLOR_BGR2RGB)
#         st.image(captured_rgb, channels="RGB", use_container_width=True)

#         # Hiển thị kết quả OCR
#         if st.session_state.captured_result and st.session_state.captured_result["detections"]:
#             st.subheader("Kết quả nhận diện")
#             for det in st.session_state.captured_result["detections"]:
#                 st.write(f"**Biển số:** {det['plate']}")
#                 st.write(f"**Độ tin cậy:** {det['confidence']:.2f}")

#             # Nút tải ảnh
#             _, buffer = cv2.imencode('.jpg', st.session_state.captured_frame)
#             b64 = base64.b64encode(buffer).decode()
#             href = f'<a href="data:image/jpeg;base64,{b64}" download="captured_plate.jpg">Tải ảnh về</a>'
#             st.markdown(href, unsafe_allow_html=True)

#         # Lưu ảnh vào thư mục
#         timestamp = time.strftime("%Y%m%d_%H%M%S")
#         save_path = os.path.join(captured_dir, f"capture_{timestamp}.jpg")
#         cv2.imwrite(save_path, st.session_state.captured_frame)
#         st.info(f"Ảnh đã được lưu tại: {save_path}")

#         # Nút xóa ảnh chụp
#         if st.button("Xóa ảnh vừa chụp"):
#             st.session_state.captured_frame = None
#             st.session_state.captured_result = None
#             st.rerun()

#     st.info("Nếu camera không mở: Cho phép quyền truy cập webcam trong trình duyệt. Backend phải chạy tại port 8000.")

# ----------------------- TAB 2: Chat bot AI -----------------------
# ----------------------- TAB 2: Chat bot AI -----------------------
# ----------------------- TAB 2: Chat bot AI -----------------------
with tab2:
    st.subheader("💬 Trợ lý Giao thông Thông minh (AI Agent)")
    
    # --- CẬP NHẬT: System Instruction mới cực mạnh ---
    # --- CẬP NHẬT: System Instruction có giải nghĩa trường dữ liệu ---
    system_instruction = """
    Bạn là Trợ lý AI của hệ thống Giao Thông Thông Minh (Smart Traffic System).
    Nhiệm vụ duy nhất của bạn là tra cứu thông tin phương tiện từ Database nội bộ và trả lời người dùng.
    
    1. QUY TẮC BẮT BUỘC KHI GỌI TOOL:
       - Khi người dùng nhập biển số xe, BẮT BUỘC gọi tool 'lookup_plate_api'.
       - Không được tự bịa thông tin. Nếu API trả về lỗi hoặc không tìm thấy, hãy báo đúng như vậy.

    2. HƯỚNG DẪN DỊCH VÀ HIỂU DỮ LIỆU TỪ API (QUAN TRỌNG):
       Khi nhận được JSON từ tool, hãy giải thích theo đúng nghĩa sau:
       - 'plate_number': Biển số xe.
       - 'vehicle_type': Loại phương tiện (Ví dụ: Ô tô, Xe máy, Xe tải...).
       - 'owner_name': Tên chủ sở hữu xe.
       - 'points': SỐ ĐIỂM BỊ TRỪ trên Giấy phép lái xe (Không phải tần suất vi phạm).
       - 'fine_amount': Số tiền phạt nguội cần đóng (Đơn vị: VNĐ).
       - 'detected_at': Thời điểm camera phát hiện vi phạm.
       - 'confidence': Độ tin cậy của AI khi nhận diện biển số (Ví dụ: 0.9 tức là chính xác 90%).

    3. ĐỊNH DẠNG CÂU TRẢ LỜI:
       - Trả lời ngắn gọn, chuyên nghiệp, dùng gạch đầu dòng cho dễ đọc.
       - Ví dụ:
         * Biển số: 30A12345
         * Chủ xe: Nguyễn Văn A
         * Tiền phạt: 3.000.000 VNĐ
    """

    # 1. Khởi tạo Session State (Chỉ chạy 1 lần đầu tiên)
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "system", 
            "content": system_instruction
        }]

    # --- HÀM PHỤ TRỢ: Lấy dữ liệu an toàn (Chống lỗi Crash) ---
    def get_message_data(msg):
        if isinstance(msg, dict):
            return msg.get("role"), msg.get("content", "")
        else:
            return getattr(msg, "role", "assistant"), getattr(msg, "content", "")

    # 2. Hiển thị lịch sử chat
    for message in st.session_state.messages:
        role, content = get_message_data(message)
        
        if role != "system":
            with st.chat_message(role):
                if role == "tool":
                    with st.expander("Dữ liệu từ hệ thống (Debug):"):
                        st.code(content, language="json")
                else:
                    st.markdown(content if content else "")

    # 3. Xử lý logic Chat
    if prompt := st.chat_input("Nhập biển số xe cần tra cứu (VD: 148A02866)..."):
        
        # A. Hiển thị câu hỏi User
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # B. Xử lý AI
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            with st.spinner("Đang truy xuất dữ liệu..."):
                bot = AIClient()
                
                # Gọi AI lần 1
                ai_msg = bot.chat_with_tools(st.session_state.messages, tools=tools.tools_schema)
                
                # --- CHUYỂN ĐỔI OBJECT -> DICT (Quan trọng) ---
                ai_msg_dict = {
                    "role": ai_msg.role,
                    "content": ai_msg.content,
                    "tool_calls": getattr(ai_msg, 'tool_calls', None)
                }

                # C. Kiểm tra xem AI có muốn gọi Tool không
                if ai_msg_dict.get("tool_calls"):
                    st.session_state.messages.append(ai_msg_dict)
                    
                    for tool_call in ai_msg_dict["tool_calls"]:
                        fn_name = tool_call.function.name
                        fn_args = json.loads(tool_call.function.arguments)
                        
                        if fn_name == "lookup_plate_api":
                            plate = fn_args.get("plate_number")
                            message_placeholder.markdown(f"🔍 *Đang tra cứu biển số: **{plate}**...*")
                            
                            # Gọi API thật
                            tool_result = tools.lookup_plate_api(plate)
                            
                            # Lưu kết quả Tool vào lịch sử
                            st.session_state.messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": tool_result
                            })
                            
                            with st.expander("Chi tiết JSON"):
                                st.code(tool_result, language="json")

                    # Gọi AI lần 2 (Tổng hợp kết quả)
                    final_response = bot.chat_with_tools(st.session_state.messages, tools=tools.tools_schema)
                    full_response = final_response.content
                
                else:
                    # AI trả lời bình thường
                    full_response = ai_msg.content

                # D. Hiển thị & Lưu kết quả cuối cùng
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})