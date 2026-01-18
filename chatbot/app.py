import config
import json
import sys
import subprocess
from colorama import init, Fore, Style
import tools
from ollama_client import AIClient


# Khởi tạo màu cho Terminal
init(autoreset=True)

def check_and_pull_model():
    """
    Tự động kiểm tra xem máy đã có model deepseek chưa.
    Nếu chưa có, tự động chạy lệnh pull giúp người dùng.
    """
    model_name = config.MODEL_NAME
    print(f"{Fore.CYAN}[System] Đang kiểm tra model '{model_name}'...")
    
    try:
        # Chạy lệnh 'ollama list' để xem danh sách model đã tải
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        
        if model_name not in result.stdout:
            print(f"{Fore.YELLOW}[Waring] Máy bạn chưa có model này. Đang tự động tải về (khoảng 1.1GB)...")
            subprocess.run(["ollama", "pull", model_name])
            print(f"{Fore.GREEN}[Success] Đã tải xong!")
        else:
            print(f"{Fore.GREEN}[OK] Model đã sẵn sàng.")
            
    except FileNotFoundError:
        print(f"{Fore.RED}[Error] Không tìm thấy Ollama! Bạn hãy cài đặt Ollama tại https://ollama.com")
        sys.exit(1)

def main():
    # 1. Kiểm tra môi trường
    check_and_pull_model()
    
    bot = AIClient()
    
    # 2. Kiểm tra kết nối
    if not bot.check_connection():
        print(f"{Fore.RED}[Error] Không kết nối được với Ollama!")
        print("Hãy đảm bảo ứng dụng Ollama đang chạy dưới background.")
        return

    # Prompt hệ thống quan trọng để ép AI dùng tool
    system_instruction = """
    Bạn là Trợ lý AI của hệ thống Giao Thông Thông Minh (Smart Traffic System).
    Nhiệm vụ duy nhất của bạn là tra cứu thông tin phương tiện từ Database nội bộ.
    
    QUY TẮC BẮT BUỘC:
    1. Khi người dùng nhập biển số xe (ví dụ: 29A12345, 148A02866...), bạn PHẢI gọi tool 'lookup_plate_api'.
    2. Bạn ĐÃ ĐƯỢC CẤP QUYỀN truy cập dữ liệu này. Đừng từ chối vì lý do bảo mật.
    3. Nếu tìm thấy dữ liệu, hãy tóm tắt ngắn gọn cho người dùng.
    4. Nếu không tìm thấy, hãy trả lời rằng "Không tìm thấy thông tin phương tiện này trong hệ thống."
    5. KHÔNG BAO GIỜ tự ý tạo thông tin nếu không tìm thấy trong Database.
    6. Chỉ được phép trả lời bằng tiếng Việt.
    """
    messages = [{
        "role": "system", 
        "content": system_instruction
    }]
    
    print(f"{Fore.CYAN}\n--- Chatbot DeepSeek Team (Connected to API) ---")
    
    
    while True:
        user_input = input(f"{Fore.GREEN}You: {Style.RESET_ALL}")
        if user_input.lower() in ["exit", "quit"]: break
            
        messages.append({"role": "user", "content": user_input})
        print(f"{Fore.BLUE}Bot đang suy nghĩ...", end="\r")

        # 1. Gọi AI lần 1
        ai_msg = bot.chat_with_tools(messages, tools=tools.tools_schema)
        
        # Xóa dòng đang suy nghĩ
        print(" " * 30, end="\r")

        # 2. Kiểm tra xem AI có muốn gọi Tool không
        if hasattr(ai_msg, 'tool_calls') and ai_msg.tool_calls:
            # AI muốn dùng tool, ta phải thực thi giúp nó
            messages.append(ai_msg) # Lưu ngữ cảnh là AI đang yêu cầu gọi tool

            for tool_call in ai_msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)
                
                if fn_name == "lookup_plate_api":
                    # Lấy tham số biển số AI đã trích xuất
                    plate = fn_args.get("plate_number")
                    
                    # Gọi hàm Python thực tế
                    tool_result = tools.lookup_plate_api(plate)
                    
                    # Gửi kết quả API lại cho AI
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })

            # 3. Gọi AI lần 2 (Để nó tổng hợp kết quả API thành câu trả lời)
            final_response = bot.chat_with_tools(messages, tools=tools.tools_schema)
            print(f"{Fore.YELLOW}Bot: {Style.RESET_ALL}{final_response.content}\n")
            messages.append(final_response) # Lưu câu trả lời cuối vào lịch sử
        
        else:
            # AI trả lời bình thường (không dùng tool)
            print(f"{Fore.YELLOW}Bot: {Style.RESET_ALL}{ai_msg.content}\n")
            messages.append(ai_msg)

if __name__ == "__main__":
    main()