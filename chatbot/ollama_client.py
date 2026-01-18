import sys
import os
from openai import OpenAI
from colorama import Fore, Style

# Import config từ thư mục cha (trick để import ngược)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chatbot.config as config

class AIClient:
    def __init__(self):
        self.client = OpenAI(
            base_url=config.OLLAMA_BASE_URL,
            api_key=config.OLLAMA_API_KEY
        )

    def check_connection(self):
        """Kiểm tra xem Ollama đã bật chưa"""
        try:
            self.client.models.list()
            return True
        except Exception:
            return False

    def chat(self, messages):
        """Hàm chat chính"""
        try:
            response = self.client.chat.completions.create(
                model=config.MODEL_NAME,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Lỗi: {str(e)}"

    def chat_with_tools(self, messages, tools=None):
        """
        Hàm chat hỗ trợ Function Calling
        """
        try:
            response = self.client.chat.completions.create(
                model=config.MODEL_NAME,
                messages=messages,
                tools=tools,           # Gửi danh sách công cụ cho AI
                tool_choice="auto"     # Để AI tự quyết định có dùng tool hay không
            )
            return response.choices[0].message
        except Exception as e:
            # Trả về object giả lập lỗi để không crash chương trình
            class ErrorMessage:
                content = f"Lỗi kết nối Ollama: {str(e)}"
                tool_calls = None
            return ErrorMessage()