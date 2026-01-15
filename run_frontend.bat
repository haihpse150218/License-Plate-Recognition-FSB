@echo off
echo --- Running Frontend (Streamlit) ---
:: Nếu dùng venv thì bỏ comment dòng dưới
:call .venv_yolo11\Scripts\activate

streamlit run frontend/app.py