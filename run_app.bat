@echo off
echo Installing dependencies...
pip install -r requirements.txt
echo Starting Streamlit App (using python module)...
python -m streamlit run app.py

pause
