Write-Host "Setting up NVIDIA Voice Pro Environment..." -ForegroundColor Cyan
pip install langchain-nvidia-ai-endpoints langgraph faiss-cpu streamlit 
pip install speechrecognition pyttsx3 pythoncom pypdf yagmail tavily-python
Write-Host "Setup Complete. Run 'streamlit run app.py' to start." -ForegroundColor Green