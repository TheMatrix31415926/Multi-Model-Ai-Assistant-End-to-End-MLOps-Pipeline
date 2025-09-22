# run_app.py - Single command to run everything
import subprocess
import time
import sys
import os
from threading import Thread
import webbrowser

def run_fastapi():
    """Run FastAPI server"""
    print(" Starting FastAPI server...")
    os.chdir("api")
    subprocess.run([sys.executable, "-m", "uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"])

def run_streamlit():
    """Run Streamlit app"""
    print(" Starting Streamlit frontend...")
    time.sleep(3)  # Wait for FastAPI to start
    os.chdir("frontend")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501"])

def main():
    print(" Multi-Modal AI Assistant - Full Stack Launcher")
    print("=" * 50)
    
    # Start FastAPI in a separate thread
    api_thread = Thread(target=run_fastapi, daemon=True)
    api_thread.start()
    
    # Wait a moment then start Streamlit
    time.sleep(2)
    
    print("📱 Opening browser...")
    try:
        webbrowser.open("http://localhost:8501")
    except:
        print("Could not open browser automatically")
    
    # Start Streamlit (this will block)
    run_streamlit()

if __name__ == "__main__":
    main()

# ---
