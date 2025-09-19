#!/usr/bin/env python3
"""
Launch script for the AI Knowledge Companion
Starts both the FastAPI backend and Streamlit frontend
"""

import subprocess
import sys
import time
import threading
import signal
import os
from pathlib import Path

def run_fastapi():
    """Run the FastAPI backend server"""
    print("🚀 Starting FastAPI backend server...")
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 FastAPI server stopped")
    except Exception as e:
        print(f"❌ FastAPI server error: {e}")

def run_streamlit():
    """Run the Streamlit frontend"""
    print("🎨 Starting Streamlit frontend...")
    time.sleep(3)  # Wait for FastAPI to start
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_app.py", 
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Streamlit frontend stopped")
    except Exception as e:
        print(f"❌ Streamlit frontend error: {e}")

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n🛑 Shutting down AI Knowledge Companion...")
    sys.exit(0)

def main():
    """Main function to launch both servers"""
    print("🧠 AI Knowledge Companion - Starting Application...")
    print("=" * 50)
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Check if required files exist
    required_files = ["main.py", "streamlit_app.py", "requirements.txt"]
    for file in required_files:
        if not Path(file).exists():
            print(f"❌ Required file missing: {file}")
            sys.exit(1)
    
    # Start both servers in separate threads
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    streamlit_thread = threading.Thread(target=run_streamlit, daemon=True)
    
    try:
        # Start FastAPI backend
        fastapi_thread.start()
        
        # Start Streamlit frontend
        streamlit_thread.start()
        
        print("\n✅ AI Knowledge Companion is running!")
        print("📊 Backend API: http://localhost:8000")
        print("🎨 Frontend UI: http://localhost:8501")
        print("📚 API Docs: http://localhost:8000/docs")
        print("\nPress Ctrl+C to stop the application")
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    except Exception as e:
        print(f"❌ Application error: {e}")
    finally:
        print("👋 AI Knowledge Companion stopped")

if __name__ == "__main__":
    main()
