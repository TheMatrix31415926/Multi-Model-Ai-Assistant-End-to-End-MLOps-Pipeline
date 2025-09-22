# deployment/scripts/start_services.py - Service orchestration
#!/usr/bin/env python3

import subprocess
import time
import sys
import os
import signal
import threading
from pathlib import Path

class ServiceManager:
    def __init__(self):
        self.processes = []
        self.running = True
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n Received signal {signum}. Shutting down services...")
        self.running = False
        self.stop_all_services()
        sys.exit(0)
    
    def start_api_server(self):
        """Start FastAPI server"""
        print(" Starting FastAPI server...")
        try:
            # Change to API directory
            api_dir = Path("/app/api")
            if not api_dir.exists():
                api_dir = Path("/app")
            
            process = subprocess.Popen([
                sys.executable, "-m", "uvicorn", 
                "api.main:app", 
                "--host", "0.0.0.0", 
                "--port", "8000",
                "--reload" if os.getenv("DEBUG", "false").lower() == "true" else "--no-reload"
            ], cwd=api_dir)
            
            self.processes.append(("FastAPI", process))
            print(" FastAPI server started on port 8000")
            return process
            
        except Exception as e:
            print(f" Failed to start FastAPI server: {e}")
            return None
    
    def start_streamlit_app(self):
        """Start Streamlit app"""
        print(" Starting Streamlit frontend...")
        try:
            # Wait for API to be ready
            time.sleep(3)
            
            frontend_dir = Path("/app/frontend")
            if not frontend_dir.exists():
                frontend_dir = Path("/app")
            
            process = subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", 
                "frontend/app.py" if Path("/app/frontend/app.py").exists() else "app.py",
                "--server.port=8501",
                "--server.address=0.0.0.0",
                "--server.headless=true",
                "--server.enableCORS=false",
                "--server.enableXsrfProtection=false"
            ], cwd=frontend_dir)
            
            self.processes.append(("Streamlit", process))
            print(" Streamlit app started on port 8501")
            return process
            
        except Exception as e:
            print(f" Failed to start Streamlit app: {e}")
            return None
    
    def check_health(self):
        """Check health of services"""
        import requests
        
        # Check API health
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            api_healthy = response.status_code == 200
        except:
            api_healthy = False
        
        # Check Streamlit health
        try:
            response = requests.get("http://localhost:8501/_stcore/health", timeout=5)
            streamlit_healthy = response.status_code == 200
        except:
            streamlit_healthy = False
        
        return api_healthy, streamlit_healthy
    
    def monitor_services(self):
        """Monitor running services"""
        print(" Starting service monitor...")
        
        while self.running:
            time.sleep(30)  # Check every 30 seconds
            
            # Check process status
            for name, process in self.processes:
                if process.poll() is not None:
                    print(f" {name} process has stopped!")
                    # Optionally restart here
            
            # Health check
            try:
                api_healthy, streamlit_healthy = self.check_health()
                status = "green" if api_healthy and streamlit_healthy else "yellow"
                print(f"{status} Health Check - API: {'green' if api_healthy else 'red'} | Streamlit: {'G' if streamlit_healthy else 'R'}")
            except Exception as e:
                print(f" Health check failed: {e}")
    
    def stop_all_services(self):
        """Stop all running services"""
        print(" Stopping all services...")
        
        for name, process in self.processes:
            if process.poll() is None:  # Process is still running
                print(f"Stopping {name}...")
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=10)
                    print(f" {name} stopped gracefully")
                except subprocess.TimeoutExpired:
                    print(f" Force killing {name}...")
                    process.kill()
        
        self.processes.clear()
    
    def run(self):
        """Main run method"""
        print(" Multi-Modal AI Assistant - Docker Container Manager")
        print("=" * 60)
        
        # Set up signal handlers
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Start services
        api_process = self.start_api_server()
        if not api_process:
            print(" Failed to start API server. Exiting...")
            sys.exit(1)
        
        streamlit_process = self.start_streamlit_app()
        if not streamlit_process:
            print("Failed to start Streamlit app, but continuing with API only...")
        
        print("\n All services started successfully!")
        print(" Service URLs:")
        print("   - API Documentation: http://localhost:8000/docs")
        print("   - API Health: http://localhost:8000/health")
        print("   - Streamlit App: http://localhost:8501")
        print("\n Starting monitoring...")
        
        # Start monitoring in a separate thread
        monitor_thread = threading.Thread(target=self.monitor_services, daemon=True)
        monitor_thread.start()
        
        # Wait for processes
        try:
            while self.running and any(p.poll() is None for _, p in self.processes):
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_all_services()

if __name__ == "__main__":
    manager = ServiceManager()
    manager.run()

