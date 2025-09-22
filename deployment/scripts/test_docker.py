
# deployment/scripts/test_docker.py - Docker testing script
#!/usr/bin/env python3

import requests
import time
import json
import sys
from datetime import datetime

class DockerTester:
    def __init__(self):
        self.base_url = "http://localhost"
        self.api_port = 8000
        self.frontend_port = 8501
        self.chromadb_port = 8002
        self.mlflow_port = 5000
        
    def wait_for_service(self, url, service_name, max_attempts=30):
        """Wait for a service to be ready"""
        print(f" Waiting for {service_name} to be ready...")
        
        for attempt in range(max_attempts):
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f" {service_name} is ready!")
                    return True
            except:
                pass
            
            print(f" Attempt {attempt + 1}/{max_attempts} for {service_name}...")
            time.sleep(2)
        
        print(f" {service_name} failed to start after {max_attempts} attempts")
        return False
    
    def test_api_health(self):
        """Test API health endpoint"""
        print("\n Testing API Health...")
        try:
            url = f"{self.base_url}:{self.api_port}/health"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                print(" API Health Check Passed")
                print(f"   Status: {health_data.get('status', 'unknown')}")
                print(f"   Timestamp: {health_data.get('timestamp', 'unknown')}")
                return True
            else:
                print(f" API Health Check Failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f" API Health Check Error: {e}")
            return False
    
    def test_api_endpoints(self):
        """Test main API endpoints"""
        print("\n Testing API Endpoints...")
        
        base_api_url = f"{self.base_url}:{self.api_port}"
        
        # Test root endpoint
        try:
            response = requests.get(f"{base_api_url}/", timeout=10)
            if response.status_code == 200:
                print(" Root endpoint working")
            else:
                print(" Root endpoint failed")
                return False
        except Exception as e:
            print(f" Root endpoint error: {e}")
            return False
        
        # Test chat endpoint
        try:
            chat_data = {
                "message": "Hello from Docker test!",
                "conversation_id": "docker_test"
            }
            response = requests.post(f"{base_api_url}/chat", json=chat_data, timeout=10)
            if response.status_code == 200:
                chat_response = response.json()
                print(" Chat endpoint working")
                print(f"   Response: {chat_response['response'][:50]}...")
            else:
                print(f" Chat endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            print(f" Chat endpoint error: {e}")
            return False
        
        return True
    
    def test_frontend_access(self):
        """Test frontend accessibility"""
        print("\n Testing Frontend Access...")
        try:
            url = f"{self.base_url}:{self.frontend_port}"
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                print(" Frontend is accessible")
                return True
            else:
                print(f" Frontend access failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f" Frontend access error: {e}")
            return False
    
    def test_chromadb(self):
        """Test ChromaDB connectivity"""
        print("\n Testing ChromaDB...")
        try:
            url = f"{self.base_url}:{self.chromadb_port}/api/v1/heartbeat"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                print(" ChromaDB is accessible")
                return True
            else:
                print(f" ChromaDB access failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f" ChromaDB access error: {e}")
            return False
    
    def test_mlflow(self):
        """Test MLflow server"""
        print("\n Testing MLflow Server...")
        try:
            url = f"{self.base_url}:{self.mlflow_port}/health"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                print(" MLflow server is accessible")
                return True
            else:
                print(f" MLflow access failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f" MLflow access error: {e}")
            return False
    
    def run_full_test_suite(self):
        """Run complete test suite"""
        print(" Multi-Modal AI Assistant - Docker Test Suite")
        print("=" * 60)
        
        test_results = {}
        
        # Wait for services to be ready
        services_to_check = [
            (f"{self.base_url}:{self.api_port}/health", "API Server"),
            (f"{self.base_url}:{self.frontend_port}", "Frontend"),
            (f"{self.base_url}:{self.chromadb_port}/api/v1/heartbeat", "ChromaDB"),
        ]
        
        print(" Waiting for services to start...")
        for url, name in services_to_check:
            if not self.wait_for_service(url, name):
                print(f" {name} failed to start. Stopping tests.")
                return False
        
        # Run individual tests
        print("\n Running Test Suite...")
        
        test_results['api_health'] = self.test_api_health()
        test_results['api_endpoints'] = self.test_api_endpoints()
        test_results['frontend_access'] = self.test_frontend_access()
        test_results['chromadb'] = self.test_chromadb()
        test_results['mlflow'] = self.test_mlflow()
        
        # Summary
        print("\n Test Results Summary:")
        print("=" * 30)
        
        passed = 0
        total = len(test_results)
        
        for test_name, result in test_results.items():
            status = " PASS" if result else " FAIL"
            print(f"{test_name.replace('_', ' ').title():<20} {status}")
            if result:
                passed += 1
        
        print(f"\n Overall Result: {passed}/{total} tests passed")
        
        if passed == total:
            print(" All tests passed! Your Docker setup is working correctly.")
            print("\n Access URLs:")
            print(f"   - API Documentation: http://localhost:{self.api_port}/docs")
            print(f"   - Frontend App: http://localhost:{self.frontend_port}")
            print(f"   - ChromaDB: http://localhost:{self.chromadb_port}")
            print(f"   - MLflow UI: http://localhost:{self.mlflow_port}")
            return True
        else:
            print(" Some tests failed. Check the logs above for details.")
            return False

if __name__ == "__main__":
    tester = DockerTester()
    success = tester.run_full_test_suite()
    sys.exit(0 if success else 1)