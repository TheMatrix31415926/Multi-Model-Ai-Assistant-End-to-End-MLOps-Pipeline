"""test_full_workflow.py module"""

# TODO: Implement functionality
# tests/e2e/test_complete_workflows.py - End-to-end workflow tests
import pytest
import requests
import time
import json
import io
from PIL import Image
import subprocess
import os
from pathlib import Path

class TestCompleteWorkflows:
    """Test complete end-to-end user workflows"""
    
    @pytest.fixture(scope="class", autouse=True)
    def setup_services(self):
        """Ensure services are running for E2E tests"""
        # Check if services are already running
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print(" Services already running")
                yield
                return
        except:
            pass
        
        # Start services if not running
        print(" Starting services for E2E tests...")
        try:
            subprocess.run(
                ["docker-compose", "up", "-d"], 
                check=True, 
                capture_output=True,
                timeout=120
            )
            
            # Wait for services to be ready
            self.wait_for_services()
            yield
            
        except subprocess.CalledProcessError as e:
            pytest.fail(f"Failed to start services: {e}")
        except subprocess.TimeoutExpired:
            pytest.fail("Service startup timed out")
    
    def wait_for_services(self, max_wait=180):
        """Wait for all services to be healthy"""
        start_time = time.time()
        
        services = [
            ("API", "http://localhost:8000/health"),
            ("Frontend", "http://localhost:8501")
        ]
        
        while time.time() - start_time < max_wait:
            all_healthy = True
            
            for service_name, url in services:
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code not in [200, 201]:
                        all_healthy = False
                        break
                except:
                    all_healthy = False
                    break
            
            if all_healthy:
                print(" All services are healthy")
                return
            
            print(f" Waiting for services... ({int(time.time() - start_time)}s)")
            time.sleep(5)
        
        pytest.fail("Services failed to become healthy within timeout")
    
    def create_test_image(self, color='blue', size=(400, 300)):
        """Create test image for uploads"""
        img = Image.new('RGB', size, color=color)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        return img_bytes
    
    def test_complete_user_journey(self):
        """Test complete user journey from start to finish"""
        base_url = "http://localhost:8000"
        
        print(" Testing complete user journey...")
        
        # Step 1: Health check
        print("1. Checking API health...")
        health_response = requests.get(f"{base_url}/health")
        assert health_response.status_code == 200
        health_data = health_response.json()
        assert health_data["status"] == "healthy"
        
        # Step 2: Upload an image
        print("2. Uploading test image...")
        test_image = self.create_test_image(color='red')
        files = {"file": ("red_car.jpg", test_image, "image/jpeg")}
        
        upload_response = requests.post(f"{base_url}/upload", files=files)
        assert upload_response.status_code == 200
        
        upload_data = upload_response.json()
        image_id = upload_data["image_id"]
        assert image_id is not None
        print(f"    Image uploaded with ID: {image_id}")
        
        # Step 3: Ask questions about the image
        questions = [
            "What do you see in this image?",
            "What color is dominant in the image?", 
            "Can you describe the image in detail?"
        ]
        
        conversation_id = "e2e_test_conversation"
        
        for i, question in enumerate(questions, 1):
            print(f"3.{i} Asking: '{question}'")
            
            chat_data = {
                "message": question,
                "image_id": image_id,
                "conversation_id": conversation_id
            }
            
            chat_response = requests.post(f"{base_url}/chat", json=chat_data)
            assert chat_response.status_code == 200
            
            chat_data = chat_response.json()
            assert "response" in chat_data
            assert chat_data["confidence"] > 0
            assert chat_data["conversation_id"] == conversation_id
            
            print(f"    AI Response: {chat_data['response'][:100]}...")
            print(f"    Confidence: {chat_data['confidence']:.0%}")
        
        # Step 4: Test conversation without image
        print("4. Testing text-only conversation...")
        text_questions = [
            "Hello, how are you?",
            "What can you help me with?",
            "Thank you for your help!"
        ]
        
        for question in text_questions:
            chat_data = {
                "message": question,
                "conversation_id": conversation_id
            }
            
            chat_response = requests.post(f"{base_url}/chat", json=chat_data)
            assert chat_response.status_code == 200
            
            response_data = chat_response.json()
            assert "response" in response_data
            print(f"    Text chat working: {len(response_data['response'])} chars")
        
        # Step 5: Check conversation history
        print("5. Checking conversation history...")
        history_response = requests.get(f"{base_url}/conversations/{conversation_id}")
        if history_response.status_code == 200:
            history_data = history_response.json()
            assert history_data["conversation_id"] == conversation_id
            print(f"    Conversation history: {history_data['message_count']} messages")
        
        # Step 6: Test monitoring endpoints
        print("6. Testing monitoring endpoints...")
        
        # Metrics
        metrics_response = requests.get(f"{base_url}/metrics")
        assert metrics_response.status_code == 200
        print("    Metrics endpoint working")
        
        # Dashboard
        dashboard_response = requests.get(f"{base_url}/monitoring/dashboard")
        if dashboard_response.status_code == 200:
            dashboard_data = dashboard_response.json()
            assert "timestamp" in dashboard_data
            print("    Monitoring dashboard working")
        
        print(" Complete user journey test passed!")
    
    def test_multiple_image_workflow(self):
        """Test workflow with multiple images"""
        base_url = "http://localhost:8000"
        
        print(" Testing multiple image workflow...")
        
        # Upload multiple images
        images = [
            ("blue_image.jpg", self.create_test_image('blue')),
            ("green_image.jpg", self.create_test_image('green')), 
            ("red_image.jpg", self.create_test_image('red'))
        ]
        
        image_ids = []
        
        for filename, image_data in images:
            files = {"file": (filename, image_data, "image/jpeg")}
            response = requests.post(f"{base_url}/upload", files=files)
            assert response.status_code == 200
            
            image_id = response.json()["image_id"]
            image_ids.append(image_id)
            print(f"    Uploaded {filename}: {image_id}")
        
        # Ask about each image
        for i, image_id in enumerate(image_ids):
            chat_data = {
                "message": f"What's the main color in image {i+1}?",
                "image_id": image_id,
                "conversation_id": f"multi_image_test_{i}"
            }
            
            response = requests.post(f"{base_url}/chat", json=chat_data)
            assert response.status_code == 200
            
            chat_result = response.json()
            print(f"    Image {i+1} response: {chat_result['response'][:50]}...")
        
        # Check image list
        images_response = requests.get(f"{base_url}/images")
        if images_response.status_code == 200:
            images_data = images_response.json()
            assert images_data["total_images"] >= 3
            print(f"    Total images stored: {images_data['total_images']}")
        
        print(" Multiple image workflow test passed!")
    
    def test_error_scenarios_e2e(self):
        """Test error handling in complete workflows"""
        base_url = "http://localhost:8000"
        
        print(" Testing error scenarios...")
        
        # Test 1: Invalid image upload
        print("1. Testing invalid file upload...")
        files = {"file": ("test.txt", io.BytesIO(b"Not an image"), "text/plain")}
        response = requests.post(f"{base_url}/upload", files=files)
        assert response.status_code == 400
        print("    Invalid file upload properly rejected")
        
        # Test 2: Chat with non-existent image
        print("2. Testing chat with invalid image ID...")
        chat_data = {
            "message": "What's in this image?",
            "image_id": "nonexistent_image_123"
        }
        response = requests.post(f"{base_url}/chat", json=chat_data)
        assert response.status_code == 200  # Should handle gracefully
        print("    Invalid image ID handled gracefully")
        
        # Test 3: Malformed requests
        print("3. Testing malformed requests...")
        
        # Missing message field
        response = requests.post(f"{base_url}/chat", json={"image_id": "test"})
        assert response.status_code in [200, 422]  # Should handle gracefully
        
        # Invalid JSON
        response = requests.post(
            f"{base_url}/chat", 
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
        print("   Malformed requests handled properly")
        
        print(" Error scenarios test passed!")
    
    @pytest.mark.slow
    def test_performance_workflow(self):
        """Test performance under typical usage"""
        base_url = "http://localhost:8000"
        
        print(" Testing performance workflow...")
        
        # Upload test image
        test_image = self.create_test_image()
        files = {"file": ("perf_test.jpg", test_image, "image/jpeg")}
        response = requests.post(f"{base_url}/upload", files=files)
        assert response.status_code == 200
        image_id = response.json()["image_id"]
        
        # Test multiple rapid requests
        questions = [
            "What do you see?",
            "What colors are present?", 
            "Describe the image",
            "What's the main subject?",
            "Tell me more details"
        ] * 4  # 20 total questions
        
        start_time = time.time()
        successful_requests = 0
        total_response_time = 0
        
        for i, question in enumerate(questions):
            try:
                request_start = time.time()
                
                chat_data = {
                    "message": question,
                    "image_id": image_id,
                    "conversation_id": f"perf_test_{i // 5}"  # Group into conversations
                }
                
                response = requests.post(f"{base_url}/chat", json=chat_data, timeout=30)
                
                request_time = time.time() - request_start
                total_response_time += request_time
                
                if response.status_code == 200:
                    successful_requests += 1
                    if (i + 1) % 5 == 0:
                        print(f"    Completed {i + 1}/{len(questions)} requests")
                
            except requests.exceptions.Timeout:
                print(f"    Request {i+1} timed out")
            except Exception as e:
                print(f"    Request {i+1} failed: {e}")
        
        total_time = time.time() - start_time
        avg_response_time = total_response_time / max(successful_requests, 1)
        
        print(f" Performance Results:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Successful requests: {successful_requests}/{len(questions)}")
        print(f"   Success rate: {(successful_requests/len(questions))*100:.1f}%")
        print(f"   Average response time: {avg_response_time:.2f}s")
        print(f"   Requests per second: {successful_requests/total_time:.2f}")
        
        # Performance assertions
        assert successful_requests >= len(questions) * 0.8  # At least 80% success rate
        assert avg_response_time < 5.0  # Average response under 5 seconds
        
        print(" Performance workflow test passed!")
    
    def test_monitoring_integration_e2e(self):
        """Test monitoring system end-to-end"""
        base_url = "http://localhost:8000"
        
        print(" Testing monitoring integration...")
        
        # Generate some activity to monitor
        print("1. Generating activity for monitoring...")
        
        # Upload images
        for i in range(3):
            test_image = self.create_test_image(color=['red', 'green', 'blue'][i])
            files = {"file": (f"monitor_test_{i}.jpg", test_image, "image/jpeg")}
            response = requests.post(f"{base_url}/upload", files=files)
            assert response.status_code == 200
        
        # Generate chat activity
        for i in range(5):
            chat_data = {
                "message": f"Test monitoring message {i}",
                "conversation_id": "monitoring_test"
            }
            response = requests.post(f"{base_url}/chat", json=chat_data)
            assert response.status_code == 200
        
        # Wait a bit for metrics to be collected
        time.sleep(2)
        
        # Check metrics endpoint
        print("2. Checking metrics collection...")
        metrics_response = requests.get(f"{base_url}/metrics")
        assert metrics_response.status_code == 200
        
        metrics_text = metrics_response.text
        
        # Verify key metrics are present
        assert "http_requests_total" in metrics_text
        assert "http_request_duration_seconds" in metrics_text
        print("    Prometheus metrics are being collected")
        
        # Check monitoring dashboard
        print("3. Checking monitoring dashboard...")
        dashboard_response = requests.get(f"{base_url}/monitoring/dashboard")
        if dashboard_response.status_code == 200:
            dashboard_data = dashboard_response.json()
            
            # Verify dashboard structure
            assert "timestamp" in dashboard_data
            assert "summary" in dashboard_data
            
            if "summary" in dashboard_data:
                summary = dashboard_data["summary"]
                print(f"    Overall status: {summary.get('overall_status', 'unknown')}")
                print(f"    Total services: {summary.get('total_services', 0)}")
                print(f"    Healthy services: {summary.get('healthy_services', 0)}")
    
        # Check logs endpoint
        print("4. Checking logs analysis...")
        logs_response = requests.get(f"{base_url}/monitoring/logs")
        if logs_response.status_code == 200:
            logs_data = logs_response.json()
            print(f"    Log analysis available: {logs_data.get('total_entries', 0)} entries")
        
        print(" Monitoring integration test passed!")
