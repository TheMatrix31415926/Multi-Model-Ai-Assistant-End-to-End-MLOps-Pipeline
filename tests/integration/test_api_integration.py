# tests/integration/test_api_integration.py - API integration tests
import pytest
import asyncio
import requests
import json
import time
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import io
from PIL import Image

# Import the main app
from api.main import app

class TestAPIIntegration:
    """Test API integration scenarios"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture  
    def sample_image_file(self):
        """Create sample image file for upload"""
        img = Image.new('RGB', (300, 300), color='blue')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        return img_bytes
    
    def test_complete_image_chat_workflow(self, client, sample_image_file):
        """Test complete workflow: upload image -> chat about it"""
        
        # Step 1: Upload image
    def test_complete_image_chat_workflow(self, client, sample_image_file):
        """Test complete workflow: upload image -> chat about it"""
        
        # Step 1: Upload image
        files = {"file": ("test_image.jpg", sample_image_file, "image/jpeg")}
        upload_response = client.post("/upload", files=files)
        
        assert upload_response.status_code == 200
        upload_data = upload_response.json()
        image_id = upload_data["image_id"]
        
        # Step 2: Chat about the image
        chat_data = {
            "message": "What do you see in this image?",
            "image_id": image_id,
            "conversation_id": "test_workflow"
        }
        
        chat_response = client.post("/chat", json=chat_data)
        assert chat_response.status_code == 200
        
        chat_result = chat_response.json()
        assert chat_result["conversation_id"] == "test_workflow"
        assert "response" in chat_result
        assert chat_result["confidence"] > 0
        
        # Step 3: Continue conversation
        followup_data = {
            "message": "Tell me more about the colors",
            "image_id": image_id,
            "conversation_id": "test_workflow"
        }
        
        followup_response = client.post("/chat", json=followup_data)
        assert followup_response.status_code == 200
        
        print(" Complete image chat workflow test passed!")
    
    def test_monitoring_endpoints_integration(self, client):
        """Test monitoring endpoints work together"""
        
        # Test health endpoint
        health_response = client.get("/health")
        assert health_response.status_code == 200
        health_data = health_response.json()
        assert health_data["status"] == "healthy"
        
        # Test metrics endpoint
        metrics_response = client.get("/metrics")
        assert metrics_response.status_code == 200
        
        # Test monitoring dashboard
        with patch('api.main.dashboard_generator') as mock_dashboard:
            mock_dashboard.generate_dashboard_data.return_value = {
                "timestamp": "2024-01-01T00:00:00",
                "summary": {"overall_status": "healthy"},
                "services": {},
                "system_resources": {}
            }
            
            dashboard_response = client.get("/monitoring/dashboard")
            assert dashboard_response.status_code == 200
            dashboard_data = dashboard_response.json()
            assert dashboard_data["summary"]["overall_status"] == "healthy"
        
        print(" Monitoring endpoints integration test passed!")
    
    def test_error_handling_integration(self, client):
        """Test error handling across endpoints"""
        
        # Test chat with invalid data
        invalid_chat = client.post("/chat", json={"invalid": "data"})
        # Should handle gracefully, not crash
        assert invalid_chat.status_code in [200, 422, 400]
        
        # Test upload with invalid file
        files = {"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}
        upload_response = client.post("/upload", files=files)
        assert upload_response.status_code == 400
        
        # Test chat with non-existent image_id
        chat_data = {
            "message": "What's in this image?",
            "image_id": "nonexistent_123"
        }
        chat_response = client.post("/chat", json=chat_data)
        assert chat_response.status_code == 200  # Should handle gracefully
        
        print(" Error handling integration test passed!")
