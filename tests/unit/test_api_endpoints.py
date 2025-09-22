# tests/unit/test_api_endpoints.py - API endpoint tests  
import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json
import io
from PIL import Image

# Import the app
from api.main import app

client = TestClient(app)

class TestAPIEndpoints:
    """Test FastAPI endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "Multi-Modal AI Assistant" in data["message"]
        assert data["status"] == "active"
        assert "monitoring" in data["capabilities"]
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "services" in data
    
    def test_upload_image_success(self, sample_image):
        """Test successful image upload"""
        sample_image.seek(0)
        files = {"file": ("test.jpg", sample_image, "image/jpeg")}
        
        response = client.post("/upload", files=files)
        assert response.status_code == 200
        
        data = response.json()
        assert "image_id" in data
        assert "successfully" in data["message"]
        assert data["image_info"]["dimensions"] == [224, 224]
    
    def test_upload_invalid_file(self):
        """Test upload of non-image file"""
        files = {"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}
        
        response = client.post("/upload", files=files)
        assert response.status_code == 400
        assert "must be an image" in response.json()["detail"]
    
    @patch('api.main.generate_ai_response')
    def test_chat_endpoint_success(self, mock_generate):
        """Test successful chat request"""
        mock_generate.return_value = {
            "response": "Test response",
            "confidence": 0.8,
            "source": "test"
        }
        
        chat_data = {
            "message": "Hello, how are you?",
            "conversation_id": "test_conv"
        }
        
        response = client.post("/chat", json=chat_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["response"] == "Test response"
        assert data["confidence"] == 0.8
        assert data["conversation_id"] == "test_conv"
    
    def test_chat_endpoint_empty_message(self):
        """Test chat with empty message"""
        response = client.post("/chat", json={"message": ""})
        # Should still work but with default response
        assert response.status_code == 200
    
    def test_monitoring_dashboard_endpoint(self):
        """Test monitoring dashboard endpoint"""
        with patch('api.main.dashboard_generator') as mock_dashboard:
            mock_dashboard.generate_dashboard_data.return_value = {
                "status": "healthy",
                "metrics": {}
            }
            
            response = client.get("/monitoring/dashboard")
            assert response.status_code == 200
    
    def test_metrics_endpoint(self):
        """Test Prometheus metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"] or \
               "application/openmetrics-text" in response.headers["content-type"]

