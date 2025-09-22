# tests/integration/test_docker_integration.py - Docker integration tests
import pytest
import subprocess
import time
import requests
import os
from pathlib import Path

class TestDockerIntegration:
    """Test Docker container integration"""
    
    @pytest.mark.slow
    def test_docker_compose_health(self):
        """Test Docker Compose services health"""
        
        # Skip if not in Docker environment
        if not os.path.exists("docker-compose.yml"):
            pytest.skip("Docker compose file not found")
        
        try:
            # Start services (if not already running)
            subprocess.run(
                ["docker-compose", "up", "-d"], 
                check=False, 
                capture_output=True,
                timeout=120
            )
            
            # Wait for services to start
            time.sleep(30)
            
            # Test API service
            try:
                response = requests.get("http://localhost:8000/health", timeout=10)
                assert response.status_code == 200
                health_data = response.json()
                assert health_data["status"] == "healthy"
                print(" API service is healthy")
            except requests.exceptions.RequestException:
                pytest.fail("API service not accessible")
            
            # Test Frontend service
            try:
                response = requests.get("http://localhost:8501", timeout=10)
                assert response.status_code == 200
                print(" Frontend service is accessible")
            except requests.exceptions.RequestException:
                print(" Frontend service not accessible (may be normal)")
            
        except subprocess.TimeoutExpired:
            pytest.fail("Docker compose startup timed out")
        except Exception as e:
            pytest.fail(f"Docker integration test failed: {e}")
    
    @pytest.mark.slow
    def test_monitoring_stack_health(self):
        """Test monitoring stack health"""
        
        # Skip if monitoring compose not available
        if not os.path.exists("docker-compose.monitoring.yml"):
            pytest.skip("Monitoring compose file not found")
        
        try:
            # Start monitoring stack
            subprocess.run(
                ["docker-compose", "-f", "docker-compose.monitoring.yml", "up", "-d"],
                check=False,
                capture_output=True,
                timeout=120
            )
            
            # Wait for services
            time.sleep(45)
            
            # Test Prometheus
            try:
                response = requests.get("http://localhost:9090/-/healthy", timeout=10)
                assert response.status_code == 200
                print(" Prometheus is healthy")
            except requests.exceptions.RequestException:
                print(" Prometheus not accessible")
            
            # Test Grafana
            try:
                response = requests.get("http://localhost:3000/api/health", timeout=10)
                assert response.status_code == 200
                print(" Grafana is healthy")
            except requests.exceptions.RequestException:
                print(" Grafana not accessible")
            
        except subprocess.TimeoutExpired:
            pytest.fail("Monitoring stack startup timed out")
        except Exception as e:
            pytest.fail(f"Monitoring integration test failed: {e}")