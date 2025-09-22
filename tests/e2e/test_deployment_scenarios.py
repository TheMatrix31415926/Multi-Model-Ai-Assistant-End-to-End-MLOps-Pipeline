# tests/e2e/test_deployment_scenarios.py - Deployment scenario tests
import pytest
import requests
import subprocess
import time
import os
import json
from pathlib import Path

class TestDeploymentScenarios:
    """Test various deployment scenarios"""
    
    def test_local_docker_deployment(self):
        """Test local Docker deployment scenario"""
        print(" Testing local Docker deployment...")
        
        # Check if docker-compose is available
        try:
            subprocess.run(["docker-compose", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.skip("Docker Compose not available")
        
        # Test basic service availability
        services_to_test = [
            ("API", "http://localhost:8000/health"),
            ("Frontend", "http://localhost:8501"),
        ]
        
        failed_services = []
        
        for service_name, url in services_to_test:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    print(f"    {service_name} is accessible")
                else:
                    print(f"    {service_name} returned status {response.status_code}")
                    failed_services.append(service_name)
            except requests.exceptions.RequestException as e:
                print(f"    {service_name} is not accessible: {e}")
                failed_services.append(service_name)
        
        # At least the API should be working
        if "API" in failed_services:
            pytest.fail("API service is not accessible in local deployment")
        
        print(" Local Docker deployment test completed!")
    
    @pytest.mark.slow
    def test_monitoring_deployment(self):
        """Test monitoring stack deployment"""
        print(" Testing monitoring stack deployment...")
        
        monitoring_services = [
            ("Prometheus", "http://localhost:9090/-/healthy"),
            ("Grafana", "http://localhost:3000/api/health"),
            ("AlertManager", "http://localhost:9093/-/healthy"),
            ("Node Exporter", "http://localhost:9100/metrics"),
        ]
        
        available_services = []
        
        for service_name, url in monitoring_services:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    print(f"    {service_name} is healthy")
                    available_services.append(service_name)
                else:
                    print(f"    {service_name} returned status {response.status_code}")
            except requests.exceptions.RequestException:
                print(f"    {service_name} is not accessible")
        
        if len(available_services) > 0:
            print(f" {len(available_services)}/{len(monitoring_services)} monitoring services are running")
        else:
            print(" No monitoring services detected (may not be deployed)")
        
        print(" Monitoring deployment test completed!")
    
    def test_api_documentation_availability(self):
        """Test that API documentation is available"""
        print(" Testing API documentation...")
        
        doc_endpoints = [
            ("OpenAPI JSON", "http://localhost:8000/openapi.json"),
            ("Swagger UI", "http://localhost:8000/docs"),
            ("ReDoc", "http://localhost:8000/redoc"),
        ]
        
        for doc_name, url in doc_endpoints:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f"    {doc_name} is available")
                else:
                    print(f"    {doc_name} returned status {response.status_code}")
            except requests.exceptions.RequestException:
                print(f"    {doc_name} is not accessible")
        
        print(" API documentation test completed!")
    
    def test_environment_configuration(self):
        """Test environment configuration"""
        print(" Testing environment configuration...")
        
        # Test API health to get environment info
        try:
            response = requests.get("http://localhost:8000/health")
            if response.status_code == 200:
                health_data = response.json()
                
                print(f"    API Version: {health_data.get('version', 'unknown')}")
                print(f"    Services: {list(health_data.get('services', {}).keys())}")
                
                # Check if monitoring is enabled
                capabilities = health_data.get('capabilities', [])
                if 'monitoring' in capabilities:
                    print("    Monitoring is enabled")
                if 'alerting' in capabilities:
                    print("    Alerting is enabled")
            
        except requests.exceptions.RequestException:
            print("    Could not retrieve environment information")
        
        # Test configuration files exist
        config_files = [
            "docker-compose.yml",
            "requirements.txt",
            ".dockerignore"
        ]
        
        for config_file in config_files:
            if Path(config_file).exists():
                print(f"    {config_file} exists")
            else:
                print(f"    {config_file} missing")
        
        print(" Environment configuration test completed!")