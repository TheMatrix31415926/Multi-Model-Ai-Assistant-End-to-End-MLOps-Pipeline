# deployment/scripts/quick_test.py - Quick deployment test
#!/usr/bin/env python3

import requests
import sys
import time
from datetime import datetime

def test_aws_deployment():
    """Test AWS deployment with comprehensive checks"""
    
    # Get instance IP from environment or user input
    import os
    instance_ip = os.environ.get('INSTANCE_IP')
    
    if not instance_ip:
        instance_ip = input("Enter your EC2 instance IP: ").strip()
    
    if not instance_ip:
        print(" Instance IP required")
        sys.exit(1)
    
    print(" Multi-Modal AI Assistant - AWS Deployment Test")
    print("=" * 50)
    print(f"Testing deployment at: {instance_ip}")
    print(f"Test started at: {datetime.now()}")
    print()
    
    base_url = f"http://{instance_ip}"
    tests_passed = 0
    total_tests = 0
    
    # Test 1: API Health Check
    total_tests += 1
    print(" Test 1: API Health Check")
    try:
        response = requests.get(f"{base_url}:8000/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(" API is healthy")
            print(f"   Status: {health_data.get('status', 'unknown')}")
            tests_passed += 1
        else:
            print(f" API health check failed: {response.status_code}")
    except Exception as e:
        print(f" API health check error: {e}")
    
    # Test 2: API Root Endpoint
    total_tests += 1
    print("\n Test 2: API Root Endpoint")
    try:
        response = requests.get(f"{base_url}:8000/", timeout=10)
        if response.status_code == 200:
            root_data = response.json()
            print(" API root endpoint working")
            print(f"   Version: {root_data.get('version', 'unknown')}")
            tests_passed += 1
        else:
            print(f" API root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f" API root endpoint error: {e}")
    
    # Test 3: Chat Endpoint
    total_tests += 1
    print("\n Test 3: Chat Functionality")
    try:
        chat_data = {
            "message": "Hello from AWS deployment test!",
            "conversation_id": "aws_test"
        }
        response = requests.post(f"{base_url}:8000/chat", json=chat_data, timeout=15)
        if response.status_code == 200:
            chat_response = response.json()
            print(" Chat endpoint working")
            print(f"   Response: {chat_response['response'][:100]}...")
            print(f"   Confidence: {chat_response['confidence']:.0%}")
            tests_passed += 1
        else:
            print(f" Chat endpoint failed: {response.status_code}")
    except Exception as e:
        print(f" Chat endpoint error: {e}")
    
    # Test 4: Frontend Accessibility
    total_tests += 1
    print("\n Test 4: Frontend Accessibility")
    try:
        response = requests.get(f"{base_url}:8501", timeout=15)
        if response.status_code == 200:
            print(" Frontend is accessible")
            print(f"   Response size: {len(response.content)} bytes")
            tests_passed += 1
        else:
            print(f" Frontend access failed: {response.status_code}")
    except Exception as e:
        print(f" Frontend access error: {e}")
    
    # Test 5: API Documentation
    total_tests += 1
    print("\n Test 5: API Documentation")
    try:
        response = requests.get(f"{base_url}:8000/docs", timeout=10)
        if response.status_code == 200:
            print(" API documentation accessible")
            tests_passed += 1
        else:
            print(f" API docs failed: {response.status_code}")
    except Exception as e:
        print(f" API docs error: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print(" Test Summary")
    print("=" * 50)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    print(f"Success rate: {(tests_passed/total_tests)*100:.0f}%")
    
    if tests_passed == total_tests:
        print(" All tests passed! Your AWS deployment is working perfectly.")
        print("\n Access URLs:")
        print(f"   - Frontend: http://{instance_ip}:8501")
        print(f"   - API Docs: http://{instance_ip}:8000/docs")
        print(f"   - API Health: http://{instance_ip}:8000/health")
    elif tests_passed >= total_tests * 0.8:
        print(" Most tests passed. Some services may still be starting up.")
        print("Wait a few minutes and try again.")
    else:
        print(" Multiple tests failed. Check your deployment.")
        print("\nDebugging steps:")
        print(f"1. SSH to server: ssh -i multimodal-ai-key.pem ec2-user@{instance_ip}")
        print("2. Check containers: docker-compose ps")
        print("3. Check logs: docker-compose logs")
    
    print(f"\nTest completed at: {datetime.now()}")
    return tests_passed == total_tests

if __name__ == "__main__":
    success = test_aws_deployment()
    sys.exit(0 if success else 1)

