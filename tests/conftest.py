"""conftest.py module"""

# TODO: Implement functionality
# tests/conftest.py - Test configuration and fixtures
import pytest
import os
import sys
import asyncio
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch
import pandas as pd
from PIL import Image
import io

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)

@pytest.fixture
def sample_image():
    """Create sample test image"""
    img = Image.new('RGB', (224, 224), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes

@pytest.fixture
def sample_vqa_data():
    """Sample VQA data for testing"""
    return [
        {
            'sample_id': 'test_001',
            'question': 'What color is the car?',
            'primary_answer': 'red',
            'answers': 'red|crimson|red',
            'question_type': 'color',
            'answer_type': 'other'
        },
        {
            'sample_id': 'test_002', 
            'question': 'How many people are there?',
            'primary_answer': 'two',
            'answers': 'two|2|two people',
            'question_type': 'count',
            'answer_type': 'number'
        }
    ]

@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response"""
    return {
        "choices": [{
            "message": {
                "content": "This is a test AI response."
            }
        }]
    }