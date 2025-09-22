import pytest
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for session"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Setup test environment variables"""
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("OPENAI_API_KEY", "test_key")
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "test_token")