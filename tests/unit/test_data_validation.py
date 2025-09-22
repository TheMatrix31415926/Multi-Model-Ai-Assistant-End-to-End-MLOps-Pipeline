# tests/unit/test_data_validation.py - Data validation tests
import pytest
import pandas as pd
from PIL import Image
import tempfile
import os
from multimodal_ai_assistant.components.data_validation import DataValidation, DataValidationConfig

class TestDataValidation:
    """Test data validation functionality"""
    
    def test_validation_config_defaults(self):
        """Test default validation configuration"""
        config = DataValidationConfig()
        assert config.min_image_width == 32
        assert config.min_image_height == 32
        assert 'JPEG' in config.supported_image_formats
    
    def test_validate_image_file_success(self, sample_image, temp_dir):
        """Test successful image validation"""
        # Save sample image to temp file
        image_path = os.path.join(temp_dir, "test_image.jpg")
        sample_image.seek(0)
        with open(image_path, 'wb') as f:
            f.write(sample_image.read())
        
        validator = DataValidation()
        result = validator.validate_image_file(image_path)
        
        assert result['is_valid'] == True
        assert result['file_path'] == image_path
        assert 'properties' in result
        assert result['properties']['width'] == 224
        assert result['properties']['height'] == 224
    
    def test_validate_image_file_not_found(self):
        """Test validation of non-existent image"""
        validator = DataValidation()
        result = validator.validate_image_file("nonexistent.jpg")
        
        assert result['is_valid'] == False
        assert 'does not exist' in result['errors'][0]
    
    def test_validate_text_data_success(self):
        """Test successful text validation"""
        validator = DataValidation()
        
        # Valid question
        result = validator.validate_text_data("What color is the car?", "question")
        assert result['is_valid'] == True
        assert result['properties']['length'] > 0
        assert result['properties']['word_count'] > 0
        
        # Valid answer
        result = validator.validate_text_data("red", "answer")
        assert result['is_valid'] == True
    
    def test_validate_text_data_too_short(self):
        """Test validation of too short text"""
        validator = DataValidation()
        
        # Too short question
        result = validator.validate_text_data("Hi", "question")
        assert result['is_valid'] == False
        assert 'below minimum' in result['errors'][0]
    
    def test_validate_text_data_empty(self):
        """Test validation of empty text"""
        validator = DataValidation()
        
        result = validator.validate_text_data("", "question")
        assert result['is_valid'] == False
        assert 'empty' in result['errors'][0].lower()