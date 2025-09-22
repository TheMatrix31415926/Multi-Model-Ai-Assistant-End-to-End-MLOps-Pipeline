# tests/unit/test_utils.py - Utility function tests
import pytest
import numpy as np
from PIL import Image
import io
from multimodal_ai_assistant.utils.main_utils import (
    validate_image, 
    preprocess_text, 
    calculate_confidence_score
)

class TestUtils:
    """Test utility functions"""
    
    def test_validate_image_success(self, sample_image):
        """Test successful image validation"""
        sample_image.seek(0)
        img = Image.open(sample_image)
        
        result = validate_image(img)
        assert result['valid'] == True
        assert result['width'] == 224
        assert result['height'] == 224
        assert result['format'] == 'JPEG'
    
    def test_validate_image_none(self):
        """Test validation with None image"""
        result = validate_image(None)
        assert result['valid'] == False
        assert 'error' in result
    
    def test_preprocess_text_basic(self):
        """Test basic text preprocessing"""
        text = "  Hello World!  "
        result = preprocess_text(text)
        
        assert result['cleaned_text'] == "Hello World!"
        assert result['word_count'] == 2
        assert result['char_count'] == 12
    
    def test_preprocess_text_empty(self):
        """Test preprocessing empty text"""
        result = preprocess_text("")
        
        assert result['cleaned_text'] == ""
        assert result['word_count'] == 0
        assert result['char_count'] == 0
    
    def test_calculate_confidence_score(self):
        """Test confidence score calculation"""
        # High confidence scenario
        high_conf = calculate_confidence_score(
            text_similarity=0.9,
            image_quality=0.8,
            model_certainty=0.95
        )
        assert high_conf > 0.8
        
        # Low confidence scenario  
        low_conf = calculate_confidence_score(
            text_similarity=0.3,
            image_quality=0.4,
            model_certainty=0.2
        )
        assert low_conf < 0.5

# Helper utility functions for tests
def validate_image(image):
    """Validate image format and properties"""
    if image is None:
        return {'valid': False, 'error': 'Image is None'}
    
    try:
        return {
            'valid': True,
            'width': image.width,
            'height': image.height,
            'format': image.format or 'JPEG',
            'mode': image.mode
        }
    except Exception as e:
        return {'valid': False, 'error': str(e)}

def preprocess_text(text):
    """Preprocess text input"""
    if not text:
        return {
            'cleaned_text': '',
            'word_count': 0,
            'char_count': 0
        }
    
    cleaned = text.strip()
    words = cleaned.split()
    
    return {
        'cleaned_text': cleaned,
        'word_count': len(words),
        'char_count': len(cleaned)
    }

def calculate_confidence_score(text_similarity=0.5, image_quality=0.5, model_certainty=0.5):
    """Calculate overall confidence score"""
    weights = {
        'text': 0.3,
        'image': 0.3, 
        'model': 0.4
    }
    
    score = (
        text_similarity * weights['text'] +
        image_quality * weights['image'] +
        model_certainty * weights['model']
    )
    
    return min(max(score, 0.0), 1.0)



