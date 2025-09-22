# tests/unit/test_data_ingestion.py - Data ingestion tests
import pytest
import pandas as pd
import os
from unittest.mock import patch, MagicMock
from multimodal_ai_assistant.components.data_ingestion import DataIngestion, DataIngestionConfig

class TestDataIngestion:
    """Test data ingestion functionality"""
    
    def test_data_ingestion_config_defaults(self):
        """Test default configuration"""
        config = DataIngestionConfig()
        assert config.dataset_name == "HuggingFaceM4/VQAv2"
        assert config.train_split == "train[:5000]"
        assert "artifacts" in config.data_ingestion_dir
    
    @patch('multimodal_ai_assistant.components.data_ingestion.load_dataset')
    def test_download_dataset_success(self, mock_load_dataset, temp_dir):
        """Test successful dataset download"""
        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__ = lambda: 100
        mock_load_dataset.return_value = mock_dataset
        
        # Configure with temp directory
        config = DataIngestionConfig()
        config.data_ingestion_dir = temp_dir
        config.train_data_path = os.path.join(temp_dir, "train")
        config.validation_data_path = os.path.join(temp_dir, "validation")  
        config.test_data_path = os.path.join(temp_dir, "test")
        
        data_ingestion = DataIngestion(config)
        
        # Test download
        train_ds, val_ds, test_ds = data_ingestion.download_dataset()
        
        # Verify calls
        assert mock_load_dataset.call_count == 3
        assert len(train_ds) == 100
    
    def test_create_directories(self, temp_dir):
        """Test directory creation"""
        config = DataIngestionConfig()
        config.data_ingestion_dir = temp_dir
        config.train_data_path = os.path.join(temp_dir, "train")
        config.validation_data_path = os.path.join(temp_dir, "validation")
        config.test_data_path = os.path.join(temp_dir, "test")
        
        data_ingestion = DataIngestion(config)
        
        # Check directories were created
        assert os.path.exists(config.train_data_path)
        assert os.path.exists(config.validation_data_path)
        assert os.path.exists(config.test_data_path)
    
    @patch('multimodal_ai_assistant.components.data_ingestion.Image')
    def test_process_sample_data(self, mock_image, sample_vqa_data, temp_dir):
        """Test processing sample data"""
        # Mock image
        mock_img = MagicMock()
        mock_img.save = MagicMock()
        mock_img.width = 224
        mock_img.height = 224
        mock_image.open.return_value = mock_img
        
        # Create mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = lambda self: iter([{
            'image': mock_img,
            'question': 'What color is the car?',
            'answers': ['red', 'crimson'],
            'question_type': 'color',
            'answer_type': 'other'
        }])
        mock_dataset.__len__ = lambda self: 1
        
        config = DataIngestionConfig()
        config.data_ingestion_dir = temp_dir
        config.train_data_path = os.path.join(temp_dir, "train")
        
        data_ingestion = DataIngestion(config)
        
        # Test processing
        result_path = data_ingestion.process_and_save_dataset(
            mock_dataset, "train", config.train_data_path
        )
        
        assert result_path.endswith("train_metadata.csv")