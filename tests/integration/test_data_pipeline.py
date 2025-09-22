# tests/integration/test_data_pipeline.py - Data pipeline integration tests
import pytest
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
from PIL import Image
import io

from multimodal_ai_assistant.components.data_ingestion import DataIngestion, DataIngestionConfig
from multimodal_ai_assistant.components.data_validation import DataValidation, DataValidationConfig  
from multimodal_ai_assistant.components.data_transformation import DataTransformation, DataTransformationConfig

class TestDataPipelineIntegration:
    """Test complete data pipeline integration"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        yield
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_mock_dataset(self, num_samples=5):
        """Create mock dataset for testing"""
        samples = []
        for i in range(num_samples):
            # Create test image
            img = Image.new('RGB', (224, 224), color=(i*50, 100, 150))
            
            sample = {
                'image': img,
                'question': f'What color is object {i}?',
                'answers': [f'color_{i}', f'hue_{i}'],
                'question_type': 'color',
                'answer_type': 'other'
            }
            samples.append(sample)
        
        # Create mock dataset object
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = lambda self: iter(samples)
        mock_dataset.__len__ = lambda self: len(samples)
        return mock_dataset
    
    @patch('multimodal_ai_assistant.components.data_ingestion.load_dataset')
    def test_complete_data_pipeline(self, mock_load_dataset):
        """Test complete data pipeline from ingestion to transformation"""
        
        # Setup mock datasets
        train_dataset = self.create_mock_dataset(10)
        val_dataset = self.create_mock_dataset(5)
        test_dataset = self.create_mock_dataset(3)
        
        mock_load_dataset.side_effect = [train_dataset, val_dataset, test_dataset]
        
        # Configure components with temp directory
        ingestion_config = DataIngestionConfig()
        ingestion_config.data_ingestion_dir = os.path.join(self.temp_dir, "ingestion")
        ingestion_config.train_data_path = os.path.join(self.temp_dir, "ingestion/train")
        ingestion_config.validation_data_path = os.path.join(self.temp_dir, "ingestion/validation")
        ingestion_config.test_data_path = os.path.join(self.temp_dir, "ingestion/test")
        
        validation_config = DataValidationConfig()
        validation_config.validation_report_dir = os.path.join(self.temp_dir, "validation")
        validation_config.validation_report_file = os.path.join(self.temp_dir, "validation/report.json")
        
        transformation_config = DataTransformationConfig()
        transformation_config.data_transformation_dir = os.path.join(self.temp_dir, "transformation")
        transformation_config.transformed_train_path = os.path.join(self.temp_dir, "transformation/train")
        transformation_config.transformed_validation_path = os.path.join(self.temp_dir, "transformation/validation")
        transformation_config.transformed_test_path = os.path.join(self.temp_dir, "transformation/test")
        
        # Step 1: Data Ingestion
        data_ingestion = DataIngestion(ingestion_config)
        ingestion_artifact = data_ingestion.initiate_data_ingestion()
        
        # Verify ingestion results
        assert os.path.exists(ingestion_artifact.train_file_path)
        assert os.path.exists(ingestion_artifact.validation_file_path)
        assert os.path.exists(ingestion_artifact.test_file_path)
        
        # Check CSV files have data
        train_df = pd.read_csv(ingestion_artifact.train_file_path)
        assert len(train_df) == 10
        assert 'question' in train_df.columns
        assert 'primary_answer' in train_df.columns
        
        # Step 2: Data Validation
        data_validation = DataValidation(validation_config)
        validation_artifact = data_validation.initiate_data_validation(ingestion_artifact)
        
        # Verify validation results
        assert validation_artifact.validation_status == True
        assert os.path.exists(validation_artifact.validation_report_path)
        
        # Step 3: Data Transformation (mock heavy dependencies)
        with patch('multimodal_ai_assistant.components.data_transformation.SentenceTransformer') as mock_st:
            mock_encoder = MagicMock()
            mock_encoder.encode.return_value = [[0.1, 0.2, 0.3]] * 10  # Mock embeddings
            mock_st.return_value = mock_encoder
            
            data_transformation = DataTransformation(transformation_config)
            transformation_artifact = data_transformation.initiate_data_transformation(validation_artifact)
            
            # Verify transformation results
            assert transformation_artifact.transformation_status == True
            assert os.path.exists(transformation_artifact.transformed_train_path)
            assert os.path.exists(transformation_artifact.preprocessors_path)
        
        print("✅ Complete data pipeline test passed!")