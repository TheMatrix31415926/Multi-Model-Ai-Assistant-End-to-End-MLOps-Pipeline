"""config_entity.py module"""
# multimodal_ai_assistant/entity/config_entity.py

from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class DataIngestionConfig:
    """Data Ingestion Configuration"""
    dataset_name: str = "HuggingFaceM4/VQAv2"
    train_split: str = "train[:5000]"
    validation_split: str = "validation[:1000]" 
    test_split: str = "test[:500]"
    data_ingestion_dir: str = "artifacts/data_ingestion"
    raw_data_path: str = "artifacts/data_ingestion/raw_data"
    train_data_path: str = "artifacts/data_ingestion/train"
    validation_data_path: str = "artifacts/data_ingestion/validation"
    test_data_path: str = "artifacts/data_ingestion/test"

@dataclass
class DataValidationConfig:
    """Data Validation Configuration"""
    data_ingestion_dir: str = "artifacts/data_ingestion"
    validation_report_dir: str = "artifacts/data_validation"
    validation_report_file: str = "artifacts/data_validation/validation_report.json"
    min_image_width: int = 32
    min_image_height: int = 32
    max_image_width: int = 2048
    max_image_height: int = 2048
    min_question_length: int = 3
    max_question_length: int = 200
    min_answer_length: int = 1
    max_answer_length: int = 50
    supported_image_formats: List[str] = None
    
    def __post_init__(self):
        if self.supported_image_formats is None:
            self.supported_image_formats = ['JPEG', 'PNG', 'RGB']

@dataclass
class DataTransformationConfig:
    """Data Transformation Configuration"""
    data_transformation_dir: str = "artifacts/data_transformation"
    transformed_train_path: str = "artifacts/data_transformation/train"
    transformed_validation_path: str = "artifacts/data_transformation/validation"
    transformed_test_path: str = "artifacts/data_transformation/test"
    target_image_size: Tuple[int, int] = (224, 224)
    image_normalization_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_normalization_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    max_question_length: int = 128
    max_answer_length: int = 32
    text_embedding_model: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# ---

# multimodal_ai_assistant/entity/artifact_entity.py

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class DataIngestionArtifact:
    """Data Ingestion Artifact"""
    train_file_path: str
    validation_file_path: str
    test_file_path: str
    train_data_dir: str
    validation_data_dir: str
    test_data_dir: str

@dataclass
class DataValidationArtifact:
    """Data Validation Artifact"""
    validation_status: bool
    validation_report_path: str
    validated_train_path: str
    validated_validation_path: str
    validated_test_path: str
    validation_summary: Dict[str, Any]

@dataclass
class DataTransformationArtifact:
    """Data Transformation Artifact"""
    transformation_status: bool
    transformed_train_path: str
    transformed_validation_path: str
    transformed_test_path: str
    preprocessors_path: str
    transformation_summary: Dict[str, Any]

# ---

# multimodal_ai_assistant/exception/__init__.py

import sys
from typing import Any

class MultiModalException(Exception):
    """
    Custom exception class for Multi-Modal AI Assistant
    """
    
    def __init__(self, error_message: str, error_detail: Any):
        super().__init__(error_message)
        self.error_message = error_message
        self.error_detail = self._get_error_details(error_detail)
    
    def _get_error_details(self, error_detail):
        """Extract detailed error information"""
        if hasattr(error_detail, '__traceback__'):
            _, _, exc_tb = error_detail
        else:
            exc_tb = sys.exc_info()[2]
            
        if exc_tb is not None:
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb
