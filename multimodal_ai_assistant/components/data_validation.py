"""data_validation.py module"""

# multimodal_ai_assistant/components/data_validation.py

import os
import sys
import pandas as pd
import json
from PIL import Image
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import logging
from pathlib import Path
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class DataValidationConfig:
    """Data Validation Configuration"""
    data_ingestion_dir: str = "artifacts/data_ingestion"
    validation_report_dir: str = "artifacts/data_validation"
    validation_report_file: str = "artifacts/data_validation/validation_report.json"
    
    # Validation thresholds
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
class DataValidationArtifact:
    """Data Validation Artifact"""
    validation_status: bool
    validation_report_path: str
    validated_train_path: str
    validated_validation_path: str
    validated_test_path: str
    validation_summary: Dict[str, Any]

class MultiModalException(Exception):
    """Custom exception for the project"""
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message = error_message
        self.error_detail = str(error_detail)

class DataValidation:
    """
    Data Validation class for Multi-Modal AI Assistant
    Validates data quality, format, and integrity
    """
    
    def __init__(self, data_validation_config: DataValidationConfig = None):
        try:
            self.data_validation_config = data_validation_config or DataValidationConfig()
            self._create_directories()
            self.validation_errors = []
            self.validation_warnings = []
            
        except Exception as e:
            raise MultiModalException(f"Error in DataValidation initialization: {str(e)}", sys.exc_info())
    
    def _create_directories(self):
        """Create necessary directories for data validation"""
        try:
            os.makedirs(self.data_validation_config.validation_report_dir, exist_ok=True)
            logging.info("Created directories for data validation")
            
        except Exception as e:
            raise MultiModalException(f"Error creating directories: {str(e)}", sys.exc_info())
    
    def validate_image_file(self, image_path: str) -> Dict[str, Any]:
        """
        Validate individual image file
        Returns: validation results for the image
        """
        validation_result = {
            'file_path': image_path,
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'properties': {}
        }
        
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                validation_result['is_valid'] = False
                validation_result['errors'].append('Image file does not exist')
                return validation_result
            
            # Check file size
            file_size = os.path.getsize(image_path)
            if file_size == 0:
                validation_result['is_valid'] = False
                validation_result['errors'].append('Image file is empty')
                return validation_result
            
            # Try to open and validate image
            try:
                with Image.open(image_path) as img:
                    # Store image properties
                    validation_result['properties'] = {
                        'width': img.width,
                        'height': img.height,
                        'mode': img.mode,
                        'format': img.format,
                        'file_size_bytes': file_size
                    }
                    
                    # Validate image dimensions
                    if img.width < self.data_validation_config.min_image_width:
                        validation_result['errors'].append(f'Image width {img.width} below minimum {self.data_validation_config.min_image_width}')
                        validation_result['is_valid'] = False
                    
                    if img.height < self.data_validation_config.min_image_height:
                        validation_result['errors'].append(f'Image height {img.height} below minimum {self.data_validation_config.min_image_height}')
                        validation_result['is_valid'] = False
                    
                    if img.width > self.data_validation_config.max_image_width:
                        validation_result['warnings'].append(f'Image width {img.width} above recommended maximum {self.data_validation_config.max_image_width}')
                    
                    if img.height > self.data_validation_config.max_image_height:
                        validation_result['warnings'].append(f'Image height {img.height} above recommended maximum {self.data_validation_config.max_image_height}')
                    
                    # Validate image format
                    if img.format not in self.data_validation_config.supported_image_formats:
                        validation_result['warnings'].append(f'Image format {img.format} not in preferred formats {self.data_validation_config.supported_image_formats}')
                    
                    # Check if image can be converted to RGB
                    try:
                        img.convert('RGB')
                    except Exception:
                        validation_result['errors'].append('Image cannot be converted to RGB format')
                        validation_result['is_valid'] = False
                        
            except Exception as img_error:
                validation_result['is_valid'] = False
                validation_result['errors'].append(f'Cannot open image file: {str(img_error)}')
                
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f'Error validating image: {str(e)}')
        
        return validation_result
    
    def validate_text_data(self, text: str, field_name: str) -> Dict[str, Any]:
        """
        Validate text data (questions and answers)
        Returns: validation results for the text
        """
        validation_result = {
            'field_name': field_name,
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'properties': {}
        }
        
        try:
            if text is None or pd.isna(text):
                validation_result['is_valid'] = False
                validation_result['errors'].append(f'{field_name} is None or NaN')
                return validation_result
            
            text_str = str(text).strip()
            text_length = len(text_str)
            
            # Store text properties
            validation_result['properties'] = {
                'length': text_length,
                'word_count': len(text_str.split()),
                'has_special_chars': bool(re.search(r'[^\w\s]', text_str)),
                'has_numbers': bool(re.search(r'\d', text_str))
            }
            
            # Validate based on field type
            if field_name == 'question':
                min_length = self.data_validation_config.min_question_length
                max_length = self.data_validation_config.max_question_length
            else:  # answer
                min_length = self.data_validation_config.min_answer_length
                max_length = self.data_validation_config.max_answer_length
            
            # Length validation
            if text_length < min_length:
                validation_result['is_valid'] = False
                validation_result['errors'].append(f'{field_name} length {text_length} below minimum {min_length}')
            
            if text_length > max_length:
                validation_result['warnings'].append(f'{field_name} length {text_length} above recommended maximum {max_length}')
            
            # Check for empty or whitespace-only text
            if not text_str:
                validation_result['is_valid'] = False
                validation_result['errors'].append(f'{field_name} is empty or whitespace only')
            
            # Check for question marks in questions
            if field_name == 'question' and not text_str.endswith('?'):
                validation_result['warnings'].append('Question does not end with question mark')
                
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f'Error validating {field_name}: {str(e)}')
        
        return validation_result
    
    def validate_dataset_split(self, split_path: str, split_name: str) -> Dict[str, Any]:
        """
        Validate entire dataset split
        Returns: validation results for the split
        """
        split_validation = {
            'split_name': split_name,
            'split_path': split_path,
            'is_valid': True,
            'total_samples': 0,
            'valid_samples': 0,
            'invalid_samples': 0,
            'errors': [],
            'warnings': [],
            'sample_validations': [],
            'summary': {}
        }
        
        try:
            # Check if metadata file exists
            metadata_path = os.path.join(split_path, f"{split_name}_metadata.csv")
            if not os.path.exists(metadata_path):
                split_validation['is_valid'] = False
                split_validation['errors'].append(f'Metadata file not found: {metadata_path}')
                return split_validation
            
            # Load metadata
            df = pd.read_csv(metadata_path)
            split_validation['total_samples'] = len(df)
            
            logging.info(f"Validating {split_name} split with {len(df)} samples...")
            
            valid_samples = 0
            image_validations = []
            text_validations = []
            
            for idx, row in df.iterrows():
                sample_validation = {
                    'sample_id': row['sample_id'],
                    'is_valid': True,
                    'errors': [],
                    'warnings': []
                }
                
                # Validate image
                image_path = row['image_path']
                img_validation = self.validate_image_file(image_path)
                image_validations.append(img_validation)
                
                if not img_validation['is_valid']:
                    sample_validation['is_valid'] = False
                    sample_validation['errors'].extend(img_validation['errors'])
                
                sample_validation['warnings'].extend(img_validation['warnings'])
                
                # Validate question
                question_validation = self.validate_text_data(row['question'], 'question')
                text_validations.append(question_validation)
                
                if not question_validation['is_valid']:
                    sample_validation['is_valid'] = False
                    sample_validation['errors'].extend(question_validation['errors'])
                
                sample_validation['warnings'].extend(question_validation['warnings'])
                
                # Validate primary answer
                answer_validation = self.validate_text_data(row['primary_answer'], 'answer')
                text_validations.append(answer_validation)
                
                if not answer_validation['is_valid']:
                    sample_validation['is_valid'] = False
                    sample_validation['errors'].extend(answer_validation['errors'])
                
                sample_validation['warnings'].extend(answer_validation['warnings'])
                
                # Count valid samples
                if sample_validation['is_valid']:
                    valid_samples += 1
                
                split_validation['sample_validations'].append(sample_validation)
                
                # Log progress
                if (idx + 1) % 100 == 0:
                    logging.info(f"Validated {idx + 1}/{len(df)} samples in {split_name}")
            
            split_validation['valid_samples'] = valid_samples
            split_validation['invalid_samples'] = split_validation['total_samples'] - valid_samples
            
            # Calculate summary statistics
            split_validation['summary'] = {
                'validation_rate': valid_samples / split_validation['total_samples'] if split_validation['total_samples'] > 0 else 0,
                'total_errors': sum(len(sample['errors']) for sample in split_validation['sample_validations']),
                'total_warnings': sum(len(sample['warnings']) for sample in split_validation['sample_validations']),
                'avg_image_width': sum(img['properties'].get('width', 0) for img in image_validations if img['is_valid']) / max(len([img for img in image_validations if img['is_valid']]), 1),
                'avg_image_height': sum(img['properties'].get('height', 0) for img in image_validations if img['is_valid']) / max(len([img for img in image_validations if img['is_valid']]), 1),
                'avg_question_length': sum(txt['properties'].get('length', 0) for txt in text_validations if txt['field_name'] == 'question' and txt['is_valid']) / max(len([txt for txt in text_validations if txt['field_name'] == 'question' and txt['is_valid']]), 1)
            }
            
            # Check if split is valid overall
            if split_validation['summary']['validation_rate'] < 0.95:  # 95% threshold
                split_validation['is_valid'] = False
                split_validation['errors'].append(f"Validation rate {split_validation['summary']['validation_rate']:.2%} below acceptable threshold of 95%")
            
            logging.info(f"Completed validation for {split_name}: {valid_samples}/{split_validation['total_samples']} valid samples")
            
        except Exception as e:
            split_validation['is_valid'] = False
            split_validation['errors'].append(f'Error validating {split_name} split: {str(e)}')
        
        return split_validation
    
    def initiate_data_validation(self, data_ingestion_artifact) -> DataValidationArtifact:
        """
        Main method to initiate data validation process
        Returns: DataValidationArtifact with validation results
        """
        try:
            logging.info("Starting data validation process...")
            
            validation_report = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'overall_status': True,
                'splits': {},
                'summary': {}
            }
            
            # Validate each split
            splits_to_validate = [
                ('train', data_ingestion_artifact.train_data_dir),
                ('validation', data_ingestion_artifact.validation_data_dir),
                ('test', data_ingestion_artifact.test_data_dir)
            ]
            
            total_samples = 0
            total_valid_samples = 0
            
            for split_name, split_path in splits_to_validate:
                split_validation = self.validate_dataset_split(split_path, split_name)
                validation_report['splits'][split_name] = split_validation
                
                total_samples += split_validation['total_samples']
                total_valid_samples += split_validation['valid_samples']
                
                if not split_validation['is_valid']:
                    validation_report['overall_status'] = False
            
            # Create overall summary
            validation_report['summary'] = {
                'total_samples': total_samples,
                'total_valid_samples': total_valid_samples,
                'overall_validation_rate': total_valid_samples / max(total_samples, 1),
                'splits_validated': len(splits_to_validate),
                'valid_splits': sum(1 for split in validation_report['splits'].values() if split['is_valid'])
            }
            
            # Save validation report
            with open(self.data_validation_config.validation_report_file, 'w') as f:
                json.dump(validation_report, f, indent=2)
            
            # Create validation artifact
            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_report['overall_status'],
                validation_report_path=self.data_validation_config.validation_report_file,
                validated_train_path=data_ingestion_artifact.train_file_path,
                validated_validation_path=data_ingestion_artifact.validation_file_path,
                validated_test_path=data_ingestion_artifact.test_file_path,
                validation_summary=validation_report['summary']
            )
            
            logging.info("Data validation completed successfully")
            logging.info(f"Overall validation status: {validation_report['overall_status']}")
            logging.info(f"Overall validation rate: {validation_report['summary']['overall_validation_rate']:.2%}")
            
            return data_validation_artifact
            
        except Exception as e:
            raise MultiModalException(f"Error in data validation process: {str(e)}", sys.exc_info())

# Test the data validation
if __name__ == "__main__":
    try:
        # You would typically get this from data ingestion
        from data_ingestion import DataIngestion, DataIngestionArtifact
        
        # Run data ingestion first
        data_ingestion = DataIngestion()
        ingestion_artifact = data_ingestion.initiate_data_ingestion()
        
        # Run data validation
        data_validation = DataValidation()
        validation_artifact = data_validation.initiate_data_validation(ingestion_artifact)
        
        print(" Data Validation completed successfully!")
        print(f" Validation Status: {validation_artifact.validation_status}")
        print(f" Validation Report: {validation_artifact.validation_report_path}")
        print(f" Overall Validation Rate: {validation_artifact.validation_summary['overall_validation_rate']:.2%}")
        
    except Exception as e:
        print(f" Error in data validation: {str(e)}")
