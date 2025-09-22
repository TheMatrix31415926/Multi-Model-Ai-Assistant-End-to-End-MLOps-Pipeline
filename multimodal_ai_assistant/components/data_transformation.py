"""data_transformation.py module"""

# multimodal_ai_assistant/components/data_transformation.py

import os
import sys
import pandas as pd
import numpy as np
import json
import pickle
from PIL import Image
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path
import torch
from torchvision import transforms
from sentence_transformers import SentenceTransformer
import cv2
from sklearn.preprocessing import LabelEncoder
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class DataTransformationConfig:
    """Data Transformation Configuration"""
    data_transformation_dir: str = "artifacts/data_transformation"
    transformed_train_path: str = "artifacts/data_transformation/train"
    transformed_validation_path: str = "artifacts/data_transformation/validation"
    transformed_test_path: str = "artifacts/data_transformation/test"
    
    # Image transformation parameters
    target_image_size: Tuple[int, int] = (224, 224)
    image_normalization_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_normalization_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # Text transformation parameters
    max_question_length: int = 128
    max_answer_length: int = 32
    text_embedding_model: str = "all-MiniLM-L6-v2"
    
    # Processing parameters
    batch_size: int = 32
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class DataTransformationArtifact:
    """Data Transformation Artifact"""
    transformation_status: bool
    transformed_train_path: str
    transformed_validation_path: str
    transformed_test_path: str
    preprocessors_path: str
    transformation_summary: Dict[str, Any]

class MultiModalException(Exception):
    """Custom exception for the project"""
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message = error_message
        self.error_detail = str(error_detail)

class DataTransformation:
    """
    Data Transformation class for Multi-Modal AI Assistant
    Transforms raw data into model-ready format with embeddings and preprocessed features
    """
    
    def __init__(self, data_transformation_config: DataTransformationConfig = None):
        try:
            self.data_transformation_config = data_transformation_config or DataTransformationConfig()
            self._create_directories()
            self._initialize_models()
            self._setup_transforms()
            
        except Exception as e:
            raise MultiModalException(f"Error in DataTransformation initialization: {str(e)}", sys.exc_info())
    
    def _create_directories(self):
        """Create necessary directories for data transformation"""
        try:
            directories = [
                self.data_transformation_config.data_transformation_dir,
                self.data_transformation_config.transformed_train_path,
                self.data_transformation_config.transformed_validation_path,
                self.data_transformation_config.transformed_test_path
            ]
            
            for directory in directories:
                os.makedirs(directory, exist_ok=True)
            
            logging.info("Created directories for data transformation")
            
        except Exception as e:
            raise MultiModalException(f"Error creating directories: {str(e)}", sys.exc_info())
    
    def _initialize_models(self):
        """Initialize text embedding model and other processors"""
        try:
            logging.info(f"Loading text embedding model: {self.data_transformation_config.text_embedding_model}")
            self.text_encoder = SentenceTransformer(self.data_transformation_config.text_embedding_model)
            self.text_encoder.to(self.data_transformation_config.device)
            
            # Initialize label encoders
            self.question_type_encoder = LabelEncoder()
            self.answer_type_encoder = LabelEncoder()
            
            logging.info("Models initialized successfully")
            
        except Exception as e:
            raise MultiModalException(f"Error initializing models: {str(e)}", sys.exc_info())
    
    def _setup_transforms(self):
        """Setup image transformation pipelines"""
        try:
            # Image transforms for training
            self.train_image_transform = transforms.Compose([
                transforms.Resize(self.data_transformation_config.target_image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.data_transformation_config.image_normalization_mean,
                    std=self.data_transformation_config.image_normalization_std
                )
            ])
            
            # Image transforms for validation/test
            self.eval_image_transform = transforms.Compose([
                transforms.Resize(self.data_transformation_config.target_image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.data_transformation_config.image_normalization_mean,
                    std=self.data_transformation_config.image_normalization_std
                )
            ])
            
            logging.info("Image transforms setup successfully")
            
        except Exception as e:
            raise MultiModalException(f"Error setting up transforms: {str(e)}", sys.exc_info())
    
    def preprocess_text(self, text: str, max_length: int = None) -> Dict[str, Any]:
        """
        Preprocess and clean text data
        """
        try:
            if pd.isna(text) or text is None:
                return {
                    'original_text': '',
                    'cleaned_text': '',
                    'text_length': 0,
                    'word_count': 0
                }
            
            # Convert to string and clean
            text_str = str(text).strip()
            
            # Basic text cleaning
            cleaned_text = re.sub(r'\s+', ' ', text_str)  # Multiple spaces to single
            cleaned_text = re.sub(r'[^\w\s\?\.\!,]', '', cleaned_text)  # Remove special chars except basic punctuation
            cleaned_text = cleaned_text.strip()
            
            # Apply length limit if specified
            if max_length and len(cleaned_text) > max_length:
                cleaned_text = cleaned_text[:max_length].strip()
            
            return {
                'original_text': text_str,
                'cleaned_text': cleaned_text,
                'text_length': len(cleaned_text),
                'word_count': len(cleaned_text.split())
            }
            
        except Exception as e:
            raise MultiModalException(f"Error preprocessing text: {str(e)}", sys.exc_info())
    
    def generate_text_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for text data using SentenceTransformer
        """
        try:
            if not texts:
                return np.array([])
            
            # Filter out empty texts
            valid_texts = [text if text and str(text).strip() else "empty" for text in texts]
            
            # Generate embeddings
            embeddings = self.text_encoder.encode(
                valid_texts,
                batch_size=self.data_transformation_config.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            return embeddings
            
        except Exception as e:
            raise MultiModalException(f"Error generating text embeddings: {str(e)}", sys.exc_info())
    
    def process_image(self, image_path: str, transform: transforms.Compose = None) -> Dict[str, Any]:
        """
        Process and transform image
        """
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Load image
            image = Image.open(image_path).convert('RGB')
            original_size = image.size
            
            # Apply transforms if provided
            if transform:
                transformed_image = transform(image)
                image_tensor = transformed_image
            else:
                # Default resize and convert to tensor
                image = image.resize(self.data_transformation_config.target_image_size)
                image_tensor = transforms.ToTensor()(image)
            
            # Extract basic image features
            image_features = {
                'original_size': original_size,
                'processed_size': self.data_transformation_config.target_image_size,
                'channels': image_tensor.shape[0] if len(image_tensor.shape) >= 3 else 1,
                'aspect_ratio': original_size[0] / original_size[1] if original_size[1] > 0 else 1.0
            }
            
            return {
                'image_tensor': image_tensor,
                'image_features': image_features,
                'is_valid': True
            }
            
        except Exception as e:
            logging.warning(f"Error processing image {image_path}: {str(e)}")
            # Return dummy tensor for failed images
            return {
                'image_tensor': torch.zeros((3, *self.data_transformation_config.target_image_size)),
                'image_features': {
                    'original_size': (0, 0),
                    'processed_size': self.data_transformation_config.target_image_size,
                    'channels': 3,
                    'aspect_ratio': 1.0
                },
                'is_valid': False
            }
    
    def extract_image_features_cv2(self, image_path: str) -> Dict[str, Any]:
        """
        Extract additional image features using OpenCV
        """
        try:
            if not os.path.exists(image_path):
                return {'features_extracted': False}
            
            # Read image with OpenCV
            image = cv2.imread(image_path)
            if image is None:
                return {'features_extracted': False}
            
            # Convert to different color spaces
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Extract features
            features = {
                'brightness_mean': float(np.mean(gray)),
                'brightness_std': float(np.std(gray)),
                'contrast': float(np.std(gray)),
                'color_variance': float(np.var(hsv)),
                'edges_count': int(np.sum(cv2.Canny(gray, 50, 150) > 0)),
                'features_extracted': True
            }
            
            return features
            
        except Exception as e:
            logging.warning(f"Error extracting CV2 features from {image_path}: {str(e)}")
            return {'features_extracted': False}
    
    def fit_encoders(self, train_df: pd.DataFrame):
        """
        Fit label encoders on training data
        """
        try:
            logging.info("Fitting label encoders on training data...")
            
            # Fit question type encoder
            question_types = train_df['question_type'].fillna('unknown').astype(str)
            self.question_type_encoder.fit(question_types)
            
            # Fit answer type encoder
            answer_types = train_df['answer_type'].fillna('unknown').astype(str)
            self.answer_type_encoder.fit(answer_types)
            
            logging.info(f"Question types: {len(self.question_type_encoder.classes_)} classes")
            logging.info(f"Answer types: {len(self.answer_type_encoder.classes_)} classes")
            
        except Exception as e:
            raise MultiModalException(f"Error fitting encoders: {str(e)}", sys.exc_info())
    
    def transform_dataset_split(self, split_path: str, split_name: str, metadata_file: str) -> str:
        """
        Transform a complete dataset split
        """
        try:
            logging.info(f"Transforming {split_name} dataset...")
            
            # Load metadata
            metadata_path = os.path.join(split_path, metadata_file)
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
            df = pd.read_csv(metadata_path)
            logging.info(f"Loaded {len(df)} samples for {split_name}")
            
            # Initialize lists for transformed data
            transformed_data = []
            
            # Choose appropriate image transform
            if split_name == 'train':
                image_transform = self.train_image_transform
            else:
                image_transform = self.eval_image_transform
            
            # Process each sample
            for idx, row in df.iterrows():
                try:
                    # Process text data
                    question_data = self.preprocess_text(
                        row['question'], 
                        self.data_transformation_config.max_question_length
                    )
                    
                    answer_data = self.preprocess_text(
                        row['primary_answer'], 
                        self.data_transformation_config.max_answer_length
                    )
                    
                    # Process image
                    image_result = self.process_image(row['image_path'], image_transform)
                    
                    # Extract additional image features
                    cv2_features = self.extract_image_features_cv2(row['image_path'])
                    
                    # Encode categorical variables
                    question_type_encoded = self.question_type_encoder.transform([
                        str(row.get('question_type', 'unknown'))
                    ])[0]
                    
                    answer_type_encoded = self.answer_type_encoder.transform([
                        str(row.get('answer_type', 'unknown'))
                    ])[0]
                    
                    # Create transformed sample
                    transformed_sample = {
                        'sample_id': row['sample_id'],
                        'image_path': row['image_path'],
                        'image_tensor_path': '',  # Will be set when saving
                        'question_original': row['question'],
                        'question_cleaned': question_data['cleaned_text'],
                        'question_length': question_data['text_length'],
                        'question_word_count': question_data['word_count'],
                        'answer_original': row['primary_answer'],
                        'answer_cleaned': answer_data['cleaned_text'],
                        'answer_length': answer_data['text_length'],
                        'answer_word_count': answer_data['word_count'],
                        'all_answers': row['answers'],
                        'num_answers': row['num_answers'],
                        'question_type': row.get('question_type', 'unknown'),
                        'answer_type': row.get('answer_type', 'unknown'),
                        'question_type_encoded': question_type_encoded,
                        'answer_type_encoded': answer_type_encoded,
                        'image_features': image_result['image_features'],
                        'cv2_features': cv2_features,
                        'is_valid_image': image_result['is_valid'],
                        'image_tensor_shape': list(image_result['image_tensor'].shape)
                    }
                    
                    transformed_data.append(transformed_sample)
                    
                    # Save image tensor
                    tensor_filename = f"{row['sample_id']}_tensor.pt"
                    tensor_path = os.path.join(
                        getattr(self.data_transformation_config, f'transformed_{split_name}_path'),
                        'tensors'
                    )
                    os.makedirs(tensor_path, exist_ok=True)
                    tensor_file_path = os.path.join(tensor_path, tensor_filename)
                    torch.save(image_result['image_tensor'], tensor_file_path)
                    transformed_sample['image_tensor_path'] = tensor_file_path
                    
                    if (idx + 1) % 100 == 0:
                        logging.info(f"Transformed {idx + 1}/{len(df)} samples for {split_name}")
                        
                except Exception as sample_error:
                    logging.warning(f"Error transforming sample {idx} in {split_name}: {str(sample_error)}")
                    continue
            
            # Create DataFrame from transformed data
            transformed_df = pd.DataFrame(transformed_data)
            
            # Generate embeddings for questions and answers
            logging.info(f"Generating embeddings for {split_name}...")
            
            question_embeddings = self.generate_text_embeddings(
                transformed_df['question_cleaned'].tolist()
            )
            
            answer_embeddings = self.generate_text_embeddings(
                transformed_df['answer_cleaned'].tolist()
            )
            
            # Save embeddings
            embeddings_dir = os.path.join(
                getattr(self.data_transformation_config, f'transformed_{split_name}_path'),
                'embeddings'
            )
            os.makedirs(embeddings_dir, exist_ok=True)
            
            np.save(os.path.join(embeddings_dir, 'question_embeddings.npy'), question_embeddings)
            np.save(os.path.join(embeddings_dir, 'answer_embeddings.npy'), answer_embeddings)
            
            # Save transformed metadata
            output_path = getattr(self.data_transformation_config, f'transformed_{split_name}_path')
            transformed_metadata_path = os.path.join(output_path, f'{split_name}_transformed_metadata.csv')
            transformed_df.to_csv(transformed_metadata_path, index=False)
            
            # Save transformation statistics
            stats = {
                'split_name': split_name,
                'total_samples': len(transformed_df),
                'valid_images': int(transformed_df['is_valid_image'].sum()),
                'invalid_images': int((~transformed_df['is_valid_image']).sum()),
                'avg_question_length': float(transformed_df['question_length'].mean()),
                'avg_answer_length': float(transformed_df['answer_length'].mean()),
                'question_embedding_shape': question_embeddings.shape if len(question_embeddings) > 0 else None,
                'answer_embedding_shape': answer_embeddings.shape if len(answer_embeddings) > 0 else None,
                'unique_question_types': int(transformed_df['question_type'].nunique()),
                'unique_answer_types': int(transformed_df['answer_type'].nunique())
            }
            
            stats_path = os.path.join(output_path, f'{split_name}_transformation_stats.json')
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            
            logging.info(f"Transformation completed for {split_name}")
            logging.info(f"Transformed metadata: {transformed_metadata_path}")
            logging.info(f"Valid images: {stats['valid_images']}/{stats['total_samples']}")
            
            return transformed_metadata_path
            
        except Exception as e:
            raise MultiModalException(f"Error transforming {split_name} dataset: {str(e)}", sys.exc_info())
    
    def save_preprocessors(self) -> str:
        """
        Save all preprocessors and encoders
        """
        try:
            preprocessors_path = os.path.join(
                self.data_transformation_config.data_transformation_dir,
                'preprocessors'
            )
            os.makedirs(preprocessors_path, exist_ok=True)
            
            # Save label encoders
            with open(os.path.join(preprocessors_path, 'question_type_encoder.pkl'), 'wb') as f:
                pickle.dump(self.question_type_encoder, f)
            
            with open(os.path.join(preprocessors_path, 'answer_type_encoder.pkl'), 'wb') as f:
                pickle.dump(self.answer_type_encoder, f)
            
            # Save configuration
            config_dict = {
                'target_image_size': self.data_transformation_config.target_image_size,
                'image_normalization_mean': self.data_transformation_config.image_normalization_mean,
                'image_normalization_std': self.data_transformation_config.image_normalization_std,
                'max_question_length': self.data_transformation_config.max_question_length,
                'max_answer_length': self.data_transformation_config.max_answer_length,
                'text_embedding_model': self.data_transformation_config.text_embedding_model,
                'device': self.data_transformation_config.device
            }
            
            with open(os.path.join(preprocessors_path, 'transformation_config.json'), 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            logging.info(f"Preprocessors saved to: {preprocessors_path}")
            return preprocessors_path
            
        except Exception as e:
            raise MultiModalException(f"Error saving preprocessors: {str(e)}", sys.exc_info())
    
    def initiate_data_transformation(self, data_validation_artifact) -> DataTransformationArtifact:
        """
        Main method to initiate data transformation process
        """
        try:
            logging.info("Starting data transformation process...")
            
            # First, fit encoders on training data
            train_metadata_path = data_validation_artifact.validated_train_path
            train_df = pd.read_csv(train_metadata_path)
            self.fit_encoders(train_df)
            
            # Transform each split
            splits_to_transform = [
                ('train', os.path.dirname(data_validation_artifact.validated_train_path), 'train_metadata.csv'),
                ('validation', os.path.dirname(data_validation_artifact.validated_validation_path), 'validation_metadata.csv'),
                ('test', os.path.dirname(data_validation_artifact.validated_test_path), 'test_metadata.csv')
            ]
            
            transformed_paths = {}
            total_samples = 0
            total_valid_images = 0
            
            for split_name, split_dir, metadata_file in splits_to_transform:
                transformed_path = self.transform_dataset_split(split_dir, split_name, metadata_file)
                transformed_paths[split_name] = transformed_path
                
                # Load stats for summary
                stats_path = os.path.join(
                    getattr(self.data_transformation_config, f'transformed_{split_name}_path'),
                    f'{split_name}_transformation_stats.json'
                )
                with open(stats_path, 'r') as f:
                    stats = json.load(f)
                
                total_samples += stats['total_samples']
                total_valid_images += stats['valid_images']
            
            # Save preprocessors
            preprocessors_path = self.save_preprocessors()
            
            # Create transformation summary
            transformation_summary = {
                'total_samples_transformed': total_samples,
                'total_valid_images': total_valid_images,
                'image_validation_rate': total_valid_images / max(total_samples, 1),
                'text_embedding_model': self.data_transformation_config.text_embedding_model,
                'target_image_size': self.data_transformation_config.target_image_size,
                'question_types_count': len(self.question_type_encoder.classes_),
                'answer_types_count': len(self.answer_type_encoder.classes_),
                'device_used': self.data_transformation_config.device
            }
            
            # Create data transformation artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformation_status=True,
                transformed_train_path=transformed_paths['train'],
                transformed_validation_path=transformed_paths['validation'],
                transformed_test_path=transformed_paths['test'],
                preprocessors_path=preprocessors_path,
                transformation_summary=transformation_summary
            )
            
            logging.info("Data transformation completed successfully")
            logging.info(f"Total samples transformed: {total_samples}")
            logging.info(f"Image validation rate: {transformation_summary['image_validation_rate']:.2%}")
            
            return data_transformation_artifact
            
        except Exception as e:
            raise MultiModalException(f"Error in data transformation process: {str(e)}", sys.exc_info())

# Test the data transformation
if __name__ == "__main__":
    try:
        # Import required classes
        from data_ingestion import DataIngestion
        from data_validation import DataValidation
        
        # Run complete pipeline
        logging.info("Running complete data pipeline...")
        
        # Data Ingestion
        data_ingestion = DataIngestion()
        ingestion_artifact = data_ingestion.initiate_data_ingestion()
        
        # Data Validation
        data_validation = DataValidation()
        validation_artifact = data_validation.initiate_data_validation(ingestion_artifact)
        
        # Data Transformation
        data_transformation = DataTransformation()
        transformation_artifact = data_transformation.initiate_data_transformation(validation_artifact)
        
        print(" Complete Data Pipeline completed successfully!")
        print(f" Transformation Status: {transformation_artifact.transformation_status}")
        print(f" Total Samples: {transformation_artifact.transformation_summary['total_samples_transformed']}")
        print(f" Image Validation Rate: {transformation_artifact.transformation_summary['image_validation_rate']:.2%}")
        print(f" Preprocessors: {transformation_artifact.preprocessors_path}")
        
    except Exception as e:
        print(f" Error in data transformation: {str(e)}")
