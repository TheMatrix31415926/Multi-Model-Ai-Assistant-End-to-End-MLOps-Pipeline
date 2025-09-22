"""data_ingestion.py module"""

# multimodal_ai_assistant/components/data_ingestion.py

import os
import sys
from typing import Tuple, Dict, Any
from datasets import load_dataset, Dataset
from PIL import Image
import pandas as pd
import json
from dataclasses import dataclass
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class DataIngestionConfig:
    """Data Ingestion Configuration"""
    dataset_name: str = "HuggingFaceM4/VQAv2"
    train_split: str = "train[:5000]"  # Start with subset for development
    validation_split: str = "validation[:1000]"
    test_split: str = "test[:500]"
    data_ingestion_dir: str = "artifacts/data_ingestion"
    raw_data_path: str = "artifacts/data_ingestion/raw_data"
    train_data_path: str = "artifacts/data_ingestion/train"
    validation_data_path: str = "artifacts/data_ingestion/validation"
    test_data_path: str = "artifacts/data_ingestion/test"

@dataclass
class DataIngestionArtifact:
    """Data Ingestion Artifact"""
    train_file_path: str
    validation_file_path: str
    test_file_path: str
    train_data_dir: str
    validation_data_dir: str
    test_data_dir: str

class MultiModalException(Exception):
    """Custom exception for the project"""
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message = error_message
        self.error_detail = str(error_detail)

class DataIngestion:
    """
    Data Ingestion class for Multi-Modal AI Assistant
    Downloads and processes VQAv2 dataset from HuggingFace
    """
    
    def __init__(self, data_ingestion_config: DataIngestionConfig = None):
        try:
            self.data_ingestion_config = data_ingestion_config or DataIngestionConfig()
            self._create_directories()
            
        except Exception as e:
            raise MultiModalException(f"Error in DataIngestion initialization: {str(e)}", sys.exc_info())
    
    def _create_directories(self):
        """Create necessary directories for data ingestion"""
        try:
            directories = [
                self.data_ingestion_config.data_ingestion_dir,
                self.data_ingestion_config.raw_data_path,
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.validation_data_path,
                self.data_ingestion_config.test_data_path
            ]
            
            for directory in directories:
                os.makedirs(directory, exist_ok=True)
            
            logging.info(f"Created directories for data ingestion")
            
        except Exception as e:
            raise MultiModalException(f"Error creating directories: {str(e)}", sys.exc_info())
    
    def download_dataset(self) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Download VQAv2 dataset from HuggingFace
        Returns: train, validation, test datasets
        """
        try:
            logging.info(f"Starting dataset download from {self.data_ingestion_config.dataset_name}")
            
            # Load different splits
            logging.info("Loading training data...")
            train_dataset = load_dataset(
                self.data_ingestion_config.dataset_name, 
                split=self.data_ingestion_config.train_split
            )
            
            logging.info("Loading validation data...")
            val_dataset = load_dataset(
                self.data_ingestion_config.dataset_name,
                split=self.data_ingestion_config.validation_split
            )
            
            logging.info("Loading test data...")
            test_dataset = load_dataset(
                self.data_ingestion_config.dataset_name,
                split=self.data_ingestion_config.test_split
            )
            
            logging.info(f"Dataset download completed")
            logging.info(f"Train samples: {len(train_dataset)}")
            logging.info(f"Validation samples: {len(val_dataset)}")
            logging.info(f"Test samples: {len(test_dataset)}")
            
            return train_dataset, val_dataset, test_dataset
            
        except Exception as e:
            raise MultiModalException(f"Error downloading dataset: {str(e)}", sys.exc_info())
    
    def process_and_save_dataset(self, dataset: Dataset, split_name: str, save_path: str) -> str:
        """
        Process dataset and save in structured format
        Args:
            dataset: HuggingFace dataset
            split_name: train/validation/test
            save_path: Path to save processed data
        """
        try:
            logging.info(f"Processing {split_name} dataset...")
            
            processed_data = []
            images_dir = os.path.join(save_path, "images")
            os.makedirs(images_dir, exist_ok=True)
            
            for idx, sample in enumerate(dataset):
                try:
                    # Extract data
                    image = sample['image']
                    question = sample['question']
                    answers = sample['answers'] if isinstance(sample['answers'], list) else [sample['answers']]
                    question_type = sample.get('question_type', 'unknown')
                    answer_type = sample.get('answer_type', 'unknown')
                    
                    # Save image
                    image_filename = f"{split_name}_image_{idx:06d}.jpg"
                    image_path = os.path.join(images_dir, image_filename)
                    
                    if isinstance(image, Image.Image):
                        # Convert to RGB if necessary
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        image.save(image_path, 'JPEG', quality=95)
                    
                    # Prepare metadata
                    processed_sample = {
                        'sample_id': f"{split_name}_{idx:06d}",
                        'image_path': image_path,
                        'image_filename': image_filename,
                        'question': question,
                        'answers': '|'.join(answers),  # Join multiple answers with |
                        'primary_answer': answers[0] if answers else '',
                        'question_type': question_type,
                        'answer_type': answer_type,
                        'num_answers': len(answers),
                        'image_size': f"{image.width}x{image.height}" if hasattr(image, 'width') else 'unknown'
                    }
                    
                    processed_data.append(processed_sample)
                    
                    if (idx + 1) % 100 == 0:
                        logging.info(f"Processed {idx + 1}/{len(dataset)} samples from {split_name}")
                        
                except Exception as sample_error:
                    logging.warning(f"Error processing sample {idx} in {split_name}: {str(sample_error)}")
                    continue
            
            # Save metadata as CSV
            metadata_df = pd.DataFrame(processed_data)
            metadata_path = os.path.join(save_path, f"{split_name}_metadata.csv")
            metadata_df.to_csv(metadata_path, index=False)
            
            # Save dataset info
            self._save_dataset_info(processed_data, metadata_df, split_name, save_path)
            
            logging.info(f"Saved {len(processed_data)} processed samples for {split_name}")
            logging.info(f"Metadata saved to: {metadata_path}")
            
            return metadata_path
            
        except Exception as e:
            raise MultiModalException(f"Error processing {split_name} dataset: {str(e)}", sys.exc_info())
    
    def _save_dataset_info(self, processed_data: list, metadata_df: pd.DataFrame, split_name: str, save_path: str):
        """Save dataset information and statistics"""
        info_path = os.path.join(save_path, f"{split_name}_info.txt")
        stats_path = os.path.join(save_path, f"{split_name}_stats.json")
        
        # Calculate statistics
        stats = {
            "dataset_name": self.data_ingestion_config.dataset_name,
            "split_name": split_name,
            "total_samples": len(processed_data),
            "unique_questions": metadata_df['question'].nunique(),
            "unique_question_types": metadata_df['question_type'].nunique(),
            "unique_answer_types": metadata_df['answer_type'].nunique(),
            "avg_answers_per_question": float(metadata_df['num_answers'].mean()),
            "question_type_distribution": metadata_df['question_type'].value_counts().to_dict(),
            "answer_type_distribution": metadata_df['answer_type'].value_counts().to_dict()
        }
        
        # Save info file
        with open(info_path, 'w') as f:
            f.write(f"Dataset: {stats['dataset_name']}\n")
            f.write(f"Split: {split_name}\n")
            f.write(f"Total samples: {stats['total_samples']}\n")
            f.write(f"Unique questions: {stats['unique_questions']}\n")
            f.write(f"Unique question types: {stats['unique_question_types']}\n")
            f.write(f"Unique answer types: {stats['unique_answer_types']}\n")
            f.write(f"Average answers per question: {stats['avg_answers_per_question']:.2f}\n")
            f.write("\nQuestion Type Distribution:\n")
            for qtype, count in stats['question_type_distribution'].items():
                f.write(f"  {qtype}: {count}\n")
        
        # Save stats as JSON
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Main method to initiate data ingestion process
        Returns: DataIngestionArtifact with paths to processed data
        """
        try:
            logging.info("Starting data ingestion process...")
            
            # Download datasets
            train_dataset, val_dataset, test_dataset = self.download_dataset()
            
            # Process and save datasets
            train_metadata_path = self.process_and_save_dataset(
                train_dataset, 
                "train", 
                self.data_ingestion_config.train_data_path
            )
            
            val_metadata_path = self.process_and_save_dataset(
                val_dataset, 
                "validation", 
                self.data_ingestion_config.validation_data_path
            )
            
            test_metadata_path = self.process_and_save_dataset(
                test_dataset, 
                "test", 
                self.data_ingestion_config.test_data_path
            )
            
            # Create data ingestion artifact
            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=train_metadata_path,
                validation_file_path=val_metadata_path,
                test_file_path=test_metadata_path,
                train_data_dir=self.data_ingestion_config.train_data_path,
                validation_data_dir=self.data_ingestion_config.validation_data_path,
                test_data_dir=self.data_ingestion_config.test_data_path
            )
            
            logging.info("Data ingestion completed successfully")
            return data_ingestion_artifact
            
        except Exception as e:
            raise MultiModalException(f"Error in data ingestion process: {str(e)}", sys.exc_info())

# Test the data ingestion
if __name__ == "__main__":
    try:
        # Initialize and run data ingestion
        data_ingestion = DataIngestion()
        artifact = data_ingestion.initiate_data_ingestion()
        print(" Data Ingestion completed successfully!")
        print(f"Train data: {artifact.train_file_path}")
        print(f"Validation data: {artifact.validation_file_path}")
        print(f"Test data: {artifact.test_file_path}")
        
    except Exception as e:
        print(f"Error in data ingestion: {str(e)}")
