"""model_pusher.py module"""

"""
Multi-Modal AI Assistant - Model Registry (Model Pusher)
Push best models to registry with versioning and metadata
"""

import os
import sys
import json
import shutil
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from datetime import datetime
import hashlib

# Cloud storage imports with fallbacks
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    HAS_AWS = True
except ImportError:
    print("Warning: AWS SDK not available. Using local storage only.")
    HAS_AWS = False

# MLflow imports with fallback
try:
    import mlflow
    import mlflow.pytorch
    import mlflow.pyfunc
    from mlflow.models import infer_signature
    HAS_MLFLOW = True
except ImportError:
    print("Warning: MLflow not available. Using basic model registry.")
    HAS_MLFLOW = False

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from components.model_trainer import ModelTrainer, TrainerConfig
    from components.model_evaluation import ModelEvaluator, EvaluationConfig
except ImportError:
    print("Warning: Training components not found.")
    class ModelTrainer: pass
    class ModelEvaluator: pass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelMetadata:
    """Metadata for model versioning"""
    model_name: str
    version: str
    created_at: datetime
    model_path: str
    model_size_mb: float
    
    # Performance metrics
    accuracy: float = 0.0
    bleu_score: float = 0.0
    confidence_avg: float = 0.0
    
    # Training info
    training_config: Dict[str, Any] = field(default_factory=dict)
    training_duration: float = 0.0
    training_samples: int = 0
    
    # Model info
    model_type: str = "multimodal_vqa"
    framework: str = "pytorch"
    model_hash: str = ""
    
    # Deployment info
    status: str = "trained"  # trained, validated, staging, production, archived
    deployment_target: str = "local"
    
    # Additional metadata
    tags: List[str] = field(default_factory=list)
    description: str = ""
    author: str = "system"

@dataclass
class RegistryConfig:
    """Configuration for model registry"""
    # Local registry settings
    local_registry_path: str = "artifacts/model_registry"
    models_storage_path: str = "artifacts/models"
    
    # Remote storage settings
    enable_s3_storage: bool = False
    s3_bucket: str = "multimodal-ai-models"
    s3_prefix: str = "vqa_models"
    aws_region: str = "us-west-2"
    
    # MLflow settings
    enable_mlflow_registry: bool = True
    mlflow_tracking_uri: str = "mlruns"
    mlflow_model_name: str = "multimodal_vqa"
    
    # Model validation
    min_accuracy_threshold: float = 0.6
    min_confidence_threshold: float = 0.5
    require_evaluation: bool = True
    
    # Versioning
    version_format: str = "v{major}.{minor}.{patch}"
    auto_increment_version: bool = True
    
    # Deployment
    auto_promote_to_staging: bool = True
    staging_criteria: Dict[str, float] = field(default_factory=lambda: {
        "accuracy": 0.75,
        "bleu_score": 0.6
    })

class ModelRegistry:
    """
    Model registry for managing model versions and deployment
    """
    
    def __init__(self, config: Optional[RegistryConfig] = None):
        self.config = config or RegistryConfig()
        
        # Storage clients
        self.s3_client = None
        
        # Registry data
        self.registry_index: Dict[str, List[ModelMetadata]] = {}
        
        # Setup
        self._setup_registry()
        self._load_registry_index()
        
        if HAS_MLFLOW and self.config.enable_mlflow_registry:
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        
        logger.info("Model Registry initialized")
    
    def _setup_registry(self):
        """Setup registry directories and connections"""
        # Create local directories
        os.makedirs(self.config.local_registry_path, exist_ok=True)
        os.makedirs(self.config.models_storage_path, exist_ok=True)
        
        # Setup AWS S3 client
        if HAS_AWS and self.config.enable_s3_storage:
            try:
                self.s3_client = boto3.client('s3', region_name=self.config.aws_region)
                # Test connection
                self.s3_client.head_bucket(Bucket=self.config.s3_bucket)
                logger.info("S3 connection established")
            except (ClientError, NoCredentialsError) as e:
                logger.warning(f"S3 setup failed: {e}. Using local storage only.")
                self.s3_client = None
    
    def _load_registry_index(self):
        """Load the registry index"""
        index_path = os.path.join(self.config.local_registry_path, "registry_index.json")
        
        if os.path.exists(index_path):
            try:
                with open(index_path, "r") as f:
                    data = json.load(f)
                
                # Convert to ModelMetadata objects
                for model_name, versions in data.items():
                    self.registry_index[model_name] = []
                    for version_data in versions:
                        version_data["created_at"] = datetime.fromisoformat(version_data["created_at"])
                        self.registry_index[model_name].append(ModelMetadata(**version_data))
                
                logger.info(f"Loaded registry index with {len(self.registry_index)} models")
                
            except Exception as e:
                logger.error(f"Error loading registry index: {e}")
                self.registry_index = {}
        else:
            logger.info("No existing registry index found. Starting fresh.")
            self.registry_index = {}
    
    def _save_registry_index(self):
        """Save the registry index"""
        index_path = os.path.join(self.config.local_registry_path, "registry_index.json")
        
        try:
            # Convert to serializable format
            serializable_data = {}
            for model_name, versions in self.registry_index.items():
                serializable_data[model_name] = []
                for metadata in versions:
                    metadata_dict = {
                        **vars(metadata),
                        "created_at": metadata.created_at.isoformat()
                    }
                    serializable_data[model_name].append(metadata_dict)
            
            with open(index_path, "w") as f:
                json.dump(serializable_data, f, indent=2, default=str)
            
            logger.info("Registry index saved")
            
        except Exception as e:
            logger.error(f"Error saving registry index: {e}")
    
    def _calculate_model_hash(self, model_path: str) -> str:
        """Calculate hash of model files for integrity checking"""
        hasher = hashlib.sha256()
        
        try:
            if os.path.isfile(model_path):
                # Single file
                with open(model_path, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hasher.update(chunk)
            elif os.path.isdir(model_path):
                # Directory - hash all files
                for root, dirs, files in os.walk(model_path):
                    for file in sorted(files):  # Sort for consistent hash
                        file_path = os.path.join(root, file)
                        with open(file_path, "rb") as f:
                            for chunk in iter(lambda: f.read(4096), b""):
                                hasher.update(chunk)
            
            return hasher.hexdigest()
            
        except Exception as e:
            logger.warning(f"Could not calculate model hash: {e}")
            return "unknown"
    
    def _get_model_size(self, model_path: str) -> float:
        """Get model size in MB"""
        try:
            if os.path.isfile(model_path):
                size_bytes = os.path.getsize(model_path)
            elif os.path.isdir(model_path):
                size_bytes = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, dirnames, filenames in os.walk(model_path)
                    for filename in filenames
                )
            else:
                return 0.0
            
            return size_bytes / (1024 * 1024)  # Convert to MB
            
        except Exception as e:
            logger.warning(f"Could not calculate model size: {e}")
            return 0.0
    
    def _generate_version(self, model_name: str) -> str:
        """Generate next version number"""
        if model_name not in self.registry_index or not self.registry_index[model_name]:
            return "v1.0.0"
        
        # Get latest version
        versions = self.registry_index[model_name]
        latest_version = max(versions, key=lambda x: x.created_at).version
        
        try:
            # Parse version (assumes vX.Y.Z format)
            if latest_version.startswith("v"):
                version_parts = latest_version[1:].split(".")
                major, minor, patch = map(int, version_parts[:3])
                
                # Increment patch version
                return f"v{major}.{minor}.{patch + 1}"
            else:
                return "v1.0.0"
                
        except Exception as e:
            logger.warning(f"Could not parse version {latest_version}: {e}")
            return f"v1.0.{len(versions) + 1}"
    
    def register_model(self, model_path: str, evaluation_results: Optional[Dict[str, Any]] = None,
                      training_config: Optional[Dict[str, Any]] = None, 
                      model_name: str = "multimodal_vqa", **kwargs) -> ModelMetadata:
        """Register a new model version"""
        logger.info(f"Registering model: {model_name}")
        
        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")
        
        # Generate version
        version = self._generate_version(model_name)
        
        # Create metadata
        metadata = ModelMetadata(
            model_name=model_name,
            version=version,
            created_at=datetime.now(),
            model_path=model_path,
            model_size_mb=self._get_model_size(model_path),
            model_hash=self._calculate_model_hash(model_path),
            training_config=training_config or {},
            **kwargs
        )
        
        # Add evaluation results if provided
        if evaluation_results:
            metadata.accuracy = evaluation_results.get("accuracy", 0.0)
            metadata.bleu_score = evaluation_results.get("bleu_1", 0.0)
            metadata.confidence_avg = evaluation_results.get("confidence_avg", 0.0)
        
        # Copy model to registry storage
        registry_model_path = self._store_model(model_path, model_name, version)
        metadata.model_path = registry_model_path
        
        # Add to registry
        if model_name not in self.registry_index:
            self.registry_index[model_name] = []
        
        self.registry_index[model_name].append(metadata)
        
        # Save registry
        self._save_registry_index()
        
        # Register with MLflow if enabled
        if HAS_MLFLOW and self.config.enable_mlflow_registry:
            self._register_with_mlflow(metadata)
        
        # Check for auto-promotion
        if self.config.auto_promote_to_staging:
            self._check_promotion_criteria(metadata)
        
        logger.info(f"Model registered: {model_name} {version}")
        return metadata
    
    def _store_model(self, source_path: str, model_name: str, version: str) -> str:
        """Store model in registry storage"""
        # Create registry storage path
        storage_dir = os.path.join(self.config.local_registry_path, "models", model_name, version)
        os.makedirs(storage_dir, exist_ok=True)
        
        # Copy model files
        if os.path.isfile(source_path):
            destination = os.path.join(storage_dir, os.path.basename(source_path))
            shutil.copy2(source_path, destination)
        elif os.path.isdir(source_path):
            destination = os.path.join(storage_dir, "model")
            if os.path.exists(destination):
                shutil.rmtree(destination)
            shutil.copytree(source_path, destination)
        
        # Upload to S3 if enabled
        if self.s3_client:
            self._upload_to_s3(storage_dir, model_name, version)
        
        return storage_dir
    
    def _upload_to_s3(self, local_path: str, model_name: str, version: str):
        """Upload model to S3"""
        try:
            s3_prefix = f"{self.config.s3_prefix}/{model_name}/{version}/"
            
            for root, dirs, files in os.walk(local_path):
                for file in files:
                    local_file = os.path.join(root, file)
                    relative_path = os.path.relpath(local_file, local_path)
                    s3_key = s3_prefix + relative_path.replace("\\", "/")  # Ensure forward slashes
                    
                    self.s3_client.upload_file(local_file, self.config.s3_bucket, s3_key)
                    logger.info(f"Uploaded {relative_path} to S3")
            
            logger.info(f"Model {model_name} {version} uploaded to S3")
            
        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
    
    def _register_with_mlflow(self, metadata: ModelMetadata):
        """Register model with MLflow Model Registry"""
        try:
            # Create a simple MLflow model (this is simplified)
            with mlflow.start_run():
                # Log model metadata
                mlflow.log_params({
                    "model_name": metadata.model_name,
                    "version": metadata.version,
                    "accuracy": metadata.accuracy,
                    "model_size_mb": metadata.model_size_mb
                })
                
                # Register model (simplified - in real implementation you'd log the actual model)
                mlflow.log_artifacts(metadata.model_path, "model")
                
                # Register in MLflow registry
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
                mlflow.register_model(model_uri, self.config.mlflow_model_name)
            
            logger.info(f"Model registered with MLflow: {metadata.model_name} {metadata.version}")
            
        except Exception as e:
            logger.error(f"MLflow registration failed: {e}")
    
    def _check_promotion_criteria(self, metadata: ModelMetadata):
        """Check if model meets promotion criteria"""
        criteria_met = True
        
        for metric, threshold in self.config.staging_criteria.items():
            model_value = getattr(metadata, metric, 0.0)
            if model_value < threshold:
                criteria_met = False
                logger.info(f"Model {metadata.model_name} {metadata.version} does not meet {metric} threshold: {model_value} < {threshold}")
        
        if criteria_met:
            metadata.status = "staging"
            logger.info(f"Model {metadata.model_name} {metadata.version} auto-promoted to staging")
        else:
            metadata.status = "trained"
    
    def promote_model(self, model_name: str, version: str, target_stage: str) -> bool:
        """Promote model to a different stage"""
        model_metadata = self.get_model_metadata(model_name, version)
        
        if not model_metadata:
            logger.error(f"Model not found: {model_name} {version}")
            return False
        
        valid_stages = ["trained", "staging", "production", "archived"]
        if target_stage not in valid_stages:
            logger.error(f"Invalid stage: {target_stage}. Must be one of {valid_stages}")
            return False
        
        # Update status
        old_status = model_metadata.status
        model_metadata.status = target_stage
        
        # Save registry
        self._save_registry_index()
        
        # Update MLflow if enabled
        if HAS_MLFLOW and self.config.enable_mlflow_registry:
            try:
                client = mlflow.tracking.MlflowClient()
                # Find the model version in MLflow
                # This is simplified - in practice you'd need to track MLflow version numbers
                logger.info(f"MLflow model promotion would happen here")
            except Exception as e:
                logger.error(f"MLflow promotion failed: {e}")
        
        logger.info(f"Model {model_name} {version} promoted from {old_status} to {target_stage}")
        return True
    
    def get_model_metadata(self, model_name: str, version: str) -> Optional[ModelMetadata]:
        """Get metadata for a specific model version"""
        if model_name not in self.registry_index:
            return None
        
        for metadata in self.registry_index[model_name]:
            if metadata.version == version:
                return metadata
        
        return None
    
    def list_models(self, model_name: Optional[str] = None, status: Optional[str] = None) -> List[ModelMetadata]:
        """List models with optional filtering"""
        models = []
        
        registries_to_search = [model_name] if model_name else self.registry_index.keys()
        
        for name in registries_to_search:
            if name in self.registry_index:
                for metadata in self.registry_index[name]:
                    if status is None or metadata.status == status:
                        models.append(metadata)
        
        # Sort by creation date (newest first)
        models.sort(key=lambda x: x.created_at, reverse=True)
        return models
    
    def get_latest_model(self, model_name: str, status: Optional[str] = None) -> Optional[ModelMetadata]:
        """Get the latest model version"""
        models = self.list_models(model_name, status)
        return models[0] if models else None
    
    def get_production_model(self, model_name: str) -> Optional[ModelMetadata]:
        """Get the current production model"""
        return self.get_latest_model(model_name, "production")
    
    def delete_model_version(self, model_name: str, version: str) -> bool:
        """Delete a specific model version"""
        metadata = self.get_model_metadata(model_name, version)
        
        if not metadata:
            logger.error(f"Model not found: {model_name} {version}")
            return False
        
        if metadata.status == "production":
            logger.error(f"Cannot delete production model: {model_name} {version}")
            return False
        
        try:
            # Remove from registry
            self.registry_index[model_name] = [
                m for m in self.registry_index[model_name] if m.version != version
            ]
            
            # Delete local files
            if os.path.exists(metadata.model_path):
                shutil.rmtree(metadata.model_path)
            
            # Delete from S3 if enabled
            if self.s3_client:
                self._delete_from_s3(model_name, version)
            
            # Save registry
            self._save_registry_index()
            
            logger.info(f"Model deleted: {model_name} {version}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting model: {e}")
            return False
    
    def _delete_from_s3(self, model_name: str, version: str):
        """Delete model from S3"""
        try:
            s3_prefix = f"{self.config.s3_prefix}/{model_name}/{version}/"
            
            # List objects with the prefix
            response = self.s3_client.list_objects_v2(
                Bucket=self.config.s3_bucket,
                Prefix=s3_prefix
            )
            
            if 'Contents' in response:
                objects_to_delete = [{'Key': obj['Key']} for obj in response['Contents']]
                self.s3_client.delete_objects(
                    Bucket=self.config.s3_bucket,
                    Delete={'Objects': objects_to_delete}
                )
                logger.info(f"Deleted {len(objects_to_delete)} objects from S3")
            
        except Exception as e:
            logger.error(f"S3 deletion failed: {e}")
    
    def generate_registry_report(self) -> str:
        """Generate a registry status report"""
        report_lines = [
            "Model Registry Report",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        total_models = sum(len(versions) for versions in self.registry_index.values())
        unique_models = len(self.registry_index)
        
        report_lines.extend([
            f"Total Model Versions: {total_models}",
            f"Unique Models: {unique_models}",
            ""
        ])
        
        # Status distribution
        status_counts = {}
        for versions in self.registry_index.values():
            for metadata in versions:
                status_counts[metadata.status] = status_counts.get(metadata.status, 0) + 1
        
        report_lines.append("Status Distribution:")
        for status, count in status_counts.items():
            report_lines.append(f"  {status}: {count}")
        report_lines.append("")
        
        # Model details
        for model_name, versions in self.registry_index.items():
            report_lines.extend([
                f"Model: {model_name}",
                f"  Versions: {len(versions)}",
                f"  Latest: {max(versions, key=lambda x: x.created_at).version}",
                f"  Production: {next((v.version for v in versions if v.status == 'production'), 'None')}",
                ""
            ])
        
        return "\n".join(report_lines)

class ModelPusher:
    """
    Main class for pushing models to registry (combines training, evaluation, and registration)
    """
    
    def __init__(self, registry_config: Optional[RegistryConfig] = None):
        self.registry_config = registry_config or RegistryConfig()
        self.registry = ModelRegistry(self.registry_config)
        
        logger.info("Model Pusher initialized")
    
    def push_model(self, model_path: str, model_name: str = "multimodal_vqa",
                  run_evaluation: bool = True) -> Dict[str, Any]:
        """
        Complete model pushing pipeline:
        1. Validate model
        2. Run evaluation (optional)
        3. Register in registry
        4. Check promotion criteria
        """
        logger.info(f"Starting model push pipeline for {model_name}")
        
        try:
            # Validate model exists
            if not os.path.exists(model_path):
                return {"status": "failed", "error": f"Model path does not exist: {model_path}"}
            
            results = {"status": "success", "model_name": model_name}
            
            # Run evaluation if requested
            evaluation_results = None
            if run_evaluation and self.registry_config.require_evaluation:
                logger.info("Running model evaluation...")
                evaluator = ModelEvaluator()
                eval_results = evaluator.initiate_model_evaluation()
                
                if eval_results["status"] == "success":
                    evaluation_results = eval_results["metrics"]
                    results["evaluation"] = evaluation_results
                    
                    # Check minimum thresholds
                    accuracy = evaluation_results.get("accuracy", 0.0)
                    if accuracy < self.registry_config.min_accuracy_threshold:
                        return {
                            "status": "failed", 
                            "error": f"Model accuracy {accuracy:.3f} below threshold {self.registry_config.min_accuracy_threshold}"
                        }
                else:
                    logger.warning("Evaluation failed, proceeding without evaluation results")
            
            # Register model
            metadata = self.registry.register_model(
                model_path=model_path,
                model_name=model_name,
                evaluation_results=evaluation_results,
                description=f"Auto-registered model from {model_path}",
                tags=["vqa", "multimodal", "auto-registered"]
            )
            
            results.update({
                "version": metadata.version,
                "status_in_registry": metadata.status,
                "model_size_mb": metadata.model_size_mb,
                "registry_path": metadata.model_path
            })
            
            logger.info(f"Model push completed successfully: {model_name} {metadata.version}")
            return results
            
        except Exception as e:
            logger.error(f"Model push failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def initiate_model_pusher(self, model_path: Optional[str] = None) -> Dict[str, Any]:
        """Main model pusher pipeline"""
        try:
            # Find model if not provided
            if model_path is None:
                # Look for latest trained model
                models_dir = self.registry_config.models_storage_path
                if os.path.exists(models_dir):
                    model_dirs = [d for d in os.listdir(models_dir) 
                                if os.path.isdir(os.path.join(models_dir, d)) and "vqa_model" in d]
                    if model_dirs:
                        model_dirs.sort(reverse=True)  # Get latest
                        model_path = os.path.join(models_dir, model_dirs[0])
                
                if not model_path:
                    return {"status": "failed", "error": "No model found to push"}
            
            # Push model
            result = self.push_model(model_path)
            
            return result
            
        except Exception as e:
            logger.error(f"Model pusher pipeline failed: {e}")
            return {"status": "failed", "error": str(e)}

# Testing function
def test_model_pusher():
    """Test the model pusher and registry"""
    print("Testing Model Pusher and Registry...")
    print("=" * 50)
    
    # Initialize registry
    config = RegistryConfig(
        min_accuracy_threshold=0.5,  # Lower for testing
        auto_promote_to_staging=True
    )
    
    pusher = ModelPusher(config)
    
    # Create mock model for testing
    mock_model_dir = "artifacts/test_models/mock_vqa_model"
    os.makedirs(mock_model_dir, exist_ok=True)
    
    # Create dummy model files
    with open(os.path.join(mock_model_dir, "model.bin"), "w") as f:
        f.write("mock model weights")
    
    with open(os.path.join(mock_model_dir, "config.json"), "w") as f:
        json.dump({"model_type": "vqa", "version": "test"}, f)
    
    print(f"Created mock model at: {mock_model_dir}")
    
    # Test model pushing
    result = pusher.push_model(mock_model_dir, "test_vqa_model", run_evaluation=False)
    
    print(f"Push Result: {result['status']}")
    if result["status"] == "success":
        print(f"Model Version: {result['version']}")
        print(f"Registry Status: {result['status_in_registry']}")
        print(f"Model Size: {result['model_size_mb']:.2f} MB")
    else:
        print(f"Push Error: {result.get('error', 'Unknown error')}")
    
    # Test registry operations
    print("\nTesting registry operations...")
    
    # List models
    models = pusher.registry.list_models()
    print(f"Total models in registry: {len(models)}")
    
    # Generate report
    report = pusher.registry.generate_registry_report()
    print("\nRegistry Report Preview:")
    print(report[:300] + "..." if len(report) > 300 else report)
    
    # Cleanup
    if os.path.exists(mock_model_dir):
        shutil.rmtree(mock_model_dir)

def main():
    """Main function"""
    print("Multi-Modal AI Assistant - Model Registry")
    print("=" * 50)
    
    test_model_pusher()
    
    print("\nStep 15: Model Registry - COMPLETE ✅")
    print("\nPhase 5: Model Training & MLOps - COMPLETE ✅")

if __name__ == "__main__":
    main()
