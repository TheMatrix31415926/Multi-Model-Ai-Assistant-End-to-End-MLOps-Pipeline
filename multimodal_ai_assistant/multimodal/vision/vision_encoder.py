"""vision_encoder.py module"""

"""
Multi-Modal AI Assistant - Vision Component
Vision Encoder using CLIP/ViT for image processing and embedding generation
"""

import os
import sys
import torch
import torch.nn as nn
from PIL import Image
import clip
import numpy as np
from typing import List, Union, Tuple, Optional
import logging
from dataclasses import dataclass
from pathlib import Path
import json
from transformers import ViTImageProcessor, ViTModel
import torchvision.transforms as transforms

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VisionConfig:
    """Configuration for Vision Encoder"""
    model_name: str = "ViT-B/32"  # CLIP model
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    image_size: int = 224
    batch_size: int = 32
    embedding_dim: int = 512
    cache_embeddings: bool = True
    cache_dir: str = "cache/vision_embeddings"

class VisionEncoder:
    """
    Vision Encoder component for processing images and generating embeddings
    Supports both CLIP and ViT models
    """
    
    def __init__(self, config: Optional[VisionConfig] = None):
        self.config = config or VisionConfig()
        self.device = torch.device(self.config.device)
        self.model = None
        self.preprocess = None
        self.model_type = None
        
        # Create cache directory
        os.makedirs(self.config.cache_dir, exist_ok=True)
        
        logger.info(f"Initializing Vision Encoder with device: {self.device}")
        self._load_model()
    
    def _load_model(self):
        """Load the vision model (CLIP or ViT)"""
        try:
            if "ViT" in self.config.model_name and "/" in self.config.model_name:
                # Load CLIP model
                self.model, self.preprocess = clip.load(self.config.model_name, device=self.device)
                self.model_type = "clip"
                logger.info(f"Loaded CLIP model: {self.config.model_name}")
            else:
                # Load standalone ViT model
                self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
                self.model = ViTModel.from_pretrained('google/vit-base-patch16-224')
                self.model.to(self.device)
                self.model_type = "vit"
                logger.info("Loaded ViT model: google/vit-base-patch16-224")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _preprocess_image(self, image: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Preprocess image for model input
        
        Args:
            image: Image path, PIL Image, or numpy array
            
        Returns:
            Preprocessed image tensor
        """
        try:
            # Convert to PIL Image if needed
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image).convert('RGB')
            elif not isinstance(image, Image.Image):
                raise ValueError(f"Unsupported image type: {type(image)}")
            
            if self.model_type == "clip":
                # CLIP preprocessing
                return self.preprocess(image).unsqueeze(0).to(self.device)
            else:
                # ViT preprocessing
                inputs = self.processor(images=image, return_tensors="pt")
                return inputs['pixel_values'].to(self.device)
                
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    def encode_image(self, image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Encode a single image to embedding
        
        Args:
            image: Image to encode
            
        Returns:
            Image embedding as numpy array
        """
        try:
            # Check cache first
            cache_key = None
            if self.config.cache_embeddings and isinstance(image, str):
                cache_key = self._get_cache_key(image)
                cached_embedding = self._load_from_cache(cache_key)
                if cached_embedding is not None:
                    return cached_embedding
            
            # Preprocess image
            image_tensor = self._preprocess_image(image)
            
            # Generate embedding
            with torch.no_grad():
                if self.model_type == "clip":
                    embedding = self.model.encode_image(image_tensor)
                    embedding = embedding.cpu().numpy().flatten()
                else:
                    outputs = self.model(image_tensor)
                    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
            
            # Cache if enabled
            if self.config.cache_embeddings and cache_key:
                self._save_to_cache(cache_key, embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            raise
    
    def encode_batch(self, images: List[Union[str, Image.Image, np.ndarray]]) -> np.ndarray:
        """
        Encode a batch of images
        
        Args:
            images: List of images to encode
            
        Returns:
            Array of embeddings
        """
        embeddings = []
        
        # Process in batches
        for i in range(0, len(images), self.config.batch_size):
            batch = images[i:i + self.config.batch_size]
            batch_embeddings = []
            
            for image in batch:
                embedding = self.encode_image(image)
                batch_embeddings.append(embedding)
            
            embeddings.extend(batch_embeddings)
            logger.info(f"Processed batch {i//self.config.batch_size + 1}/{(len(images)-1)//self.config.batch_size + 1}")
        
        return np.array(embeddings)
    
    def similarity_search(self, query_image: Union[str, Image.Image, np.ndarray], 
                         image_embeddings: np.ndarray, 
                         top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Find most similar images based on embeddings
        
        Args:
            query_image: Query image
            image_embeddings: Array of image embeddings to search
            top_k: Number of top results to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        try:
            # Get query embedding
            query_embedding = self.encode_image(query_image)
            
            # Calculate cosine similarities
            similarities = np.dot(image_embeddings, query_embedding) / (
                np.linalg.norm(image_embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            return [(int(idx), float(similarities[idx])) for idx in top_indices]
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            raise
    
    def extract_features(self, image: Union[str, Image.Image, np.ndarray], 
                        layer: str = "last") -> np.ndarray:
        """
        Extract features from specific layer
        
        Args:
            image: Input image
            layer: Which layer to extract features from
            
        Returns:
            Feature array
        """
        try:
            image_tensor = self._preprocess_image(image)
            
            with torch.no_grad():
                if self.model_type == "clip":
                    # For CLIP, return the visual features
                    features = self.model.encode_image(image_tensor)
                else:
                    # For ViT, extract from specified layer
                    outputs = self.model(image_tensor, output_hidden_states=True)
                    if layer == "last":
                        features = outputs.last_hidden_state
                    else:
                        # Extract from specific layer index
                        layer_idx = int(layer) if layer.isdigit() else -1
                        features = outputs.hidden_states[layer_idx]
                    
                    features = features.mean(dim=1)  # Global average pooling
            
            return features.cpu().numpy()
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise
    
    def _get_cache_key(self, image_path: str) -> str:
        """Generate cache key for image"""
        return f"{hash(image_path)}_{self.config.model_name.replace('/', '_')}"
    
    def _load_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """Load embedding from cache"""
        cache_file = Path(self.config.cache_dir) / f"{cache_key}.npy"
        if cache_file.exists():
            try:
                return np.load(cache_file)
            except:
                return None
        return None
    
    def _save_to_cache(self, cache_key: str, embedding: np.ndarray):
        """Save embedding to cache"""
        cache_file = Path(self.config.cache_dir) / f"{cache_key}.npy"
        try:
            np.save(cache_file, embedding)
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")
    
    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            "model_name": self.config.model_name,
            "model_type": self.model_type,
            "device": str(self.device),
            "embedding_dim": self.config.embedding_dim,
            "image_size": self.config.image_size
        }

# Test function
def test_vision_encoder():
    """Test the Vision Encoder component"""
    print("Testing Vision Encoder...")
    
    # Initialize encoder
    config = VisionConfig(model_name="ViT-B/32", device="cpu")  # Use CPU for testing
    encoder = VisionEncoder(config)
    
    # Print model info
    print("Model Info:", encoder.get_model_info())
    
    # Create a dummy image for testing
    dummy_image = Image.new('RGB', (224, 224), color='red')
    
    try:
        # Test single image encoding
        embedding = encoder.encode_image(dummy_image)
        print(f"Generated embedding shape: {embedding.shape}")
        print(f"Embedding sample: {embedding[:5]}")
        
        # Test batch encoding
        batch_embeddings = encoder.encode_batch([dummy_image, dummy_image])
        print(f"Batch embeddings shape: {batch_embeddings.shape}")
        
        # Test feature extraction
        features = encoder.extract_features(dummy_image)
        print(f"Features shape: {features.shape}")
        
        print(" Vision Encoder test passed!")
        
    except Exception as e:
        print(f" Vision Encoder test failed: {e}")

if __name__ == "__main__":
    test_vision_encoder()
