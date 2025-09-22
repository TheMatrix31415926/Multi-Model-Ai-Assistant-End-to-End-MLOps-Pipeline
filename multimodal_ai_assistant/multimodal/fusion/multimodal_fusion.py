"""multimodal_fusion.py module"""

"""
Multi-Modal AI Assistant - Multi-Modal Fusion Component
Combines vision and text features for unified understanding
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Union, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from pathlib import Path
import json
from PIL import Image
import clip
from transformers import AutoTokenizer, AutoModel

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import our custom components
try:
    from multimodal.vision.vision_encoder import VisionEncoder, VisionConfig
    from multimodal.nlp.language_model import LanguageModel, LanguageConfig
except ImportError:
    # Fallback for testing
    print("Warning: Could not import custom components. Using mock classes for testing.")
    
    class VisionEncoder:
        def __init__(self, config=None):
            self.config = config
        def encode_image(self, image):
            return np.random.rand(512)
    
    class LanguageModel:
        def __init__(self, config=None):
            self.config = config
        def encode_text(self, text):
            return np.random.rand(1, 384)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FusionConfig:
    """Configuration for Multi-Modal Fusion"""
    vision_model: str = "ViT-B/32"
    text_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    fusion_method: str = "attention"  # "concat", "attention", "cross_attention", "transformer"
    hidden_dim: int = 512
    output_dim: int = 256
    dropout: float = 0.1
    num_attention_heads: int = 8
    temperature: float = 0.07  # For contrastive learning
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cache_fused_features: bool = True
    cache_dir: str = "cache/fusion_features"

class AttentionFusion(nn.Module):
    """Attention-based fusion module"""
    
    def __init__(self, vision_dim: int, text_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, vision_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        # Project to common space
        v_proj = self.vision_proj(vision_features)  # [batch, hidden_dim]
        t_proj = self.text_proj(text_features)      # [batch, hidden_dim]
        
        # Add sequence dimension for attention
        v_proj = v_proj.unsqueeze(1)  # [batch, 1, hidden_dim]
        t_proj = t_proj.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Cross attention
        v_attended, _ = self.attention(v_proj, t_proj, t_proj)  # Vision attending to text
        t_attended, _ = self.attention(t_proj, v_proj, v_proj)  # Text attending to vision
        
        # Remove sequence dimension
        v_attended = v_attended.squeeze(1)
        t_attended = t_attended.squeeze(1)
        
        # Concatenate and fuse
        fused = torch.cat([v_attended, t_attended], dim=-1)
        output = self.fusion_layer(fused)
        
        return output

class CrossModalTransformer(nn.Module):
    """Cross-modal transformer for deep fusion"""
    
    def __init__(self, vision_dim: int, text_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Positional embeddings
        self.vision_pos_emb = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.text_pos_emb = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, vision_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        batch_size = vision_features.size(0)
        
        # Project features
        v_proj = self.vision_proj(vision_features).unsqueeze(1)  # [batch, 1, hidden_dim]
        t_proj = self.text_proj(text_features).unsqueeze(1)      # [batch, 1, hidden_dim]
        
        # Add positional embeddings
        v_proj += self.vision_pos_emb
        t_proj += self.text_pos_emb
        
        # Concatenate modalities
        combined = torch.cat([v_proj, t_proj], dim=1)  # [batch, 2, hidden_dim]
        
        # Apply transformer
        transformed = self.transformer(combined)  # [batch, 2, hidden_dim]
        
        # Global average pooling
        pooled = transformed.mean(dim=1)  # [batch, hidden_dim]
        
        # Output projection
        output = self.output_proj(pooled)
        
        return output

class MultiModalFusion:
    """
    Multi-Modal Fusion component that combines vision and text features
    for unified understanding and reasoning
    """
    
    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()
        self.device = torch.device(self.config.device)
        
        # Initialize vision and text encoders
        vision_config = VisionConfig(model_name=self.config.vision_model, device=self.config.device)
        text_config = LanguageConfig(model_name=self.config.text_model, device=self.config.device)
        
        self.vision_encoder = VisionEncoder(vision_config)
        self.text_encoder = LanguageModel(text_config)
        
        # Initialize fusion module
        self.fusion_model = None
        self._setup_fusion_model()
        
        # Create cache directory
        os.makedirs(self.config.cache_dir, exist_ok=True)
        
        logger.info(f"Initialized Multi-Modal Fusion with method: {self.config.fusion_method}")
    
    def _setup_fusion_model(self):
        """Setup the fusion model based on configuration"""
        try:
            # Get feature dimensions
            vision_dim = 512  # CLIP/ViT standard
            text_dim = 384    # Sentence transformer standard
            
            if self.config.fusion_method == "attention":
                self.fusion_model = AttentionFusion(
                    vision_dim=vision_dim,
                    text_dim=text_dim,
                    hidden_dim=self.config.hidden_dim,
                    output_dim=self.config.output_dim
                )
            elif self.config.fusion_method == "transformer":
                self.fusion_model = CrossModalTransformer(
                    vision_dim=vision_dim,
                    text_dim=text_dim,
                    hidden_dim=self.config.hidden_dim,
                    output_dim=self.config.output_dim
                )
            elif self.config.fusion_method == "concat":
                # Simple concatenation + MLP
                total_dim = vision_dim + text_dim
                self.fusion_model = nn.Sequential(
                    nn.Linear(total_dim, self.config.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.config.dropout),
                    nn.Linear(self.config.hidden_dim, self.config.output_dim)
                )
            else:
                raise ValueError(f"Unknown fusion method: {self.config.fusion_method}")
            
            self.fusion_model.to(self.device)
            logger.info(f"Fusion model setup complete: {self.config.fusion_method}")
            
        except Exception as e:
            logger.error(f"Error setting up fusion model: {e}")
            raise
    
    def encode_multimodal(self, image: Union[str, Image.Image, np.ndarray], 
                         text: str, return_individual: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Encode image-text pair into unified representation
        
        Args:
            image: Input image
            text: Input text
            return_individual: Whether to return individual modality features
            
        Returns:
            Fused features or tuple of (fused, vision, text) features
        """
        try:
            # Get individual modality features
            vision_features = self.vision_encoder.encode_image(image)
            text_features = self.text_encoder.encode_text([text])[0]
            
            # Convert to tensors
            vision_tensor = torch.from_numpy(vision_features).float().unsqueeze(0).to(self.device)
            text_tensor = torch.from_numpy(text_features).float().unsqueeze(0).to(self.device)
            
            # Fuse features
            with torch.no_grad():
                if self.config.fusion_method == "concat":
                    combined_features = torch.cat([vision_tensor, text_tensor], dim=-1)
                    fused_features = self.fusion_model(combined_features)
                else:
                    fused_features = self.fusion_model(vision_tensor, text_tensor)
            
            fused_np = fused_features.cpu().numpy().flatten()
            
            if return_individual:
                return fused_np, vision_features, text_features
            return fused_np
            
        except Exception as e:
            logger.error(f"Error in multimodal encoding: {e}")
            raise
    
    def similarity_search_multimodal(self, query_image: Union[str, Image.Image, np.ndarray],
                                   query_text: str,
                                   candidate_pairs: List[Tuple[Union[str, Image.Image], str]],
                                   top_k: int = 5) -> List[Tuple[int, float, Tuple]]:
        """
        Search for most similar image-text pairs
        
        Args:
            query_image: Query image
            query_text: Query text
            candidate_pairs: List of (image, text) pairs to search
            top_k: Number of top results
            
        Returns:
            List of (index, similarity_score, (image, text)) tuples
        """
        try:
            # Encode query
            query_features = self.encode_multimodal(query_image, query_text)
            
            # Encode all candidates
            candidate_features = []
            for image, text in candidate_pairs:
                features = self.encode_multimodal(image, text)
                candidate_features.append(features)
            
            candidate_features = np.array(candidate_features)
            
            # Calculate similarities
            similarities = np.dot(candidate_features, query_features) / (
                np.linalg.norm(candidate_features, axis=1) * np.linalg.norm(query_features)
            )
            
            # Get top-k results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                results.append((int(idx), float(similarities[idx]), candidate_pairs[idx]))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in multimodal similarity search: {e}")
            raise
    
    def visual_question_answering(self, image: Union[str, Image.Image, np.ndarray], 
                                question: str) -> Dict[str, Any]:
        """
        Answer questions about images
        
        Args:
            image: Input image
            question: Question about the image
            
        Returns:
            Dictionary with answer and confidence
        """
        try:
            # Get multimodal features
            fused_features, vision_features, text_features = self.encode_multimodal(
                image, question, return_individual=True
            )
            
            # For now, use similarity-based approach
            # In a full implementation, you'd train a QA head on the fused features
            
            # Generate contextual response using text model
            # Create a simple prompt for VQA
            context_prompt = f"Based on the image content and question '{question}', provide a concise answer:"
            
            try:
                answer = self.text_encoder.generate_response(context_prompt, max_length=50)
            except:
                # Fallback answer based on feature analysis
                answer = "I can see the image and understand your question, but cannot provide a specific answer at the moment."
            
            return {
                "answer": answer,
                "confidence": 0.7,  # Placeholder confidence
                "fused_features": fused_features,
                "vision_features": vision_features,
                "text_features": text_features
            }
            
        except Exception as e:
            logger.error(f"Error in VQA: {e}")
            raise
    
    def image_text_matching(self, image: Union[str, Image.Image, np.ndarray], 
                          text: str) -> float:
        """
        Calculate how well an image matches given text
        
        Args:
            image: Input image
            text: Text description
            
        Returns:
            Matching score (0-1)
        """
        try:
            # Get individual features
            vision_features = self.vision_encoder.encode_image(image)
            text_features = self.text_encoder.encode_text([text])[0]
            
            # Calculate cosine similarity
            similarity = np.dot(vision_features, text_features) / (
                np.linalg.norm(vision_features) * np.linalg.norm(text_features)
            )
            
            # Normalize to 0-1 range
            normalized_score = (similarity + 1) / 2
            
            return float(normalized_score)
            
        except Exception as e:
            logger.error(f"Error in image-text matching: {e}")
            raise
    
    def cross_modal_retrieval(self, query: Union[str, Union[str, Image.Image, np.ndarray]], 
                            database: List[Tuple[Union[str, Image.Image], str]],
                            query_type: str = "text", top_k: int = 5) -> List[Tuple[int, float, Tuple]]:
        """
        Cross-modal retrieval: text->image or image->text
        
        Args:
            query: Query (text or image)
            database: Database of (image, text) pairs
            query_type: "text" or "image"
            top_k: Number of results
            
        Returns:
            List of (index, score, (image, text)) tuples
        """
        try:
            if query_type == "text":
                # Text query, find matching images
                query_features = self.text_encoder.encode_text([query])[0]
                
                # Compare with vision features of database
                scores = []
                for i, (image, text) in enumerate(database):
                    image_features = self.vision_encoder.encode_image(image)
                    score = np.dot(query_features, image_features) / (
                        np.linalg.norm(query_features) * np.linalg.norm(image_features)
                    )
                    scores.append((i, score, (image, text)))
                
            else:  # query_type == "image"
                # Image query, find matching texts
                query_features = self.vision_encoder.encode_image(query)
                
                # Compare with text features of database
                scores = []
                for i, (image, text) in enumerate(database):
                    text_features = self.text_encoder.encode_text([text])[0]
                    score = np.dot(query_features, text_features) / (
                        np.linalg.norm(query_features) * np.linalg.norm(text_features)
                    )
                    scores.append((i, score, (image, text)))
            
            # Sort by score and return top-k
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:top_k]
            
        except Exception as e:
            logger.error(f"Error in cross-modal retrieval: {e}")
            raise
    
    def get_fusion_info(self) -> Dict[str, Any]:
        """Get fusion model information"""
        return {
            "fusion_method": self.config.fusion_method,
            "vision_model": self.config.vision_model,
            "text_model": self.config.text_model,
            "hidden_dim": self.config.hidden_dim,
            "output_dim": self.config.output_dim,
            "device": str(self.device),
            "num_parameters": sum(p.numel() for p in self.fusion_model.parameters()) if self.fusion_model else 0
        }

# Test function
def test_multimodal_fusion():
    """Test the Multi-Modal Fusion component"""
    print("Testing Multi-Modal Fusion...")
    
    # Initialize fusion model
    config = FusionConfig(
        fusion_method="attention",
        device="cpu",  # Use CPU for testing
        hidden_dim=256,
        output_dim=128
    )
    
    try:
        fusion = MultiModalFusion(config)
        
        # Print model info
        print("Fusion Info:", fusion.get_fusion_info())
        
        # Create dummy data for testing
        dummy_image = Image.new('RGB', (224, 224), color='blue')
        sample_text = "A blue colored image with simple content"
        
        # Test multimodal encoding
        fused_features = fusion.encode_multimodal(dummy_image, sample_text)
        print(f"Fused features shape: {fused_features.shape}")
        
        # Test with individual features
        fused, vision, text = fusion.encode_multimodal(dummy_image, sample_text, return_individual=True)
        print(f"Individual features - Vision: {vision.shape}, Text: {text.shape}, Fused: {fused.shape}")
        
        # Test image-text matching
        match_score = fusion.image_text_matching(dummy_image, sample_text)
        print(f"Image-text match score: {match_score:.3f}")
        
        # Test VQA
        vqa_result = fusion.visual_question_answering(dummy_image, "What color is this image?")
        print(f"VQA Answer: {vqa_result['answer']}")
        print(f"VQA Confidence: {vqa_result['confidence']}")
        
        # Test similarity search
        candidate_pairs = [
            (dummy_image, "blue image"),
            (dummy_image, "red image"),
            (dummy_image, "colorful picture")
        ]
        
        search_results = fusion.similarity_search_multimodal(
            dummy_image, "blue colored image", candidate_pairs, top_k=2
        )
        print("Similarity search results:")
        for idx, score, (img, txt) in search_results:
            print(f"  {idx}: {score:.3f} - {txt}")
        
        print(" Multi-Modal Fusion test passed!")
        
    except Exception as e:
        print(f" Multi-Modal Fusion test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_multimodal_fusion()
