"""language_model.py module"""

"""
Multi-Modal AI Assistant - NLP Component
Language Model for text processing using OpenAI/HuggingFace models
"""

import os
import sys
import torch
import openai
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM,
    GPT2LMHeadModel, GPT2Tokenizer,
    BertModel, BertTokenizer,
    T5ForConditionalGeneration, T5Tokenizer,
    pipeline
)
import numpy as np
from typing import List, Dict, Union, Optional, Tuple
import logging
from dataclasses import dataclass
from pathlib import Path
import json
import time
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LanguageConfig:
    """Configuration for Language Model"""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"  # Default embedding model
    generative_model: str = "gpt2"  # For text generation
    openai_model: str = "gpt-3.5-turbo"  # OpenAI model
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_length: int = 512
    batch_size: int = 16
    temperature: float = 0.7
    use_openai: bool = False
    openai_api_key: Optional[str] = None
    cache_embeddings: bool = True
    cache_dir: str = "cache/text_embeddings"

class LanguageModel:
    """
    Language Model component for text processing, embedding generation, and response generation
    Supports both local models (HuggingFace) and OpenAI API
    """
    
    def __init__(self, config: Optional[LanguageConfig] = None):
        self.config = config or LanguageConfig()
        self.device = torch.device(self.config.device)
        
        # Initialize models
        self.embedding_model = None
        self.generative_model = None
        self.tokenizer = None
        
        # OpenAI setup
        if self.config.use_openai and self.config.openai_api_key:
            openai.api_key = self.config.openai_api_key
        
        # Create cache directory
        os.makedirs(self.config.cache_dir, exist_ok=True)
        
        logger.info(f"Initializing Language Model with device: {self.device}")
        self._load_models()
    
    def _load_models(self):
        """Load embedding and generative models"""
        try:
            # Load embedding model
            logger.info(f"Loading embedding model: {self.config.model_name}")
            if "sentence-transformers" in self.config.model_name:
                self.embedding_model = SentenceTransformer(self.config.model_name, device=self.device)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
                self.embedding_model = AutoModel.from_pretrained(self.config.model_name)
                self.embedding_model.to(self.device)
            
            # Load generative model (if not using OpenAI)
            if not self.config.use_openai:
                logger.info(f"Loading generative model: {self.config.generative_model}")
                if self.config.generative_model.startswith("gpt2"):
                    self.generative_tokenizer = GPT2Tokenizer.from_pretrained(self.config.generative_model)
                    self.generative_model = GPT2LMHeadModel.from_pretrained(self.config.generative_model)
                    self.generative_tokenizer.pad_token = self.generative_tokenizer.eos_token
                elif self.config.generative_model.startswith("t5"):
                    self.generative_tokenizer = T5Tokenizer.from_pretrained(self.config.generative_model)
                    self.generative_model = T5ForConditionalGeneration.from_pretrained(self.config.generative_model)
                else:
                    self.generative_tokenizer = AutoTokenizer.from_pretrained(self.config.generative_model)
                    self.generative_model = AutoModelForCausalLM.from_pretrained(self.config.generative_model)
                
                self.generative_model.to(self.device)
            
            logger.info("Models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text to embeddings
        
        Args:
            text: Single text or list of texts
            
        Returns:
            Text embeddings as numpy array
        """
        try:
            if isinstance(text, str):
                text = [text]
            
            # Check cache
            if self.config.cache_embeddings:
                cached_results = []
                uncached_texts = []
                uncached_indices = []
                
                for i, t in enumerate(text):
                    cache_key = self._get_cache_key(t)
                    cached_embedding = self._load_from_cache(cache_key)
                    if cached_embedding is not None:
                        cached_results.append((i, cached_embedding))
                    else:
                        uncached_texts.append(t)
                        uncached_indices.append(i)
                
                if not uncached_texts:
                    # All results were cached
                    embeddings = np.zeros((len(text), cached_results[0][1].shape[0]))
                    for idx, embedding in cached_results:
                        embeddings[idx] = embedding
                    return embeddings
            else:
                uncached_texts = text
                uncached_indices = list(range(len(text)))
            
            # Generate embeddings for uncached texts
            if hasattr(self.embedding_model, 'encode'):
                # SentenceTransformer model
                new_embeddings = self.embedding_model.encode(uncached_texts)
            else:
                # HuggingFace model
                new_embeddings = []
                for batch_start in range(0, len(uncached_texts), self.config.batch_size):
                    batch = uncached_texts[batch_start:batch_start + self.config.batch_size]
                    
                    inputs = self.tokenizer(
                        batch, 
                        padding=True, 
                        truncation=True, 
                        max_length=self.config.max_length,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.embedding_model(**inputs)
                        # Use mean pooling
                        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                        new_embeddings.extend(embeddings)
                
                new_embeddings = np.array(new_embeddings)
            
            # Combine cached and new embeddings
            if self.config.cache_embeddings and cached_results:
                all_embeddings = np.zeros((len(text), new_embeddings.shape[1]))
                
                # Place cached results
                for idx, embedding in cached_results:
                    all_embeddings[idx] = embedding
                
                # Place new results and cache them
                for i, (original_idx, new_embedding) in enumerate(zip(uncached_indices, new_embeddings)):
                    all_embeddings[original_idx] = new_embedding
                    cache_key = self._get_cache_key(uncached_texts[i])
                    self._save_to_cache(cache_key, new_embedding)
                
                return all_embeddings
            else:
                # Cache new embeddings
                if self.config.cache_embeddings:
                    for i, embedding in enumerate(new_embeddings):
                        cache_key = self._get_cache_key(uncached_texts[i])
                        self._save_to_cache(cache_key, embedding)
                
                return new_embeddings
            
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            raise
    
    def generate_response(self, prompt: str, max_length: Optional[int] = None, 
                         temperature: Optional[float] = None) -> str:
        """
        Generate text response
        
        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        try:
            max_length = max_length or self.config.max_length
            temperature = temperature or self.config.temperature
            
            if self.config.use_openai:
                # Use OpenAI API
                response = openai.ChatCompletion.create(
                    model=self.config.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_length,
                    temperature=temperature
                )
                return response.choices[0].message.content.strip()
            
            else:
                # Use local model
                inputs = self.generative_tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.generative_model.generate(
                        inputs,
                        max_length=len(inputs[0]) + max_length,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=self.generative_tokenizer.eos_token_id,
                        num_return_sequences=1
                    )
                
                response = self.generative_tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Remove the original prompt from response
                response = response[len(prompt):].strip()
                return response
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def question_answering(self, context: str, question: str) -> Dict[str, Union[str, float]]:
        """
        Answer question based on context using QA pipeline
        
        Args:
            context: Context text
            question: Question to answer
            
        Returns:
            Dictionary with answer and confidence score
        """
        try:
            # Create QA pipeline
            qa_pipeline = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad",
                tokenizer="distilbert-base-cased-distilled-squad",
                device=0 if self.device.type == "cuda" else -1
            )
            
            result = qa_pipeline(question=question, context=context)
            
            return {
                "answer": result["answer"],
                "confidence": result["score"],
                "start": result["start"],
                "end": result["end"]
            }
            
        except Exception as e:
            logger.error(f"Error in question answering: {e}")
            raise
    
    def text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        try:
            embeddings = self.encode_text([text1, text2])
            
            # Calculate cosine similarity
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            raise
    
    def semantic_search(self, query: str, corpus: List[str], top_k: int = 5) -> List[Tuple[int, float, str]]:
        """
        Perform semantic search in text corpus
        
        Args:
            query: Search query
            corpus: List of texts to search
            top_k: Number of top results
            
        Returns:
            List of (index, score, text) tuples
        """
        try:
            # Get embeddings
            query_embedding = self.encode_text([query])[0]
            corpus_embeddings = self.encode_text(corpus)
            
            # Calculate similarities
            similarities = np.dot(corpus_embeddings, query_embedding) / (
                np.linalg.norm(corpus_embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            
            # Get top-k results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                results.append((int(idx), float(similarities[idx]), corpus[idx]))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            raise
    
    def summarize_text(self, text: str, max_length: int = 150) -> str:
        """
        Summarize text using T5 or other summarization model
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            
        Returns:
            Summary text
        """
        try:
            summarizer = pipeline(
                "summarization",
                model="t5-small",
                tokenizer="t5-small",
                device=0 if self.device.type == "cuda" else -1
            )
            
            # Split text if too long
            max_chunk_length = 512
            if len(text) > max_chunk_length:
                # Simple chunking - could be improved
                chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
                summaries = []
                
                for chunk in chunks:
                    if len(chunk.strip()) > 50:  # Only summarize substantial chunks
                        summary = summarizer(chunk, max_length=max_length//len(chunks), min_length=20)[0]['summary_text']
                        summaries.append(summary)
                
                return " ".join(summaries)
            else:
                return summarizer(text, max_length=max_length, min_length=50)[0]['summary_text']
                
        except Exception as e:
            logger.error(f"Error summarizing text: {e}")
            # Fallback: return first few sentences
            sentences = text.split('.')[:3]
            return '.'.join(sentences) + '.'
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return f"{hash(text)}_{self.config.model_name.replace('/', '_')}"
    
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
    
    def get_model_info(self) -> Dict[str, str]:
        """Get model information"""
        return {
            "embedding_model": self.config.model_name,
            "generative_model": self.config.generative_model if not self.config.use_openai else self.config.openai_model,
            "device": str(self.device),
            "max_length": self.config.max_length,
            "use_openai": self.config.use_openai
        }

# Test function
def test_language_model():
    """Test the Language Model component"""
    print("Testing Language Model...")
    
    # Initialize model
    config = LanguageConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        generative_model="gpt2",
        device="cpu",  # Use CPU for testing
        use_openai=False
    )
    
    model = LanguageModel(config)
    
    # Print model info
    print("Model Info:", model.get_model_info())
    
    try:
        # Test text encoding
        sample_texts = [
            "The cat is sitting on the mat.",
            "A feline is resting on a rug.",
            "The dog is running in the park."
        ]
        
        embeddings = model.encode_text(sample_texts)
        print(f"Generated embeddings shape: {embeddings.shape}")
        
        # Test similarity
        similarity = model.text_similarity(sample_texts[0], sample_texts[1])
        print(f"Similarity between text 1 and 2: {similarity:.3f}")
        
        # Test semantic search
        query = "cat on mat"
        results = model.semantic_search(query, sample_texts, top_k=2)
        print(f"Search results for '{query}':")
        for idx, score, text in results:
            print(f"  {idx}: {score:.3f} - {text[:50]}...")
        
        # Test text generation
        response = model.generate_response("The weather today is", max_length=20)
        print(f"Generated response: {response}")
        
        # Test summarization
        long_text = "This is a long text that needs to be summarized. " * 10
        summary = model.summarize_text(long_text, max_length=50)
        print(f"Summary: {summary}")
        
        print(" Language Model test passed!")
        
    except Exception as e:
        print(f" Language Model test failed: {e}")

if __name__ == "__main__":
    test_language_model()
