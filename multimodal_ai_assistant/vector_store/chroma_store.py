"""chroma_store.py module"""

"""
Multi-Modal AI Assistant - RAG System with ChromaDB
Vector store for storing and retrieving image-text pairs
"""

import os
import sys
import uuid
import json
import numpy as np
from typing import List, Dict, Union, Optional, Tuple, Any
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import base64
from PIL import Image
import io
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChromaConfig:
    """Configuration for ChromaDB Vector Store"""
    persist_directory: str = "data/chroma_db"
    collection_name: str = "multimodal_collection"
    embedding_function: str = "default"  # "default", "openai", "sentence_transformers"
    distance_metric: str = "cosine"  # "cosine", "l2", "ip"
    max_batch_size: int = 100
    openai_api_key: Optional[str] = None
    sentence_transformers_model: str = "all-MiniLM-L6-v2"

@dataclass
class MultiModalDocument:
    """Represents a multimodal document"""
    id: str
    text: str
    image_path: Optional[str] = None
    image_base64: Optional[str] = None
    metadata: Dict[str, Any] = None
    embedding: Optional[List[float]] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.metadata is None:
            self.metadata = {}
        if self.id is None:
            self.id = str(uuid.uuid4())

class ChromaStore:
    """
    ChromaDB-based vector store for multimodal RAG system
    Supports storing and retrieving image-text pairs with embeddings
    """
    
    def __init__(self, config: Optional[ChromaConfig] = None):
        self.config = config or ChromaConfig()
        self.client = None
        self.collection = None
        self.embedding_function = None
        
        # Initialize ChromaDB
        self._initialize_chroma()
        
        logger.info(f"ChromaDB Vector Store initialized with collection: {self.config.collection_name}")
    
    def _initialize_chroma(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Create persist directory
            os.makedirs(self.config.persist_directory, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.config.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Setup embedding function
            self._setup_embedding_function()
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=self.config.collection_name,
                    embedding_function=self.embedding_function
                )
                logger.info(f"Loaded existing collection: {self.config.collection_name}")
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.config.collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"hnsw:space": self.config.distance_metric}
                )
                logger.info(f"Created new collection: {self.config.collection_name}")
                
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            raise
    
    def _setup_embedding_function(self):
        """Setup the embedding function"""
        try:
            if self.config.embedding_function == "openai":
                if not self.config.openai_api_key:
                    raise ValueError("OpenAI API key required for OpenAI embeddings")
                self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=self.config.openai_api_key,
                    model_name="text-embedding-ada-002"
                )
            elif self.config.embedding_function == "sentence_transformers":
                self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=self.config.sentence_transformers_model
                )
            else:
                # Use ChromaDB's default embedding function
                self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
            
            logger.info(f"Embedding function setup: {self.config.embedding_function}")
            
        except Exception as e:
            logger.error(f"Error setting up embedding function: {e}")
            raise
    
    def _image_to_base64(self, image: Union[str, Image.Image, np.ndarray]) -> str:
        """Convert image to base64 string for storage"""
        try:
            if isinstance(image, str):
                # Image path
                with open(image, "rb") as f:
                    return base64.b64encode(f.read()).decode('utf-8')
            elif isinstance(image, Image.Image):
                # PIL Image
                buffer = io.BytesIO()
                image.save(buffer, format='PNG')
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
            elif isinstance(image, np.ndarray):
                # Numpy array
                image_pil = Image.fromarray(image)
                buffer = io.BytesIO()
                image_pil.save(buffer, format='PNG')
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
                
        except Exception as e:
            logger.error(f"Error converting image to base64: {e}")
            raise
    
    def _base64_to_image(self, base64_str: str) -> Image.Image:
        """Convert base64 string back to PIL Image"""
        try:
            image_data = base64.b64decode(base64_str)
            return Image.open(io.BytesIO(image_data))
        except Exception as e:
            logger.error(f"Error converting base64 to image: {e}")
            raise
    
    def add_document(self, document: MultiModalDocument) -> str:
        """
        Add a single multimodal document to the vector store
        
        Args:
            document: MultiModalDocument to add
            
        Returns:
            Document ID
        """
        try:
            # Prepare document data
            doc_text = document.text
            doc_id = document.id
            metadata = document.metadata.copy()
            
            # Add image data to metadata if present
            if document.image_path:
                metadata["image_path"] = document.image_path
                metadata["has_image"] = True
                # Convert image to base64 for storage
                document.image_base64 = self._image_to_base64(document.image_path)
            elif document.image_base64:
                metadata["has_image"] = True
            else:
                metadata["has_image"] = False
            
            # Add image base64 to metadata (ChromaDB can handle large metadata)
            if document.image_base64:
                metadata["image_base64"] = document.image_base64
            
            metadata["timestamp"] = document.timestamp
            metadata["doc_type"] = "multimodal" if metadata.get("has_image") else "text"
            
            # Add to collection
            self.collection.add(
                documents=[doc_text],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            logger.info(f"Added document {doc_id} to collection")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            raise
    
    def add_documents_batch(self, documents: List[MultiModalDocument]) -> List[str]:
        """
        Add multiple documents in batch
        
        Args:
            documents: List of MultiModalDocument objects
            
        Returns:
            List of document IDs
        """
        try:
            doc_ids = []
            doc_texts = []
            doc_metadatas = []
            
            for doc in documents:
                # Prepare document data
                doc_text = doc.text
                doc_id = doc.id
                metadata = doc.metadata.copy()
                
                # Handle images
                if doc.image_path:
                    metadata["image_path"] = doc.image_path
                    metadata["has_image"] = True
                    doc.image_base64 = self._image_to_base64(doc.image_path)
                elif doc.image_base64:
                    metadata["has_image"] = True
                else:
                    metadata["has_image"] = False
                
                if doc.image_base64:
                    metadata["image_base64"] = doc.image_base64
                
                metadata["timestamp"] = doc.timestamp
                metadata["doc_type"] = "multimodal" if metadata.get("has_image") else "text"
                
                doc_ids.append(doc_id)
                doc_texts.append(doc_text)
                doc_metadatas.append(metadata)
            
            # Process in batches
            all_ids = []
            for i in range(0, len(documents), self.config.max_batch_size):
                batch_end = min(i + self.config.max_batch_size, len(documents))
                
                batch_ids = doc_ids[i:batch_end]
                batch_texts = doc_texts[i:batch_end]
                batch_metadata = doc_metadatas[i:batch_end]
                
                self.collection.add(
                    documents=batch_texts,
                    metadatas=batch_metadata,
                    ids=batch_ids
                )
                
                all_ids.extend(batch_ids)
                logger.info(f"Added batch {i//self.config.max_batch_size + 1} with {len(batch_ids)} documents")
            
            logger.info(f"Added {len(all_ids)} documents to collection in total")
            return all_ids
            
        except Exception as e:
            logger.error(f"Error adding documents batch: {e}")
            raise
    
    def query_documents(self, query_text: str, n_results: int = 5, 
                       where: Optional[Dict] = None, 
                       include_images: bool = True) -> Dict[str, Any]:
        """
        Query documents using text similarity
        
        Args:
            query_text: Text query
            n_results: Number of results to return
            where: Metadata filter conditions
            include_images: Whether to include image data in results
            
        Returns:
            Query results with documents, metadata, and distances
        """
        try:
            # Set what to include in results
            include = ["documents", "metadatas", "distances"]
            
            # Query the collection
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where,
                include=include
            )
            
            # Process results
            processed_results = {
                "query": query_text,
                "n_results": len(results["documents"][0]),
                "documents": [],
                "metadata": results["metadatas"][0],
                "distances": results["distances"][0],
                "ids": results["ids"][0] if "ids" in results else []
            }
            
            # Process each result
            for i, (doc_text, metadata, distance) in enumerate(zip(
                results["documents"][0], 
                results["metadatas"][0], 
                results["distances"][0]
            )):
                result_item = {
                    "id": results["ids"][0][i] if "ids" in results else f"result_{i}",
                    "text": doc_text,
                    "distance": distance,
                    "score": 1 - distance,  # Convert distance to similarity score
                    "metadata": metadata.copy(),
                    "image": None
                }
                
                # Include image if requested and available
                if include_images and metadata.get("has_image") and metadata.get("image_base64"):
                    try:
                        result_item["image"] = self._base64_to_image(metadata["image_base64"])
                        # Remove base64 from metadata to reduce size
                        result_item["metadata"].pop("image_base64", None)
                    except Exception as e:
                        logger.warning(f"Error loading image for result {i}: {e}")
                
                processed_results["documents"].append(result_item)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error querying documents: {e}")
            raise
    
    def query_by_image_description(self, image_description: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Query documents by image description (find images matching description)
        
        Args:
            image_description: Description of desired image
            n_results: Number of results
            
        Returns:
            Query results filtered for documents with images
        """
        try:
            # Query with filter for documents that have images
            where_filter = {"has_image": True}
            
            results = self.query_documents(
                query_text=image_description,
                n_results=n_results * 2,  # Get more to account for filtering
                where=where_filter,
                include_images=True
            )
            
            # Further filter and limit results
            filtered_results = {
                "query": image_description,
                "query_type": "image_description",
                "documents": results["documents"][:n_results]
            }
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error querying by image description: {e}")
            raise
    
    def hybrid_query(self, text_query: str, image_query: Optional[Union[str, Image.Image]] = None,
                    n_results: int = 5, text_weight: float = 0.7) -> Dict[str, Any]:
        """
        Perform hybrid text and image-based query
        
        Args:
            text_query: Text part of query
            image_query: Image part of query (path or PIL Image)
            n_results: Number of results
            text_weight: Weight for text vs image similarity (0-1)
            
        Returns:
            Hybrid query results
        """
        try:
            # For now, implement as text-based query
            # In a full implementation, you'd combine text and image embeddings
            
            if image_query:
                # Enhance text query with image-based description
                enhanced_query = f"{text_query} visual content image"
                logger.info(f"Enhanced query with image context: {enhanced_query}")
            else:
                enhanced_query = text_query
            
            results = self.query_documents(
                query_text=enhanced_query,
                n_results=n_results,
                include_images=True
            )
            
            results["query_type"] = "hybrid"
            results["text_weight"] = text_weight
            results["has_image_query"] = image_query is not None
            
            return results
            
        except Exception as e:
            logger.error(f"Error in hybrid query: {e}")
            raise
    
    def get_document_by_id(self, doc_id: str, include_image: bool = True) -> Optional[MultiModalDocument]:
        """
        Retrieve a specific document by ID
        
        Args:
            doc_id: Document ID
            include_image: Whether to include image data
            
        Returns:
            MultiModalDocument or None if not found
        """
        try:
            results = self.collection.get(
                ids=[doc_id],
                include=["documents", "metadatas"]
            )
            
            if not results["documents"]:
                return None
            
            doc_text = results["documents"][0]
            metadata = results["metadatas"][0]
            
            # Create document object
            doc = MultiModalDocument(
                id=doc_id,
                text=doc_text,
                metadata=metadata.copy(),
                timestamp=metadata.get("timestamp")
            )
            
            # Add image data if available
            if include_image and metadata.get("has_image") and metadata.get("image_base64"):
                doc.image_base64 = metadata["image_base64"]
                doc.image_path = metadata.get("image_path")
            
            return doc
            
        except Exception as e:
            logger.error(f"Error getting document by ID: {e}")
            return None
    
    def update_document(self, doc_id: str, updated_document: MultiModalDocument) -> bool:
        """
        Update an existing document
        
        Args:
            doc_id: ID of document to update
            updated_document: Updated document data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete existing document
            self.collection.delete(ids=[doc_id])
            
            # Add updated document with same ID
            updated_document.id = doc_id
            self.add_document(updated_document)
            
            logger.info(f"Updated document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating document: {e}")
            return False
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document by ID
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.collection.delete(ids=[doc_id])
            logger.info(f"Deleted document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            
            # Get sample of metadata to analyze
            sample_results = self.collection.get(
                include=["metadatas"],
                limit=min(100, count) if count > 0 else 0
            )
            
            stats = {
                "total_documents": count,
                "collection_name": self.config.collection_name,
                "embedding_function": self.config.embedding_function,
                "distance_metric": self.config.distance_metric
            }
            
            if sample_results["metadatas"]:
                # Analyze document types
                doc_types = {}
                has_image_count = 0
                
                for metadata in sample_results["metadatas"]:
                    doc_type = metadata.get("doc_type", "unknown")
                    doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                    
                    if metadata.get("has_image"):
                        has_image_count += 1
                
                stats.update({
                    "document_types": doc_types,
                    "documents_with_images": has_image_count,
                    "sample_size": len(sample_results["metadatas"])
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection"""
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(self.config.collection_name)
            
            self.collection = self.client.create_collection(
                name=self.config.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": self.config.distance_metric}
            )
            
            logger.info(f"Cleared collection: {self.config.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False
    
    def backup_collection(self, backup_path: str) -> bool:
        """
        Backup collection data to JSON file
        
        Args:
            backup_path: Path to save backup file
            
        Returns:
            True if successful
        """
        try:
            # Get all documents
            all_docs = self.collection.get(include=["documents", "metadatas"])
            
            backup_data = {
                "collection_name": self.config.collection_name,
                "backup_timestamp": datetime.now().isoformat(),
                "config": asdict(self.config),
                "documents": []
            }
            
            # Process documents
            for i, (doc_text, metadata) in enumerate(zip(all_docs["documents"], all_docs["metadatas"])):
                doc_data = {
                    "id": all_docs["ids"][i] if "ids" in all_docs else f"doc_{i}",
                    "text": doc_text,
                    "metadata": metadata
                }
                backup_data["documents"].append(doc_data)
            
            # Save to file
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Backup saved to {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return False

# Test function
def test_chroma_store():
    """Test the ChromaDB Vector Store component"""
    print("Testing ChromaDB Vector Store...")
    
    # Initialize store
    config = ChromaConfig(
        persist_directory="test_chroma_db",
        collection_name="test_collection",
        embedding_function="default"
    )
    
    try:
        store = ChromaStore(config)
        
        # Print collection stats
        stats = store.get_collection_stats()
        print("Collection Stats:", stats)
        
        # Create test documents
        doc1 = MultiModalDocument(
            id="doc1",
            text="A beautiful sunset over the mountains with orange and pink colors",
            metadata={"category": "landscape", "colors": ["orange", "pink"]}
        )
        
        doc2 = MultiModalDocument(
            id="doc2", 
            text="A cute cat sitting on a windowsill looking outside",
            metadata={"category": "animal", "subject": "cat"}
        )
        
        doc3 = MultiModalDocument(
            id="doc3",
            text="Modern city skyline with tall buildings and bright lights",
            metadata={"category": "urban", "time": "night"}
        )
        
        # Test adding single document
        doc_id = store.add_document(doc1)
        print(f"Added document: {doc_id}")
        
        # Test adding batch
        batch_ids = store.add_documents_batch([doc2, doc3])
        print(f"Added batch: {batch_ids}")
        
        # Test querying
        query_results = store.query_documents("sunset mountains", n_results=2)
        print(f"Query results for 'sunset mountains': {len(query_results['documents'])} documents")
        for i, doc in enumerate(query_results["documents"]):
            print(f"  {i+1}. Score: {doc['score']:.3f} - {doc['text'][:50]}...")
        
        # Test getting document by ID
        retrieved_doc = store.get_document_by_id("doc1")
        if retrieved_doc:
            print(f"Retrieved document: {retrieved_doc.text[:50]}...")
        
        # Test collection stats after adding documents
        final_stats = store.get_collection_stats()
        print("Final Collection Stats:", final_stats)
        
        print(" ChromaDB Vector Store test passed!")
        
    except Exception as e:
        print(f" ChromaDB Vector Store test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_chroma_store()
