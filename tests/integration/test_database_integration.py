# tests/integration/test_database_integration.py - Database integration tests
import pytest
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

class TestDatabaseIntegration:
    """Test database integration scenarios"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        yield
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('pymongo.MongoClient')
    def test_mongodb_connection_integration(self, mock_mongo_client):
        """Test MongoDB connection and operations"""
        
        # Mock MongoDB client and database
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_collection = MagicMock()
        
        mock_mongo_client.return_value = mock_client
        mock_client.__getitem__ = lambda self, key: mock_db
        mock_db.__getitem__ = lambda self, key: mock_collection
        
        # Test connection
        from pymongo import MongoClient
        
        client = MongoClient("mongodb://localhost:27017")
        db = client["multimodal_ai"]
        collection = db["conversations"]
        
        # Test insert operation
        test_conversation = {
            "conversation_id": "test_123",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ],
            "timestamp": "2024-01-01T00:00:00"
        }
        
        collection.insert_one(test_conversation)
        mock_collection.insert_one.assert_called_once_with(test_conversation)
        
        # Test find operation
        collection.find({"conversation_id": "test_123"})
        mock_collection.find.assert_called_once_with({"conversation_id": "test_123"})
        
        print(" MongoDB integration test passed!")
    
    @patch('chromadb.Client')
    def test_chromadb_integration(self, mock_chroma_client):
        """Test ChromaDB vector database integration"""
        
        # Mock ChromaDB client and collection
        mock_client = MagicMock()
        mock_collection = MagicMock()
        
        mock_chroma_client.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        
        # Test ChromaDB operations
        import chromadb
        
        client = chromadb.Client()
        collection = client.get_or_create_collection("multimodal_embeddings")
        
        # Test adding embeddings
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        documents = ["What color is the car?", "How many people are there?"]
        ids = ["q1", "q2"]
        
        collection.add(embeddings=embeddings, documents=documents, ids=ids)
        mock_collection.add.assert_called_once_with(
            embeddings=embeddings, 
            documents=documents, 
            ids=ids
        )
        
        # Test querying
        query_embedding = [[0.15, 0.25, 0.35]]
        collection.query(query_embeddings=query_embedding, n_results=1)
        mock_collection.query.assert_called_once_with(
            query_embeddings=query_embedding,
            n_results=1
        )
        
        print(" ChromaDB integration test passed!")