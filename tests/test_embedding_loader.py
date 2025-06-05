import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from classes.embedding_loader import EmbeddingLoader


class TestEmbeddingLoader:
    
    @patch('classes.embedding_loader.chromadb')
    def test_init(self, mock_chromadb):
        mock_client = Mock()
        mock_collection = Mock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cleaned_text_dir = Path(temp_dir) / "cleaned"
            embeddings_dir = Path(temp_dir) / "embeddings"
            vectordb_dir = Path(temp_dir) / "vectordb"
            
            loader = EmbeddingLoader(
                cleaned_text_file_list=["test.txt"],
                cleaned_text_dir=str(cleaned_text_dir),
                embeddings_dir=str(embeddings_dir),
                vectordb_dir=str(vectordb_dir),
                collection_name="test_collection",
                batch_size=32
            )
            
            assert loader.cleaned_text_file_list == ["test.txt"]
            assert loader.cleaned_text_path == cleaned_text_dir
            assert loader.embeddings_path == embeddings_dir
            assert loader.vectordb_path == vectordb_dir
            assert loader.collection_name == "test_collection"
            assert loader.batch_size == 32
            
            mock_chromadb.PersistentClient.assert_called_once_with(path=str(vectordb_dir))
            mock_client.get_or_create_collection.assert_called_once_with("test_collection")
    
    @patch('classes.embedding_loader.chromadb')
    def test_load_cleaned_text_success(self, mock_chromadb):
        mock_client = Mock()
        mock_collection = Mock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cleaned_text_dir = Path(temp_dir) / "cleaned"
            cleaned_text_dir.mkdir()
            
            test_file = cleaned_text_dir / "test.txt"
            test_content = "  Test content  "
            test_file.write_text(test_content, encoding="utf-8")
            
            loader = EmbeddingLoader(
                cleaned_text_file_list=[],
                cleaned_text_dir=str(cleaned_text_dir),
                embeddings_dir=str(temp_dir),
                vectordb_dir=str(temp_dir),
                collection_name="test"
            )
            
            result = loader._load_cleaned_text(test_file)
            assert result == "Test content"
    
    @patch('classes.embedding_loader.chromadb')
    def test_load_cleaned_text_file_error(self, mock_chromadb):
        mock_client = Mock()
        mock_collection = Mock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = EmbeddingLoader(
                cleaned_text_file_list=[],
                cleaned_text_dir=str(temp_dir),
                embeddings_dir=str(temp_dir),
                vectordb_dir=str(temp_dir),
                collection_name="test"
            )
            
            result = loader._load_cleaned_text(Path("nonexistent.txt"))
            assert result == ""
    
    @patch('classes.embedding_loader.chromadb')
    def test_load_embeddings_success(self, mock_chromadb):
        mock_client = Mock()
        mock_collection = Mock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        
        with tempfile.TemporaryDirectory() as temp_dir:
            embeddings_dir = Path(temp_dir) / "embeddings"
            embeddings_dir.mkdir()
            
            test_embeddings = [0.1, 0.2, 0.3, 0.4]
            test_file = embeddings_dir / "test_embeddings.json"
            with open(test_file, 'w') as f:
                json.dump(test_embeddings, f)
            
            loader = EmbeddingLoader(
                cleaned_text_file_list=[],
                cleaned_text_dir=str(temp_dir),
                embeddings_dir=str(embeddings_dir),
                vectordb_dir=str(temp_dir),
                collection_name="test"
            )
            
            result = loader._load_embeddings(test_file)
            assert result == test_embeddings
    
    @patch('classes.embedding_loader.chromadb')
    def test_load_embeddings_invalid_format(self, mock_chromadb):
        mock_client = Mock()
        mock_collection = Mock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        
        with tempfile.TemporaryDirectory() as temp_dir:
            embeddings_dir = Path(temp_dir) / "embeddings"
            embeddings_dir.mkdir()
            
            # Invalid format: nested list
            test_embeddings = [[0.1, 0.2], [0.3, 0.4]]
            test_file = embeddings_dir / "test_embeddings.json"
            with open(test_file, 'w') as f:
                json.dump(test_embeddings, f)
            
            loader = EmbeddingLoader(
                cleaned_text_file_list=[],
                cleaned_text_dir=str(temp_dir),
                embeddings_dir=str(embeddings_dir),
                vectordb_dir=str(temp_dir),
                collection_name="test"
            )
            
            result = loader._load_embeddings(test_file)
            assert result == []
    
    @patch('classes.embedding_loader.chromadb')
    def test_load_embeddings_file_not_found(self, mock_chromadb):
        mock_client = Mock()
        mock_collection = Mock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = EmbeddingLoader(
                cleaned_text_file_list=[],
                cleaned_text_dir=str(temp_dir),
                embeddings_dir=str(temp_dir),
                vectordb_dir=str(temp_dir),
                collection_name="test"
            )
            
            result = loader._load_embeddings(Path("nonexistent.json"))
            assert result == []
    
    @patch('classes.embedding_loader.chromadb')
    def test_process_files_success(self, mock_chromadb):
        mock_client = Mock()
        mock_collection = Mock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cleaned_text_dir = Path(temp_dir) / "cleaned"
            embeddings_dir = Path(temp_dir) / "embeddings"
            cleaned_text_dir.mkdir()
            embeddings_dir.mkdir()
            
            # Create test files
            test_text_file = cleaned_text_dir / "test.txt"
            test_text_file.write_text("Test content", encoding="utf-8")
            
            test_embeddings = [0.1, 0.2, 0.3]
            test_embedding_file = embeddings_dir / "test_embeddings.json"
            with open(test_embedding_file, 'w') as f:
                json.dump(test_embeddings, f)
            
            loader = EmbeddingLoader(
                cleaned_text_file_list=["test.txt"],
                cleaned_text_dir=str(cleaned_text_dir),
                embeddings_dir=str(embeddings_dir),
                vectordb_dir=str(temp_dir),
                collection_name="test"
            )
            
            loader.process_files()
            
            # Verify ChromaDB add was called
            mock_collection.add.assert_called_once_with(
                ids=["test.txt"],
                embeddings=[test_embeddings],
                metadatas=[{"text": "Test content", "source": "test.txt"}]
            )
    
    @patch('classes.embedding_loader.chromadb')
    def test_process_files_missing_text_file(self, mock_chromadb):
        mock_client = Mock()
        mock_collection = Mock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cleaned_text_dir = Path(temp_dir) / "cleaned"
            embeddings_dir = Path(temp_dir) / "embeddings"
            cleaned_text_dir.mkdir()
            embeddings_dir.mkdir()
            
            loader = EmbeddingLoader(
                cleaned_text_file_list=["nonexistent.txt"],
                cleaned_text_dir=str(cleaned_text_dir),
                embeddings_dir=str(embeddings_dir),
                vectordb_dir=str(temp_dir),
                collection_name="test"
            )
            
            loader.process_files()
            
            # Should not call add if text file is missing
            mock_collection.add.assert_not_called()
    
    @patch('classes.embedding_loader.chromadb')
    def test_process_files_missing_embedding_file(self, mock_chromadb):
        mock_client = Mock()
        mock_collection = Mock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cleaned_text_dir = Path(temp_dir) / "cleaned"
            embeddings_dir = Path(temp_dir) / "embeddings"
            cleaned_text_dir.mkdir()
            embeddings_dir.mkdir()
            
            # Only create text file, no embedding file
            test_text_file = cleaned_text_dir / "test.txt"
            test_text_file.write_text("Test content", encoding="utf-8")
            
            loader = EmbeddingLoader(
                cleaned_text_file_list=["test.txt"],
                cleaned_text_dir=str(cleaned_text_dir),
                embeddings_dir=str(embeddings_dir),
                vectordb_dir=str(temp_dir),
                collection_name="test"
            )
            
            loader.process_files()
            
            # Should not call add if embedding file is missing
            mock_collection.add.assert_not_called()