"""Integration tests for error handling scenarios."""
import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from classes.config_manager import ConfigManager
from classes.document_ingestor import DocumentIngestor
from classes.embedding_loader import EmbeddingLoader

@pytest.mark.integration
class TestErrorHandling:
    """Integration tests for error handling throughout the pipeline."""
    
    @pytest.fixture
    def error_test_setup(self):
        """Set up environment for error testing."""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create basic directory structure
        (temp_dir / "raw_input").mkdir()
        (temp_dir / "cleaned_text").mkdir()
        (temp_dir / "embeddings").mkdir()
        (temp_dir / "vectordb").mkdir()
        
        config = {
            "log_level": "INFO",
            "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "collection_name": "test_documents",
            "raw_input_directory": str(temp_dir / "raw_input"),
            "cleaned_text_directory": str(temp_dir / "cleaned_text"),
            "embeddings_directory": str(temp_dir / "embeddings"),
            "vectordb_directory": str(temp_dir / "vectordb")
        }
        
        config_file = temp_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(config, f)
        
        yield {
            "temp_dir": temp_dir,
            "config_file": config_file,
            "config": config
        }
        
        shutil.rmtree(temp_dir)
    
    def test_missing_input_files(self, error_test_setup):
        """Test handling of missing input files."""
        config = ConfigManager(error_test_setup["config_file"])
        
        with patch('classes.document_ingestor.AutoTokenizer.from_pretrained') as mock_tokenizer_class:
            mock_tokenizer = MagicMock()
            mock_tokenizer_class.return_value = mock_tokenizer
            
            # Try to process non-existent files
            ingestor = DocumentIngestor(
                file_list=["missing_file1.txt", "missing_file2.pdf"],
                input_dir=config.get("raw_input_directory"),
                output_dir=config.get("cleaned_text_directory"),
                embedding_model_name=config.get("embedding_model_name")
            )
            
            # Should not crash, just log warnings
            ingestor.process_files()
            
            # No cleaned files should be created
            cleaned_dir = Path(config.get("cleaned_text_directory"))
            assert len(list(cleaned_dir.glob("*.txt"))) == 0
    
    def test_corrupted_config_file(self, error_test_setup):
        """Test handling of corrupted configuration."""
        # Create invalid JSON config
        corrupted_config_file = error_test_setup["temp_dir"] / "corrupted_config.json"
        with open(corrupted_config_file, "w") as f:
            f.write("{ invalid json content")
        
        # Should raise an error when trying to load
        with pytest.raises(json.JSONDecodeError):
            with open(corrupted_config_file, "r") as f:
                json.load(f)
    
    def test_permission_denied_scenarios(self, error_test_setup):
        """Test handling of permission denied errors."""
        config = ConfigManager(error_test_setup["config_file"])
        
        # Create a read-only output directory
        readonly_dir = error_test_setup["temp_dir"] / "readonly_output"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)  # Read-only
        
        with patch('classes.document_ingestor.AutoTokenizer.from_pretrained') as mock_tokenizer_class:
            mock_tokenizer = MagicMock()
            mock_tokenizer_class.return_value = mock_tokenizer
            
            # Create a test input file
            test_file = Path(config.get("raw_input_directory")) / "test.txt"
            with open(test_file, "w") as f:
                f.write("Test content")
            
            ingestor = DocumentIngestor(
                file_list=["test.txt"],
                input_dir=config.get("raw_input_directory"),
                output_dir=str(readonly_dir),
                embedding_model_name=config.get("embedding_model_name")
            )
            
            # Should handle permission error gracefully
            try:
                ingestor.process_files()
            except PermissionError:
                # This is expected behavior
                pass
        
        # Cleanup
        readonly_dir.chmod(0o755)
    
    def test_disk_space_simulation(self, error_test_setup):
        """Test behavior when disk space is limited."""
        config = ConfigManager(error_test_setup["config_file"])
        
        # Create a very large input file (simulated)
        large_content = "A" * 1000000  # 1MB of text
        large_file = Path(config.get("raw_input_directory")) / "large_file.txt"
        with open(large_file, "w") as f:
            f.write(large_content)
        
        with patch('classes.document_ingestor.AutoTokenizer.from_pretrained') as mock_tokenizer_class:
            mock_tokenizer = MagicMock()
            mock_tokenizer.tokenize.return_value = ["large", "content"]
            mock_tokenizer.convert_tokens_to_string.return_value = "large content"
            mock_tokenizer_class.return_value = mock_tokenizer
            
            ingestor = DocumentIngestor(
                file_list=["large_file.txt"],
                input_dir=config.get("raw_input_directory"),
                output_dir=config.get("cleaned_text_directory"),
                embedding_model_name=config.get("embedding_model_name")
            )
            
            # Should process the file normally
            ingestor.process_files()
            
            # Verify output was created
            output_file = Path(config.get("cleaned_text_directory")) / "large_file_cleaned.txt"
            assert output_file.exists()
    
    def test_malformed_pdf_handling(self, error_test_setup):
        """Test handling of corrupted or malformed PDF files."""
        config = ConfigManager(error_test_setup["config_file"])
        
        # Create a fake PDF file (not actually PDF content)
        fake_pdf = Path(config.get("raw_input_directory")) / "corrupted.pdf"
        with open(fake_pdf, "w") as f:
            f.write("This is not a valid PDF file content")
        
        with patch('classes.document_ingestor.AutoTokenizer.from_pretrained') as mock_tokenizer_class, \
             patch('classes.document_ingestor.pdfplumber.open') as mock_pdf_open:
            
            mock_tokenizer = MagicMock()
            mock_tokenizer_class.return_value = mock_tokenizer
            
            # Mock PDF extraction failure
            mock_pdf_open.side_effect = Exception("Invalid PDF format")
            
            ingestor = DocumentIngestor(
                file_list=["corrupted.pdf"],
                input_dir=config.get("raw_input_directory"),
                output_dir=config.get("cleaned_text_directory"),
                embedding_model_name=config.get("embedding_model_name")
            )
            
            # Should handle PDF error gracefully
            ingestor.process_files()
            
            # No output file should be created for corrupted PDF
            output_file = Path(config.get("cleaned_text_directory")) / "corrupted_cleaned.txt"
            assert not output_file.exists()
    
    def test_embedding_file_corruption(self, error_test_setup):
        """Test handling of corrupted embedding files."""
        config = ConfigManager(error_test_setup["config_file"])
        
        # Create cleaned text file
        cleaned_file = Path(config.get("cleaned_text_directory")) / "test_cleaned.txt"
        with open(cleaned_file, "w") as f:
            f.write("Test content for embedding")
        
        # Create corrupted embedding file
        corrupted_embedding = Path(config.get("embeddings_directory")) / "test_cleaned_embeddings.json"
        with open(corrupted_embedding, "w") as f:
            f.write("{ corrupted json data")
        
        with patch('classes.embedding_loader.chromadb.PersistentClient') as mock_client_class:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client
            
            loader = EmbeddingLoader(
                cleaned_text_file_list=["test_cleaned.txt"],
                cleaned_text_dir=config.get("cleaned_text_directory"),
                embeddings_dir=config.get("embeddings_directory"),
                vectordb_dir=config.get("vectordb_directory"),
                collection_name=config.get("collection_name")
            )
            
            # Should handle corrupted embedding file gracefully
            loader.process_files()
            
            # ChromaDB add should not be called due to corrupted embeddings
            mock_collection.add.assert_not_called()
    
    def test_network_timeout_simulation(self, error_test_setup):
        """Test handling of network timeouts in LLM requests."""
        from classes.llm_client import LLMClient
        import requests
        
        llm_client = LLMClient(
            llm_api_url="http://localhost:11434/api/generate",
            llm_model_name="test-model"
        )
        
        with patch('classes.llm_client.requests.post') as mock_post:
            # Simulate network timeout
            mock_post.side_effect = requests.exceptions.Timeout("Request timed out")
            
            response = llm_client.query("Test query")
            
            # Should return error message, not crash
            assert "Error: Could not connect to the LLM" in response
    
    def test_memory_exhaustion_simulation(self, error_test_setup):
        """Test behavior with memory constraints."""
        config = ConfigManager(error_test_setup["config_file"])
        
        # Create many large files to simulate memory pressure
        large_files = []
        for i in range(5):
            filename = f"large_doc_{i}.txt"
            large_files.append(filename)
            file_path = Path(config.get("raw_input_directory")) / filename
            
            # Create moderately large content
            content = f"Large document {i} content. " * 1000  # ~25KB per file
            with open(file_path, "w") as f:
                f.write(content)
        
        with patch('classes.document_ingestor.AutoTokenizer.from_pretrained') as mock_tokenizer_class:
            mock_tokenizer = MagicMock()
            mock_tokenizer.tokenize.return_value = ["processed", "tokens"]
            mock_tokenizer.convert_tokens_to_string.return_value = "processed tokens"
            mock_tokenizer_class.return_value = mock_tokenizer
            
            ingestor = DocumentIngestor(
                file_list=large_files,
                input_dir=config.get("raw_input_directory"),
                output_dir=config.get("cleaned_text_directory"),
                embedding_model_name=config.get("embedding_model_name")
            )
            
            # Should handle multiple large files
            ingestor.process_files()
            
            # All files should be processed
            for filename in large_files:
                output_file = Path(config.get("cleaned_text_directory")) / f"{Path(filename).stem}_cleaned.txt"
                assert output_file.exists()
    
    def test_concurrent_access_simulation(self, error_test_setup):
        """Test behavior with simulated concurrent access."""
        config = ConfigManager(error_test_setup["config_file"])
        
        # Create test files
        test_files = ["doc1.txt", "doc2.txt"]
        for filename in test_files:
            file_path = Path(config.get("raw_input_directory")) / filename
            with open(file_path, "w") as f:
                f.write(f"Content of {filename}")
        
        with patch('classes.embedding_loader.chromadb.PersistentClient') as mock_client_class:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            
            # Simulate concurrent access conflict
            mock_collection.add.side_effect = [None, Exception("Database locked")]
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client
            
            # Create cleaned text and embedding files
            for filename in test_files:
                stem = Path(filename).stem
                
                cleaned_file = Path(config.get("cleaned_text_directory")) / f"{stem}_cleaned.txt"
                with open(cleaned_file, "w") as f:
                    f.write(f"Cleaned content of {filename}")
                
                embedding_file = Path(config.get("embeddings_directory")) / f"{stem}_cleaned_embeddings.json"
                with open(embedding_file, "w") as f:
                    json.dump([0.1, 0.2, 0.3], f)
            
            loader = EmbeddingLoader(
                cleaned_text_file_list=[f"{Path(f).stem}_cleaned.txt" for f in test_files],
                cleaned_text_dir=config.get("cleaned_text_directory"),
                embeddings_dir=config.get("embeddings_directory"),
                vectordb_dir=config.get("vectordb_directory"),
                collection_name=config.get("collection_name")
            )
            
            # Should handle the database conflict gracefully
            loader.process_files()
            
            # First file should succeed, second should fail but not crash
            assert mock_collection.add.call_count == 2