"""Integration tests for document processing flow."""
import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from classes.config_manager import ConfigManager
from classes.document_ingestor import DocumentIngestor
from classes.embedding_preparer import EmbeddingPreparer
from classes.embedding_loader import EmbeddingLoader

@pytest.mark.integration
class TestDocumentProcessingFlow:
    """Integration tests for document processing pipeline."""
    
    @pytest.fixture
    def pipeline_setup(self):
        """Set up document processing environment."""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create directory structure
        (temp_dir / "raw_input").mkdir()
        (temp_dir / "cleaned_text").mkdir()
        (temp_dir / "embeddings").mkdir()
        (temp_dir / "vectordb").mkdir()
        
        # Create config
        config = {
            "log_level": "INFO",
            "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "collection_name": "test_documents",
            "retriever_min_score_threshold": "0.5",
            "llm_api_url": "http://localhost:11434/api/generate",
            "llm_model_name": "test-model",
            "raw_input_directory": str(temp_dir / "raw_input"),
            "cleaned_text_directory": str(temp_dir / "cleaned_text"),
            "embeddings_directory": str(temp_dir / "embeddings"),
            "vectordb_directory": str(temp_dir / "vectordb")
        }
        
        config_file = temp_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(config, f)
        
        # Create sample documents
        sample_doc1 = temp_dir / "raw_input" / "ai_basics.txt"
        with open(sample_doc1, "w") as f:
            f.write("Artificial intelligence (AI) is a field of computer science that aims to create "
                   "intelligent machines. Machine learning is a subset of AI that enables computers "
                   "to learn without being explicitly programmed.")
        
        sample_doc2 = temp_dir / "raw_input" / "deep_learning.txt"
        with open(sample_doc2, "w") as f:
            f.write("Deep learning uses neural networks with multiple layers to process data. "
                   "Convolutional neural networks are particularly effective for image recognition "
                   "and computer vision tasks.")
        
        yield {
            "temp_dir": temp_dir,
            "config_file": config_file,
            "config": config,
            "sample_docs": ["ai_basics.txt", "deep_learning.txt"]
        }
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_complete_document_ingestion_flow(self, pipeline_setup):
        """Test complete document ingestion from raw files to cleaned text."""
        config = ConfigManager(pipeline_setup["config_file"])
        
        with patch('classes.document_ingestor.AutoTokenizer.from_pretrained') as mock_tokenizer_class:
            # Mock tokenizer
            mock_tokenizer = MagicMock()
            mock_tokenizer.tokenize.return_value = ["artificial", "intelligence", "machine", "learning", "deep", "neural", "networks"]
            mock_tokenizer.convert_tokens_to_string.return_value = "artificial intelligence machine learning deep neural networks"
            mock_tokenizer_class.return_value = mock_tokenizer
            
            # Create ingestor
            ingestor = DocumentIngestor(
                file_list=pipeline_setup["sample_docs"],
                input_dir=config.get("raw_input_directory"),
                output_dir=config.get("cleaned_text_directory"),
                embedding_model_name=config.get("embedding_model_name")
            )
            
            # Process all files
            ingestor.process_files()
            
            # Verify all cleaned files were created
            cleaned_dir = Path(config.get("cleaned_text_directory"))
            expected_files = ["ai_basics_cleaned.txt", "deep_learning_cleaned.txt"]
            
            for expected_file in expected_files:
                cleaned_file = cleaned_dir / expected_file
                assert cleaned_file.exists(), f"Missing cleaned file: {expected_file}"
                
                # Verify content was processed
                with open(cleaned_file, "r") as f:
                    content = f.read()
                    assert "artificial intelligence machine learning deep neural networks" in content
                    assert len(content) > 0
    
    def test_embedding_generation_flow(self, pipeline_setup):
        """Test embedding generation from cleaned text files."""
        config = ConfigManager(pipeline_setup["config_file"])
        
        # First create cleaned text files
        cleaned_dir = Path(config.get("cleaned_text_directory"))
        cleaned_files = []
        
        for i, doc in enumerate(pipeline_setup["sample_docs"]):
            cleaned_file = cleaned_dir / f"{Path(doc).stem}_cleaned.txt"
            cleaned_files.append(cleaned_file.name)
            
            with open(cleaned_file, "w") as f:
                f.write(f"processed content from document {i}: artificial intelligence and machine learning")
        
        # Mock embedding generation
        with patch('classes.embedding_preparer.AutoTokenizer.from_pretrained') as mock_tokenizer_class, \
             patch('classes.embedding_preparer.AutoModel.from_pretrained') as mock_model_class, \
             patch('classes.embedding_preparer.torch.cuda.is_available', return_value=False):
            
            # Mock tokenizer
            mock_tokenizer = MagicMock()
            mock_tokenizer_class.return_value = mock_tokenizer
            
            # Mock model and outputs
            mock_model = MagicMock()
            mock_model.to.return_value = mock_model
            
            # Create mock embedding output
            mock_outputs = MagicMock()
            mock_tensor = MagicMock()
            mock_tensor.mean.return_value.squeeze.return_value.cpu.return_value.numpy.return_value.tolist.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
            mock_outputs.last_hidden_state = mock_tensor
            mock_model.return_value = mock_outputs
            mock_model_class.return_value = mock_model
            
            # Create embedding preparer
            preparer = EmbeddingPreparer(
                file_list=cleaned_files,
                input_dir=config.get("cleaned_text_directory"),
                output_dir=config.get("embeddings_directory"),
                embedding_model_name=config.get("embedding_model_name")
            )
            
            # Generate embeddings
            preparer.process_files()
            
            # Verify embedding files were created
            embeddings_dir = Path(config.get("embeddings_directory"))
            for cleaned_file in cleaned_files:
                embedding_file = embeddings_dir / f"{Path(cleaned_file).stem}_embeddings.json"
                assert embedding_file.exists(), f"Missing embedding file for {cleaned_file}"
                
                # Verify embedding content
                with open(embedding_file, "r") as f:
                    embeddings = json.load(f)
                    assert isinstance(embeddings, list)
                    assert len(embeddings) == 5
                    assert all(isinstance(x, (int, float)) for x in embeddings)
    
    def test_vector_storage_flow(self, pipeline_setup):
        """Test storing embeddings in ChromaDB."""
        config = ConfigManager(pipeline_setup["config_file"])
        
        # Create cleaned text and embedding files
        cleaned_files = []
        embeddings_data = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        for doc in pipeline_setup["sample_docs"]:
            # Create cleaned text file
            cleaned_file_name = f"{Path(doc).stem}_cleaned.txt"
            cleaned_file_path = Path(config.get("cleaned_text_directory")) / cleaned_file_name
            cleaned_files.append(cleaned_file_name)
            
            with open(cleaned_file_path, "w") as f:
                f.write(f"Content from {doc}: artificial intelligence and deep learning concepts")
            
            # Create embedding file
            embedding_file_path = Path(config.get("embeddings_directory")) / f"{Path(doc).stem}_cleaned_embeddings.json"
            with open(embedding_file_path, "w") as f:
                json.dump(embeddings_data, f)
        
        # Mock ChromaDB
        with patch('classes.embedding_loader.chromadb.PersistentClient') as mock_client_class:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client
            
            # Create embedding loader
            loader = EmbeddingLoader(
                cleaned_text_file_list=cleaned_files,
                cleaned_text_dir=config.get("cleaned_text_directory"),
                embeddings_dir=config.get("embeddings_directory"),
                vectordb_dir=config.get("vectordb_directory"),
                collection_name=config.get("collection_name")
            )
            
            # Load embeddings into vector database
            loader.process_files()
            
            # Verify ChromaDB operations
            mock_client.get_or_create_collection.assert_called_once_with(config.get("collection_name"))
            assert mock_collection.add.call_count == len(cleaned_files)
            
            # Verify each file was added correctly
            add_calls = mock_collection.add.call_args_list
            for i, call_args in enumerate(add_calls):
                call_kwargs = call_args[1]
                
                # Check IDs
                assert call_kwargs["ids"] == [cleaned_files[i]]
                
                # Check embeddings
                assert call_kwargs["embeddings"] == [embeddings_data]
                
                # Check metadata
                metadata = call_kwargs["metadatas"][0]
                assert "artificial intelligence and deep learning" in metadata["text"]
                assert metadata["source"] == cleaned_files[i]
    
    def test_end_to_end_document_processing(self, pipeline_setup):
        """Test complete document processing pipeline from raw to vector storage."""
        config = ConfigManager(pipeline_setup["config_file"])
        
        # Mock all external dependencies
        with patch('classes.document_ingestor.AutoTokenizer.from_pretrained') as mock_ing_tokenizer, \
             patch('classes.embedding_preparer.AutoTokenizer.from_pretrained') as mock_emb_tokenizer, \
             patch('classes.embedding_preparer.AutoModel.from_pretrained') as mock_model_class, \
             patch('classes.embedding_preparer.torch.cuda.is_available', return_value=False), \
             patch('classes.embedding_loader.chromadb.PersistentClient') as mock_client_class:
            
            # Mock ingestor tokenizer
            mock_ing_tok = MagicMock()
            mock_ing_tok.tokenize.return_value = ["clean", "tokens"]
            mock_ing_tok.convert_tokens_to_string.return_value = "clean tokens"
            mock_ing_tokenizer.return_value = mock_ing_tok
            
            # Mock embedding components
            mock_emb_tok = MagicMock()
            mock_emb_tokenizer.return_value = mock_emb_tok
            
            mock_model = MagicMock()
            mock_model.to.return_value = mock_model
            mock_outputs = MagicMock()
            mock_outputs.last_hidden_state.mean.return_value.squeeze.return_value.cpu.return_value.numpy.return_value.tolist.return_value = [0.1, 0.2, 0.3]
            mock_model.return_value = mock_outputs
            mock_model_class.return_value = mock_model
            
            # Mock ChromaDB
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client
            
            # Step 1: Document Ingestion
            ingestor = DocumentIngestor(
                file_list=pipeline_setup["sample_docs"],
                input_dir=config.get("raw_input_directory"),
                output_dir=config.get("cleaned_text_directory"),
                embedding_model_name=config.get("embedding_model_name")
            )
            ingestor.process_files()
            
            # Step 2: Embedding Generation
            cleaned_files = [f"{Path(doc).stem}_cleaned.txt" for doc in pipeline_setup["sample_docs"]]
            preparer = EmbeddingPreparer(
                file_list=cleaned_files,
                input_dir=config.get("cleaned_text_directory"),
                output_dir=config.get("embeddings_directory"),
                embedding_model_name=config.get("embedding_model_name")
            )
            preparer.process_files()
            
            # Step 3: Vector Storage
            loader = EmbeddingLoader(
                cleaned_text_file_list=cleaned_files,
                cleaned_text_dir=config.get("cleaned_text_directory"),
                embeddings_dir=config.get("embeddings_directory"),
                vectordb_dir=config.get("vectordb_directory"),
                collection_name=config.get("collection_name")
            )
            loader.process_files()
            
            # Verify complete pipeline execution
            # Check that all intermediate files exist
            for doc in pipeline_setup["sample_docs"]:
                doc_stem = Path(doc).stem
                
                # Cleaned text file
                cleaned_file = Path(config.get("cleaned_text_directory")) / f"{doc_stem}_cleaned.txt"
                assert cleaned_file.exists()
                
                # Embedding file
                embedding_file = Path(config.get("embeddings_directory")) / f"{doc_stem}_cleaned_embeddings.json"
                assert embedding_file.exists()
            
            # Verify ChromaDB was called for each document
            assert mock_collection.add.call_count == len(pipeline_setup["sample_docs"])
    
    def test_error_handling_in_document_flow(self, pipeline_setup):
        """Test error handling throughout document processing flow."""
        config = ConfigManager(pipeline_setup["config_file"])
        
        # Test with some missing files
        file_list_with_missing = pipeline_setup["sample_docs"] + ["missing_file.txt"]
        
        with patch('classes.document_ingestor.AutoTokenizer.from_pretrained') as mock_tokenizer_class:
            mock_tokenizer = MagicMock()
            mock_tokenizer.tokenize.return_value = ["tokens"]
            mock_tokenizer.convert_tokens_to_string.return_value = "tokens"
            mock_tokenizer_class.return_value = mock_tokenizer
            
            ingestor = DocumentIngestor(
                file_list=file_list_with_missing,
                input_dir=config.get("raw_input_directory"),
                output_dir=config.get("cleaned_text_directory"),
                embedding_model_name=config.get("embedding_model_name")
            )
            
            # Should not crash, just skip missing files
            ingestor.process_files()
            
            # Only existing files should be processed
            for existing_doc in pipeline_setup["sample_docs"]:
                cleaned_file = Path(config.get("cleaned_text_directory")) / f"{Path(existing_doc).stem}_cleaned.txt"
                assert cleaned_file.exists()
            
            # Missing file should not create a cleaned version
            missing_cleaned = Path(config.get("cleaned_text_directory")) / "missing_file_cleaned.txt"
            assert not missing_cleaned.exists()
    
    @pytest.mark.slow
    def test_large_document_processing(self, pipeline_setup):
        """Test processing multiple documents efficiently."""
        config = ConfigManager(pipeline_setup["config_file"])
        
        # Create many documents
        large_doc_list = []
        for i in range(20):
            doc_name = f"large_doc_{i}.txt"
            doc_path = pipeline_setup["temp_dir"] / "raw_input" / doc_name
            large_doc_list.append(doc_name)
            
            with open(doc_path, "w") as f:
                f.write(f"Large document {i} with content about AI, machine learning, "
                       f"and deep learning concepts. Document number {i} contains "
                       f"information about neural networks and artificial intelligence.")
        
        with patch('classes.document_ingestor.AutoTokenizer.from_pretrained') as mock_tokenizer_class:
            mock_tokenizer = MagicMock()
            mock_tokenizer.tokenize.return_value = ["ai", "machine", "learning", "neural", "networks"]
            mock_tokenizer.convert_tokens_to_string.return_value = "ai machine learning neural networks"
            mock_tokenizer_class.return_value = mock_tokenizer
            
            ingestor = DocumentIngestor(
                file_list=large_doc_list,
                input_dir=config.get("raw_input_directory"),
                output_dir=config.get("cleaned_text_directory"),
                embedding_model_name=config.get("embedding_model_name")
            )
            
            # Should handle large batch efficiently
            ingestor.process_files()
            
            # Verify all documents were processed
            cleaned_dir = Path(config.get("cleaned_text_directory"))
            processed_files = list(cleaned_dir.glob("*_cleaned.txt"))
            assert len(processed_files) == len(large_doc_list)
            
            # Verify content is correct
            for processed_file in processed_files:
                with open(processed_file, "r") as f:
                    content = f.read()
                    assert "ai machine learning neural networks" in content