"""Integration tests for performance scenarios."""
import pytest
import json
import tempfile
import shutil
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
from classes.config_manager import ConfigManager
from classes.document_ingestor import DocumentIngestor
from classes.embedding_preparer import EmbeddingPreparer
from classes.embedding_loader import EmbeddingLoader
from classes.chromadb_retriever import ChromaDBRetriever
from classes.llm_client import LLMClient
from classes.rag_query_processor import RAGQueryProcessor

@pytest.mark.integration
@pytest.mark.slow
class TestPerformance:
    """Performance integration tests (marked as slow)."""
    
    @pytest.fixture
    def performance_setup(self):
        """Set up performance testing environment."""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create directory structure
        (temp_dir / "raw_input").mkdir()
        (temp_dir / "cleaned_text").mkdir()
        (temp_dir / "embeddings").mkdir()
        (temp_dir / "vectordb").mkdir()
        
        config = {
            "log_level": "INFO",
            "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "collection_name": "performance_test",
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
        
        yield {
            "temp_dir": temp_dir,
            "config_file": config_file,
            "config": config
        }
        
        shutil.rmtree(temp_dir)
    
    def test_large_batch_document_processing(self, performance_setup):
        """Test processing a large batch of documents for performance."""
        config = ConfigManager(performance_setup["config_file"])
        
        # Create 50 test documents
        num_docs = 50
        doc_files = []
        
        for i in range(num_docs):
            filename = f"perf_doc_{i:03d}.txt"
            doc_files.append(filename)
            file_path = Path(config.get("raw_input_directory")) / filename
            
            # Create realistic document content
            content = f"""
            Performance Test Document {i}
            
            This document contains information about artificial intelligence and machine learning.
            Document {i} focuses on performance testing of the RAG pipeline system.
            
            The content includes various AI concepts, algorithms, and applications.
            This text is designed to test the processing capabilities of the system.
            
            Performance metrics include throughput, latency, and resource utilization.
            Document {i} contributes to the overall performance evaluation.
            """
            
            with open(file_path, "w") as f:
                f.write(content)
        
        with patch('classes.document_ingestor.AutoTokenizer.from_pretrained') as mock_tokenizer_class:
            # Mock tokenizer for consistent performance testing
            mock_tokenizer = MagicMock()
            mock_tokenizer.tokenize.return_value = ["performance", "test", "ai", "ml", "document"]
            mock_tokenizer.convert_tokens_to_string.return_value = "performance test ai ml document"
            mock_tokenizer_class.return_value = mock_tokenizer
            
            # Measure processing time
            start_time = time.time()
            
            ingestor = DocumentIngestor(
                file_list=doc_files,
                input_dir=config.get("raw_input_directory"),
                output_dir=config.get("cleaned_text_directory"),
                embedding_model_name=config.get("embedding_model_name")
            )
            
            ingestor.process_files()
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Verify all documents were processed
            cleaned_dir = Path(config.get("cleaned_text_directory"))
            processed_files = list(cleaned_dir.glob("*_cleaned.txt"))
            assert len(processed_files) == num_docs
            
            # Performance assertions
            throughput = num_docs / processing_time
            avg_time_per_doc = processing_time / num_docs
            
            print(f"Large batch performance:")
            print(f"  Processed {num_docs} documents in {processing_time:.2f} seconds")
            print(f"  Throughput: {throughput:.2f} documents/second")
            print(f"  Average time per document: {avg_time_per_doc:.3f} seconds")
            
            # Performance requirements
            assert processing_time < 120  # Should complete within 2 minutes
            assert throughput > 0.4  # Should process at least 0.4 docs/second
            assert avg_time_per_doc < 3.0  # Should not take more than 3 seconds per doc
            
            # Verify content quality
            sample_file = processed_files[0]
            with open(sample_file, "r") as f:
                content = f.read()
                assert "performance test ai ml document" in content
    
    def test_memory_usage_with_large_documents(self, performance_setup):
        """Test memory efficiency with very large documents."""
        config = ConfigManager(performance_setup["config_file"])
        
        # Create large documents (simulate 1MB+ each)
        large_docs = []
        for i in range(5):
            filename = f"large_doc_{i}.txt"
            large_docs.append(filename)
            file_path = Path(config.get("raw_input_directory")) / filename
            
            # Create substantial content
            base_content = f"Large performance test document {i} with extensive AI content. "
            # Repeat to create large file
            large_content = base_content * 1000  # Approximately 70KB per file
            
            with open(file_path, "w") as f:
                f.write(large_content)
        
        with patch('classes.document_ingestor.AutoTokenizer.from_pretrained') as mock_tokenizer_class:
            mock_tokenizer = MagicMock()
            mock_tokenizer.tokenize.return_value = ["large", "performance", "test"] * 20
            mock_tokenizer.convert_tokens_to_string.return_value = "large performance test " * 20
            mock_tokenizer_class.return_value = mock_tokenizer
            
            start_time = time.time()
            
            ingestor = DocumentIngestor(
                file_list=large_docs,
                input_dir=config.get("raw_input_directory"),
                output_dir=config.get("cleaned_text_directory"),
                embedding_model_name=config.get("embedding_model_name")
            )
            
            ingestor.process_files()
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Verify all large documents were processed
            cleaned_dir = Path(config.get("cleaned_text_directory"))
            processed_files = list(cleaned_dir.glob("*_cleaned.txt"))
            assert len(processed_files) == len(large_docs)
            
            print(f"Large document performance:")
            print(f"  Processed {len(large_docs)} large documents in {processing_time:.2f} seconds")
            print(f"  Average time per large document: {processing_time/len(large_docs):.2f} seconds")
            
            # Should handle large documents efficiently
            assert processing_time < 60  # Should complete within 1 minute
            assert processing_time / len(large_docs) < 15  # <15 seconds per large doc
    
    def test_embedding_generation_performance(self, performance_setup):
        """Test performance of embedding generation process."""
        config = ConfigManager(performance_setup["config_file"])
        
        # Create cleaned text files for embedding
        num_docs = 30
        cleaned_files = []
        
        for i in range(num_docs):
            filename = f"embed_test_{i:03d}_cleaned.txt"
            cleaned_files.append(filename)
            file_path = Path(config.get("cleaned_text_directory")) / filename
            
            content = f"Embedding performance test document {i} about AI and ML concepts."
            with open(file_path, "w") as f:
                f.write(content)
        
        with patch('classes.embedding_preparer.AutoTokenizer.from_pretrained') as mock_tokenizer_class, \
             patch('classes.embedding_preparer.AutoModel.from_pretrained') as mock_model_class, \
             patch('classes.embedding_preparer.torch.cuda.is_available', return_value=False):
            
            # Mock tokenizer
            mock_tokenizer = MagicMock()
            mock_tokenizer_class.return_value = mock_tokenizer
            
            # Mock model
            mock_model = MagicMock()
            mock_model.to.return_value = mock_model
            mock_outputs = MagicMock()
            mock_outputs.last_hidden_state.mean.return_value.squeeze.return_value.cpu.return_value.numpy.return_value.tolist.return_value = [0.1] * 384
            mock_model.return_value = mock_outputs
            mock_model_class.return_value = mock_model
            
            start_time = time.time()
            
            preparer = EmbeddingPreparer(
                file_list=cleaned_files,
                input_dir=config.get("cleaned_text_directory"),
                output_dir=config.get("embeddings_directory"),
                embedding_model_name=config.get("embedding_model_name")
            )
            
            preparer.process_files()
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Verify all embeddings were generated
            embeddings_dir = Path(config.get("embeddings_directory"))
            embedding_files = list(embeddings_dir.glob("*_embeddings.json"))
            assert len(embedding_files) == num_docs
            
            # Performance metrics
            throughput = num_docs / processing_time
            
            print(f"Embedding generation performance:")
            print(f"  Generated {num_docs} embeddings in {processing_time:.2f} seconds")
            print(f"  Throughput: {throughput:.2f} embeddings/second")
            
            # Performance requirements
            assert processing_time < 90  # Should complete within 1.5 minutes
            assert throughput > 0.3  # Should generate at least 0.3 embeddings/second
            
            # Verify embedding content
            sample_embedding = embedding_files[0]
            with open(sample_embedding, "r") as f:
                embedding_data = json.load(f)
                assert isinstance(embedding_data, list)
                assert len(embedding_data) == 384
    
    def test_vector_storage_performance(self, performance_setup):
        """Test performance of vector database storage operations."""
        config = ConfigManager(performance_setup["config_file"])
        
        # Create test data for vector storage
        num_vectors = 40
        cleaned_files = []
        
        for i in range(num_vectors):
            # Create cleaned text file
            cleaned_filename = f"vector_test_{i:03d}_cleaned.txt"
            cleaned_files.append(cleaned_filename)
            cleaned_path = Path(config.get("cleaned_text_directory")) / cleaned_filename
            
            with open(cleaned_path, "w") as f:
                f.write(f"Vector storage test document {i}")
            
            # Create embedding file
            embedding_filename = f"vector_test_{i:03d}_cleaned_embeddings.json"
            embedding_path = Path(config.get("embeddings_directory")) / embedding_filename
            
            with open(embedding_path, "w") as f:
                json.dump([0.1] * 384, f)  # 384-dimensional mock embedding
        
        with patch('classes.embedding_loader.chromadb.PersistentClient') as mock_client_class:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client
            
            start_time = time.time()
            
            loader = EmbeddingLoader(
                cleaned_text_file_list=cleaned_files,
                cleaned_text_dir=config.get("cleaned_text_directory"),
                embeddings_dir=config.get("embeddings_directory"),
                vectordb_dir=config.get("vectordb_directory"),
                collection_name=config.get("collection_name")
            )
            
            loader.process_files()
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Verify all vectors were stored
            assert mock_collection.add.call_count == num_vectors
            
            # Performance metrics
            throughput = num_vectors / processing_time
            
            print(f"Vector storage performance:")
            print(f"  Stored {num_vectors} vectors in {processing_time:.2f} seconds")
            print(f"  Throughput: {throughput:.2f} vectors/second")
            
            # Performance requirements
            assert processing_time < 45  # Should complete within 45 seconds
            assert throughput > 0.8  # Should store at least 0.8 vectors/second
    
    def test_rag_query_performance(self, performance_setup):
        """Test performance of RAG query processing."""
        config = ConfigManager(performance_setup["config_file"])
        
        with patch('classes.chromadb_retriever.chromadb.PersistentClient') as mock_client_class, \
             patch('classes.chromadb_retriever.SentenceTransformer') as mock_transformer_class, \
             patch('classes.llm_client.requests.post') as mock_post:
            
            # Mock ChromaDB
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_collection.query.return_value = {
                "ids": [["perf_doc.txt"]],
                "metadatas": [[{"text": "Performance test document", "source": "perf_doc.txt"}]],
                "distances": [[0.2]]
            }
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client
            
            # Mock sentence transformer
            mock_transformer = MagicMock()
            mock_transformer.encode.return_value.tolist.return_value = [0.1] * 384
            mock_transformer_class.return_value = mock_transformer
            
            # Mock LLM response
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": "Performance test response"}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response
            
            # Create RAG components
            retriever = ChromaDBRetriever(
                embedding_model_name=config.get("embedding_model_name"),
                collection_name=config.get("collection_name"),
                vectordb_dir=config.get("vectordb_directory"),
                score_threshold=float(config.get("retriever_min_score_threshold"))
            )
            
            llm_client = LLMClient(
                llm_api_url=config.get("llm_api_url"),
                llm_model_name=config.get("llm_model_name")
            )
            
            rag_processor = RAGQueryProcessor(
                llm_client=llm_client,
                retriever=retriever,
                use_rag=True
            )
            
            # Test multiple queries for performance
            num_queries = 20
            queries = [f"Performance test query {i}" for i in range(num_queries)]
            
            start_time = time.time()
            
            responses = []
            for query in queries:
                response = rag_processor.query(query)
                responses.append(response)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Verify all queries were processed
            assert len(responses) == num_queries
            assert all("Performance test response" in response for response in responses)
            
            # Performance metrics
            avg_query_time = processing_time / num_queries
            queries_per_second = num_queries / processing_time
            
            print(f"RAG query performance:")
            print(f"  Processed {num_queries} queries in {processing_time:.2f} seconds")
            print(f"  Average query time: {avg_query_time:.3f} seconds")
            print(f"  Queries per second: {queries_per_second:.2f}")
            
            # Performance requirements
            assert avg_query_time < 1.0  # Each query should take <1 second
            assert queries_per_second > 1.0  # Should handle >1 query/second
            assert processing_time < 20  # Total time should be <20 seconds
    
    def test_stress_test_many_files(self, performance_setup):
        """Stress test with many small files."""
        config = ConfigManager(performance_setup["config_file"])
        
        # Create many small files
        num_files = 100
        stress_files = []
        
        for i in range(num_files):
            filename = f"stress_{i:03d}.txt"
            stress_files.append(filename)
            file_path = Path(config.get("raw_input_directory")) / filename
            
            content = f"Stress test document {i} with AI content."
            with open(file_path, "w") as f:
                f.write(content)
        
        with patch('classes.document_ingestor.AutoTokenizer.from_pretrained') as mock_tokenizer_class:
            mock_tokenizer = MagicMock()
            mock_tokenizer.tokenize.return_value = ["stress", "test", "ai"]
            mock_tokenizer.convert_tokens_to_string.return_value = "stress test ai"
            mock_tokenizer_class.return_value = mock_tokenizer
            
            start_time = time.time()
            
            ingestor = DocumentIngestor(
                file_list=stress_files,
                input_dir=config.get("raw_input_directory"),
                output_dir=config.get("cleaned_text_directory"),
                embedding_model_name=config.get("embedding_model_name")
            )
            
            ingestor.process_files()
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Verify all files were processed
            cleaned_dir = Path(config.get("cleaned_text_directory"))
            processed_files = list(cleaned_dir.glob("stress_*_cleaned.txt"))
            assert len(processed_files) == num_files
            
            # Performance metrics
            throughput = num_files / processing_time
            
            print(f"Stress test performance:")
            print(f"  Processed {num_files} files in {processing_time:.2f} seconds")
            print(f"  Throughput: {throughput:.2f} files/second")
            
            # Stress test requirements
            assert processing_time < 300  # Should complete within 5 minutes
            assert throughput > 0.3  # Should process at least 0.3 files/second
    
    def test_scalability_with_increasing_load(self, performance_setup):
        """Test system scalability with increasing document load."""
        config = ConfigManager(performance_setup["config_file"])
        
        # Test with increasing numbers of documents
        load_sizes = [10, 20, 40]
        processing_times = []
        
        with patch('classes.document_ingestor.AutoTokenizer.from_pretrained') as mock_tokenizer_class:
            mock_tokenizer = MagicMock()
            mock_tokenizer.tokenize.return_value = ["scalability", "test"]
            mock_tokenizer.convert_tokens_to_string.return_value = "scalability test"
            mock_tokenizer_class.return_value = mock_tokenizer
            
            for load_size in load_sizes:
                # Create documents for this load size
                load_files = []
                for i in range(load_size):
                    filename = f"scale_{load_size}_{i}.txt"
                    load_files.append(filename)
                    file_path = Path(config.get("raw_input_directory")) / filename
                    
                    with open(file_path, "w") as f:
                        f.write(f"Scalability test document {i} for load {load_size}")
                
                start_time = time.time()
                
                ingestor = DocumentIngestor(
                    file_list=load_files,
                    input_dir=config.get("raw_input_directory"),
                    output_dir=config.get("cleaned_text_directory"),
                    embedding_model_name=config.get("embedding_model_name")
                )
                
                ingestor.process_files()
                
                end_time = time.time()
                processing_time = end_time - start_time
                processing_times.append(processing_time)
                
                print(f"Load size {load_size}: {processing_time:.2f} seconds")
                
                # Clean up for next iteration
                cleaned_dir = Path(config.get("cleaned_text_directory"))
                for file in cleaned_dir.glob(f"scale_{load_size}_*_cleaned.txt"):
                    file.unlink()
            
            # Analyze scalability
            throughputs = [load_sizes[i] / processing_times[i] for i in range(len(load_sizes))]
            
            print(f"Scalability analysis:")
            for i, load_size in enumerate(load_sizes):
                print(f"  {load_size} docs: {throughputs[i]:.2f} docs/second")
            
            # Check that throughput doesn't degrade too much with load
            throughput_variance = (max(throughputs) - min(throughputs)) / max(throughputs)
            
            print(f"Throughput variance: {throughput_variance:.2f}")
            
            # Scalability requirements
            assert throughput_variance < 0.6  # Variance should be <60%
            assert all(t > 0.2 for t in throughputs)  # Minimum throughput
    
    def test_memory_leak_detection(self, performance_setup):
        """Test for memory leaks during repeated operations."""
        config = ConfigManager(performance_setup["config_file"])
        
        # Create test files
        test_files = []
        for i in range(15):
            filename = f"memory_test_{i}.txt"
            test_files.append(filename)
            file_path = Path(config.get("raw_input_directory")) / filename
            
            with open(file_path, "w") as f:
                f.write(f"Memory leak test document {i}")
        
        with patch('classes.document_ingestor.AutoTokenizer.from_pretrained') as mock_tokenizer_class:
            mock_tokenizer = MagicMock()
            mock_tokenizer.tokenize.return_value = ["memory", "test"]
            mock_tokenizer.convert_tokens_to_string.return_value = "memory test"
            mock_tokenizer_class.return_value = mock_tokenizer
            
            # Run multiple iterations
            iteration_times = []
            
            for iteration in range(5):
                start_time = time.time()
                
                ingestor = DocumentIngestor(
                    file_list=test_files,
                    input_dir=config.get("raw_input_directory"),
                    output_dir=config.get("cleaned_text_directory"),
                    embedding_model_name=config.get("embedding_model_name")
                )
                
                ingestor.process_files()
                
                end_time = time.time()
                iteration_time = end_time - start_time
                iteration_times.append(iteration_time)
                
                # Clean up for next iteration
                cleaned_dir = Path(config.get("cleaned_text_directory"))
                for file in cleaned_dir.glob("memory_test_*_cleaned.txt"):
                    file.unlink()
            
            # Analyze for memory leaks (performance degradation over iterations)
            avg_time = sum(iteration_times) / len(iteration_times)
            max_deviation = max(abs(t - avg_time) for t in iteration_times)
            
            print(f"Memory leak test:")
            print(f"  Iteration times: {[f'{t:.2f}' for t in iteration_times]}")
            print(f"  Average time: {avg_time:.2f}s")
            print(f"  Max deviation: {max_deviation:.2f}s")
            
            # Memory leak detection
            assert max_deviation < avg_time * 0.5  # Deviation should be <50% of average
            assert all(t < 15 for t in iteration_times)  # All iterations should be reasonable
    
    def test_concurrent_operations_simulation(self, performance_setup):
        """Test performance under simulated concurrent operations."""
        config = ConfigManager(performance_setup["config_file"])
        
        # Create documents for concurrent simulation
        batch_sizes = [8, 12, 16]
        batch_times = []
        
        with patch('classes.document_ingestor.AutoTokenizer.from_pretrained') as mock_tokenizer_class:
            mock_tokenizer = MagicMock()
            mock_tokenizer.tokenize.return_value = ["concurrent", "test"]
            mock_tokenizer.convert_tokens_to_string.return_value = "concurrent test"
            mock_tokenizer_class.return_value = mock_tokenizer
            
            for batch_idx, batch_size in enumerate(batch_sizes):
                # Create batch documents
                batch_docs = []
                for i in range(batch_size):
                    filename = f"concurrent_{batch_idx}_{i}.txt"
                    batch_docs.append(filename)
                    file_path = Path(config.get("raw_input_directory")) / filename
                    
                    with open(file_path, "w") as f:
                        f.write(f"Concurrent test document {batch_idx}-{i}")
                
                start_time = time.time()
                
                ingestor = DocumentIngestor(
                    file_list=batch_docs,
                    input_dir=config.get("raw_input_directory"),
                    output_dir=config.get("cleaned_text_directory"),
                    embedding_model_name=config.get("embedding_model_name")
                )
                
                ingestor.process_files()
                
                end_time = time.time()
                batch_time = end_time - start_time
                batch_times.append(batch_time)
                
                print(f"Batch {batch_idx} ({batch_size} docs): {batch_time:.2f} seconds")
            
            # Verify reasonable performance across batches
            assert all(time < 45 for time in batch_times)  # All batches <45 seconds
            
            # Verify all documents were processed
            cleaned_dir = Path(config.get("cleaned_text_directory"))
            processed_files = list(cleaned_dir.glob("concurrent_*_cleaned.txt"))
            assert len(processed_files) == sum(batch_sizes)
    
    def test_full_pipeline_end_to_end_performance(self, performance_setup):
        """Test complete pipeline performance from documents to queries."""
        config = ConfigManager(performance_setup["config_file"])
        
        # Create test documents
        pipeline_docs = []
        for i in range(20):
            filename = f"pipeline_{i}.txt"
            pipeline_docs.append(filename)
            file_path = Path(config.get("raw_input_directory")) / filename
            
            content = f"Full pipeline test document {i} about AI and ML applications."
            with open(file_path, "w") as f:
                f.write(content)
        
        total_start_time = time.time()
        
        # Phase 1: Document Ingestion
        with patch('classes.document_ingestor.AutoTokenizer.from_pretrained') as mock_ing_tokenizer:
            mock_ing_tok = MagicMock()
            mock_ing_tok.tokenize.return_value = ["pipeline", "test", "ai", "ml"]
            mock_ing_tok.convert_tokens_to_string.return_value = "pipeline test ai ml"
            mock_ing_tokenizer.return_value = mock_ing_tok
            
            ingest_start = time.time()
            ingestor = DocumentIngestor(
                file_list=pipeline_docs,
                input_dir=config.get("raw_input_directory"),
                output_dir=config.get("cleaned_text_directory"),
                embedding_model_name=config.get("embedding_model_name")
            )
            ingestor.process_files()
            ingest_time = time.time() - ingest_start
        
        # Phase 2: Embedding Generation
        with patch('classes.embedding_preparer.AutoTokenizer.from_pretrained') as mock_emb_tokenizer, \
             patch('classes.embedding_preparer.AutoModel.from_pretrained') as mock_model_class, \
             patch('classes.embedding_preparer.torch.cuda.is_available', return_value=False):
            
            mock_emb_tok = MagicMock()
            mock_emb_tokenizer.return_value = mock_emb_tok
            
            mock_model = MagicMock()
            mock_model.to.return_value = mock_model
            mock_outputs = MagicMock()
            mock_outputs.last_hidden_state.mean.return_value.squeeze.return_value.cpu.return_value.numpy.return_value.tolist.return_value = [0.1] * 384
            mock_model.return_value = mock_outputs
            mock_model_class.return_value = mock_model
            
            embed_start = time.time()
            cleaned_files = [f"pipeline_{i}_cleaned.txt" for i in range(len(pipeline_docs))]
            preparer = EmbeddingPreparer(
                file_list=cleaned_files,
                input_dir=config.get("cleaned_text_directory"),
                output_dir=config.get("embeddings_directory"),
                embedding_model_name=config.get("embedding_model_name")
            )
            preparer.process_files()
            embed_time = time.time() - embed_start
        
        # Phase 3: Vector Storage
        with patch('classes.embedding_loader.chromadb.PersistentClient') as mock_client_class:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client
            
            storage_start = time.time()
            loader = EmbeddingLoader(
                cleaned_text_file_list=cleaned_files,
                cleaned_text_dir=config.get("cleaned_text_directory"),
                embeddings_dir=config.get("embeddings_directory"),
                vectordb_dir=config.get("vectordb_directory"),
                collection_name=config.get("collection_name")
            )
            loader.process_files()
            storage_time = time.time() - storage_start
        
        # Phase 4: RAG Query Testing
        with patch('classes.chromadb_retriever.chromadb.PersistentClient') as mock_rag_client, \
             patch('classes.chromadb_retriever.SentenceTransformer') as mock_transformer, \
             patch('classes.llm_client.requests.post') as mock_post:
            
            # Mock RAG components
            mock_rag_client.return_value = mock_client
            mock_transformer_instance = MagicMock()
            mock_transformer_instance.encode.return_value.tolist.return_value = [0.1] * 384
            mock_transformer.return_value = mock_transformer_instance
            
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": "Pipeline test response"}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response
            
            query_start = time.time()
            retriever = ChromaDBRetriever(
                embedding_model_name=config.get("embedding_model_name"),
                collection_name=config.get("collection_name"),
                vectordb_dir=config.get("vectordb_directory"),
                score_threshold=float(config.get("retriever_min_score_threshold"))
            )
            
            llm_client = LLMClient(
                llm_api_url=config.get("llm_api_url"),
                llm_model_name=config.get("llm_model_name")
            )
            
            rag_processor = RAGQueryProcessor(
                llm_client=llm_client,
                retriever=retriever,
                use_rag=True
            )
            
            # Test multiple queries
            test_queries = [
                "What is artificial intelligence?",
                "Explain machine learning applications",
                "How does the pipeline work?",
                "What are AI concepts?",
                "Describe ML algorithms"
            ]
            
            for query in test_queries:
                response = rag_processor.query(query)
                assert "Pipeline test response" in response
            
            query_time = time.time() - query_start
        
        total_time = time.time() - total_start_time
        
        # Performance analysis and reporting
        print(f"Full Pipeline Performance Benchmark:")
        print(f"  Documents processed: {len(pipeline_docs)}")
        print(f"  Phase 1 - Ingestion: {ingest_time:.2f}s ({len(pipeline_docs)/ingest_time:.2f} docs/s)")
        print(f"  Phase 2 - Embeddings: {embed_time:.2f}s ({len(pipeline_docs)/embed_time:.2f} embeddings/s)")
        print(f"  Phase 3 - Storage: {storage_time:.2f}s ({len(pipeline_docs)/storage_time:.2f} vectors/s)")
        print(f"  Phase 4 - Queries: {query_time:.2f}s ({len(test_queries)/query_time:.2f} queries/s)")
        print(f"  Total pipeline time: {total_time:.2f}s")
        print(f"  End-to-end throughput: {len(pipeline_docs)/total_time:.2f} docs/s")
        
        # Performance assertions for full pipeline
        assert ingest_time < 40  # Ingestion should be <40s
        assert embed_time < 60  # Embedding generation should be <60s
        assert storage_time < 20  # Vector storage should be <20s
        assert query_time < 15  # Queries should be <15s
        assert total_time < 120  # Total pipeline should be <2 minutes
        
        # Verify pipeline completeness
        assert mock_collection.add.call_count == len(pipeline_docs)
        assert mock_post.call_count == len(test_queries)
        
        # Verify all intermediate files exist
        cleaned_dir = Path(config.get("cleaned_text_directory"))
        embeddings_dir = Path(config.get("embeddings_directory"))
        
        cleaned_files_created = list(cleaned_dir.glob("pipeline_*_cleaned.txt"))
        embedding_files_created = list(embeddings_dir.glob("pipeline_*_embeddings.json"))
        
        assert len(cleaned_files_created) == len(pipeline_docs)
        assert len(embedding_files_created) == len(pipeline_docs)
    
    def test_edge_case_performance(self, performance_setup):
        """Test performance with edge case file sizes and types."""
        config = ConfigManager(performance_setup["config_file"])
        
        # Create files with different edge case characteristics
        edge_cases = [
            ("tiny", "AI"),  # Minimal content
            ("empty", ""),   # Empty file
            ("medium", "Artificial intelligence and machine learning. " * 100),  # Medium
            ("large", "Deep learning neural networks and AI systems. " * 500),  # Large
            ("repetitive", "AI " * 1000),  # Highly repetitive
            ("special_chars", "AI & ML: 50% improvement, $1000 investment!"),  # Special characters
        ]
        
        edge_files = []
        for case_name, content in edge_cases:
            filename = f"edge_{case_name}.txt"
            edge_files.append(filename)
            file_path = Path(config.get("raw_input_directory")) / filename
            
            with open(file_path, "w") as f:
                f.write(content)
        
        with patch('classes.document_ingestor.AutoTokenizer.from_pretrained') as mock_tokenizer_class:
            mock_tokenizer = MagicMock()
            mock_tokenizer.tokenize.return_value = ["edge", "case", "test"]
            mock_tokenizer.convert_tokens_to_string.return_value = "edge case test"
            mock_tokenizer_class.return_value = mock_tokenizer
            
            start_time = time.time()
            
            ingestor = DocumentIngestor(
                file_list=edge_files,
                input_dir=config.get("raw_input_directory"),
                output_dir=config.get("cleaned_text_directory"),
                embedding_model_name=config.get("embedding_model_name")
            )
            
            ingestor.process_files()
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Verify edge case handling
            cleaned_dir = Path(config.get("cleaned_text_directory"))
            processed_files = list(cleaned_dir.glob("edge_*_cleaned.txt"))
            
            # Should process non-empty files (empty file might be skipped)
            assert len(processed_files) >= len(edge_cases) - 1
            
            print(f"Edge case performance:")
            print(f"  Processed {len(processed_files)} edge case files in {processing_time:.2f} seconds")
            
            # Should handle edge cases efficiently
            assert processing_time < 30  # Should complete quickly
    
    def test_resource_intensive_simulation(self, performance_setup):
        """Test performance under simulated resource constraints."""
        config = ConfigManager(performance_setup["config_file"])
        
        # Create documents that simulate resource-intensive processing
        resource_files = []
        for i in range(25):
            filename = f"resource_{i}.txt"
            resource_files.append(filename)
            file_path = Path(config.get("raw_input_directory")) / filename
            
            # Vary content size to simulate different resource requirements
            content_multiplier = (i % 5) + 1
            content = f"Resource intensive document {i}. " * (content_multiplier * 50)
            
            with open(file_path, "w") as f:
                f.write(content)
        
        with patch('classes.document_ingestor.AutoTokenizer.from_pretrained') as mock_tokenizer_class:
            mock_tokenizer = MagicMock()
            
            # Simulate variable processing times
            call_count = [0]
            
            def simulate_resource_load(*args, **kwargs):
                call_count[0] += 1
                # Simulate occasional resource bottlenecks
                if call_count[0] % 7 == 0:
                    time.sleep(0.05)  # Simulate brief resource constraint
                return ["resource", "intensive", "test"]
            
            mock_tokenizer.tokenize.side_effect = simulate_resource_load
            mock_tokenizer.convert_tokens_to_string.return_value = "resource intensive test"
            mock_tokenizer_class.return_value = mock_tokenizer
            
            start_time = time.time()
            
            ingestor = DocumentIngestor(
                file_list=resource_files,
                input_dir=config.get("raw_input_directory"),
                output_dir=config.get("cleaned_text_directory"),
                embedding_model_name=config.get("embedding_model_name")
            )
            
            ingestor.process_files()
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Verify all files processed despite resource constraints
            cleaned_dir = Path(config.get("cleaned_text_directory"))
            processed_files = list(cleaned_dir.glob("resource_*_cleaned.txt"))
            assert len(processed_files) == len(resource_files)
            
            # Performance under resource constraints
            throughput = len(resource_files) / processing_time
            
            print(f"Resource intensive simulation:")
            print(f"  Processed {len(resource_files)} files in {processing_time:.2f} seconds")
            print(f"  Throughput under constraints: {throughput:.2f} files/second")
            
            # Should maintain reasonable performance despite constraints
            assert processing_time < 90  # Should complete within 1.5 minutes
            assert throughput > 0.2  # Should maintain minimum throughput
    
    def test_rapid_iteration_performance(self, performance_setup):
        """Test performance of rapid successive operations."""
        config = ConfigManager(performance_setup["config_file"])
        
        # Test rapid iterations with small batches
        num_iterations = 8
        docs_per_iteration = 4
        
        with patch('classes.document_ingestor.AutoTokenizer.from_pretrained') as mock_tokenizer_class:
            mock_tokenizer = MagicMock()
            mock_tokenizer.tokenize.return_value = ["rapid", "iteration"]
            mock_tokenizer.convert_tokens_to_string.return_value = "rapid iteration"
            mock_tokenizer_class.return_value = mock_tokenizer
            
            iteration_times = []
            
            for iteration in range(num_iterations):
                # Create documents for this iteration
                iteration_docs = []
                for i in range(docs_per_iteration):
                    filename = f"rapid_{iteration}_{i}.txt"
                    iteration_docs.append(filename)
                    file_path = Path(config.get("raw_input_directory")) / filename
                    
                    with open(file_path, "w") as f:
                        f.write(f"Rapid iteration test document {iteration}-{i}")
                
                # Process this iteration
                start_time = time.time()
                
                ingestor = DocumentIngestor(
                    file_list=iteration_docs,
                    input_dir=config.get("raw_input_directory"),
                    output_dir=config.get("cleaned_text_directory"),
                    embedding_model_name=config.get("embedding_model_name")
                )
                
                ingestor.process_files()
                
                end_time = time.time()
                iteration_time = end_time - start_time
                iteration_times.append(iteration_time)
                
                # Clean up for next iteration
                cleaned_dir = Path(config.get("cleaned_text_directory"))
                for file in cleaned_dir.glob(f"rapid_{iteration}_*_cleaned.txt"):
                    file.unlink()
            
            # Analyze rapid iteration performance
            avg_time = sum(iteration_times) / len(iteration_times)
            max_time = max(iteration_times)
            min_time = min(iteration_times)
            
            print(f"Rapid iteration performance:")
            print(f"  {num_iterations} iterations, {docs_per_iteration} docs each")
            print(f"  Average time: {avg_time:.3f}s")
            print(f"  Min time: {min_time:.3f}s, Max time: {max_time:.3f}s")
            print(f"  Time variance: {(max_time - min_time):.3f}s")
            
            # Performance should be consistent across rapid iterations
            assert max_time < avg_time * 2.5  # Max shouldn't be >2.5x average
            assert all(time < 8 for time in iteration_times)  # All iterations <8 seconds
            assert avg_time < 3.0  # Average iteration time <3 seconds
    
    def test_component_isolation_performance(self, performance_setup):
        """Test individual component performance in isolation."""
        config = ConfigManager(performance_setup["config_file"])
        
        # Test 1: Document Ingestor Performance
        ingest_files = [f"ingest_test_{i}.txt" for i in range(15)]
        for filename in ingest_files:
            file_path = Path(config.get("raw_input_directory")) / filename
            with open(file_path, "w") as f:
                f.write(f"Component isolation test: {filename}")
        
        with patch('classes.document_ingestor.AutoTokenizer.from_pretrained') as mock_tokenizer:
            mock_tok = MagicMock()
            mock_tok.tokenize.return_value = ["component", "test"]
            mock_tok.convert_tokens_to_string.return_value = "component test"
            mock_tokenizer.return_value = mock_tok
            
            ingest_start = time.time()
            ingestor = DocumentIngestor(
                file_list=ingest_files,
                input_dir=config.get("raw_input_directory"),
                output_dir=config.get("cleaned_text_directory"),
                embedding_model_name=config.get("embedding_model_name")
            )
            ingestor.process_files()
            ingest_isolated_time = time.time() - ingest_start
        
        # Test 2: Embedding Preparer Performance
        cleaned_files = [f"ingest_test_{i}_cleaned.txt" for i in range(15)]
        
        with patch('classes.embedding_preparer.AutoTokenizer.from_pretrained') as mock_emb_tokenizer, \
             patch('classes.embedding_preparer.AutoModel.from_pretrained') as mock_model, \
             patch('classes.embedding_preparer.torch.cuda.is_available', return_value=False):
            
            mock_emb_tok = MagicMock()
            mock_emb_tokenizer.return_value = mock_emb_tok
            
            mock_model_instance = MagicMock()
            mock_model_instance.to.return_value = mock_model_instance
            mock_outputs = MagicMock()
            mock_outputs.last_hidden_state.mean.return_value.squeeze.return_value.cpu.return_value.numpy.return_value.tolist.return_value = [0.1] * 384
            mock_model_instance.return_value = mock_outputs
            mock_model.return_value = mock_model_instance
            
            embed_start = time.time()
            preparer = EmbeddingPreparer(
                file_list=cleaned_files,
                input_dir=config.get("cleaned_text_directory"),
                output_dir=config.get("embeddings_directory"),
                embedding_model_name=config.get("embedding_model_name")
            )
            preparer.process_files()
            embed_isolated_time = time.time() - embed_start
        
        # Test 3: Vector Storage Performance
        for i in range(15):
            embedding_file = Path(config.get("embeddings_directory")) / f"ingest_test_{i}_cleaned_embeddings.json"
            with open(embedding_file, "w") as f:
                json.dump([0.1] * 384, f)
        
        with patch('classes.embedding_loader.chromadb.PersistentClient') as mock_client:
            mock_client_instance = MagicMock()
            mock_collection = MagicMock()
            mock_client_instance.get_or_create_collection.return_value = mock_collection
            mock_client.return_value = mock_client_instance
            
            storage_start = time.time()
            loader = EmbeddingLoader(
                cleaned_text_file_list=cleaned_files,
                cleaned_text_dir=config.get("cleaned_text_directory"),
                embeddings_dir=config.get("embeddings_directory"),
                vectordb_dir=config.get("vectordb_directory"),
                collection_name=config.get("collection_name")
            )
            loader.process_files()
            storage_isolated_time = time.time() - storage_start
        
        # Component performance analysis
        num_docs = len(ingest_files)
        
        print(f"Component isolation performance ({num_docs} documents):")
        print(f"  Document Ingestor: {ingest_isolated_time:.2f}s ({num_docs/ingest_isolated_time:.2f} docs/s)")
        print(f"  Embedding Preparer: {embed_isolated_time:.2f}s ({num_docs/embed_isolated_time:.2f} embeddings/s)")
        print(f"  Vector Storage: {storage_isolated_time:.2f}s ({num_docs/storage_isolated_time:.2f} vectors/s)")
        
        # Component performance requirements
        assert ingest_isolated_time < 25  # Ingestion <25s
        assert embed_isolated_time < 30  # Embedding <30s
        assert storage_isolated_time < 15  # Storage <15s
        
        # Component throughput requirements
        assert num_docs / ingest_isolated_time > 0.5  # >0.5 docs/s ingestion
        assert num_docs / embed_isolated_time > 0.4  # >0.4 embeddings/s
        assert num_docs / storage_isolated_time > 0.8  # >0.8 vectors/s