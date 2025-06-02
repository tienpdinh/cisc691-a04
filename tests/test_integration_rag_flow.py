"""Integration tests for RAG query flow."""
import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from classes.config_manager import ConfigManager
from classes.chromadb_retriever import ChromaDBRetriever
from classes.llm_client import LLMClient
from classes.rag_query_processor import RAGQueryProcessor

@pytest.mark.integration
class TestRAGQueryFlow:
    """Integration tests for RAG query processing."""
    
    @pytest.fixture
    def rag_setup(self):
        """Set up RAG query environment."""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create directory structure
        (temp_dir / "vectordb").mkdir()
        
        # Create config
        config = {
            "log_level": "INFO",
            "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "collection_name": "test_documents",
            "retriever_min_score_threshold": "0.5",
            "llm_api_url": "http://localhost:11434/api/generate",
            "llm_model_name": "llama3.1:8b",
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
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_end_to_end_rag_query_with_results(self, rag_setup):
        """Test complete RAG query flow with document retrieval."""
        config = ConfigManager(rag_setup["config_file"])
        
        # Mock ChromaDB, SentenceTransformer, and LLM
        with patch('classes.chromadb_retriever.chromadb.PersistentClient') as mock_client_class, \
             patch('classes.chromadb_retriever.SentenceTransformer') as mock_transformer_class, \
             patch('classes.llm_client.requests.post') as mock_post:
            
            # Mock ChromaDB client and collection
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_collection.query.return_value = {
                "ids": [["ai_basics_cleaned.txt"]],
                "metadatas": [[{
                    "text": "Artificial intelligence (AI) is a field of computer science that aims to create intelligent machines. Machine learning is a subset of AI.",
                    "source": "ai_basics.txt"
                }]],
                "distances": [[0.25]]  # Good similarity score
            }
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client
            
            # Mock sentence transformer
            mock_transformer = MagicMock()
            mock_transformer.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
            mock_transformer_class.return_value = mock_transformer
            
            # Mock LLM response
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "response": "Based on the provided context, artificial intelligence (AI) is a field of computer science that aims to create intelligent machines. Machine learning is a specific subset of AI that enables systems to learn and improve from data."
            }
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
            
            # Execute RAG query
            query = "What is artificial intelligence and machine learning?"
            response = rag_processor.query(query)
            
            # Verify the complete flow
            assert "artificial intelligence" in response.lower()
            assert "machine learning" in response.lower()
            
            # Verify each component was called
            mock_transformer.encode.assert_called_once_with(query, normalize_embeddings=True)
            mock_collection.query.assert_called_once()
            mock_post.assert_called_once()
            
            # Verify LLM received context
            call_args = mock_post.call_args
            request_data = json.loads(call_args[1]['data'])
            assert query in request_data['prompt']
            assert "Artificial intelligence (AI) is a field of computer science" in request_data['prompt']
    
    def test_rag_query_with_no_relevant_documents(self, rag_setup):
        """Test RAG query when no relevant documents are found."""
        config = ConfigManager(rag_setup["config_file"])
        
        with patch('classes.chromadb_retriever.chromadb.PersistentClient') as mock_client_class, \
             patch('classes.chromadb_retriever.SentenceTransformer') as mock_transformer_class, \
             patch('classes.llm_client.requests.post') as mock_post:
            
            # Mock ChromaDB with no relevant results
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_collection.query.return_value = {
                "ids": [[]],
                "metadatas": [[]],
                "distances": [[]]
            }
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client
            
            # Mock sentence transformer
            mock_transformer = MagicMock()
            mock_transformer.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
            mock_transformer_class.return_value = mock_transformer
            
            # Mock LLM response
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "response": "I don't have enough context to answer that question accurately."
            }
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
            
            # Execute query
            response = rag_processor.query("What is quantum computing?")
            
            # Should still get LLM response even without context
            assert "don't have enough context" in response
            
            # Verify LLM prompt included "No relevant context found"
            call_args = mock_post.call_args
            request_data = json.loads(call_args[1]['data'])
            assert "No relevant context found" in request_data['prompt']
    
    def test_direct_llm_query_without_rag(self, rag_setup):
        """Test direct LLM query without RAG retrieval."""
        config = ConfigManager(rag_setup["config_file"])
        
        with patch('classes.llm_client.requests.post') as mock_post:
            # Mock LLM response
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "response": "This is a direct response from the LLM without any document context."
            }
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response
            
            # Create LLM client
            llm_client = LLMClient(
                llm_api_url=config.get("llm_api_url"),
                llm_model_name=config.get("llm_model_name")
            )
            
            # Create RAG processor without retrieval
            rag_processor = RAGQueryProcessor(
                llm_client=llm_client,
                retriever=None,
                use_rag=False
            )
            
            # Execute query
            response = rag_processor.query("What is artificial intelligence?")
            
            # Verify response
            assert "direct response from the LLM" in response
            
            # Verify LLM was called with no context
            call_args = mock_post.call_args
            request_data = json.loads(call_args[1]['data'])
            assert "What is artificial intelligence?" in request_data['prompt']
            assert "No relevant context found" in request_data['prompt']
    
    def test_rag_query_with_multiple_documents(self, rag_setup):
        """Test RAG query that returns multiple relevant documents."""
        config = ConfigManager(rag_setup["config_file"])
        
        with patch('classes.chromadb_retriever.chromadb.PersistentClient') as mock_client_class, \
             patch('classes.chromadb_retriever.SentenceTransformer') as mock_transformer_class, \
             patch('classes.llm_client.requests.post') as mock_post:
            
            # Mock ChromaDB with multiple results
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_collection.query.return_value = {
                "ids": [["ai_basics_cleaned.txt", "deep_learning_cleaned.txt", "ml_paper_cleaned.txt"]],
                "metadatas": [[
                    {
                        "text": "Artificial intelligence (AI) is a field of computer science that creates intelligent machines.",
                        "source": "ai_basics.txt"
                    },
                    {
                        "text": "Deep learning uses neural networks with multiple layers to process complex data patterns.",
                        "source": "deep_learning.txt"
                    },
                    {
                        "text": "Machine learning algorithms can learn from data without explicit programming instructions.",
                        "source": "ml_paper.txt"
                    }
                ]],
                "distances": [[0.2, 0.3, 0.4]]  # Different similarity scores
            }
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client
            
            # Mock sentence transformer
            mock_transformer = MagicMock()
            mock_transformer.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
            mock_transformer_class.return_value = mock_transformer
            
            # Mock LLM response
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "response": "AI encompasses machine learning and deep learning. AI creates intelligent machines, machine learning learns from data, and deep learning uses neural networks."
            }
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
            
            # Execute query
            response = rag_processor.query("Explain AI, machine learning, and deep learning")
            
            # Verify comprehensive response
            assert "AI" in response or "artificial intelligence" in response.lower()
            assert "machine learning" in response.lower()
            assert "deep learning" in response.lower()
            
            # ChromaDBRetriever should return only the best match (lowest distance)
            # Verify retrieval was called
            mock_collection.query.assert_called_once()
    
    def test_rag_error_handling_llm_failure(self, rag_setup):
        """Test RAG behavior when LLM service fails."""
        config = ConfigManager(rag_setup["config_file"])
        
        with patch('classes.chromadb_retriever.chromadb.PersistentClient') as mock_client_class, \
             patch('classes.chromadb_retriever.SentenceTransformer') as mock_transformer_class, \
             patch('classes.llm_client.requests.post') as mock_post:
            
            # Mock successful retrieval
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_collection.query.return_value = {
                "ids": [["doc1.txt"]],
                "metadatas": [[{"text": "Some AI content", "source": "doc1.txt"}]],
                "distances": [[0.3]]
            }
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client
            
            mock_transformer = MagicMock()
            mock_transformer.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
            mock_transformer_class.return_value = mock_transformer
            
            # Mock LLM failure
            mock_post.side_effect = Exception("LLM service unavailable")
            
            # Create components
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
            
            # Should handle error gracefully
            response = rag_processor.query("What is AI?")
            
            # Should return error message, not crash
            assert "Error: Could not connect to the LLM" in response
            
            # Retrieval should still work
            mock_collection.query.assert_called_once()
    
    def test_query_with_low_similarity_scores(self, rag_setup):
        """Test RAG behavior with low similarity scores."""
        config = ConfigManager(rag_setup["config_file"])
        
        with patch('classes.chromadb_retriever.chromadb.PersistentClient') as mock_client_class, \
             patch('classes.chromadb_retriever.SentenceTransformer') as mock_transformer_class:
            
            # Mock ChromaDB with high distance (low similarity)
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_collection.query.return_value = {
                "ids": [["irrelevant_doc.txt"]],
                "metadatas": [[{"text": "Completely unrelated content about cooking", "source": "irrelevant_doc.txt"}]],
                "distances": [[0.9]]  # Very high distance (low similarity)
            }
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client
            
            mock_transformer = MagicMock()
            mock_transformer.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
            mock_transformer_class.return_value = mock_transformer
            
            # Create retriever
            retriever = ChromaDBRetriever(
                embedding_model_name=config.get("embedding_model_name"),
                collection_name=config.get("collection_name"),
                vectordb_dir=config.get("vectordb_directory"),
                score_threshold=float(config.get("retriever_min_score_threshold"))
            )
            
            # Query should return no results due to low similarity
            results = retriever.query("artificial intelligence", top_k=5)
            
            # Should filter out low-similarity results
            assert len(results) == 0
    
    def test_context_extraction_and_relevance(self, rag_setup):
        """Test context extraction and relevance filtering."""
        config = ConfigManager(rag_setup["config_file"])
        
        with patch('classes.chromadb_retriever.chromadb.PersistentClient') as mock_client_class, \
             patch('classes.chromadb_retriever.SentenceTransformer') as mock_transformer_class:
            
            # Mock ChromaDB with document containing paragraphs
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_collection.query.return_value = {
                "ids": [["multi_paragraph_doc.txt"]],
                "metadatas": [[{
                    "text": "First paragraph about general topics.\n\nSecond paragraph discusses artificial intelligence and machine learning concepts in detail.\n\nThird paragraph covers unrelated subjects.",
                    "source": "multi_paragraph_doc.txt"
                }]],
                "distances": [[0.25]]
            }
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client
            
            mock_transformer = MagicMock()
            mock_transformer.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
            mock_transformer_class.return_value = mock_transformer
            
            # Create retriever
            retriever = ChromaDBRetriever(
                embedding_model_name=config.get("embedding_model_name"),
                collection_name=config.get("collection_name"),
                vectordb_dir=config.get("vectordb_directory"),
                score_threshold=float(config.get("retriever_min_score_threshold"))
            )
            
            # Query for AI content
            results = retriever.query("artificial intelligence", top_k=1)
            
            # Should find relevant paragraph
            assert len(results) == 1
            result = results[0]
            
            # Context should contain the relevant paragraph
            assert "artificial intelligence and machine learning" in result["context"]
            assert result["score"] == 0.25
            assert result["source"] == "multi_paragraph_doc.txt"
    
    def test_query_word_relevance_filtering(self, rag_setup):
        """Test that irrelevant documents are filtered out based on query words."""
        config = ConfigManager(rag_setup["config_file"])
        
        with patch('classes.chromadb_retriever.chromadb.PersistentClient') as mock_client_class, \
             patch('classes.chromadb_retriever.SentenceTransformer') as mock_transformer_class:
            
            # Mock ChromaDB with document that has good similarity but no query words
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_collection.query.return_value = {
                "ids": [["irrelevant_content.txt"]],
                "metadatas": [[{
                    "text": "This document discusses cooking recipes and food preparation techniques.",
                    "source": "irrelevant_content.txt"
                }]],
                "distances": [[0.3]]  # Good similarity score
            }
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client
            
            mock_transformer = MagicMock()
            mock_transformer.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
            mock_transformer_class.return_value = mock_transformer
            
            # Create retriever
            retriever = ChromaDBRetriever(
                embedding_model_name=config.get("embedding_model_name"),
                collection_name=config.get("collection_name"),
                vectordb_dir=config.get("vectordb_directory"),
                score_threshold=float(config.get("retriever_min_score_threshold"))
            )
            
            # Query for AI content
            results = retriever.query("artificial intelligence machine learning", top_k=1)
            
            # Should filter out document with no relevant words
            assert len(results) == 0
    
    def test_rag_processor_context_integration(self, rag_setup):
        """Test how RAG processor integrates retrieved context with LLM."""
        config = ConfigManager(rag_setup["config_file"])
        
        with patch('classes.chromadb_retriever.chromadb.PersistentClient') as mock_client_class, \
             patch('classes.chromadb_retriever.SentenceTransformer') as mock_transformer_class, \
             patch('classes.llm_client.requests.post') as mock_post:
            
            # Mock retrieval with specific context
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_collection.query.return_value = {
                "ids": [["ai_definition.txt"]],
                "metadatas": [[{
                    "text": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans.",
                    "source": "ai_definition.txt"
                }]],
                "distances": [[0.2]]
            }
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client
            
            mock_transformer = MagicMock()
            mock_transformer.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
            mock_transformer_class.return_value = mock_transformer
            
            # Mock LLM response
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "response": "Based on the provided context, AI is the simulation of human intelligence in machines that can think and learn."
            }
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response
            
            # Create components
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
            
            # Execute query
            response = rag_processor.query("What is artificial intelligence?")
            
            # Verify response uses context
            assert "simulation of human intelligence" in response
            
            # Verify LLM received the context
            call_args = mock_post.call_args
            request_data = json.loads(call_args[1]['data'])
            prompt = request_data['prompt']
            
            # Check that context was included in prompt
            assert "Artificial Intelligence (AI) refers to the simulation" in prompt
            assert "What is artificial intelligence?" in prompt
            assert "Context:" in prompt
    
    def test_retrieval_scoring_and_ranking(self, rag_setup):
        """Test that retrieval properly scores and ranks results."""
        config = ConfigManager(rag_setup["config_file"])
        
        with patch('classes.chromadb_retriever.chromadb.PersistentClient') as mock_client_class, \
             patch('classes.chromadb_retriever.SentenceTransformer') as mock_transformer_class:
            
            # Mock ChromaDB with multiple results at different distances
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_collection.query.return_value = {
                "ids": [["doc1.txt", "doc2.txt", "doc3.txt"]],
                "metadatas": [[
                    {"text": "Machine learning is a subset of artificial intelligence", "source": "doc1.txt"},
                    {"text": "Artificial intelligence encompasses machine learning and deep learning", "source": "doc2.txt"},
                    {"text": "Deep learning and artificial intelligence are related fields", "source": "doc3.txt"}
                ]],
                "distances": [[0.4, 0.1, 0.3]]  # doc2 has best score (0.1)
            }
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client
            
            mock_transformer = MagicMock()
            mock_transformer.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
            mock_transformer_class.return_value = mock_transformer
            
            # Create retriever
            retriever = ChromaDBRetriever(
                embedding_model_name=config.get("embedding_model_name"),
                collection_name=config.get("collection_name"),
                vectordb_dir=config.get("vectordb_directory"),
                score_threshold=float(config.get("retriever_min_score_threshold"))
            )
            
            # Query should return only the best result
            results = retriever.query("artificial intelligence", top_k=5)
            
            # Should return only the best match (ChromaDBRetriever returns top 1)
            assert len(results) == 1
            assert results[0]["id"] == "doc2.txt"  # Best score (0.1)
            assert results[0]["score"] == 0.1
            assert "encompasses machine learning and deep learning" in results[0]["context"]
    
    def test_empty_query_handling(self, rag_setup):
        """Test handling of empty or whitespace-only queries."""
        config = ConfigManager(rag_setup["config_file"])
        
        with patch('classes.chromadb_retriever.chromadb.PersistentClient') as mock_client_class, \
             patch('classes.chromadb_retriever.SentenceTransformer') as mock_transformer_class, \
             patch('classes.llm_client.requests.post') as mock_post:
            
            # Mock components
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_collection.query.return_value = {"ids": [[]], "metadatas": [[]], "distances": [[]]}
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client
            
            mock_transformer = MagicMock()
            mock_transformer.encode.return_value.tolist.return_value = [0.0, 0.0, 0.0]
            mock_transformer_class.return_value = mock_transformer
            
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": "I need a specific question to provide a helpful answer."}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response
            
            # Create components
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
            
            # Test empty query
            response = rag_processor.query("")
            assert "need a specific question" in response
            
            # Test whitespace query
            response = rag_processor.query("   \n\t   ")
            assert "need a specific question" in response
    
    def test_very_long_query_handling(self, rag_setup):
        """Test handling of very long queries."""
        config = ConfigManager(rag_setup["config_file"])
        
        with patch('classes.chromadb_retriever.chromadb.PersistentClient') as mock_client_class, \
             patch('classes.chromadb_retriever.SentenceTransformer') as mock_transformer_class, \
             patch('classes.llm_client.requests.post') as mock_post:
            
            # Mock components
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_collection.query.return_value = {
                "ids": [["long_doc.txt"]],
                "metadatas": [[{"text": "Document about artificial intelligence", "source": "long_doc.txt"}]],
                "distances": [[0.3]]
            }
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client
            
            mock_transformer = MagicMock()
            mock_transformer.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
            mock_transformer_class.return_value = mock_transformer
            
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": "Here's information about AI based on your detailed question."}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response
            
            # Create components
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
            
            # Create very long query
            long_query = "Please explain artificial intelligence " * 50  # Very long query
            
            # Should handle long query without issues
            response = rag_processor.query(long_query)
            assert "information about AI" in response
            
            # Verify embedding was called with the long query
            mock_transformer.encode.assert_called_once()
            call_args = mock_transformer.encode.call_args[0]
            assert len(call_args[0]) > 1000  # Verify it's a long query
    
    def test_rag_with_special_characters_in_query(self, rag_setup):
        """Test RAG handling of queries with special characters."""
        config = ConfigManager(rag_setup["config_file"])
        
        with patch('classes.chromadb_retriever.chromadb.PersistentClient') as mock_client_class, \
             patch('classes.chromadb_retriever.SentenceTransformer') as mock_transformer_class, \
             patch('classes.llm_client.requests.post') as mock_post:
            
            # Mock components
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_collection.query.return_value = {
                "ids": [["special_doc.txt"]],
                "metadatas": [[{"text": "Document about AI & ML technologies", "source": "special_doc.txt"}]],
                "distances": [[0.3]]
            }
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client
            
            mock_transformer = MagicMock()
            mock_transformer.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
            mock_transformer_class.return_value = mock_transformer
            
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": "AI & ML are related technologies."}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response
            
            # Create components
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
            
            # Test query with special characters
            special_query = "What's the difference between AI & ML? How do they work together?"
            response = rag_processor.query(special_query)
            
            # Should handle special characters without issues
            assert "AI & ML are related" in response
            
            # Verify the query was processed
            mock_transformer.encode.assert_called_once()
            mock_post.assert_called_once()
    
    def test_rag_chunked_document_retrieval(self, rag_setup):
        """Test RAG retrieval from chunked documents."""
        config = ConfigManager(rag_setup["config_file"])
        
        with patch('classes.chromadb_retriever.chromadb.PersistentClient') as mock_client_class, \
             patch('classes.chromadb_retriever.SentenceTransformer') as mock_transformer_class, \
             patch('classes.llm_client.requests.post') as mock_post:
            
            # Mock ChromaDB with chunked document results
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_collection.query.return_value = {
                "ids": [["large_doc_chunk_3.txt"]],
                "metadatas": [[{
                    "text": "Chapter 3: Machine Learning Applications\n\nMachine learning has numerous applications in healthcare, including medical image analysis, drug discovery, and patient diagnosis. The technology enables doctors to make more accurate predictions about patient outcomes.",
                    "source": "large_doc_chunk_3.txt"
                }]],
                "distances": [[0.15]]
            }
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client
            
            mock_transformer = MagicMock()
            mock_transformer.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
            mock_transformer_class.return_value = mock_transformer
            
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "response": "Machine learning applications in healthcare include medical image analysis, drug discovery, and patient diagnosis."
            }
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response
            
            # Create components
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
            
            # Query about healthcare applications
            response = rag_processor.query("How is machine learning used in healthcare?")
            
            # Should return information from the relevant chunk
            assert "healthcare" in response.lower()
            assert "medical image analysis" in response
            
            # Verify context from specific chunk was used
            call_args = mock_post.call_args
            request_data = json.loads(call_args[1]['data'])
            assert "Chapter 3: Machine Learning Applications" in request_data['prompt']
    
    def test_rag_performance_with_concurrent_queries(self, rag_setup):
        """Test RAG system behavior with multiple concurrent-like queries."""
        config = ConfigManager(rag_setup["config_file"])
        
        with patch('classes.chromadb_retriever.chromadb.PersistentClient') as mock_client_class, \
             patch('classes.chromadb_retriever.SentenceTransformer') as mock_transformer_class, \
             patch('classes.llm_client.requests.post') as mock_post:
            
            # Mock components with different responses for different queries
            mock_client = MagicMock()
            mock_collection = MagicMock()
            
            # Set up different responses for different calls
            query_responses = [
                {
                    "ids": [["ai_doc.txt"]],
                    "metadatas": [[{"text": "AI content", "source": "ai_doc.txt"}]],
                    "distances": [[0.2]]
                },
                {
                    "ids": [["ml_doc.txt"]],
                    "metadatas": [[{"text": "ML content", "source": "ml_doc.txt"}]],
                    "distances": [[0.3]]
                },
                {
                    "ids": [["dl_doc.txt"]],
                    "metadatas": [[{"text": "DL content", "source": "dl_doc.txt"}]],
                    "distances": [[0.25]]
                }
            ]
            mock_collection.query.side_effect = query_responses
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client
            
            mock_transformer = MagicMock()
            mock_transformer.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
            mock_transformer_class.return_value = mock_transformer
            
            # Mock LLM responses
            llm_responses = [
                {"response": "AI is artificial intelligence"},
                {"response": "ML is machine learning"},
                {"response": "DL is deep learning"}
            ]
            mock_response = MagicMock()
            mock_response.json.side_effect = llm_responses
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response
            
            # Create components
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
            
            # Simulate multiple queries
            queries = [
                "What is artificial intelligence?",
                "Explain machine learning",
                "How does deep learning work?"
            ]
            
            responses = []
            for query in queries:
                response = rag_processor.query(query)
                responses.append(response)
            
            # Verify all queries were processed
            assert len(responses) == 3
            assert "artificial intelligence" in responses[0]
            assert "machine learning" in responses[1]
            assert "deep learning" in responses[2]
            
            # Verify all components were called for each query
            assert mock_collection.query.call_count == 3
            assert mock_post.call_count == 3
    
    def test_rag_fallback_behavior(self, rag_setup):
        """Test RAG fallback behavior when retrieval fails."""
        config = ConfigManager(rag_setup["config_file"])
        
        with patch('classes.chromadb_retriever.chromadb.PersistentClient') as mock_client_class, \
             patch('classes.chromadb_retriever.SentenceTransformer') as mock_transformer_class, \
             patch('classes.llm_client.requests.post') as mock_post:
            
            # Mock ChromaDB failure
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_collection.query.side_effect = Exception("Database connection failed")
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client
            
            mock_transformer = MagicMock()
            mock_transformer.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
            mock_transformer_class.return_value = mock_transformer
            
            # Mock LLM response for fallback
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "response": "I apologize, but I cannot access the document database right now. However, I can still help with general questions."
            }
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response
            
            # Create components
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
            
            # Should handle retrieval failure gracefully
            response = rag_processor.query("What is AI?")
            
            # Should still get LLM response even with retrieval failure
            assert "general questions" in response
            
            # LLM should still be called
            mock_post.assert_called_once()
    
    def test_rag_memory_efficiency(self, rag_setup):
        """Test RAG system memory efficiency with large contexts."""
        config = ConfigManager(rag_setup["config_file"])
        
        with patch('classes.chromadb_retriever.chromadb.PersistentClient') as mock_client_class, \
             patch('classes.chromadb_retriever.SentenceTransformer') as mock_transformer_class, \
             patch('classes.llm_client.requests.post') as mock_post:
            
            # Mock retrieval with very large document
            large_text = "AI and machine learning content. " * 1000  # Large text content
            
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_collection.query.return_value = {
                "ids": [["large_document.txt"]],
                "metadatas": [[{"text": large_text, "source": "large_document.txt"}]],
                "distances": [[0.2]]
            }
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client
            
            mock_transformer = MagicMock()
            mock_transformer.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
            mock_transformer_class.return_value = mock_transformer
            
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": "Summary of the large document content."}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response
            
            # Create components
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
            
            # Should handle large context efficiently
            response = rag_processor.query("Summarize the document")
            
            assert "Summary of the large document" in response
            
            # Verify context was processed (should extract relevant portion)
            call_args = mock_post.call_args
            request_data = json.loads(call_args[1]['data'])
            
            # Context should be included but may be truncated for efficiency
            assert "AI and machine learning content" in request_data['prompt']