import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from src.langchain_rag_processor import LangChainRAGProcessor
from src.langchain_vector_store import LangChainVectorStore


class TestLangChainRAGProcessor:
    """Test cases for LangChainRAGProcessor."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        return Mock(spec=LangChainVectorStore)
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(page_content="This is about artificial intelligence", metadata={"source": "ai.txt"}),
            Document(page_content="Machine learning is a subset of AI", metadata={"source": "ml.txt"}),
            Document(page_content="Deep learning uses neural networks", metadata={"source": "dl.txt"})
        ]
    
    @patch('src.langchain_rag_processor.OllamaLLM')
    def test_init_ollama_default(self, mock_ollama, mock_vector_store):
        """Test initialization with default Ollama settings."""
        mock_llm = Mock()
        mock_ollama.return_value = mock_llm
        
        processor = LangChainRAGProcessor(
            vector_store=mock_vector_store,
            llm_provider="ollama",
            llm_model_name="llama3.2"
        )
        
        assert processor.vector_store == mock_vector_store
        assert processor.llm_provider == "ollama"
        assert processor.llm_model_name == "llama3.2"
        assert processor.max_context_length == 4000
        
        # Verify Ollama LLM initialization
        mock_ollama.assert_called_once_with(model="llama3.2", base_url="http://localhost:11434")
    
    @patch('src.langchain_rag_processor.OllamaLLM')
    def test_init_ollama_with_url(self, mock_ollama, mock_vector_store):
        """Test initialization with Ollama API URL."""
        mock_llm = Mock()
        mock_ollama.return_value = mock_llm
        
        processor = LangChainRAGProcessor(
            vector_store=mock_vector_store,
            llm_provider="ollama",
            llm_model_name="llama3.2",
            llm_api_url="http://localhost:11434/api/generate"
        )
        
        # Should strip /api/generate from URL
        mock_ollama.assert_called_once_with(
            model="llama3.2",
            base_url="http://localhost:11434"
        )
    
    @patch('src.langchain_rag_processor.OllamaLLM')
    def test_init_ollama_url_formats(self, mock_ollama, mock_vector_store):
        """Test different URL format handling."""
        test_cases = [
            ("http://localhost:11434/api/generate", "http://localhost:11434"),
            ("http://localhost:11434/", "http://localhost:11434"),
            ("http://localhost:11434", "http://localhost:11434"),
        ]
        
        for input_url, expected_base in test_cases:
            mock_ollama.reset_mock()
            
            LangChainRAGProcessor(
                vector_store=mock_vector_store,
                llm_provider="ollama",
                llm_model_name="test",
                llm_api_url=input_url
            )
            
            mock_ollama.assert_called_once_with(
                model="test",
                base_url=expected_base
            )
    
    @patch('src.langchain_rag_processor.ChatOpenAI')
    def test_init_openai(self, mock_openai, mock_vector_store):
        """Test initialization with OpenAI."""
        mock_llm = Mock()
        mock_openai.return_value = mock_llm
        
        processor = LangChainRAGProcessor(
            vector_store=mock_vector_store,
            llm_provider="openai",
            llm_model_name="gpt-3.5-turbo",
            openai_api_key="test-key"
        )
        
        assert processor.llm_provider == "openai"
        
        # Verify OpenAI LLM initialization
        mock_openai.assert_called_once_with(
            model="gpt-3.5-turbo",
            api_key="test-key",
            temperature=0.7
        )
    
    @patch('src.langchain_rag_processor.OllamaLLM')
    def test_init_unsupported_provider(self, mock_ollama, mock_vector_store):
        """Test initialization with unsupported provider."""
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            LangChainRAGProcessor(
                vector_store=mock_vector_store,
                llm_provider="unsupported"
            )
    
    @patch('src.langchain_rag_processor.OllamaLLM')
    def test_init_llm_failure(self, mock_ollama, mock_vector_store):
        """Test LLM initialization failure."""
        mock_ollama.side_effect = Exception("LLM initialization failed")
        
        with pytest.raises(Exception, match="LLM initialization failed"):
            LangChainRAGProcessor(
                vector_store=mock_vector_store,
                llm_provider="ollama"
            )
    
    @patch('src.langchain_rag_processor.OllamaLLM')
    def test_query_with_rag_success(self, mock_ollama, mock_vector_store, sample_documents):
        """Test successful RAG query."""
        # Setup mocks
        mock_llm = Mock()
        mock_ollama.return_value = mock_llm
        
        # Mock vector store to return documents with scores
        mock_vector_store.similarity_search_with_scores.return_value = [
            (sample_documents[0], 0.9),
            (sample_documents[1], 0.8)
        ]
        
        processor = LangChainRAGProcessor(
            vector_store=mock_vector_store,
            llm_provider="ollama"
        )
        
        # Mock the RAG chain
        with patch.object(processor, 'rag_chain') as mock_chain:
            mock_chain.invoke.return_value = "This is a test response about AI."
            
            result = processor.query("What is AI?", use_rag=True)
            
            assert result["response"] == "This is a test response about AI."
            assert result["context_used"] is True
            assert len(result["retrieved_docs"]) == 2
            assert len(result["scores"]) == 2
            assert result["scores"] == [0.9, 0.8]
            
            # Verify vector store was called
            mock_vector_store.similarity_search_with_scores.assert_called_once_with("What is AI?", k=4)
            
            # Verify RAG chain was invoked
            mock_chain.invoke.assert_called_once_with({"question": "What is AI?"})
    
    @patch('src.langchain_rag_processor.OllamaLLM')
    def test_query_with_rag_no_docs(self, mock_ollama, mock_vector_store):
        """Test RAG query with no relevant documents."""
        mock_llm = Mock()
        mock_ollama.return_value = mock_llm
        
        # Mock vector store to return no documents
        mock_vector_store.similarity_search_with_scores.return_value = []
        
        processor = LangChainRAGProcessor(
            vector_store=mock_vector_store,
            llm_provider="ollama"
        )
        
        result = processor.query("What is AI?", use_rag=True)
        
        assert "don't have any relevant information" in result["response"]
        assert result["context_used"] is False
        assert result["retrieved_docs"] == []
        assert result["scores"] == []
    
    @patch('src.langchain_rag_processor.OllamaLLM')
    def test_query_with_rag_context_truncation(self, mock_ollama, mock_vector_store):
        """Test context truncation for long content."""
        mock_llm = Mock()
        mock_ollama.return_value = mock_llm
        
        # Create a document with very long content
        long_content = "A" * 5000  # Longer than default max_context_length
        long_doc = Document(page_content=long_content, metadata={"source": "long.txt"})
        
        mock_vector_store.similarity_search_with_scores.return_value = [(long_doc, 0.9)]
        
        processor = LangChainRAGProcessor(
            vector_store=mock_vector_store,
            llm_provider="ollama",
            max_context_length=1000
        )
        
        with patch.object(processor, 'rag_chain') as mock_chain:
            mock_chain.invoke.return_value = "Truncated response"
            
            result = processor.query("Test query", use_rag=True)
            
            # Context should be truncated
            assert result["context_length"] == 1003  # 1000 + "..."
            assert result["response"] == "Truncated response"
    
    @patch('src.langchain_rag_processor.OllamaLLM')
    def test_query_without_rag(self, mock_ollama, mock_vector_store):
        """Test direct LLM query without RAG."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = "Direct LLM response"
        mock_ollama.return_value = mock_llm
        
        processor = LangChainRAGProcessor(
            vector_store=mock_vector_store,
            llm_provider="ollama"
        )
        
        result = processor.query("What is AI?", use_rag=False)
        
        assert result["response"] == "Direct LLM response"
        assert result["context_used"] is False
        assert result["retrieved_docs"] == []
        assert result["scores"] == []
        
        # Verify LLM was called directly
        mock_llm.invoke.assert_called_once_with("What is AI?")
        
        # Verify vector store was not used
        mock_vector_store.similarity_search_with_scores.assert_not_called()
    
    @patch('src.langchain_rag_processor.OllamaLLM')
    def test_query_error_handling(self, mock_ollama, mock_vector_store):
        """Test query error handling."""
        mock_llm = Mock()
        mock_ollama.return_value = mock_llm
        
        # Mock vector store to raise exception
        mock_vector_store.similarity_search_with_scores.side_effect = Exception("Search failed")
        
        processor = LangChainRAGProcessor(
            vector_store=mock_vector_store,
            llm_provider="ollama"
        )
        
        result = processor.query("What is AI?", use_rag=True)
        
        assert "Error processing query" in result["response"]
        assert result["context_used"] is False
        assert result["retrieved_docs"] == []
        assert "error" in result
    
    @patch('src.langchain_rag_processor.OllamaLLM')
    def test_retrieve_documents(self, mock_ollama, mock_vector_store, sample_documents):
        """Test document retrieval."""
        mock_llm = Mock()
        mock_ollama.return_value = mock_llm
        
        # Mock vector store response
        mock_vector_store.similarity_search_with_scores.return_value = [
            (sample_documents[0], 0.9),
            (sample_documents[1], 0.7)
        ]
        
        processor = LangChainRAGProcessor(
            vector_store=mock_vector_store,
            llm_provider="ollama"
        )
        
        result = processor.retrieve_documents("test query", k=2)
        
        assert len(result) == 2
        assert result[0]["content"] == sample_documents[0].page_content
        assert result[0]["metadata"] == sample_documents[0].metadata
        assert result[0]["score"] == 0.9
        assert result[1]["content"] == sample_documents[1].page_content
        assert result[1]["score"] == 0.7
        
        mock_vector_store.similarity_search_with_scores.assert_called_once_with("test query", k=2)
    
    @patch('src.langchain_rag_processor.OllamaLLM')
    def test_retrieve_documents_with_threshold(self, mock_ollama, mock_vector_store, sample_documents):
        """Test document retrieval with score threshold."""
        mock_llm = Mock()
        mock_ollama.return_value = mock_llm
        
        # Mock vector store response with varying scores
        mock_vector_store.similarity_search_with_scores.return_value = [
            (sample_documents[0], 0.9),
            (sample_documents[1], 0.7),
            (sample_documents[2], 0.3)
        ]
        
        processor = LangChainRAGProcessor(
            vector_store=mock_vector_store,
            llm_provider="ollama"
        )
        
        result = processor.retrieve_documents("test query", k=3, score_threshold=0.5)
        
        # Should only return documents with score >= 0.5
        assert len(result) == 2
        assert result[0]["score"] == 0.9
        assert result[1]["score"] == 0.7
    
    @patch('src.langchain_rag_processor.OllamaLLM')
    def test_retrieve_documents_error(self, mock_ollama, mock_vector_store):
        """Test document retrieval error handling."""
        mock_llm = Mock()
        mock_ollama.return_value = mock_llm
        
        # Mock vector store to raise exception
        mock_vector_store.similarity_search_with_scores.side_effect = Exception("Retrieval failed")
        
        processor = LangChainRAGProcessor(
            vector_store=mock_vector_store,
            llm_provider="ollama"
        )
        
        result = processor.retrieve_documents("test query")
        
        assert result == []