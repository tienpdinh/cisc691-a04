import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from langchain_core.documents import Document
from langchain.schema import Document as LangChainDocument

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
    @pytest.mark.asyncio
    async def test_query_with_rag_success(self, mock_ollama, mock_vector_store, sample_documents):
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
            
            result = await processor.query("What is AI?", use_rag=True)
            
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
    @pytest.mark.asyncio
    async def test_query_with_rag_no_docs(self, mock_ollama, mock_vector_store):
        """Test RAG query with no relevant documents."""
        mock_llm = Mock()
        mock_ollama.return_value = mock_llm
        
        # Mock vector store to return no documents
        mock_vector_store.similarity_search_with_scores.return_value = []
        
        processor = LangChainRAGProcessor(
            vector_store=mock_vector_store,
            llm_provider="ollama"
        )
        
        result = await processor.query("What is AI?", use_rag=True)
        
        assert "don't have any relevant information" in result["response"]
        assert result["context_used"] is False
        assert result["retrieved_docs"] == []
        assert result["scores"] == []
    
    @patch('src.langchain_rag_processor.OllamaLLM')
    @pytest.mark.asyncio
    async def test_query_with_rag_context_truncation(self, mock_ollama, mock_vector_store):
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
            
            result = await processor.query("Test query", use_rag=True)
            
            # Context should be truncated
            assert result["context_length"] == 1003  # 1000 + "..."
            assert result["response"] == "Truncated response"
    
    @patch('src.langchain_rag_processor.OllamaLLM')
    @pytest.mark.asyncio
    async def test_query_without_rag(self, mock_ollama, mock_vector_store):
        """Test direct LLM query without RAG."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = "Direct LLM response"
        mock_ollama.return_value = mock_llm
        
        processor = LangChainRAGProcessor(
            vector_store=mock_vector_store,
            llm_provider="ollama"
        )
        
        result = await processor.query("What is AI?", use_rag=False)
        
        assert result["response"] == "Direct LLM response"
        assert result["context_used"] is False
        assert result["retrieved_docs"] == []
        assert result["scores"] == []
        
        # Verify LLM was called directly
        mock_llm.invoke.assert_called_once_with("What is AI?")
        
        # Verify vector store was not used
        mock_vector_store.similarity_search_with_scores.assert_not_called()
    
    @patch('src.langchain_rag_processor.OllamaLLM')
    @pytest.mark.asyncio
    async def test_query_error_handling(self, mock_ollama, mock_vector_store):
        """Test query error handling."""
        mock_llm = Mock()
        mock_ollama.return_value = mock_llm
        
        # Mock vector store to raise exception
        mock_vector_store.similarity_search_with_scores.side_effect = Exception("Search failed")
        
        processor = LangChainRAGProcessor(
            vector_store=mock_vector_store,
            llm_provider="ollama"
        )
        
        result = await processor.query("What is AI?", use_rag=True)
        
        assert "Error processing query" in result["response"]
        assert result["context_used"] is False
        assert result["retrieved_docs"] == []
        assert "error" in result
    
    @patch('src.langchain_rag_processor.OllamaLLM')
    @pytest.mark.asyncio
    async def test_retrieve_documents(self, mock_ollama, mock_vector_store, sample_documents):
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
        
        result = await processor.retrieve_documents("test query", k=2)
        
        assert len(result) == 2
        assert result[0]["content"] == sample_documents[0].page_content
        assert result[0]["metadata"] == sample_documents[0].metadata
        assert result[0]["score"] == 0.9
        assert result[1]["content"] == sample_documents[1].page_content
        assert result[1]["score"] == 0.7
        
        mock_vector_store.similarity_search_with_scores.assert_called_once_with("test query", k=2)
    
    @patch('src.langchain_rag_processor.OllamaLLM')
    @pytest.mark.asyncio
    async def test_retrieve_documents_with_threshold(self, mock_ollama, mock_vector_store, sample_documents):
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
        
        result = await processor.retrieve_documents("test query", k=3, score_threshold=0.5)
        
        # Should only return documents with score >= 0.5
        assert len(result) == 2
        assert result[0]["score"] == 0.9
        assert result[1]["score"] == 0.7
    
    @patch('src.langchain_rag_processor.OllamaLLM')
    @pytest.mark.asyncio
    async def test_retrieve_documents_error(self, mock_ollama, mock_vector_store):
        """Test document retrieval error handling."""
        mock_llm = Mock()
        mock_ollama.return_value = mock_llm
        
        # Mock vector store to raise exception
        mock_vector_store.similarity_search_with_scores.side_effect = Exception("Retrieval failed")
        
        processor = LangChainRAGProcessor(
            vector_store=mock_vector_store,
            llm_provider="ollama"
        )
        
        result = await processor.retrieve_documents("test query")
        
        assert result == []


class TestLangChainRAGProcessorCaching:
    """Test cases for RAG processor caching functionality."""

    @pytest.fixture
    def cache_config(self):
        """Sample cache configuration."""
        return {
            'cache': {
                'enabled': True,
                'redis_host': 'localhost',
                'redis_port': 6379,
                'redis_db': 0,
                'ttl_seconds': {'query_responses': 900}
            }
        }

    @pytest.fixture
    def mock_cache_manager(self):
        """Mock cache manager for testing."""
        mock_cache = Mock()
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.set = AsyncMock(return_value=True)
        return mock_cache

    @pytest.fixture
    def mock_vector_store(self):
        """Mock vector store with caching."""
        mock_store = Mock()
        mock_store.similarity_search_with_scores = AsyncMock(return_value=[
            (LangChainDocument(page_content="Test document content", metadata={"source": "test.txt"}), 0.85)
        ])
        return mock_store

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM for testing."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = "Test LLM response"
        return mock_llm

    def test_init_with_cache_config(self, cache_config, mock_vector_store):
        """Test RAG processor initialization with cache configuration."""
        with patch('src.langchain_rag_processor.get_cache_manager') as mock_get_cache:
            with patch('src.langchain_rag_processor.OllamaLLM'):
                with patch('src.langchain_rag_processor.ChatPromptTemplate'):
                    with patch('src.langchain_rag_processor.StrOutputParser'):
                        with patch('src.langchain_rag_processor.RunnablePassthrough'):
                            mock_cache = Mock()
                            mock_get_cache.return_value = mock_cache
                            
                            rag_processor = LangChainRAGProcessor(
                                vector_store=mock_vector_store,
                                config=cache_config
                            )
                            
                            assert rag_processor.cache_manager is mock_cache
                            mock_get_cache.assert_called_once_with(cache_config)

    def test_init_without_cache_config(self, mock_vector_store):
        """Test RAG processor initialization without cache configuration."""
        with patch('src.langchain_rag_processor.OllamaLLM'):
            with patch('src.langchain_rag_processor.ChatPromptTemplate'):
                with patch('src.langchain_rag_processor.StrOutputParser'):
                    with patch('src.langchain_rag_processor.RunnablePassthrough'):
                        rag_processor = LangChainRAGProcessor(
                            vector_store=mock_vector_store
                        )
                        
                        assert rag_processor.cache_manager is None

    def test_generate_query_cache_key(self, cache_config, mock_vector_store):
        """Test query cache key generation."""
        with patch('src.langchain_rag_processor.get_cache_manager'):
            with patch('src.langchain_rag_processor.OllamaLLM'):
                with patch('src.langchain_rag_processor.ChatPromptTemplate'):
                    with patch('src.langchain_rag_processor.StrOutputParser'):
                        with patch('src.langchain_rag_processor.RunnablePassthrough'):
                            rag_processor = LangChainRAGProcessor(
                                vector_store=mock_vector_store,
                                llm_model_name="llama3.1",
                                llm_provider="ollama",
                                config=cache_config
                            )
                            
                            key = rag_processor._generate_query_cache_key("test query", True)
                            
                            assert key.startswith('query:')
                            assert len(key.split(':')[1]) == 16  # Hash length

    def test_generate_query_cache_key_consistency(self, cache_config, mock_vector_store):
        """Test that same query parameters generate same cache key."""
        with patch('src.langchain_rag_processor.get_cache_manager'):
            with patch('src.langchain_rag_processor.OllamaLLM'):
                with patch('src.langchain_rag_processor.ChatPromptTemplate'):
                    with patch('src.langchain_rag_processor.StrOutputParser'):
                        with patch('src.langchain_rag_processor.RunnablePassthrough'):
                            rag_processor = LangChainRAGProcessor(
                                vector_store=mock_vector_store,
                                config=cache_config
                            )
                            
                            key1 = rag_processor._generate_query_cache_key("test query", True)
                            key2 = rag_processor._generate_query_cache_key("test query", True)
                            
                            assert key1 == key2

    def test_generate_query_cache_key_different_params(self, cache_config, mock_vector_store):
        """Test that different query parameters generate different cache keys."""
        with patch('src.langchain_rag_processor.get_cache_manager'):
            with patch('src.langchain_rag_processor.OllamaLLM'):
                with patch('src.langchain_rag_processor.ChatPromptTemplate'):
                    with patch('src.langchain_rag_processor.StrOutputParser'):
                        with patch('src.langchain_rag_processor.RunnablePassthrough'):
                            rag_processor = LangChainRAGProcessor(
                                vector_store=mock_vector_store,
                                config=cache_config
                            )
                            
                            key1 = rag_processor._generate_query_cache_key("test query", True)
                            key2 = rag_processor._generate_query_cache_key("test query", False)  # Different use_rag
                            key3 = rag_processor._generate_query_cache_key("different query", True)  # Different query
                            
                            assert key1 != key2
                            assert key1 != key3
                            assert key2 != key3

    @pytest.mark.asyncio
    async def test_query_cache_hit(self, cache_config, mock_cache_manager, mock_vector_store):
        """Test query with cache hit."""
        with patch('src.langchain_rag_processor.get_cache_manager', return_value=mock_cache_manager):
            with patch('src.langchain_rag_processor.OllamaLLM'):
                with patch('src.langchain_rag_processor.ChatPromptTemplate'):
                    with patch('src.langchain_rag_processor.StrOutputParser'):
                        with patch('src.langchain_rag_processor.RunnablePassthrough'):
                            # Mock cached response
                            cached_response = {
                                "response": "Cached RAG response",
                                "context_used": True,
                                "retrieved_docs": ["cached doc"],
                                "scores": [0.9]
                            }
                            mock_cache_manager.get.return_value = cached_response
                            
                            rag_processor = LangChainRAGProcessor(
                                vector_store=mock_vector_store,
                                config=cache_config
                            )
                            
                            result = await rag_processor.query("test query", use_rag=True)
                            
                            assert result == cached_response
                            mock_cache_manager.get.assert_called_once()
                            mock_cache_manager.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_query_cache_miss_with_rag(self, cache_config, mock_cache_manager, mock_vector_store, mock_llm):
        """Test query with cache miss using RAG."""
        with patch('src.langchain_rag_processor.get_cache_manager', return_value=mock_cache_manager):
            with patch('src.langchain_rag_processor.OllamaLLM', return_value=mock_llm):
                with patch('src.langchain_rag_processor.ChatPromptTemplate'):
                    with patch('src.langchain_rag_processor.StrOutputParser'):
                        with patch('src.langchain_rag_processor.RunnablePassthrough'):
                            # Cache miss
                            mock_cache_manager.get.return_value = None
                            
                            # Mock RAG chain
                            mock_rag_chain = Mock()
                            mock_rag_chain.invoke.return_value = "Fresh RAG response"
                            
                            rag_processor = LangChainRAGProcessor(
                                vector_store=mock_vector_store,
                                config=cache_config
                            )
                            rag_processor.rag_chain = mock_rag_chain
                            
                            result = await rag_processor.query("test query", use_rag=True)
                            
                            assert result["response"] == "Fresh RAG response"
                            assert result["context_used"] is True
                            assert len(result["retrieved_docs"]) > 0
                            
                            mock_cache_manager.get.assert_called_once()
                            mock_cache_manager.set.assert_called_once()
                            
                            # Verify the response was cached
                            set_call_args = mock_cache_manager.set.call_args
                            assert set_call_args[0][2] == 'query_responses'  # Cache type

    @pytest.mark.asyncio
    async def test_query_cache_miss_without_rag(self, cache_config, mock_cache_manager, mock_vector_store, mock_llm):
        """Test query with cache miss without RAG."""
        with patch('src.langchain_rag_processor.get_cache_manager', return_value=mock_cache_manager):
            with patch('src.langchain_rag_processor.OllamaLLM', return_value=mock_llm):
                with patch('src.langchain_rag_processor.ChatPromptTemplate'):
                    with patch('src.langchain_rag_processor.StrOutputParser'):
                        with patch('src.langchain_rag_processor.RunnablePassthrough'):
                            # Cache miss
                            mock_cache_manager.get.return_value = None
                            
                            rag_processor = LangChainRAGProcessor(
                                vector_store=mock_vector_store,
                                config=cache_config
                            )
                            
                            result = await rag_processor.query("test query", use_rag=False)
                            
                            assert result["response"] == "Test LLM response"
                            assert result["context_used"] is False
                            assert result["retrieved_docs"] == []
                            
                            mock_cache_manager.get.assert_called_once()
                            mock_cache_manager.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_error_not_cached(self, cache_config, mock_cache_manager, mock_vector_store):
        """Test that error responses are not cached."""
        with patch('src.langchain_rag_processor.get_cache_manager', return_value=mock_cache_manager):
            with patch('src.langchain_rag_processor.OllamaLLM') as mock_llm_class:
                with patch('src.langchain_rag_processor.ChatPromptTemplate'):
                    with patch('src.langchain_rag_processor.StrOutputParser'):
                        with patch('src.langchain_rag_processor.RunnablePassthrough'):
                            # Cache miss
                            mock_cache_manager.get.return_value = None
                            
                            # Mock LLM - let initialization succeed but make query fail
                            mock_llm = Mock()
                            mock_llm.invoke.side_effect = Exception("LLM error")
                            mock_llm_class.return_value = mock_llm
                            
                            rag_processor = LangChainRAGProcessor(
                                vector_store=mock_vector_store,
                                config=cache_config
                            )
                            
                            result = await rag_processor.query("test query", use_rag=False)
                            
                            assert "error" in result
                            mock_cache_manager.get.assert_called_once()
                            mock_cache_manager.set.assert_not_called()  # Error responses not cached

    @pytest.mark.asyncio
    async def test_retrieve_documents_cache_integration(self, cache_config, mock_cache_manager, mock_vector_store):
        """Test retrieve_documents method with caching."""
        with patch('src.langchain_rag_processor.get_cache_manager', return_value=mock_cache_manager):
            with patch('src.langchain_rag_processor.OllamaLLM'):
                with patch('src.langchain_rag_processor.ChatPromptTemplate'):
                    with patch('src.langchain_rag_processor.StrOutputParser'):
                        with patch('src.langchain_rag_processor.RunnablePassthrough'):
                            rag_processor = LangChainRAGProcessor(
                                vector_store=mock_vector_store,
                                config=cache_config
                            )
                            
                            results = await rag_processor.retrieve_documents("test query", k=3)
                            
                            assert len(results) == 1  # Based on mock_vector_store fixture
                            assert results[0]["content"] == "Test document content"
                            assert abs(results[0]["score"] - 0.85) < 1e-10
                            
                            # Vector store should have been called with caching
                            mock_vector_store.similarity_search_with_scores.assert_called_once_with("test query", k=3)

    @pytest.mark.asyncio
    async def test_query_no_cache_manager(self, mock_vector_store, mock_llm):
        """Test query when cache manager is not available."""
        with patch('src.langchain_rag_processor.OllamaLLM', return_value=mock_llm):
            with patch('src.langchain_rag_processor.ChatPromptTemplate'):
                with patch('src.langchain_rag_processor.StrOutputParser'):
                    with patch('src.langchain_rag_processor.RunnablePassthrough'):
                        rag_processor = LangChainRAGProcessor(
                            vector_store=mock_vector_store
                            # No cache config
                        )
                        
                        result = await rag_processor.query("test query", use_rag=False)
                        
                        assert result["response"] == "Test LLM response"
                        # Should still work without caching

    @pytest.mark.asyncio
    async def test_rag_query_no_documents_found(self, cache_config, mock_cache_manager, mock_vector_store):
        """Test RAG query when no documents are found."""
        with patch('src.langchain_rag_processor.get_cache_manager', return_value=mock_cache_manager):
            with patch('src.langchain_rag_processor.OllamaLLM'):
                with patch('src.langchain_rag_processor.ChatPromptTemplate'):
                    with patch('src.langchain_rag_processor.StrOutputParser'):
                        with patch('src.langchain_rag_processor.RunnablePassthrough'):
                            # Cache miss
                            mock_cache_manager.get.return_value = None
                            # No documents found
                            mock_vector_store.similarity_search_with_scores = AsyncMock(return_value=[])
                            
                            rag_processor = LangChainRAGProcessor(
                                vector_store=mock_vector_store,
                                config=cache_config
                            )
                            
                            result = await rag_processor.query("test query", use_rag=True)
                            
                            assert "don't have any relevant information" in result["response"]
                            assert result["context_used"] is False
                            assert result["retrieved_docs"] == []
                            
                            # Should still cache the "no documents" response
                            mock_cache_manager.set.assert_called_once()