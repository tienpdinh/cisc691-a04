import pytest
import tempfile
import shutil
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from langchain.schema import Document

from src.langchain_vector_store import LangChainVectorStore


class TestLangChainVectorStore:
    """Test cases for LangChainVectorStore."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(page_content="This is the first document", metadata={"source": "doc1.txt"}),
            Document(page_content="This is the second document", metadata={"source": "doc2.txt"}),
            Document(page_content="This is the third document", metadata={"source": "doc3.txt"})
        ]
    
    @patch('src.langchain_vector_store.HuggingFaceEmbeddings')
    @patch('src.langchain_vector_store.Chroma')
    def test_init_local(self, mock_chroma, mock_embeddings, temp_dir):
        """Test initialization with local ChromaDB."""
        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance
        
        vector_store = LangChainVectorStore(
            collection_name="test_collection",
            embedding_model_name="test-model",
            persist_directory=temp_dir
        )
        
        assert vector_store.collection_name == "test_collection"
        assert vector_store.chromadb_host == "localhost"
        assert vector_store.chromadb_port == 8000
        assert vector_store.persist_directory == temp_dir
        
        # Verify embeddings initialization
        mock_embeddings.assert_called_once_with(
            model_name="test-model",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Verify Chroma initialization
        mock_chroma.assert_called_once_with(
            collection_name="test_collection",
            embedding_function=mock_embeddings_instance,
            persist_directory=temp_dir
        )
    
    @patch('src.langchain_vector_store.HuggingFaceEmbeddings')
    @patch('src.langchain_vector_store.Chroma')
    @patch('chromadb.HttpClient')
    def test_init_remote(self, mock_http_client, mock_chroma, mock_embeddings):
        """Test initialization with remote ChromaDB."""
        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance
        mock_client = Mock()
        mock_http_client.return_value = mock_client
        
        vector_store = LangChainVectorStore(
            collection_name="test_collection",
            chromadb_host="remote-host",
            chromadb_port=9000
        )
        
        assert vector_store.chromadb_host == "remote-host"
        assert vector_store.chromadb_port == 9000
        
        # Verify HTTP client initialization
        mock_http_client.assert_called_once_with(host="remote-host", port=9000)
        
        # Verify Chroma initialization with client
        mock_chroma.assert_called_once_with(
            collection_name="test_collection",
            embedding_function=mock_embeddings_instance,
            client=mock_client
        )
    
    @patch('src.langchain_vector_store.HuggingFaceEmbeddings')
    @patch('src.langchain_vector_store.Chroma')
    def test_init_failure(self, mock_chroma, mock_embeddings):
        """Test initialization failure handling."""
        mock_embeddings.return_value = Mock()
        mock_chroma.side_effect = Exception("Chroma initialization failed")
        
        with pytest.raises(Exception, match="Chroma initialization failed"):
            LangChainVectorStore(collection_name="test_collection")
    
    @patch('src.langchain_vector_store.HuggingFaceEmbeddings')
    @patch('src.langchain_vector_store.Chroma')
    def test_add_documents_success(self, mock_chroma, mock_embeddings, sample_documents):
        """Test successful document addition."""
        mock_vector_store = Mock()
        mock_vector_store.add_documents.return_value = ["id1", "id2", "id3"]
        mock_chroma.return_value = mock_vector_store
        
        vector_store = LangChainVectorStore(collection_name="test_collection")
        result = vector_store.add_documents(sample_documents)
        
        assert result == ["id1", "id2", "id3"]
        mock_vector_store.add_documents.assert_called_once_with(sample_documents)
    
    @patch('src.langchain_vector_store.HuggingFaceEmbeddings')
    @patch('src.langchain_vector_store.Chroma')
    def test_add_documents_failure(self, mock_chroma, mock_embeddings, sample_documents):
        """Test document addition failure."""
        mock_vector_store = Mock()
        mock_vector_store.add_documents.side_effect = Exception("Addition failed")
        mock_chroma.return_value = mock_vector_store
        
        vector_store = LangChainVectorStore(collection_name="test_collection")
        result = vector_store.add_documents(sample_documents)
        
        assert result == []
    
    @patch('src.langchain_vector_store.HuggingFaceEmbeddings')
    @patch('src.langchain_vector_store.Chroma')
    @pytest.mark.asyncio
    async def test_similarity_search_basic(self, mock_chroma, mock_embeddings, sample_documents):
        """Test basic similarity search."""
        mock_vector_store = Mock()
        mock_vector_store.similarity_search.return_value = sample_documents[:2]
        mock_chroma.return_value = mock_vector_store
        
        vector_store = LangChainVectorStore(collection_name="test_collection")
        result = await vector_store.similarity_search("test query", k=2)
        
        assert len(result) == 2
        assert result == sample_documents[:2]
        mock_vector_store.similarity_search.assert_called_once_with("test query", k=2)
    
    @patch('src.langchain_vector_store.HuggingFaceEmbeddings')
    @patch('src.langchain_vector_store.Chroma')
    @pytest.mark.asyncio
    async def test_similarity_search_with_threshold(self, mock_chroma, mock_embeddings, sample_documents):
        """Test similarity search with score threshold."""
        mock_vector_store = Mock()
        mock_vector_store.similarity_search_with_score.return_value = [
            (sample_documents[0], 0.8),
            (sample_documents[1], 0.6),
            (sample_documents[2], 0.4)
        ]
        mock_chroma.return_value = mock_vector_store
        
        vector_store = LangChainVectorStore(collection_name="test_collection")
        result = await vector_store.similarity_search("test query", k=3, score_threshold=0.5)
        
        # Should return only documents with score >= 0.5
        assert len(result) == 2
        assert result == [sample_documents[0], sample_documents[1]]
    
    @patch('src.langchain_vector_store.HuggingFaceEmbeddings')
    @patch('src.langchain_vector_store.Chroma')
    @pytest.mark.asyncio
    async def test_similarity_search_with_scores(self, mock_chroma, mock_embeddings, sample_documents):
        """Test similarity search with scores."""
        mock_vector_store = Mock()
        expected_result = [(sample_documents[0], 0.8), (sample_documents[1], 0.6)]
        mock_vector_store.similarity_search_with_score.return_value = expected_result
        mock_chroma.return_value = mock_vector_store
        
        vector_store = LangChainVectorStore(collection_name="test_collection")
        result = await vector_store.similarity_search_with_scores("test query", k=2)
        
        assert result == expected_result
        mock_vector_store.similarity_search_with_score.assert_called_once_with("test query", k=2)
    
    @patch('src.langchain_vector_store.HuggingFaceEmbeddings')
    @patch('src.langchain_vector_store.Chroma')
    @pytest.mark.asyncio
    async def test_similarity_search_error(self, mock_chroma, mock_embeddings):
        """Test similarity search error handling."""
        mock_vector_store = Mock()
        mock_vector_store.similarity_search.side_effect = Exception("Search failed")
        mock_chroma.return_value = mock_vector_store
        
        vector_store = LangChainVectorStore(collection_name="test_collection")
        result = await vector_store.similarity_search("test query")
        
        assert result == []
    
    @patch('src.langchain_vector_store.HuggingFaceEmbeddings')
    @patch('src.langchain_vector_store.Chroma')
    def test_delete_collection(self, mock_chroma, mock_embeddings):
        """Test collection deletion."""
        mock_vector_store = Mock()
        mock_chroma.return_value = mock_vector_store
        
        vector_store = LangChainVectorStore(collection_name="test_collection")
        vector_store.delete_collection()
        
        mock_vector_store.delete_collection.assert_called_once()
    
    @patch('src.langchain_vector_store.HuggingFaceEmbeddings')
    @patch('src.langchain_vector_store.Chroma')
    def test_get_collection_info(self, mock_chroma, mock_embeddings):
        """Test getting collection information."""
        mock_collection = Mock()
        mock_collection.count.return_value = 5
        
        mock_vector_store = Mock()
        mock_vector_store._collection = mock_collection
        mock_chroma.return_value = mock_vector_store
        
        mock_embeddings_instance = Mock()
        mock_embeddings_instance.model_name = "test-model"
        mock_embeddings.return_value = mock_embeddings_instance
        
        vector_store = LangChainVectorStore(collection_name="test_collection")
        info = vector_store.get_collection_info()
        
        expected_info = {
            "collection_name": "test_collection",
            "document_count": 5,
            "embedding_model": "test-model"
        }
        assert info == expected_info
    
    @patch('src.langchain_vector_store.HuggingFaceEmbeddings')
    @patch('src.langchain_vector_store.Chroma')
    def test_get_collection_info_error(self, mock_chroma, mock_embeddings):
        """Test collection info error handling."""
        mock_vector_store = Mock()
        mock_vector_store._collection.count.side_effect = Exception("Count failed")
        mock_chroma.return_value = mock_vector_store
        
        vector_store = LangChainVectorStore(collection_name="test_collection")
        info = vector_store.get_collection_info()
        
        assert info == {}


class TestLangChainVectorStoreCaching:
    """Test cases for vector store caching functionality."""

    @pytest.fixture
    def cache_config(self):
        """Sample cache configuration."""
        return {
            'cache': {
                'enabled': True,
                'redis_host': 'localhost',
                'redis_port': 6379,
                'redis_db': 0,
                'ttl_seconds': {'document_retrieval': 1800}
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
    def mock_chroma_vector_store(self):
        """Mock LangChain vector store."""
        mock_store = Mock()
        mock_store.similarity_search.return_value = [
            Document(page_content="Test content 1", metadata={"source": "doc1"}),
            Document(page_content="Test content 2", metadata={"source": "doc2"})
        ]
        mock_store.similarity_search_with_score.return_value = [
            (Document(page_content="Test content 1", metadata={"source": "doc1"}), 0.85),
            (Document(page_content="Test content 2", metadata={"source": "doc2"}), 0.75)
        ]
        return mock_store

    def test_init_with_cache_config(self, cache_config):
        """Test vector store initialization with cache configuration."""
        with patch('src.langchain_vector_store.get_cache_manager') as mock_get_cache:
            with patch('src.langchain_vector_store.HuggingFaceEmbeddings'):
                with patch('src.langchain_vector_store.Chroma'):
                    mock_cache = Mock()
                    mock_get_cache.return_value = mock_cache
                    
                    vector_store = LangChainVectorStore(
                        collection_name="test_collection",
                        config=cache_config
                    )
                    
                    assert vector_store.cache_manager is mock_cache
                    mock_get_cache.assert_called_once_with(cache_config)

    def test_init_without_cache_config(self):
        """Test vector store initialization without cache configuration."""
        with patch('src.langchain_vector_store.HuggingFaceEmbeddings'):
            with patch('src.langchain_vector_store.Chroma'):
                vector_store = LangChainVectorStore(
                    collection_name="test_collection"
                )
                
                assert vector_store.cache_manager is None

    def test_generate_search_cache_key(self, cache_config):
        """Test search cache key generation."""
        with patch('src.langchain_vector_store.get_cache_manager'):
            with patch('src.langchain_vector_store.HuggingFaceEmbeddings'):
                with patch('src.langchain_vector_store.Chroma'):
                    vector_store = LangChainVectorStore(
                        collection_name="test_collection",
                        config=cache_config
                    )
                    
                    key = vector_store._generate_search_cache_key("test query", 5, 0.7)
                    
                    assert key.startswith('search:')
                    assert len(key.split(':')[1]) == 16  # Hash length

    def test_generate_search_cache_key_consistency(self, cache_config):
        """Test that same search parameters generate same cache key."""
        with patch('src.langchain_vector_store.get_cache_manager'):
            with patch('src.langchain_vector_store.HuggingFaceEmbeddings'):
                with patch('src.langchain_vector_store.Chroma'):
                    vector_store = LangChainVectorStore(
                        collection_name="test_collection",
                        config=cache_config
                    )
                    
                    key1 = vector_store._generate_search_cache_key("test query", 5, 0.7)
                    key2 = vector_store._generate_search_cache_key("test query", 5, 0.7)
                    
                    assert key1 == key2

    def test_generate_search_cache_key_different_params(self, cache_config):
        """Test that different search parameters generate different cache keys."""
        with patch('src.langchain_vector_store.get_cache_manager'):
            with patch('src.langchain_vector_store.HuggingFaceEmbeddings'):
                with patch('src.langchain_vector_store.Chroma'):
                    vector_store = LangChainVectorStore(
                        collection_name="test_collection",
                        config=cache_config
                    )
                    
                    key1 = vector_store._generate_search_cache_key("test query", 5, 0.7)
                    key2 = vector_store._generate_search_cache_key("test query", 10, 0.7)  # Different k
                    key3 = vector_store._generate_search_cache_key("different query", 5, 0.7)  # Different query
                    
                    assert key1 != key2
                    assert key1 != key3
                    assert key2 != key3

    @pytest.mark.asyncio
    async def test_similarity_search_cache_hit(self, cache_config, mock_cache_manager, mock_chroma_vector_store):
        """Test similarity search with cache hit."""
        with patch('src.langchain_vector_store.get_cache_manager', return_value=mock_cache_manager):
            with patch('src.langchain_vector_store.HuggingFaceEmbeddings'):
                with patch('src.langchain_vector_store.Chroma', return_value=mock_chroma_vector_store):
                    # Mock cached documents
                    cached_docs = [
                        {'page_content': 'Cached content 1', 'metadata': {'source': 'cached1'}},
                        {'page_content': 'Cached content 2', 'metadata': {'source': 'cached2'}}
                    ]
                    mock_cache_manager.get.return_value = cached_docs
                    
                    vector_store = LangChainVectorStore(
                        collection_name="test_collection",
                        config=cache_config
                    )
                    
                    results = await vector_store.similarity_search("test query", k=2)
                    
                    assert len(results) == 2
                    assert results[0].page_content == 'Cached content 1'
                    assert results[0].metadata == {'source': 'cached1'}
                    
                    mock_cache_manager.get.assert_called_once()
                    mock_cache_manager.set.assert_not_called()
                    mock_chroma_vector_store.similarity_search.assert_not_called()

    @pytest.mark.asyncio
    async def test_similarity_search_cache_miss(self, cache_config, mock_cache_manager, mock_chroma_vector_store):
        """Test similarity search with cache miss."""
        with patch('src.langchain_vector_store.get_cache_manager', return_value=mock_cache_manager):
            with patch('src.langchain_vector_store.HuggingFaceEmbeddings'):
                with patch('src.langchain_vector_store.Chroma', return_value=mock_chroma_vector_store):
                    # Cache miss
                    mock_cache_manager.get.return_value = None
                    
                    vector_store = LangChainVectorStore(
                        collection_name="test_collection",
                        config=cache_config
                    )
                    
                    results = await vector_store.similarity_search("test query", k=2)
                    
                    assert len(results) == 2
                    assert results[0].page_content == "Test content 1"
                    
                    mock_cache_manager.get.assert_called_once()
                    mock_cache_manager.set.assert_called_once()
                    mock_chroma_vector_store.similarity_search.assert_called_once_with("test query", k=2)
                    
                    # Verify the data was cached
                    set_call_args = mock_cache_manager.set.call_args
                    cached_data = set_call_args[0][1]
                    assert len(cached_data) == 2
                    assert cached_data[0]['page_content'] == "Test content 1"

    @pytest.mark.asyncio
    async def test_similarity_search_with_scores_cache_hit(self, cache_config, mock_cache_manager, mock_chroma_vector_store):
        """Test similarity search with scores and cache hit."""
        with patch('src.langchain_vector_store.get_cache_manager', return_value=mock_cache_manager):
            with patch('src.langchain_vector_store.HuggingFaceEmbeddings'):
                with patch('src.langchain_vector_store.Chroma', return_value=mock_chroma_vector_store):
                    # Mock cached results with scores
                    cached_results = [
                        {
                            'doc': {'page_content': 'Cached content 1', 'metadata': {'source': 'cached1'}},
                            'score': 0.9
                        },
                        {
                            'doc': {'page_content': 'Cached content 2', 'metadata': {'source': 'cached2'}},
                            'score': 0.8
                        }
                    ]
                    mock_cache_manager.get.return_value = cached_results
                    
                    vector_store = LangChainVectorStore(
                        collection_name="test_collection",
                        config=cache_config
                    )
                    
                    results = await vector_store.similarity_search_with_scores("test query", k=2)
                    
                    assert len(results) == 2
                    doc, score = results[0]
                    assert doc.page_content == 'Cached content 1'
                    assert abs(score - 0.9) < 1e-10
                    
                    mock_cache_manager.get.assert_called_once()
                    mock_cache_manager.set.assert_not_called()
                    mock_chroma_vector_store.similarity_search_with_score.assert_not_called()

    @pytest.mark.asyncio
    async def test_similarity_search_with_scores_cache_miss(self, cache_config, mock_cache_manager, mock_chroma_vector_store):
        """Test similarity search with scores and cache miss."""
        with patch('src.langchain_vector_store.get_cache_manager', return_value=mock_cache_manager):
            with patch('src.langchain_vector_store.HuggingFaceEmbeddings'):
                with patch('src.langchain_vector_store.Chroma', return_value=mock_chroma_vector_store):
                    # Cache miss
                    mock_cache_manager.get.return_value = None
                    
                    vector_store = LangChainVectorStore(
                        collection_name="test_collection",
                        config=cache_config
                    )
                    
                    results = await vector_store.similarity_search_with_scores("test query", k=2)
                    
                    assert len(results) == 2
                    doc, score = results[0]
                    assert doc.page_content == "Test content 1"
                    assert abs(score - 0.85) < 1e-10
                    
                    mock_cache_manager.get.assert_called_once()
                    mock_cache_manager.set.assert_called_once()
                    mock_chroma_vector_store.similarity_search_with_score.assert_called_once_with("test query", k=2)
                    
                    # Verify the data was cached with scores
                    set_call_args = mock_cache_manager.set.call_args
                    cached_data = set_call_args[0][1]
                    assert len(cached_data) == 2
                    assert cached_data[0]['doc']['page_content'] == "Test content 1"
                    assert abs(cached_data[0]['score'] - 0.85) < 1e-10

    @pytest.mark.asyncio
    async def test_similarity_search_with_threshold(self, cache_config, mock_cache_manager, mock_chroma_vector_store):
        """Test similarity search with score threshold."""
        with patch('src.langchain_vector_store.get_cache_manager', return_value=mock_cache_manager):
            with patch('src.langchain_vector_store.HuggingFaceEmbeddings'):
                with patch('src.langchain_vector_store.Chroma', return_value=mock_chroma_vector_store):
                    # Cache miss
                    mock_cache_manager.get.return_value = None
                    
                    # Mock vector store to return documents with scores
                    mock_chroma_vector_store.similarity_search_with_score.return_value = [
                        (Document(page_content="High score content", metadata={"source": "doc1"}), 0.9),
                        (Document(page_content="Low score content", metadata={"source": "doc2"}), 0.6)
                    ]
                    
                    vector_store = LangChainVectorStore(
                        collection_name="test_collection",
                        config=cache_config
                    )
                    
                    results = await vector_store.similarity_search("test query", k=2, score_threshold=0.8)
                    
                    # Should only return documents above threshold
                    assert len(results) == 1
                    assert results[0].page_content == "High score content"
                    
                    mock_cache_manager.get.assert_called_once()
                    mock_cache_manager.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_similarity_search_no_cache_manager(self, mock_chroma_vector_store):
        """Test similarity search when cache manager is not available."""
        with patch('src.langchain_vector_store.HuggingFaceEmbeddings'):
            with patch('src.langchain_vector_store.Chroma', return_value=mock_chroma_vector_store):
                vector_store = LangChainVectorStore(
                    collection_name="test_collection"
                    # No cache config
                )
                
                results = await vector_store.similarity_search("test query", k=2)
                
                assert len(results) == 2
                assert results[0].page_content == "Test content 1"
                mock_chroma_vector_store.similarity_search.assert_called_once_with("test query", k=2)

    @pytest.mark.asyncio
    async def test_similarity_search_error_handling(self, cache_config, mock_cache_manager, mock_chroma_vector_store):
        """Test similarity search error handling."""
        with patch('src.langchain_vector_store.get_cache_manager', return_value=mock_cache_manager):
            with patch('src.langchain_vector_store.HuggingFaceEmbeddings'):
                with patch('src.langchain_vector_store.Chroma', return_value=mock_chroma_vector_store):
                    # Cache miss
                    mock_cache_manager.get.return_value = None
                    # Vector store error
                    mock_chroma_vector_store.similarity_search.side_effect = Exception("Vector store error")
                    
                    vector_store = LangChainVectorStore(
                        collection_name="test_collection",
                        config=cache_config
                    )
                    
                    results = await vector_store.similarity_search("test query", k=2)
                    
                    # Should return empty list on error
                    assert results == []
                    mock_cache_manager.set.assert_not_called()  # Don't cache errors

    def test_similarity_search_sync_basic(self, mock_chroma_vector_store):
        """Test synchronous similarity search."""
        with patch('src.langchain_vector_store.HuggingFaceEmbeddings'):
            with patch('src.langchain_vector_store.Chroma', return_value=mock_chroma_vector_store):
                vector_store = LangChainVectorStore(
                    collection_name="test_collection"
                )
                
                results = vector_store.similarity_search_sync("test query", k=3)
                
                assert len(results) == 2
                assert results[0].page_content == "Test content 1"
                mock_chroma_vector_store.similarity_search.assert_called_once_with("test query", k=3)

    def test_similarity_search_sync_with_threshold(self, mock_chroma_vector_store):
        """Test synchronous similarity search with score threshold."""
        with patch('src.langchain_vector_store.HuggingFaceEmbeddings'):
            with patch('src.langchain_vector_store.Chroma', return_value=mock_chroma_vector_store):
                vector_store = LangChainVectorStore(
                    collection_name="test_collection"
                )
                
                results = vector_store.similarity_search_sync("test query", k=3, score_threshold=0.8)
                
                # Should filter by threshold
                assert len(results) == 1  # Only first doc has score >= 0.8
                assert results[0].page_content == "Test content 1"
                mock_chroma_vector_store.similarity_search_with_score.assert_called_once_with("test query", k=3)

    def test_similarity_search_sync_error(self, mock_chroma_vector_store):
        """Test synchronous similarity search error handling."""
        with patch('src.langchain_vector_store.HuggingFaceEmbeddings'):
            with patch('src.langchain_vector_store.Chroma', return_value=mock_chroma_vector_store):
                # Vector store error
                mock_chroma_vector_store.similarity_search.side_effect = Exception("Vector store error")
                
                vector_store = LangChainVectorStore(
                    collection_name="test_collection"
                )
                
                results = vector_store.similarity_search_sync("test query", k=3)
                
                # Should return empty list on error
                assert results == []