import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
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
    def test_similarity_search_basic(self, mock_chroma, mock_embeddings, sample_documents):
        """Test basic similarity search."""
        mock_vector_store = Mock()
        mock_vector_store.similarity_search.return_value = sample_documents[:2]
        mock_chroma.return_value = mock_vector_store
        
        vector_store = LangChainVectorStore(collection_name="test_collection")
        result = vector_store.similarity_search("test query", k=2)
        
        assert len(result) == 2
        assert result == sample_documents[:2]
        mock_vector_store.similarity_search.assert_called_once_with("test query", k=2)
    
    @patch('src.langchain_vector_store.HuggingFaceEmbeddings')
    @patch('src.langchain_vector_store.Chroma')
    def test_similarity_search_with_threshold(self, mock_chroma, mock_embeddings, sample_documents):
        """Test similarity search with score threshold."""
        mock_vector_store = Mock()
        mock_vector_store.similarity_search_with_score.return_value = [
            (sample_documents[0], 0.8),
            (sample_documents[1], 0.6),
            (sample_documents[2], 0.4)
        ]
        mock_chroma.return_value = mock_vector_store
        
        vector_store = LangChainVectorStore(collection_name="test_collection")
        result = vector_store.similarity_search("test query", k=3, score_threshold=0.5)
        
        # Should return only documents with score >= 0.5
        assert len(result) == 2
        assert result == [sample_documents[0], sample_documents[1]]
    
    @patch('src.langchain_vector_store.HuggingFaceEmbeddings')
    @patch('src.langchain_vector_store.Chroma')
    def test_similarity_search_with_scores(self, mock_chroma, mock_embeddings, sample_documents):
        """Test similarity search with scores."""
        mock_vector_store = Mock()
        expected_result = [(sample_documents[0], 0.8), (sample_documents[1], 0.6)]
        mock_vector_store.similarity_search_with_score.return_value = expected_result
        mock_chroma.return_value = mock_vector_store
        
        vector_store = LangChainVectorStore(collection_name="test_collection")
        result = vector_store.similarity_search_with_scores("test query", k=2)
        
        assert result == expected_result
        mock_vector_store.similarity_search_with_score.assert_called_once_with("test query", k=2)
    
    @patch('src.langchain_vector_store.HuggingFaceEmbeddings')
    @patch('src.langchain_vector_store.Chroma')
    def test_similarity_search_error(self, mock_chroma, mock_embeddings):
        """Test similarity search error handling."""
        mock_vector_store = Mock()
        mock_vector_store.similarity_search.side_effect = Exception("Search failed")
        mock_chroma.return_value = mock_vector_store
        
        vector_store = LangChainVectorStore(collection_name="test_collection")
        result = vector_store.similarity_search("test query")
        
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