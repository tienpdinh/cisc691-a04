"""Tests for ChromaDBRetriever class."""
import pytest
from unittest.mock import patch, MagicMock
from classes.chromadb_retriever import ChromaDBRetriever

class TestChromaDBRetriever:
    """Test cases for ChromaDBRetriever."""
    
    @pytest.fixture
    def mock_sentence_transformer(self, sample_embeddings):
        """Mock SentenceTransformer."""
        mock_model = MagicMock()
        mock_model.encode.return_value.tolist.return_value = sample_embeddings
        return mock_model
    
    @pytest.fixture
    def retriever(self, temp_dir, mock_chromadb_client, mock_sentence_transformer):
        """Create ChromaDBRetriever instance for testing."""
        with patch('classes.chromadb_retriever.chromadb.PersistentClient', return_value=mock_chromadb_client), \
             patch('classes.chromadb_retriever.SentenceTransformer', return_value=mock_sentence_transformer):
            return ChromaDBRetriever(
                embedding_model_name="test-model",
                collection_name="test_collection",
                vectordb_dir=str(temp_dir),
                score_threshold=0.5
            )
    
    def test_init(self, retriever, temp_dir):
        """Test ChromaDBRetriever initialization."""
        assert retriever.vectordb_path == temp_dir
        assert retriever.score_threshold == 0.5
        assert retriever.collection is not None
        assert retriever.embedding_model is not None
    
    def test_embed_text(self, retriever, sample_embeddings):
        """Test text embedding generation."""
        result = retriever.embed_text("test text")
        assert result == sample_embeddings
        retriever.embedding_model.encode.assert_called_once_with(
            "test text", normalize_embeddings=True
        )
    
    def test_extract_context_with_match(self, retriever):
        """Test context extraction when search term is found."""
        full_text = "First paragraph.\n\nSecond paragraph with AI content.\n\nThird paragraph."
        search_str = "AI content"
        
        result = retriever.extract_context(full_text, search_str)
        assert result == "Second paragraph with AI content."
    
    def test_extract_context_no_match(self, retriever):
        """Test context extraction when search term is not found."""
        full_text = "This is a long text that doesn't contain the search term."
        search_str = "missing term"
        
        result = retriever.extract_context(full_text, search_str)
        assert result == full_text[:300]  # Should return first 300 chars
    
    def test_query_with_results(self, retriever, mock_chromadb_collection):
        """Test querying with valid results."""
        mock_chromadb_collection.query.return_value = {
            "ids": [["doc1"]],
            "metadatas": [[{"text": "This text contains AI information", "source": "doc1.txt"}]],
            "distances": [[0.3]]  # Below threshold
        }
        
        results = retriever.query("AI information", top_k=5)
        
        assert len(results) == 1
        assert results[0]["id"] == "doc1"
        assert results[0]["score"] == 0.3
        assert results[0]["source"] == "doc1.txt"
        assert "AI information" in results[0]["context"]
    
    def test_query_no_results_high_distance(self, retriever, mock_chromadb_collection):
        """Test querying with high distance (low similarity)."""
        mock_chromadb_collection.query.return_value = {
            "ids": [["doc1"]],
            "metadatas": [[{"text": "Some random text", "source": "doc1.txt"}]],
            "distances": [[0.8]]  # Above threshold
        }
        
        results = retriever.query("AI information", top_k=5)
        
        assert len(results) == 0
    
    def test_query_no_results_irrelevant_content(self, retriever, mock_chromadb_collection):
        """Test querying with irrelevant content."""
        mock_chromadb_collection.query.return_value = {
            "ids": [["doc1"]],
            "metadatas": [[{"text": "Completely different topic", "source": "doc1.txt"}]],
            "distances": [[0.3]]  # Below threshold but irrelevant
        }
        
        results = retriever.query("AI machine learning", top_k=5)
        
        assert len(results) == 0  # Should be filtered out due to lack of relevant words
    
    def test_query_multiple_results_sorted(self, retriever, mock_chromadb_collection):
        """Test querying with multiple results sorted by score."""
        mock_chromadb_collection.query.return_value = {
            "ids": [["doc1", "doc2", "doc3"]],
            "metadatas": [[
                {"text": "AI content here", "source": "doc1.txt"},
                {"text": "More AI content", "source": "doc2.txt"},
                {"text": "Different AI approach", "source": "doc3.txt"}
            ]],
            "distances": [[0.4, 0.2, 0.3]]  # Different scores
        }
        
        results = retriever.query("AI", top_k=5)
        
        # Should return only the best result (lowest distance)
        assert len(results) == 1
        assert results[0]["id"] == "doc2"  # Best score (0.2)
        assert results[0]["score"] == 0.2
    
    def test_query_empty_response(self, retriever, mock_chromadb_collection):
        """Test querying with empty response from ChromaDB."""
        mock_chromadb_collection.query.return_value = {
            "ids": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }
        
        results = retriever.query("AI", top_k=5)
        
        assert len(results) == 0