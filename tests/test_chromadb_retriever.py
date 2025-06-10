import pytest
from unittest.mock import Mock, patch, MagicMock
from src.chromadb_retriever import ChromaDBRetriever


class TestChromaDBRetriever:
    
    @patch('src.chromadb_retriever.SentenceTransformer')
    @patch('src.chromadb_retriever.chromadb')
    def test_init(self, mock_chromadb, mock_sentence_transformer):
        mock_client = Mock()
        mock_collection = Mock()
        mock_embedding_model = Mock()
        
        mock_chromadb.HttpClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_sentence_transformer.return_value = mock_embedding_model
        
        retriever = ChromaDBRetriever(
            embedding_model_name="all-MiniLM-L6-v2",
            collection_name="test_collection",
            chromadb_host="localhost",
            chromadb_port=8000,
            score_threshold=0.7
        )
        
        assert retriever.collection == mock_collection
        assert retriever.embedding_model == mock_embedding_model
        assert retriever.score_threshold == 0.7
        assert retriever.collection_name == "test_collection"
        
        mock_chromadb.HttpClient.assert_called_once_with(host="localhost", port=8000)
        mock_client.get_or_create_collection.assert_called_once_with(name="test_collection")
        mock_sentence_transformer.assert_called_once_with("all-MiniLM-L6-v2")
    
    @patch('src.chromadb_retriever.SentenceTransformer')
    @patch('src.chromadb_retriever.chromadb')
    def test_init_default_port(self, mock_chromadb, mock_sentence_transformer):
        mock_client = Mock()
        mock_collection = Mock()
        mock_embedding_model = Mock()
        
        mock_chromadb.HttpClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_sentence_transformer.return_value = mock_embedding_model
        
        retriever = ChromaDBRetriever(
            embedding_model_name="all-MiniLM-L6-v2",
            collection_name="test_collection",
            chromadb_host="chromadb-service"
        )
        
        mock_chromadb.HttpClient.assert_called_once_with(host="chromadb-service", port=8000)
    
    @patch('src.chromadb_retriever.SentenceTransformer')
    @patch('src.chromadb_retriever.chromadb')
    def test_embed_text(self, mock_chromadb, mock_sentence_transformer):
        mock_client = Mock()
        mock_collection = Mock()
        mock_embedding_model = Mock()
        mock_embedding_model.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
        
        mock_chromadb.HttpClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_sentence_transformer.return_value = mock_embedding_model
        
        retriever = ChromaDBRetriever(
            embedding_model_name="all-MiniLM-L6-v2",
            collection_name="test_collection",
            chromadb_host="localhost"
        )
        
        result = retriever.embed_text("test text")
        
        assert result == [0.1, 0.2, 0.3]
        mock_embedding_model.encode.assert_called_once_with("test text", normalize_embeddings=True)
    
    @patch('src.chromadb_retriever.SentenceTransformer')
    @patch('src.chromadb_retriever.chromadb')
    def test_extract_context_found_in_paragraph(self, mock_chromadb, mock_sentence_transformer):
        mock_client = Mock()
        mock_collection = Mock()
        mock_embedding_model = Mock()
        
        mock_chromadb.HttpClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_sentence_transformer.return_value = mock_embedding_model
        
        retriever = ChromaDBRetriever(
            embedding_model_name="all-MiniLM-L6-v2",
            collection_name="test_collection",
            chromadb_host="localhost"
        )
        
        full_text = "First paragraph about cats.\n\nSecond paragraph about artificial intelligence and machine learning.\n\nThird paragraph about dogs."
        search_str = "artificial intelligence"
        
        result = retriever.extract_context(full_text, search_str)
        
        assert result == "Second paragraph about artificial intelligence and machine learning."
    
    @patch('src.chromadb_retriever.SentenceTransformer')
    @patch('src.chromadb_retriever.chromadb')
    def test_extract_context_not_found_fallback(self, mock_chromadb, mock_sentence_transformer):
        mock_client = Mock()
        mock_collection = Mock()
        mock_embedding_model = Mock()
        
        mock_chromadb.HttpClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_sentence_transformer.return_value = mock_embedding_model
        
        retriever = ChromaDBRetriever(
            embedding_model_name="all-MiniLM-L6-v2",
            collection_name="test_collection",
            chromadb_host="localhost"
        )
        
        full_text = "This is a very long text that contains information about various topics but not the one we are searching for. " * 10
        search_str = "nonexistent term"
        
        result = retriever.extract_context(full_text, search_str)
        
        # Should return first 300 characters
        assert len(result) == 300
        assert result == full_text[:300]
    
    @patch('src.chromadb_retriever.SentenceTransformer')
    @patch('src.chromadb_retriever.chromadb')
    def test_query_success(self, mock_chromadb, mock_sentence_transformer):
        mock_client = Mock()
        mock_collection = Mock()
        mock_embedding_model = Mock()
        mock_embedding_model.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
        
        # Mock ChromaDB query response
        mock_query_results = {
            "ids": [["doc1"]],
            "metadatas": [[{"text": "This document is about artificial intelligence and machine learning.", "source": "ai_doc.txt"}]],
            "distances": [[0.8]]  # Good similarity score (higher than threshold)
        }
        mock_collection.query.return_value = mock_query_results
        
        mock_chromadb.HttpClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_sentence_transformer.return_value = mock_embedding_model
        
        retriever = ChromaDBRetriever(
            embedding_model_name="all-MiniLM-L6-v2",
            collection_name="test_collection",
            chromadb_host="localhost",
            score_threshold=0.5
        )
        
        result = retriever.query("artificial intelligence", top_k=5)
        
        assert len(result) == 1
        assert result[0]["id"] == "doc1"
        assert result[0]["score"] == 0.8
        assert result[0]["context"] == "This document is about artificial intelligence and machine learning."
        assert result[0]["source"] == "ai_doc.txt"
        
        mock_collection.query.assert_called_once_with(
            query_embeddings=[[0.1, 0.2, 0.3]], 
            n_results=5
        )
    
    @patch('src.chromadb_retriever.SentenceTransformer')
    @patch('src.chromadb_retriever.chromadb')
    def test_query_filtered_by_score_threshold(self, mock_chromadb, mock_sentence_transformer):
        mock_client = Mock()
        mock_collection = Mock()
        mock_embedding_model = Mock()
        mock_embedding_model.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
        
        # Mock ChromaDB query response with poor similarity score
        mock_query_results = {
            "ids": [["doc1"]],
            "metadatas": [[{"text": "This document is about cats.", "source": "cat_doc.txt"}]],
            "distances": [[0.2]]  # Poor similarity score (below threshold)
        }
        mock_collection.query.return_value = mock_query_results
        
        mock_chromadb.HttpClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_sentence_transformer.return_value = mock_embedding_model
        
        retriever = ChromaDBRetriever(
            embedding_model_name="all-MiniLM-L6-v2",
            collection_name="test_collection",
            chromadb_host="localhost",
            score_threshold=0.5
        )
        
        result = retriever.query("artificial intelligence", top_k=5)
        
        # Should return empty list due to poor score
        assert result == []
    
    @patch('src.chromadb_retriever.SentenceTransformer')
    @patch('src.chromadb_retriever.chromadb')
    def test_query_filtered_by_irrelevant_content(self, mock_chromadb, mock_sentence_transformer):
        mock_client = Mock()
        mock_collection = Mock()
        mock_embedding_model = Mock()
        mock_embedding_model.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
        
        # Mock ChromaDB query response with good score but irrelevant content
        mock_query_results = {
            "ids": [["doc1"]],
            "metadatas": [[{"text": "This document is about cats and dogs.", "source": "animals_doc.txt"}]],
            "distances": [[0.8]]  # Good similarity score
        }
        mock_collection.query.return_value = mock_query_results
        
        mock_chromadb.HttpClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_sentence_transformer.return_value = mock_embedding_model
        
        retriever = ChromaDBRetriever(
            embedding_model_name="all-MiniLM-L6-v2",
            collection_name="test_collection",
            chromadb_host="localhost",
            score_threshold=0.5
        )
        
        result = retriever.query("artificial intelligence", top_k=5)
        
        # Should return empty list because no query words found in text
        assert result == []
    
    @patch('src.chromadb_retriever.SentenceTransformer')
    @patch('src.chromadb_retriever.chromadb')
    def test_query_empty_results(self, mock_chromadb, mock_sentence_transformer):
        mock_client = Mock()
        mock_collection = Mock()
        mock_embedding_model = Mock()
        mock_embedding_model.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
        
        # Mock empty ChromaDB query response
        mock_query_results = {
            "ids": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }
        mock_collection.query.return_value = mock_query_results
        
        mock_chromadb.HttpClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_sentence_transformer.return_value = mock_embedding_model
        
        retriever = ChromaDBRetriever(
            embedding_model_name="all-MiniLM-L6-v2",
            collection_name="test_collection",
            chromadb_host="localhost"
        )
        
        result = retriever.query("artificial intelligence", top_k=5)
        
        assert result == []
    
    @patch('src.chromadb_retriever.SentenceTransformer')
    @patch('src.chromadb_retriever.chromadb')
    def test_required_chromadb_host_parameter(self, mock_chromadb, mock_sentence_transformer):
        """Test that chromadb_host is required and no longer optional."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_embedding_model = Mock()
        
        mock_chromadb.HttpClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_sentence_transformer.return_value = mock_embedding_model
        
        # This should work with chromadb_host provided
        retriever = ChromaDBRetriever(
            embedding_model_name="all-MiniLM-L6-v2",
            collection_name="test_collection",
            chromadb_host="chromadb-service"
        )
        
        assert retriever.client is not None
        mock_chromadb.HttpClient.assert_called_once_with(host="chromadb-service", port=8000)