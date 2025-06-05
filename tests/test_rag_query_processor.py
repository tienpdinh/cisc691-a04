import pytest
from unittest.mock import Mock, patch
from src.rag_query_processor import RAGQueryProcessor
from src.llm_client import LLMClient
from src.chromadb_retriever import ChromaDBRetriever


class TestRAGQueryProcessor:
    
    def test_init_without_rag(self):
        mock_llm_client = Mock(spec=LLMClient)
        mock_retriever = Mock(spec=ChromaDBRetriever)
        
        processor = RAGQueryProcessor(
            llm_client=mock_llm_client,
            retriever=mock_retriever,
            use_rag=False
        )
        
        assert processor.use_rag is False
        assert processor.llm_client == mock_llm_client
        assert processor.retriever is None
    
    def test_init_with_rag(self):
        mock_llm_client = Mock(spec=LLMClient)
        mock_retriever = Mock(spec=ChromaDBRetriever)
        
        processor = RAGQueryProcessor(
            llm_client=mock_llm_client,
            retriever=mock_retriever,
            use_rag=True
        )
        
        assert processor.use_rag is True
        assert processor.llm_client == mock_llm_client
        assert processor.retriever == mock_retriever
    
    def test_query_without_rag(self):
        mock_llm_client = Mock(spec=LLMClient)
        mock_llm_client.query.return_value = "LLM response without RAG"
        mock_retriever = Mock(spec=ChromaDBRetriever)
        
        processor = RAGQueryProcessor(
            llm_client=mock_llm_client,
            retriever=mock_retriever,
            use_rag=False
        )
        
        result = processor.query("What is AI?")
        
        assert result == "LLM response without RAG"
        
        # Verify LLM was called with correct prompt structure
        expected_prompt = """
        You are an AI assistant answering user queries using retrieved context.
        If the context is insufficient, say 'I don't know'. 

        Context:
        No relevant context found.

        Question:
        What is AI?
        """
        mock_llm_client.query.assert_called_once_with(expected_prompt)
    
    def test_query_with_rag_no_documents_found(self):
        mock_llm_client = Mock(spec=LLMClient)
        mock_llm_client.query.return_value = "I don't know"
        mock_retriever = Mock(spec=ChromaDBRetriever)
        mock_retriever.query.return_value = []  # No documents found
        
        processor = RAGQueryProcessor(
            llm_client=mock_llm_client,
            retriever=mock_retriever,
            use_rag=True
        )
        
        result = processor.query("What is AI?")
        
        assert result == "I don't know"
        
        # Verify retriever was called
        mock_retriever.query.assert_called_once_with("What is AI?")
        
        # Verify LLM was called with empty context
        expected_prompt = """
        You are an AI assistant answering user queries using retrieved context.
        If the context is insufficient, say 'I don't know'. 

        Context:
        No relevant context found.

        Question:
        What is AI?
        """
        mock_llm_client.query.assert_called_once_with(expected_prompt)
    
    def test_query_with_rag_documents_found(self):
        mock_llm_client = Mock(spec=LLMClient)
        mock_llm_client.query.return_value = "AI is artificial intelligence based on retrieved context"
        
        mock_retriever = Mock(spec=ChromaDBRetriever)
        mock_retrieved_docs = [{
            "id": "doc1",
            "score": 0.2,
            "context": "Artificial intelligence (AI) is the simulation of human intelligence in machines.",
            "source": "ai_doc.txt",
            "text": "Full document text about AI..."
        }]
        mock_retriever.query.return_value = mock_retrieved_docs
        
        processor = RAGQueryProcessor(
            llm_client=mock_llm_client,
            retriever=mock_retriever,
            use_rag=True
        )
        
        result = processor.query("What is AI?")
        
        assert result == "AI is artificial intelligence based on retrieved context"
        
        # Verify retriever was called
        mock_retriever.query.assert_called_once_with("What is AI?")
        
        # Verify LLM was called with retrieved context
        expected_prompt = """
        You are an AI assistant answering user queries using retrieved context.
        If the context is insufficient, say 'I don't know'. 

        Context:
        Artificial intelligence (AI) is the simulation of human intelligence in machines.

        Question:
        What is AI?
        """
        mock_llm_client.query.assert_called_once_with(expected_prompt)
    
    def test_query_with_rag_document_missing_context(self):
        mock_llm_client = Mock(spec=LLMClient)
        mock_llm_client.query.return_value = "Response based on empty context"
        
        mock_retriever = Mock(spec=ChromaDBRetriever)
        mock_retrieved_docs = [{
            "id": "doc1",
            "score": 0.2,
            "source": "ai_doc.txt"
            # Missing 'context' key
        }]
        mock_retriever.query.return_value = mock_retrieved_docs
        
        processor = RAGQueryProcessor(
            llm_client=mock_llm_client,
            retriever=mock_retriever,
            use_rag=True
        )
        
        result = processor.query("What is AI?")
        
        assert result == "Response based on empty context"
        
        # Verify LLM was called with fallback context (empty context defaults to "No relevant context found.")
        expected_prompt = """
        You are an AI assistant answering user queries using retrieved context.
        If the context is insufficient, say 'I don't know'. 

        Context:
        No relevant context found.

        Question:
        What is AI?
        """
        mock_llm_client.query.assert_called_once_with(expected_prompt)
    
    def test_query_with_rag_multiple_documents_uses_first_only(self):
        mock_llm_client = Mock(spec=LLMClient)
        mock_llm_client.query.return_value = "Response using first document context"
        
        mock_retriever = Mock(spec=ChromaDBRetriever)
        mock_retrieved_docs = [
            {
                "id": "doc1",
                "score": 0.1,
                "context": "First document context",
                "source": "doc1.txt"
            },
            {
                "id": "doc2", 
                "score": 0.2,
                "context": "Second document context",
                "source": "doc2.txt"
            }
        ]
        mock_retriever.query.return_value = mock_retrieved_docs
        
        processor = RAGQueryProcessor(
            llm_client=mock_llm_client,
            retriever=mock_retriever,
            use_rag=True
        )
        
        result = processor.query("What is AI?")
        
        assert result == "Response using first document context"
        
        # Verify only first document's context was used
        expected_prompt = """
        You are an AI assistant answering user queries using retrieved context.
        If the context is insufficient, say 'I don't know'. 

        Context:
        First document context

        Question:
        What is AI?
        """
        mock_llm_client.query.assert_called_once_with(expected_prompt)