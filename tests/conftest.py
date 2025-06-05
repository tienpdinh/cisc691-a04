import pytest
import tempfile
import json
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_config_data():
    """Sample configuration data for testing."""
    return {
        "data_directory": "/path/to/data",
        "output_directory": "/path/to/output",
        "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "llm_model_name": "llama2",
        "llm_api_url": "http://localhost:11434/api/generate",
        "collection_name": "documents",
        "regular_setting": "value"
    }


@pytest.fixture
def sample_config_file(temp_dir, sample_config_data):
    """Create a temporary config file for testing."""
    config_file = temp_dir / "test_config.json"
    with open(config_file, 'w') as f:
        json.dump(sample_config_data, f)
    return config_file


@pytest.fixture
def sample_text_content():
    """Sample text content for testing document processing."""
    return "This is a sample document about artificial intelligence and machine learning. AI systems can process natural language and provide intelligent responses."


@pytest.fixture
def sample_embedding():
    """Sample embedding vector for testing."""
    return [0.1, 0.2, 0.3, 0.4, 0.5, -0.1, -0.2, 0.8, 0.9, -0.3]


@pytest.fixture
def sample_cleaned_text_file(temp_dir, sample_text_content):
    """Create a sample cleaned text file for testing."""
    text_file = temp_dir / "sample_cleaned.txt"
    text_file.write_text(sample_text_content, encoding="utf-8")
    return text_file


@pytest.fixture
def sample_embedding_file(temp_dir, sample_embedding):
    """Create a sample embedding file for testing."""
    embedding_file = temp_dir / "sample_cleaned_embeddings.json"
    with open(embedding_file, 'w') as f:
        json.dump(sample_embedding, f)
    return embedding_file


@pytest.fixture
def sample_pdf_content():
    """Sample PDF-like content for testing."""
    return "PDF Document Content\n\nThis is page 1 of the PDF.\n\nThis is page 2 of the PDF."


@pytest.fixture
def sample_chromadb_query_results():
    """Sample ChromaDB query results for testing."""
    return {
        "ids": [["doc1", "doc2"]],
        "metadatas": [[
            {"text": "Document about artificial intelligence and machine learning.", "source": "ai_doc.txt"},
            {"text": "Document about natural language processing.", "source": "nlp_doc.txt"}
        ]],
        "distances": [[0.2, 0.4]]
    }


@pytest.fixture
def sample_retrieved_documents():
    """Sample retrieved documents for RAG testing."""
    return [
        {
            "id": "doc1",
            "score": 0.2,
            "context": "Artificial intelligence (AI) refers to the simulation of human intelligence in machines.",
            "source": "ai_doc.txt"
        }
    ]