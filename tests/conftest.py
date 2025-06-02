"""Test configuration and fixtures."""
import json
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def config_dict():
    """Sample configuration dictionary."""
    return {
        "log_level": "INFO",
        "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "collection_name": "test_documents",
        "retriever_min_score_threshold": "0.5",
        "llm_api_url": "http://localhost:11434/api/generate",
        "llm_model_name": "llama3.1:8b",
        "raw_input_directory": "data/raw_input",
        "cleaned_text_directory": "data/cleaned_text",
        "embeddings_directory": "data/embeddings",
        "vectordb_directory": "data/vectordb"
    }

@pytest.fixture
def config_file(temp_dir, config_dict):
    """Create a temporary config file."""
    config_path = temp_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f)
    return config_path

@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return "This is a sample document about artificial intelligence and machine learning."

@pytest.fixture
def sample_pdf_content():
    """Sample PDF content for testing."""
    return "PDF content about AI and ML technologies in healthcare applications."

@pytest.fixture
def sample_embeddings():
    """Sample embedding vector."""
    return [0.1, 0.2, 0.3, 0.4, 0.5] * 10  # 50-dimensional vector

@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer.tokenize.return_value = ["sample", "tokens"]
    tokenizer.convert_tokens_to_string.return_value = "sample tokens"
    return tokenizer

@pytest.fixture
def mock_model():
    """Mock transformer model for testing."""
    model = MagicMock()
    model.to.return_value = model
    return model

@pytest.fixture
def mock_chromadb_collection():
    """Mock ChromaDB collection."""
    collection = MagicMock()
    collection.count.return_value = 5
    collection.add.return_value = None
    collection.query.return_value = {
        "ids": [["doc1"]],
        "metadatas": [[{"text": "sample text", "source": "doc1.txt"}]],
        "distances": [[0.3]]
    }
    return collection

@pytest.fixture
def mock_chromadb_client(mock_chromadb_collection):
    """Mock ChromaDB client."""
    client = MagicMock()
    client.get_or_create_collection.return_value = mock_chromadb_collection
    return client