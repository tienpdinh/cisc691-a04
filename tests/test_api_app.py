import pytest
from fastapi.testclient import TestClient
from src.api_app import create_app
from src.config_manager import ConfigManager

@pytest.fixture
def mock_config():
    """Create a mock config for testing."""
    config = ConfigManager("config.json")
    # Override with test values if needed
    return config

@pytest.fixture
def app(mock_config):
    """Create FastAPI app for testing."""
    return create_app(mock_config)

@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)

class TestAPIApp:
    def test_app_creation(self, mock_config):
        """Test that the app is created with correct configuration."""
        app = create_app(mock_config)
        assert app.title == "RAG API"
        assert app.description == "API for querying the RAG system"
        assert app.version == "1.1.0"

    def test_config_stored_in_app_state(self, app, mock_config):
        """Test that config is properly stored in app state."""
        assert hasattr(app.state, 'config')
        assert app.state.config == mock_config

    def test_app_has_openapi_schema(self, client):
        """Test that OpenAPI schema is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert schema["info"]["title"] == "RAG API"
        assert schema["info"]["version"] == "1.1.0"