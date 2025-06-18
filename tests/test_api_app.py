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
        assert app.version == "1.1.2"

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
        assert schema["info"]["version"] == "1.1.2"

    def test_cors_middleware_added(self, app):
        """Test that CORS middleware is properly added to the app."""
        middleware_names = [middleware.cls.__name__ for middleware in app.user_middleware]
        assert "CORSMiddleware" in middleware_names

    def test_cors_configuration_default(self):
        """Test CORS configuration with default origins."""
        import tempfile
        import json
        from pathlib import Path
        
        config_data = {"other_setting": "value"}
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            config = ConfigManager(config_file)
            app = create_app(config)
            cors_middleware = None
            for middleware in app.user_middleware:
                if middleware.cls.__name__ == "CORSMiddleware":
                    cors_middleware = middleware
                    break
            assert cors_middleware is not None
            assert cors_middleware.kwargs["allow_origins"] == ["*"]
        finally:
            Path(config_file).unlink()

    def test_cors_configuration_custom_origins(self):
        """Test CORS configuration with custom origins."""
        import tempfile
        import json
        from pathlib import Path
        
        custom_origins = ["http://localhost:3000", "https://example.com"]
        config_data = {"cors_allowed_origins": custom_origins}
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            config = ConfigManager(config_file)
            app = create_app(config)
            cors_middleware = None
            for middleware in app.user_middleware:
                if middleware.cls.__name__ == "CORSMiddleware":
                    cors_middleware = middleware
                    break
            assert cors_middleware is not None
            assert cors_middleware.kwargs["allow_origins"] == custom_origins
        finally:
            Path(config_file).unlink()

    def test_cors_preflight_request(self, client):
        """Test CORS preflight request handling."""
        response = client.options(
            "/",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "content-type"
            }
        )
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers

    def test_cors_actual_request(self, client):
        """Test CORS actual request with origin header."""
        response = client.get(
            "/openapi.json",
            headers={"Origin": "http://localhost:3000"}
        )
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers