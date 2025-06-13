from fastapi import FastAPI
from .config_manager import ConfigManager

def create_app(config: ConfigManager) -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="RAG API", 
        description="API for querying the RAG system", 
        version="1.2.0"
    )
    
    # Store config in app state for access in endpoints
    app.state.config = config
    
    return app