from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config_manager import ConfigManager

def create_app(config: ConfigManager) -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="RAG API", 
        description="API for querying the RAG system", 
        version="1.1.2"
    )
    
    # Add CORS middleware
    cors_origins = config.get("cors_allowed_origins", ["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],  # Allows all methods
        allow_headers=["*"],  # Allows all headers
    )
    
    # Store config in app state for access in endpoints
    app.state.config = config
    
    return app