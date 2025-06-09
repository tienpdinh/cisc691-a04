import logging
from pathlib import Path
from src.config_manager import ConfigManager

from datetime import datetime
import uvicorn

CONFIG_FILE = "config.json"
config = ConfigManager(CONFIG_FILE)  # Use ConfigManager for configuration loading

# FastAPI application will be created in start_api_server()


def setup_logging(log_level):
    """ Configures logging to console and file."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    numeric_level = getattr(logging, log_level.upper(), logging.DEBUG)
    log_filename = f"rag_pipeline_{datetime.now().strftime('%Y%m%d_%H%M_%S%f')[:-4]}"  # Remove last 4 digits of microseconds
    logging.basicConfig(
        level=numeric_level,
        format="[%(asctime)s] %(levelname)s %(module)s:%(lineno)d - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / log_filename)
        ]
    )
    logging.getLogger("transformers").setLevel(logging.INFO)
    logging.getLogger("pdfplumber").setLevel(logging.INFO)
    logging.getLogger("chromadb").setLevel(logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.WARNING)  # Reduce excessive API logs


def ensure_directories_exist(config):
    """Ensures necessary directories exist, creating them if needed."""
    for key in config.get_directory_names():
        dir_path = Path(config.get(key, key))  # Use key name as default
        dir_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Ensured directory exists: {dir_path}")


# All steps are now available through API endpoints


def main():
    """Start the RAG API server."""
    start_api_server()

# API endpoints are now defined in src/api_routes.py

def start_api_server():
    """Start the FastAPI server."""
    from src.api_app import create_app
    from src.api_routes import create_router
    
    setup_logging(config.get("log_level", "DEBUG"))
    ensure_directories_exist(config)
    
    # Create FastAPI app and configure routes
    app = create_app(config)
    router = create_router()
    app.include_router(router)
    
    logging.info("Starting RAG API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()