import logging
from pathlib import Path
import chromadb
import json
from typing import List

class EmbeddingLoader:
    def __init__(self,
                 cleaned_text_file_list: List[str],
                 cleaned_text_dir: str,
                 embeddings_dir: str,
                 collection_name: str,
                 chromadb_host: str,
                 chromadb_port: int = 8000,
                 batch_size: int = 16,
                 vectordb_dir: str = None,
                 store_vectors_locally: bool = False):

        self.cleaned_text_file_list = cleaned_text_file_list
        self.cleaned_text_path = Path(cleaned_text_dir)
        self.embeddings_path = Path(embeddings_dir)
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.store_vectors_locally = store_vectors_locally
        self.logger = logging.getLogger(__name__)
        
        # Setup local vector storage if enabled
        if self.store_vectors_locally and vectordb_dir:
            self.vectordb_path = Path(vectordb_dir)
            self.vectordb_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Local vector storage enabled: {self.vectordb_path}")
        else:
            self.vectordb_path = None

        # Always use HTTP client for containerized setup
        self.client = chromadb.HttpClient(host=chromadb_host, port=chromadb_port)
        self.logger.info(f"Connected to ChromaDB at {chromadb_host}:{chromadb_port}")
            
        self.collection = self.client.get_or_create_collection(collection_name)

    def _load_cleaned_text(self, file_path: Path) -> str:
        """Loads the cleaned text from a file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            self.logger.error(f"Error loading text file {file_path}: {e}")
            return ""

    def _load_embeddings(self, file_path: Path) -> List[float]:
        """Loads embeddings from a JSON file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                embeddings = json.load(f)
                if isinstance(embeddings, list) and all(isinstance(e, (int, float)) for e in embeddings):
                    return embeddings
                else:
                    raise ValueError("Invalid embedding format.")
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Error parsing embeddings file {file_path}: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error loading embeddings file {file_path}: {e}")

        return []

    def _store_vector_locally(self, filename: str, embeddings: List[float], text: str):
        """Store vector data locally in JSON format."""
        try:
            vector_data = {
                "id": filename,
                "embeddings": embeddings,
                "metadata": {
                    "text": text,
                    "source": filename,
                    "collection": self.collection_name
                }
            }
            
            output_file = self.vectordb_path / f"{Path(filename).stem}_vector.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(vector_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"Error storing vector locally for {filename}: {e}")

    def process_files(self):
        """Processes and stores cleaned text and embeddings into ChromaDB."""
        for cleaned_text_file in self.cleaned_text_file_list:
            cleaned_text_file_path = self.cleaned_text_path / cleaned_text_file
            embedding_file_path = self.embeddings_path / f"{Path(cleaned_text_file).stem}_embeddings.json"

            if not cleaned_text_file_path.exists():
                self.logger.warning(f"Missing cleaned text file: {cleaned_text_file_path}")
                continue
            if not embedding_file_path.exists():
                self.logger.warning(f"Missing embedding file for {cleaned_text_file}, skipping.")
                continue

            text = self._load_cleaned_text(cleaned_text_file_path)
            embeddings = self._load_embeddings(embedding_file_path)

            if not text or not embeddings:
                self.logger.warning(f"Skipping {cleaned_text_file} due to missing text or embeddings.")
                continue

            self.logger.info(f"Storing {cleaned_text_file} in ChromaDB...")

            self.collection.add(
                ids=[cleaned_text_file],
                embeddings=[embeddings],
                metadatas=[{"text": text, "source": cleaned_text_file}]
            )

            self.logger.info(f"Stored {cleaned_text_file} in ChromaDB successfully.")
            
            # Also store locally if enabled
            if self.store_vectors_locally and self.vectordb_path:
                self._store_vector_locally(cleaned_text_file, embeddings, text)
                self.logger.info(f"Stored {cleaned_text_file} locally successfully.")