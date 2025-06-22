import logging
import hashlib
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from .cache_manager import get_cache_manager


class LangChainVectorStore:
    """
    LangChain-based vector store handler using ChromaDB for document storage and retrieval.
    Replaces custom embedding and vector storage logic.
    """
    
    def __init__(self,
                 collection_name: str,
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 chromadb_host: str = "localhost",
                 chromadb_port: int = 8000,
                 persist_directory: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LangChain vector store.
        
        :param collection_name: Name of the ChromaDB collection
        :param embedding_model_name: HuggingFace embedding model name
        :param chromadb_host: ChromaDB server host
        :param chromadb_port: ChromaDB server port
        :param persist_directory: Directory for local persistence
        :param config: Configuration dictionary for caching
        """
        self.collection_name = collection_name
        self.chromadb_host = chromadb_host
        self.chromadb_port = chromadb_port
        self.persist_directory = persist_directory
        self.config = config or {}
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize cache manager
        self.cache_manager = get_cache_manager(self.config) if self.config else None
        
        # Initialize LangChain embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.logger.info(f"Initialized HuggingFace embeddings: {embedding_model_name}")
        
        # Initialize ChromaDB vector store
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize the ChromaDB vector store with LangChain integration."""
        try:
            if self.chromadb_host != "localhost" or self.chromadb_port != 8000:
                # Use HTTP client for remote ChromaDB
                import chromadb
                client = chromadb.HttpClient(
                    host=self.chromadb_host, 
                    port=self.chromadb_port
                )
                
                self.vector_store = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                    client=client
                )
                self.logger.info(f"Connected to remote ChromaDB at {self.chromadb_host}:{self.chromadb_port}")
            else:
                # Use local ChromaDB with persistence
                self.vector_store = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=self.persist_directory
                )
                self.logger.info(f"Initialized local ChromaDB with persistence: {self.persist_directory}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.
        
        :param documents: List of LangChain Document objects
        :return: List of document IDs
        """
        try:
            self.logger.info(f"Adding {len(documents)} documents to vector store")
            
            # Add documents to ChromaDB via LangChain
            ids = self.vector_store.add_documents(documents)
            
            self.logger.info(f"Successfully added {len(ids)} documents to ChromaDB")
            return ids
            
        except Exception as e:
            self.logger.error(f"Error adding documents to vector store: {e}")
            return []
    
    async def similarity_search(self, 
                               query: str, 
                               k: int = 4,
                               score_threshold: Optional[float] = None) -> List[Document]:
        """
        Perform similarity search for relevant documents with caching.
        
        :param query: Search query
        :param k: Number of documents to return
        :param score_threshold: Minimum similarity score threshold
        :return: List of relevant documents
        """
        # Check cache first
        cache_key = self._generate_search_cache_key(query, k, score_threshold)
        if self.cache_manager:
            cached_docs = await self.cache_manager.get(cache_key, 'document_retrieval')
            if cached_docs:
                self.logger.debug(f"Cache hit for similarity search: {query[:50]}...")
                # Convert cached dict back to Document objects
                return [Document(page_content=doc['page_content'], metadata=doc['metadata']) 
                       for doc in cached_docs]

        try:
            self.logger.info(f"Searching for '{query}' with k={k}")
            
            if score_threshold is not None:
                # Use similarity search with score threshold
                docs_and_scores = self.vector_store.similarity_search_with_score(
                    query, k=k
                )
                
                # Filter by score threshold
                filtered_docs = [
                    doc for doc, score in docs_and_scores 
                    if score >= score_threshold
                ]
                
                self.logger.info(f"Found {len(filtered_docs)} documents above threshold {score_threshold}")
                result_docs = filtered_docs
            else:
                # Regular similarity search
                docs = self.vector_store.similarity_search(query, k=k)
                self.logger.info(f"Found {len(docs)} similar documents")
                result_docs = docs

            # Cache the results
            if self.cache_manager and result_docs:
                # Convert Documents to dict for caching
                cacheable_docs = [
                    {'page_content': doc.page_content, 'metadata': doc.metadata}
                    for doc in result_docs
                ]
                await self.cache_manager.set(cache_key, cacheable_docs, 'document_retrieval')
                self.logger.debug(f"Cached similarity search results for: {query[:50]}...")

            return result_docs
                
        except Exception as e:
            self.logger.error(f"Error during similarity search: {e}")
            return []

    def _generate_search_cache_key(self, query: str, k: int, score_threshold: Optional[float]) -> str:
        """Generate cache key for similarity search."""
        key_data = {
            'query': query,
            'k': k,
            'score_threshold': score_threshold,
            'collection': self.collection_name
        }
        return f"search:{hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()[:16]}"
    
    async def similarity_search_with_scores(self, 
                                          query: str, 
                                          k: int = 4) -> List[tuple]:
        """
        Perform similarity search with scores and caching.
        
        :param query: Search query
        :param k: Number of documents to return
        :return: List of (document, score) tuples
        """
        # Check cache first
        cache_key = self._generate_search_cache_key(query, k, None)
        if self.cache_manager:
            cached_results = await self.cache_manager.get(cache_key, 'document_retrieval')
            if cached_results:
                self.logger.debug(f"Cache hit for similarity search with scores: {query[:50]}...")
                # Convert cached dict back to (Document, score) tuples
                return [(Document(page_content=item['doc']['page_content'], 
                                metadata=item['doc']['metadata']), 
                        item['score']) for item in cached_results]

        try:
            docs_and_scores = self.vector_store.similarity_search_with_score(query, k=k)
            self.logger.info(f"Found {len(docs_and_scores)} documents with scores")
            
            # Cache the results
            if self.cache_manager and docs_and_scores:
                # Convert to cacheable format
                cacheable_results = [
                    {
                        'doc': {'page_content': doc.page_content, 'metadata': doc.metadata},
                        'score': score
                    }
                    for doc, score in docs_and_scores
                ]
                await self.cache_manager.set(cache_key, cacheable_results, 'document_retrieval')
                self.logger.debug(f"Cached similarity search with scores for: {query[:50]}...")
            
            return docs_and_scores
            
        except Exception as e:
            self.logger.error(f"Error during similarity search with scores: {e}")
            return []
    
    def similarity_search_sync(self, 
                              query: str, 
                              k: int = 4,
                              score_threshold: Optional[float] = None) -> List[Document]:
        """
        Synchronous version of similarity search for LangChain chains.
        
        :param query: Search query
        :param k: Number of documents to return
        :param score_threshold: Minimum similarity score threshold
        :return: List of relevant documents
        """
        try:
            self.logger.info(f"Sync searching for '{query}' with k={k}")
            
            if score_threshold is not None:
                # Use similarity search with score threshold
                docs_and_scores = self.vector_store.similarity_search_with_score(
                    query, k=k
                )
                
                # Filter by score threshold
                filtered_docs = [
                    doc for doc, score in docs_and_scores 
                    if score >= score_threshold
                ]
                
                self.logger.info(f"Found {len(filtered_docs)} documents above threshold {score_threshold}")
                return filtered_docs
            else:
                # Regular similarity search
                docs = self.vector_store.similarity_search(query, k=k)
                self.logger.info(f"Found {len(docs)} similar documents")
                return docs
                
        except Exception as e:
            self.logger.error(f"Error during sync similarity search: {e}")
            return []
    
    def delete_collection(self):
        """Delete the entire collection."""
        try:
            self.vector_store.delete_collection()
            self.logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            self.logger.error(f"Error deleting collection: {e}")
    
    def get_collection_info(self) -> dict:
        """Get information about the collection."""
        try:
            # Get collection stats
            collection = self.vector_store._collection
            count = collection.count()
            
            info = {
                "collection_name": self.collection_name,
                "document_count": count,
                "embedding_model": self.embeddings.model_name
            }
            
            self.logger.info(f"Collection info: {info}")
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting collection info: {e}")
            return {}