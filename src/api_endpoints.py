import logging
from fastapi import HTTPException, UploadFile, Request
from pathlib import Path
import shutil

from .api_models import QueryRequest, QueryResponse, UploadResponse, HealthResponse, RetrieveRequest, RetrieveResponse, RetrieveResult
from .langchain_document_processor import LangChainDocumentProcessor
from .langchain_vector_store import LangChainVectorStore
from .langchain_rag_processor import LangChainRAGProcessor

# Global variables to store LangChain components
_vector_store = None
_rag_processor = None


def _get_or_create_vector_store(config) -> LangChainVectorStore:
    """Get or create the global vector store instance."""
    global _vector_store
    
    if _vector_store is None:
        _vector_store = LangChainVectorStore(
            collection_name=config.get("collection_name"),
            embedding_model_name=config.get("embedding_model_name"),
            chromadb_host=config.get("chromadb_host"),
            chromadb_port=config.get("chromadb_port", 8000),
            persist_directory=config.get("vectordb_directory")
        )
    
    return _vector_store


def _get_or_create_rag_processor(config) -> LangChainRAGProcessor:
    """Get or create the global RAG processor instance."""
    global _rag_processor
    
    if _rag_processor is None:
        vector_store = _get_or_create_vector_store(config)
        
        _rag_processor = LangChainRAGProcessor(
            vector_store=vector_store,
            llm_provider=config.get("llm_provider", "ollama"),
            llm_model_name=config.get("llm_model_name"),
            llm_api_url=config.get("llm_api_url"),
            openai_api_key=config.get("openai_api_key")
        )
    
    return _rag_processor


async def query_rag(request: QueryRequest, app_request: Request) -> QueryResponse:
    """Query the RAG system using LangChain components."""
    config = app_request.app.state.config
    
    try:
        logging.info(f"Processing query: {request.query}")
        
        # Get RAG processor
        rag_processor = _get_or_create_rag_processor(config)
        
        # Process query
        result = rag_processor.query(
            question=request.query,
            use_rag=request.use_rag
        )
        
        # Format response
        response = QueryResponse(
            query=request.query,
            response=result["response"],
            use_rag=request.use_rag
        )
        
        logging.info(f"Generated response, use_rag: {response.use_rag}")
        return response
        
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

async def upload_document(file: UploadFile, app_request: Request) -> UploadResponse:
    """Upload and process a document using LangChain components."""
    config = app_request.app.state.config
    
    # Validate file type
    allowed_extensions = ['.pdf', '.txt', '.docx']
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Save uploaded file to raw_input directory
        raw_input_dir = Path(config.get("raw_input_directory"))
        raw_input_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = raw_input_dir / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logging.info(f"Uploaded file saved: {file_path}")
        
        # Step 1: Document processing with LangChain
        logging.info("Starting Step 1: Document processing with LangChain")
        
        processor = LangChainDocumentProcessor(
            file_list=[file.filename],
            input_dir=config.get("raw_input_directory"),
            output_dir=config.get("cleaned_text_directory"),
            chunk_size=config.get("chunk_size", 1000),
            chunk_overlap=config.get("chunk_overlap", 200)
        )
        
        # Process documents and get LangChain Document objects
        documents = processor.process_files()
        
        if not documents:
            raise HTTPException(status_code=400, detail="Failed to process document")
        
        logging.info(f"Step 1 completed - processed {len(documents)} document chunks")
        
        # Step 2: Add documents to vector store
        logging.info("Starting Step 2: Adding to vector store")
        
        vector_store = _get_or_create_vector_store(config)
        document_ids = vector_store.add_documents(documents)
        
        if not document_ids:
            raise HTTPException(status_code=500, detail="Failed to add documents to vector store")
        
        logging.info(f"Step 2 completed - added {len(document_ids)} documents to vector store")
        
        return UploadResponse(
            message="Document processed successfully with LangChain",
            filename=file.filename,
            steps_completed=["langchain_document_processing", "langchain_vector_storage"],
            status="success",
            document_chunks=len(documents)
        )
        
    except Exception as e:
        logging.error(f"Error processing document upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

async def retrieve_chunks(request: RetrieveRequest, app_request: Request) -> RetrieveResponse:
    """Retrieve relevant text chunks using LangChain vector store."""
    config = app_request.app.state.config
    
    try:
        logging.info(f"Retrieving chunks for query: {request.query}")
        
        # Get RAG processor
        rag_processor = _get_or_create_rag_processor(config)
        
        # Retrieve documents
        results = rag_processor.retrieve_documents(
            query=request.query,
            k=request.top_k,
            score_threshold=float(config.get("retriever_min_score_threshold", 0.5))
        )
        
        # Format results
        retrieve_results = []
        for i, result in enumerate(results):
            # Extract ID from metadata or create one
            doc_id = result["metadata"].get("source", f"doc_{i}")
            
            retrieve_results.append(RetrieveResult(
                id=doc_id,
                score=result["score"],
                text=result["content"],
                context=result["content"]  # Use same content for context
            ))
        
        response = RetrieveResponse(
            query=request.query,
            results=retrieve_results,
            count=len(retrieve_results)
        )
        
        logging.info(f"Retrieved {len(retrieve_results)} chunks")
        return response
        
    except Exception as e:
        logging.error(f"Error retrieving chunks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving chunks: {str(e)}")

async def health_check() -> HealthResponse:
    """Health check endpoint for LangChain-based API."""
    try:
        # Check if global components are initialized
        vector_store_status = "initialized" if _vector_store is not None else "not_initialized"
        rag_processor_status = "initialized" if _rag_processor is not None else "not_initialized"
        
        # Get collection info if vector store is available
        collection_info = {}
        if _vector_store is not None:
            try:
                collection_info = _vector_store.get_collection_info()
            except:
                collection_info = {"error": "Failed to get collection info"}
        
        return HealthResponse(
            status="healthy",
            message="LangChain RAG API is running",
            components={
                "vector_store": vector_store_status,
                "rag_processor": rag_processor_status,
                "collection_info": collection_info
            }
        )
        
    except Exception as e:
        logging.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            message=f"Health check failed: {str(e)}",
            components={}
        )