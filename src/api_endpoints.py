import logging
from fastapi import HTTPException, UploadFile, File, Request
from pathlib import Path
import shutil

from .api_models import QueryRequest, QueryResponse, UploadResponse, HealthResponse, RetrieveRequest, RetrieveResponse, RetrieveResult
from .llm_client import LLMClient
from .chromadb_retriever import ChromaDBRetriever
from .rag_query_processor import RAGQueryProcessor
from .document_ingestor import DocumentIngestor
from .embedding_preparer import EmbeddingPreparer
from .embedding_loader import EmbeddingLoader

async def query_rag(request: QueryRequest, app_request: Request) -> QueryResponse:
    """Query the RAG system with a given question."""
    try:
        config = app_request.app.state.config
        
        # Initialize LLM client
        llm_client = LLMClient(
            llm_api_url=config.get("llm_api_url"),
            llm_model_name=config.get("llm_model_name"),
            llm_provider=config.get("llm_provider", "ollama"),
            project_id=config.get("project_id"),
            location=config.get("location")
        )

        # Initialize retriever
        retriever = ChromaDBRetriever(
            vectordb_dir=config.get("vectordb_directory"),
            embedding_model_name=config.get("embedding_model_name"),
            collection_name=config.get("collection_name")
        )

        # Initialize RAG processor
        processor = RAGQueryProcessor(
            llm_client=llm_client,
            retriever=retriever,
            use_rag=request.use_rag
        )
        
        # Process query
        response = processor.query(request.query)
        
        return QueryResponse(
            query=request.query, 
            response=response, 
            use_rag=request.use_rag
        )
        
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

async def upload_document(file: UploadFile, app_request: Request) -> UploadResponse:
    """Upload a document and process it through steps 1-3 (ingest, embed, store)."""
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
        
        # Step 1: Document ingestion
        logging.info("[API] Starting Step 1: Document ingestion")
        ingestor = DocumentIngestor(
            file_list=[file.filename],
            input_dir=config.get("raw_input_directory"),
            output_dir=config.get("cleaned_text_directory"),
            embedding_model_name=config.get("embedding_model_name")
        )
        ingestor.process_files()
        logging.info("[API] Step 1 completed")
        
        # Step 2: Generate embeddings
        logging.info("[API] Starting Step 2: Embedding generation")
        cleaned_filename = f"{Path(file.filename).stem}_cleaned.txt"
        preparer = EmbeddingPreparer(
            file_list=[cleaned_filename],
            input_dir=config.get("cleaned_text_directory"),
            output_dir=config.get("embeddings_directory"),
            embedding_model_name=config.get("embedding_model_name")
        )
        preparer.process_files()
        logging.info("[API] Step 2 completed")
        
        # Step 3: Store vectors
        logging.info("[API] Starting Step 3: Vector storage")
        loader = EmbeddingLoader(
            cleaned_text_file_list=[cleaned_filename],
            cleaned_text_dir=config.get("cleaned_text_directory"),
            embeddings_dir=config.get("embeddings_directory"),
            vectordb_dir=config.get("vectordb_directory"),
            collection_name=config.get("collection_name")
        )
        loader.process_files()
        logging.info("[API] Step 3 completed")
        
        return UploadResponse(
            message="Document processed successfully",
            filename=file.filename,
            steps_completed=["ingest", "embed", "store"],
            status="success"
        )
        
    except Exception as e:
        logging.error(f"Error processing document upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

async def retrieve_chunks(request: RetrieveRequest, app_request: Request) -> RetrieveResponse:
    """Retrieve relevant text chunks based on a query."""
    try:
        config = app_request.app.state.config
        
        logging.info(f"Retrieving chunks for query: {request.query}")
        
        retriever = ChromaDBRetriever(
            vectordb_dir=config.get("vectordb_directory"),
            embedding_model_name=config.get("embedding_model_name"),
            collection_name=config.get("collection_name"),
            score_threshold=float(config.get("retriever_min_score_threshold"))
        )
        
        search_results = retriever.query(request.query, top_k=request.top_k)
        
        # Convert results to response format
        results = []
        if search_results:
            for result in search_results:
                results.append(RetrieveResult(
                    id=result.get('id', 'N/A'),
                    score=result.get('score', 0.0),
                    text=result.get('text', ''),
                    context=result.get('context', '')
                ))
        
        logging.info(f"Found {len(results)} relevant chunks")
        
        return RetrieveResponse(
            query=request.query,
            results=results,
            count=len(results)
        )
        
    except Exception as e:
        logging.error(f"Error retrieving chunks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving chunks: {str(e)}")

async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="healthy", message="RAG API is running")