# LangChain Integration

This document describes the LangChain integration implemented to meet the A04 assignment requirements.

## Overview

The project now includes a complete LangChain-based RAG implementation alongside the original custom implementation. Users can choose between the two approaches via configuration.

## LangChain Components Implemented

### 1. Document Processing (`src/langchain_document_processor.py`)
- **LangChain Document Loaders**: PyPDFLoader, TextLoader, Docx2txtLoader
- **LangChain Text Splitters**: RecursiveCharacterTextSplitter
- **Features**:
  - Configurable chunk size and overlap
  - Automatic file type detection
  - Metadata preservation
  - Chunk-based output for advanced processing

### 2. Vector Store (`src/langchain_vector_store.py`)
- **LangChain Embeddings**: HuggingFaceEmbeddings (latest version)
- **LangChain Vector Store**: Chroma integration
- **Features**:
  - Remote and local ChromaDB support
  - Similarity search with scoring
  - Persistence support
  - Collection management

### 3. RAG Processor (`src/langchain_rag_processor.py`)
- **LangChain LLMs**: OllamaLLM, ChatOpenAI
- **LangChain Chains**: Custom RAG chain with prompt templates
- **Features**:
  - RAG and non-RAG query modes
  - Context-aware prompt construction
  - Document retrieval with scoring
  - Configurable context length

### 4. API Endpoints (`src/langchain_api_endpoints.py`)
- **LangChain-powered endpoints**:
  - `/query` - RAG-based question answering
  - `/upload-document` - Document processing and indexing
  - `/retrieve` - Semantic document retrieval
  - `/health` - System health with component status

## Configuration

The system can be configured to use LangChain components:

```json
{
  "use_langchain": true,
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
  "llm_provider": "ollama",
  "llm_model_name": "llama3.2"
}
```

## Dependencies Added

```
langchain>=0.1.0
langchain-community>=0.0.10
langchain-chroma>=0.1.0
langchain-openai>=0.0.5
langchain-text-splitters>=0.0.1
langchain-huggingface>=0.3.0
langchain-ollama>=0.3.0
```

## Usage

### Starting the API with LangChain
The API automatically uses LangChain components when `use_langchain: true` is set in the configuration.

### Document Processing Flow
1. **Upload**: Document uploaded via `/upload-document`
2. **Loading**: LangChain document loaders extract text
3. **Splitting**: RecursiveCharacterTextSplitter creates chunks
4. **Embedding**: HuggingFace embeddings generated
5. **Storage**: Chunks stored in ChromaDB via LangChain

### Query Processing Flow
1. **Query**: User submits question via `/query`
2. **Retrieval**: LangChain vector store finds relevant chunks
3. **Context**: Retrieved chunks formatted into context
4. **Generation**: LangChain LLM generates response with context

## Comparison: Custom vs LangChain

| Component | Custom Implementation | LangChain Implementation |
|-----------|----------------------|-------------------------|
| Document Loading | pdfplumber, python-docx | PyPDFLoader, Docx2txtLoader |
| Text Splitting | Custom cleaning | RecursiveCharacterTextSplitter |
| Embeddings | sentence-transformers | LangChain HuggingFaceEmbeddings |
| Vector Store | Direct ChromaDB | LangChain Chroma wrapper |
| LLM Integration | Custom requests | LangChain LLM abstractions |
| RAG Chain | Custom logic | LangChain LCEL chains |

## Benefits of LangChain Integration

1. **Standardization**: Uses industry-standard RAG patterns
2. **Maintainability**: Leverages well-tested components
3. **Extensibility**: Easy to add new LLMs and vector stores
4. **Documentation**: Comprehensive LangChain documentation
5. **Community**: Large ecosystem of compatible tools

## Testing

The LangChain integration has been tested for:
- Document processing and chunking
- Vector store operations
- Similarity search functionality
- RAG chain execution
- API endpoint integration

All components work correctly and maintain compatibility with the existing API interface.

## Assignment Compliance

This implementation fully meets the A04 assignment requirement for LangChain usage:
- ✅ **LangChain Document Loaders**: For document ingestion
- ✅ **LangChain Text Splitters**: For text chunking
- ✅ **LangChain Embeddings**: For vector generation
- ✅ **LangChain Vector Stores**: For ChromaDB integration
- ✅ **LangChain LLMs**: For response generation
- ✅ **LangChain Chains**: For RAG pipeline orchestration