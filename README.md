# RAG API

[![CI](https://github.com/tienpdinh/cisc691-a04/workflows/CI/badge.svg)](https://github.com/tienpdinh/cisc691-a04/actions)
[![codecov](https://codecov.io/gh/tienpdinh/cisc691-a04/graph/badge.svg?token=Oot2JmamNl)](https://codecov.io/gh/tienpdinh/cisc691-a04)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A modern REST API for Retrieval-Augmented Generation that processes documents and answers questions using either local LLMs (Ollama) or cloud AI services (Vertex AI). Built with FastAPI for high performance and easy integration.

## 🚀 Deployment Options

### Local Development
- **LLM**: Ollama (privacy-focused, local processing)
- **Setup**: Simple pip install + Ollama
- **Use case**: Development, testing, privacy requirements

### Production (GKE)
- **LLM**: Google Vertex AI (managed, scalable)
- **Setup**: Kubernetes deployment on Google Cloud
- **Use case**: Production workloads, high availability

## Quick Start (Local)

**Full containerized setup with Ollama, ChromaDB, and RAG API:**

```bash
# Start all services
docker-compose up -d

# Download the LLM model (first time only)
./scripts/setup-ollama.sh

# API will be available at http://localhost:8001
```

**Service endpoints:**
- RAG API: http://localhost:8001 (with docs at http://localhost:8001/docs)
- ChromaDB: http://localhost:8000
- Ollama: http://localhost:11434

**Benefits:**
- No local installations required (except Docker)
- Isolated services with persistent storage
- Matches production microservices architecture
- Easy cleanup with `docker-compose down`

### Development Setup

For code development while keeping services containerized:

```bash
# 1. Start only ChromaDB and Ollama
docker-compose up -d chromadb ollama

# 2. Install Python dependencies locally
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# 3. Configure for local development
cp config.local.json config.json
# Edit config.json: set chromadb_host to "localhost" for local development

# 4. Run API locally for development
python main.py
```

This hybrid approach allows code editing/debugging locally while using containerized services.

## Using the API

### Upload and Process Documents
```bash
# Docker Compose (port 8001)
curl -X POST "http://localhost:8001/upload-document" \
  -F "file=@your-document.pdf"

# Local Python (port 8000)  
curl -X POST "http://localhost:8000/upload-document" \
  -F "file=@your-document.pdf"
```

### Query Documents
```bash
# Ask questions about your documents
curl -X POST "http://localhost:8001/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this document about?", "use_rag": true}'
```

### Retrieve Relevant Chunks
```bash
# Get relevant document chunks for a query
curl -X POST "http://localhost:8001/retrieve" \
  -H "Content-Type: application/json" \
  -d '{"query": "artificial intelligence", "top_k": 3}'
```

### Health Check
```bash
# Check API status
curl "http://localhost:8001/health"
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload-document` | Upload and process documents (PDF, TXT, DOCX) |
| `POST` | `/query` | Query documents with RAG |
| `POST` | `/retrieve` | Retrieve relevant document chunks |
| `GET` | `/health` | API health check |
| `GET` | `/docs` | Interactive API documentation |

## Production Deployment (GKE)

### Prerequisites
- Google Cloud Project with billing enabled
- GKE cluster
- kubectl configured

### Deploy
```bash
# Build and push Docker image
docker build -t gcr.io/cisc691-a04/rag-api:latest .
docker push gcr.io/cisc691-a04/rag-api:latest

# Set up GCP authentication (see k8s/setup-gcp.md)
# Then deploy
kubectl apply -f k8s/
```

For detailed deployment instructions, see [`k8s/README.md`](k8s/README.md).

## Architecture

This RAG API uses a **microservices architecture** with separate containers for different components:

### Local Development (Docker Compose)
- **RAG API**: FastAPI server handling document processing and queries
- **ChromaDB**: Vector database service for embeddings storage
- **Ollama**: Local LLM service with `llama3.1:8b`
- **Communication**: All services communicate via HTTP APIs

### Production (GKE)
- **RAG API**: Load-balanced FastAPI pods with auto-scaling
- **ChromaDB**: Dedicated service with persistent storage
- **Vertex AI**: Google's managed LLM service (`gemini-1.5-flash`)
- **Communication**: Kubernetes service mesh with internal networking

## Configuration

### Local Configuration (`config.local.json`)
- **LLM**: Containerized Ollama service
- **Vector DB**: HTTP connection to ChromaDB service
- **Storage**: Docker volumes for persistence
- **Networking**: Docker Compose internal networking

### Production Configuration (Kubernetes ConfigMap)
- **LLM**: Vertex AI with Workload Identity
- **Vector DB**: HTTP connection to ChromaDB service
- **Storage**: Kubernetes persistent volumes
- **Networking**: Service mesh with load balancing

## Testing & Development

### Run Tests
```bash
pytest --cov=src --cov=. --cov-report=html --cov-report=term-missing
```

### View Coverage Report
```bash
open htmlcov/index.html  # View detailed coverage report
```

## Project Structure

```
├── src/                        # Source code
│   ├── api_app.py             # FastAPI application setup
│   ├── api_routes.py          # API route definitions
│   ├── api_endpoints.py       # Endpoint business logic
│   ├── api_models.py          # Pydantic request/response models
│   ├── chromadb_retriever.py  # ChromaDB HTTP client
│   ├── embedding_loader.py    # ChromaDB HTTP client for storage
│   └── ...                    # Core RAG modules
├── tests/                     # Unit tests
│   ├── test_api_*.py          # API endpoint tests
│   ├── test_chromadb_*.py     # ChromaDB HTTP client tests
│   └── ...                    # Core module tests
├── k8s/                       # Kubernetes manifests
│   ├── chromadb-*.yaml        # ChromaDB service manifests
│   ├── deployment.yaml        # RAG API deployment
│   └── ...                    # Other K8s resources
├── data/                      # Data directories (Docker volumes)
├── docker-compose.yml         # Local development setup
├── config.local.json          # Local development config
├── main.py                    # API server entry point
└── Dockerfile                 # RAG API container image
```

## Troubleshooting

### Docker Compose Issues
- **Services won't start**: Run `docker-compose logs` to check errors
- **Model not found**: Run `./scripts/setup-ollama.sh` to download models
- **Port conflicts**: Change ports in `docker-compose.yml` if needed
- **Storage issues**: Run `docker-compose down -v` to reset volumes

### Local Python Issues  
- **API won't start**: Make sure `ollama serve` is running first
- **Upload failures**: Check file types (PDF, TXT, DOCX only)
- **Query returns empty**: Upload documents first via `/upload-document`
- **CUDA errors**: Install CUDA drivers for GPU support

### GKE Issues
- **Authentication**: Verify service account setup
- **Pod failures**: Check `kubectl logs` and resource limits
- **Vertex AI errors**: Ensure APIs are enabled and billing is active
- **API timeout**: Check load balancer and service configuration

## Contributing

1. Run tests: `pytest`
2. Check coverage: Coverage reports available in CI
3. Follow code style: Automated linting in CI