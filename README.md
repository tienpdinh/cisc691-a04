# RAG API

[![CI](https://github.com/tienpdinh/cisc691-a04/workflows/CI/badge.svg)](https://github.com/tienpdinh/cisc691-a04/actions)
[![codecov](https://codecov.io/gh/tienpdinh/cisc691-a04/graph/badge.svg?token=Oot2JmamNl)](https://codecov.io/gh/tienpdinh/cisc691-a04)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A modern REST API for Retrieval-Augmented Generation that processes documents and answers questions using either local LLMs (Ollama) or cloud AI services (Vertex AI). Built with FastAPI and designed for cloud-native deployment with Google Cloud Storage and Kubernetes.

## 🚀 Deployment Options

### Local Development
- **LLM**: Ollama (privacy-focused, local processing)
- **Storage**: Fake GCS Server (containerized S3-compatible storage)
- **Setup**: Simple docker compose command
- **Use case**: Development, testing, privacy requirements

### Production (GKE)
- **LLM**: Google Vertex AI (managed, scalable)
- **Storage**: Google Cloud Storage (managed, durable)
- **Setup**: Terraform + Kubernetes deployment on Google Cloud
- **Use case**: Production workloads, high availability

## Prerequisites

**Docker is required** for running this application. Install Docker:

- **Windows/Mac**: Download [Docker Desktop](https://www.docker.com/products/docker-desktop)
- **Linux**: `sudo apt install docker.io docker-compose` (Ubuntu/Debian) or check your distro's package manager

## Quick Start (Local)

**Full containerized setup with Ollama, ChromaDB, and RAG API:**

```bash
# Start all services, this will take a while if you're running this the first time, grab a coffee while you wait
docker compose up -d

# Download the LLM model (first time only)
./scripts/setup-ollama.sh

# API will be available at http://localhost:8001
```

**Service endpoints:**
- RAG API: http://localhost:8001 (with docs at http://localhost:8001/docs)
- ChromaDB: http://localhost:8000
- Ollama: http://localhost:11434
- Fake GCS Server: http://localhost:4443


### Development Setup

For code development while keeping services containerized:

```bash
# 1. Start only ChromaDB, Ollama, and Fake GCS
docker-compose up -d chromadb ollama fake-gcs

# 2. Install Python dependencies locally
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# 3. Configure for local development
cp config.local.json config.json
# Edit config.json: set chromadb_host to "localhost" for local development

# 4. Set environment variable for fake GCS
export STORAGE_EMULATOR_HOST=http://localhost:4443

# 5. Run API locally for development
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
- Terraform and gcloud CLI configured

### Deploy
```bash
# 1. Deploy infrastructure (GKE, GCS, IAM)
cd terraform/
# See terraform/README.md for detailed instructions
terraform apply

# 2. Deploy application
kubectl apply -f k8s/
```

For detailed deployment instructions:
- **Infrastructure**: [`terraform/README.md`](terraform/README.md)
- **Kubernetes**: [`k8s/README.md`](k8s/README.md)

## Architecture

This RAG API uses a **cloud-native microservices architecture** with unified storage abstraction:

### Local Development (Docker Compose)
- **RAG API**: FastAPI server handling document processing and queries
- **ChromaDB**: Vector database service for embeddings storage
- **Ollama**: Local LLM service with `llama3.1:8b`
- **Fake GCS Server**: S3-compatible storage emulator
- **Storage**: GCS abstraction layer with fake-gcs-server backend
- **Communication**: All services communicate via HTTP APIs

### Production (GKE)
- **RAG API**: Load-balanced FastAPI pods with auto-scaling
- **ChromaDB**: Dedicated service with persistent storage
- **Vertex AI**: Google's managed LLM service (`gemini-2.5-flash-preview`)
- **Google Cloud Storage**: Managed object storage for documents
- **Storage**: GCS abstraction layer with native GCS backend
- **Communication**: Kubernetes service mesh with Workload Identity

### Storage Architecture
- **Unified Interface**: Same GCS client code works in both environments
- **Local**: `fake-gcs-server` provides S3-compatible API
- **Production**: Native Google Cloud Storage with bucket lifecycle management
- **Buckets**: Separate buckets for raw input, cleaned text, and embeddings

## Configuration

### Local Configuration (`config.local.json`)
- **LLM**: Containerized Ollama service
- **Vector DB**: HTTP connection to ChromaDB service
- **Storage**: GCS buckets via fake-gcs-server
- **Environment**: `STORAGE_EMULATOR_HOST=http://fake-gcs:4443`
- **Networking**: Docker Compose internal networking

### Production Configuration (Kubernetes ConfigMap)
- **LLM**: Vertex AI with Workload Identity authentication
- **Vector DB**: HTTP connection to ChromaDB service
- **Storage**: Native Google Cloud Storage buckets
- **Authentication**: Workload Identity (no service account keys)
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
├── src/                       # Source code
│   ├── api_app.py             # FastAPI application setup
│   ├── api_routes.py          # API route definitions
│   ├── api_endpoints.py       # Endpoint business logic
│   ├── api_models.py          # Pydantic request/response models
│   ├── chromadb_retriever.py  # ChromaDB HTTP client
│   ├── embedding_loader.py    # ChromaDB HTTP client for storage
│   ├── gcs_storage.py         # GCS storage abstraction layer
│   └── ...                    # Core RAG modules
├── tests/                     # Unit tests
│   ├── test_api_*.py          # API endpoint tests
│   ├── test_chromadb_*.py     # ChromaDB HTTP client tests
│   ├── test_gcs_storage.py    # GCS storage layer tests
│   └── ...                    # Core module tests
├── terraform/                 # Infrastructure as Code
│   ├── main.tf                # GKE cluster, GCS buckets, IAM
│   ├── variables.tf           # Terraform variables
│   └── outputs.tf             # Infrastructure outputs
├── k8s/                       # Kubernetes manifests
│   ├── chromadb-*.yaml        # ChromaDB service manifests
│   ├── deployment.yaml        # RAG API deployment
│   ├── configmap.yaml         # Application configuration
│   └── ...                    # Other K8s resources
├── data/                      # Data directories (legacy)
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
- **Storage issues**: Check fake-gcs-server logs with `docker-compose logs fake-gcs`

### Local Python Issues  
- **API won't start**: Make sure `ollama` and `fake-gcs` containers are running
- **Upload failures**: Check file types (PDF, TXT, DOCX only) and GCS connectivity
- **Query returns empty**: Upload documents first via `/upload-document`
- **GCS errors**: Ensure `STORAGE_EMULATOR_HOST=http://localhost:4443` is set

### GKE Issues
- **Pod scheduling**: Check zone affinity and node availability
- **Storage errors**: Verify GCS bucket permissions and Workload Identity
- **Authentication**: Ensure Workload Identity binding is correct
- **Vertex AI errors**: Ensure APIs are enabled and billing is active
- **API timeout**: Check load balancer and service configuration


## Contributing

1. Run tests: `pytest`
2. Check coverage: Coverage reports available in CI
3. Follow code style: Automated linting in CI