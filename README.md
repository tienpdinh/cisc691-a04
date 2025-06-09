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

### 1. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Install Ollama
- Download from [ollama.ai](https://ollama.ai)
- Run: `ollama serve`
- Install model: `ollama pull llama3.1:8b`

### 3. Configure for Local Use
```bash
cp config.local.json config.json
```

### 4. Start the API Server
```bash
python main.py
```

The API will be available at `http://localhost:8000` with interactive docs at `http://localhost:8000/docs`

### 5. Use the API

#### Upload and Process Documents
```bash
# Upload a PDF or TXT file (automatically processes through ingestion, embedding, and storage)
curl -X POST "http://localhost:8000/upload-document" \
  -F "file=@your-document.pdf"
```

#### Query Documents
```bash
# Ask questions about your documents
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this document about?", "use_rag": true}'
```

#### Retrieve Relevant Chunks
```bash
# Get relevant document chunks for a query
curl -X POST "http://localhost:8000/retrieve" \
  -H "Content-Type: application/json" \
  -d '{"query": "artificial intelligence", "top_k": 3}'
```

#### Health Check
```bash
# Check API status
curl "http://localhost:8000/health"
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

## Configuration

### Local Configuration (`config.local.json`)
- **LLM**: Ollama with `llama3.1:8b`
- **Storage**: Local directories
- **Privacy**: Complete data privacy
- **API**: FastAPI server on port 8000

### GKE Configuration (`config.gke.json`)
- **LLM**: Vertex AI with `gemini-1.5-flash`
- **Storage**: Persistent volumes
- **Scalability**: Kubernetes auto-scaling
- **API**: Load-balanced FastAPI service

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
├── src/                 # Source code
│   ├── api_app.py      # FastAPI application setup
│   ├── api_routes.py   # API route definitions
│   ├── api_endpoints.py # Endpoint business logic
│   ├── api_models.py   # Pydantic request/response models
│   └── ...             # Core RAG modules
├── tests/              # Unit tests
│   ├── test_api_*.py   # API tests
│   └── ...             # Core module tests
├── k8s/                # Kubernetes manifests
├── data/               # Data directories
├── config.local.json   # Local development config
├── config.gke.json     # GKE production config
├── main.py             # API server entry point
└── Dockerfile          # Container image
```

## Troubleshooting

### Local Issues
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