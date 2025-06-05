# RAG Pipeline

[![CI](https://github.com/tienpdinh/cisc691-a04/workflows/CI/badge.svg)](https://github.com/tienpdinh/cisc691-a04/actions)
[![codecov](https://codecov.io/gh/tienpdinh/cisc691-a04/graph/badge.svg?token=Oot2JmamNl)](https://codecov.io/gh/tienpdinh/cisc691-a04)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A Retrieval-Augmented Generation pipeline that processes documents and answers questions using either local LLMs (Ollama) or cloud AI services (Vertex AI).

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

### 4. Add Documents
```bash
# Put your PDF or TXT files here
cp your-document.pdf data/raw_input/
```

### 5. Run Pipeline
```bash
# Process documents
python main.py step01_ingest --input_filename your-file.txt
python main.py step02_generate_embeddings --input_filename your-file_cleaned.txt
python main.py step03_store_vectors --input_filename your-file_cleaned.txt

# Ask questions
python main.py step05_generate_response --query_args "Your question here" --use_rag
```

## Production Deployment (GKE)

### Prerequisites
- Google Cloud Project with billing enabled
- GKE cluster
- kubectl configured

### Deploy
```bash
# Build and push Docker image
docker build -t gcr.io/cisc691-a04/rag-pipeline:latest .
docker push gcr.io/cisc691-a04/rag-pipeline:latest

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

### GKE Configuration (`config.gke.json`)
- **LLM**: Vertex AI with `gemini-1.5-flash`
- **Storage**: Persistent volumes
- **Scalability**: Kubernetes auto-scaling

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
├── tests/              # Unit tests
├── k8s/                # Kubernetes manifests
├── data/               # Data directories
├── config.local.json   # Local development config
├── config.gke.json     # GKE production config
└── Dockerfile          # Container image
```

## Troubleshooting

### Local Issues
- **Ollama errors**: Make sure `ollama serve` is running
- **File not found**: Check files exist in `data/raw_input/`
- **CUDA errors**: Install CUDA drivers for GPU support

### GKE Issues
- **Authentication**: Verify service account setup
- **Pod failures**: Check `kubectl logs` and resource limits
- **Vertex AI errors**: Ensure APIs are enabled and billing is active

## Contributing

1. Run tests: `pytest`
2. Check coverage: Coverage reports available in CI
3. Follow code style: Automated linting in CI