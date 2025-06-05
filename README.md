# RAG Pipeline

[![CI](https://github.com/USERNAME/REPO-NAME/workflows/CI/badge.svg)](https://github.com/USERNAME/REPO-NAME/actions)
[![codecov](https://codecov.io/gh/tienpdinh/cisc691-a04/graph/badge.svg?token=Oot2JmamNl)](https://codecov.io/gh/tienpdinh/cisc691-a04)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A simple Retrieval-Augmented Generation pipeline that processes documents and answers questions using local LLMs.

## Quick Setup

### 1. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Install Ollama
- Download from [ollama.ai](https://ollama.ai)
- Run: `ollama serve`
- Install model: `ollama pull llama2`

### 3. Add Documents
```bash
# Put your PDF or TXT files here
cp your-document.pdf data/raw_input/
```

## Usage

### Process Documents
```bash
# Replace "your-file.txt" with your actual filename
python main.py step01_ingest --input_filename your-file.txt
python main.py step02_generate_embeddings --input_filename your-file_cleaned.txt
python main.py step03_store_vectors --input_filename your-file_cleaned.txt
```

### Ask Questions
```bash
python main.py step05_generate_response --query_args "Your question here" --use_rag
```

## Quick Test
```bash
# Create test document
echo "AI is artificial intelligence." > data/raw_input/test.txt

# Run pipeline
python main.py step01_ingest --input_filename test.txt
python main.py step02_generate_embeddings --input_filename test_cleaned.txt
python main.py step03_store_vectors --input_filename test_cleaned.txt

# Ask question
python main.py step05_generate_response --query_args "What is AI?" --use_rag
```

## Configuration

Edit `config.json` to change models:
- For RTX 4090: `"llm_model_name": "llama3.1:70b-instruct-q4_0"`
- For other GPUs: `"llm_model_name": "llama3.1:8b"`

## Testing & Coverage

### Run Tests
```bash
pytest
```

### Run Tests with Coverage
```bash
pytest --cov=classes --cov=. --cov-report=html --cov-report=term-missing
```

### View Coverage Report
```bash
# Open htmlcov/index.html in your browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

## Troubleshooting

- **Ollama errors**: Make sure `ollama serve` is running
- **File not found**: Check files exist in `data/raw_input/`
- **CUDA errors**: Install CUDA drivers for GPU support