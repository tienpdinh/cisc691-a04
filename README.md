# GKE Deployment Guide

Deploy the RAG Pipeline on Google Kubernetes Engine (GKE).

## Prerequisites

1. **GKE Cluster**: Create a GKE cluster with at least 2 nodes (e2-standard-4 recommended)
2. **kubectl**: Configure kubectl to connect to your GKE cluster
3. **Docker**: For building and pushing images

## Quick Start

### 1. Build and Push Docker Image

```bash
# Build the Docker image
docker build -t gcr.io/cisc691-a04/rag-pipeline:latest .

# Push to Google Container Registry
docker push gcr.io/cisc691-a04/rag-pipeline:latest
```

### 2. Deploy to GKE

Choose your deployment method:

#### Option A: Development Environment
```bash
kubectl apply -k overlays/dev/
```

#### Option B: Production Environment
```bash
kubectl apply -k overlays/prod/
```

#### Option C: Direct Apply (All Environments)
```bash
kubectl apply -f k8s/
```

### 3. Verify Deployment

```bash
# Check all resources
kubectl get all -n rag-pipeline

# Check pod status
kubectl get pods -n rag-pipeline

# View logs
kubectl logs deployment/rag-pipeline -n rag-pipeline -f
```

## Usage

### Interactive Mode

```bash
# Access the running container
kubectl exec -it deployment/rag-pipeline -n rag-pipeline -- /bin/bash

# Run pipeline steps
python main.py step01_ingest --input_filename your-file.txt
python main.py step02_generate_embeddings --input_filename your-file_cleaned.txt
python main.py step03_store_vectors --input_filename your-file_cleaned.txt
python main.py step05_generate_response --query_args "Your question" --use_rag
```

### File Upload

```bash
# Copy files to the pod
kubectl cp your-document.pdf rag-pipeline/deployment/rag-pipeline:/data/raw_input/
```

### Batch Processing

```bash
# Run processing jobs
kubectl apply -f k8s/jobs.yaml

# Monitor job progress
kubectl get jobs -n rag-pipeline
kubectl logs job/rag-ingest-job -n rag-pipeline
```

## Cleanup

```bash
# Delete all resources
kubectl delete namespace rag-pipeline
```