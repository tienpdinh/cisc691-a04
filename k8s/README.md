# GKE Deployment Guide

This directory contains Kubernetes manifests to deploy the RAG Pipeline on Google Kubernetes Engine (GKE).

## Prerequisites

1. **GKE Cluster**: Create a GKE cluster with at least 2 nodes (e2-standard-4 recommended)
2. **Docker Image**: Build and push the Docker image to Google Container Registry
3. **kubectl**: Configure kubectl to connect to your GKE cluster

## Repository Structure Recommendations

### Current Setup: Monorepo ✅
This setup works well for:
- Small to medium teams
- Rapid development and prototyping
- Simple deployment scenarios

### When to Consider Alternatives:
- **Separate Infra Repo**: Large teams, complex multi-app infrastructure
- **GitOps Branches**: When using ArgoCD/FluxCD for deployment
- **Helm Charts**: For templating and packaging

### Deployment Options

#### Option 1: Direct Apply (Current)
```bash
kubectl apply -f k8s/
```

#### Option 2: Kustomize (Recommended)
```bash
# Development
kubectl apply -k k8s/overlays/dev/

# Production  
kubectl apply -k k8s/overlays/prod/
```

#### Option 3: GitOps (Advanced)
Use ArgoCD or FluxCD to monitor this repo and auto-deploy changes.

## Quick Start

### 1. Build and Push Docker Image

```bash
# Set your GCP project ID
export PROJECT_ID=your-gcp-project-id

# Build the Docker image
docker build -t gcr.io/$PROJECT_ID/rag-pipeline:latest .

# Push to Google Container Registry
docker push gcr.io/$PROJECT_ID/rag-pipeline:latest
```

### 2. Update Kubernetes Manifests

Replace `PROJECT_ID` in the following files with your actual GCP project ID:
- `rag-pipeline-deployment.yaml`
- `jobs.yaml`

```bash
sed -i "s/PROJECT_ID/$PROJECT_ID/g" k8s/rag-pipeline-deployment.yaml
sed -i "s/PROJECT_ID/$PROJECT_ID/g" k8s/jobs.yaml
```

### 3. Deploy to GKE

```bash
# Deploy all resources
kubectl apply -f k8s/

# Verify deployments
kubectl get all -n rag-pipeline
```

## Architecture

### Components

1. **Namespace**: `rag-pipeline` - Isolated environment for all resources
2. **Ollama**: Local LLM server for text generation
3. **RAG Pipeline**: Main application container
4. **Persistent Volumes**: Storage for data, vectordb, and logs
5. **ConfigMap**: Application configuration
6. **Jobs**: Batch processing for pipeline steps

### Storage

- **rag-pipeline-data** (10Gi): Document storage and processing
- **rag-pipeline-vectordb** (5Gi): ChromaDB vector database
- **rag-pipeline-logs** (1Gi): Application logs
- **ollama-data** (20Gi): Ollama models and cache

## Usage

### Interactive Mode

```bash
# Access the running container
kubectl exec -it deployment/rag-pipeline -n rag-pipeline -- /bin/bash

# Run pipeline steps manually
python main.py step01_ingest --input_filename your-file.txt
python main.py step02_generate_embeddings --input_filename your-file_cleaned.txt
python main.py step03_store_vectors --input_filename your-file_cleaned.txt
python main.py step05_generate_response --query_args "Your question" --use_rag
```

### Batch Processing with Jobs

```bash
# Run document ingestion job
kubectl apply -f k8s/jobs.yaml

# Monitor job progress
kubectl get jobs -n rag-pipeline
kubectl logs job/rag-ingest-job -n rag-pipeline

# Run subsequent jobs in order
kubectl create job --from=job/rag-embeddings-job rag-embeddings-$(date +%s) -n rag-pipeline
kubectl create job --from=job/rag-vectors-job rag-vectors-$(date +%s) -n rag-pipeline
```

### File Upload

```bash
# Copy files to the pod
kubectl cp your-document.pdf rag-pipeline/rag-pipeline-pod:/data/raw_input/

# Or mount a volume from your local machine
kubectl run -it --rm debug --image=alpine --restart=Never -n rag-pipeline -- sh
```

## Monitoring

### Check Pod Status

```bash
kubectl get pods -n rag-pipeline
kubectl describe pod <pod-name> -n rag-pipeline
```

### View Logs

```bash
# Application logs
kubectl logs deployment/rag-pipeline -n rag-pipeline -f

# Ollama logs
kubectl logs deployment/ollama -n rag-pipeline -f

# Job logs
kubectl logs job/rag-ingest-job -n rag-pipeline
```

### Resource Usage

```bash
kubectl top pods -n rag-pipeline
kubectl top nodes
```

## Scaling

### Horizontal Pod Autoscaler

```bash
kubectl autoscale deployment rag-pipeline --cpu-percent=70 --min=1 --max=3 -n rag-pipeline
```

### Manual Scaling

```bash
kubectl scale deployment rag-pipeline --replicas=2 -n rag-pipeline
```

## Troubleshooting

### Common Issues

1. **Ollama not ready**: Wait for Ollama to download models
2. **Storage issues**: Check PVC status and node disk space
3. **Memory issues**: Increase resource limits or use larger nodes

### Debug Commands

```bash
# Check events
kubectl get events -n rag-pipeline --sort-by='.lastTimestamp'

# Check resource usage
kubectl describe node <node-name>

# Check storage
kubectl get pvc -n rag-pipeline
```

## Cleanup

```bash
# Delete all resources
kubectl delete namespace rag-pipeline

# Or delete specific resources
kubectl delete -f k8s/
```

## Security Considerations

1. **Service Accounts**: Create dedicated service accounts with minimal permissions
2. **Network Policies**: Implement network segmentation
3. **Secrets**: Store sensitive data in Kubernetes secrets
4. **Image Security**: Scan images for vulnerabilities before deployment

## Cost Optimization

1. **Node Types**: Use appropriate node types for workloads
2. **Spot Instances**: Use preemptible instances for batch jobs
3. **Storage**: Use appropriate storage classes
4. **Resource Limits**: Set proper resource requests and limits