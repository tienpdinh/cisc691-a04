# GCP Setup for Vertex AI

## Prerequisites Setup

### 1. Create Service Account

```bash
# Create service account
gcloud iam service-accounts create rag-pipeline \
    --display-name="RAG Pipeline Service Account" \
    --project=cisc691-a04

# Add required permissions
gcloud projects add-iam-policy-binding cisc691-a04 \
    --member="serviceAccount:rag-pipeline@cisc691-a04.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding cisc691-a04 \
    --member="serviceAccount:rag-pipeline@cisc691-a04.iam.gserviceaccount.com" \
    --role="roles/ml.developer"

# Create and download key
gcloud iam service-accounts keys create ~/rag-pipeline-key.json \
    --iam-account=rag-pipeline@cisc691-a04.iam.gserviceaccount.com
```

### 2. Enable APIs

```bash
# Enable required APIs
gcloud services enable aiplatform.googleapis.com --project=cisc691-a04
gcloud services enable ml.googleapis.com --project=cisc691-a04
gcloud services enable container.googleapis.com --project=cisc691-a04
```

### 3. Create Kubernetes Secret

```bash
# Create secret with service account key
kubectl create secret generic gcp-service-account-key \
    --from-file=key.json=~/rag-pipeline-key.json \
    -n rag-pipeline

# Clean up local key file
rm ~/rag-pipeline-key.json
```

## Usage

### Local Development (Ollama)
```bash
# Use local config
cp config.local.json config.json
python main.py step05_generate_response --query_args "What is AI?" --use_rag
```

### GKE Deployment (Vertex AI)
```bash
# Deploy with Vertex AI configuration
kubectl apply -f k8s/

# Test in cluster
kubectl exec -it deployment/rag-pipeline -n rag-pipeline -- \
    python main.py step05_generate_response --query_args "What is AI?" --use_rag
```

## Model Options

- **gemini-1.5-flash**: Fast, cost-effective
- **gemini-1.5-pro**: Higher quality, more expensive
- **text-bison**: Classic text generation

## Monitoring

```bash
# Check logs
kubectl logs deployment/rag-pipeline -n rag-pipeline -f

# Monitor costs
gcloud billing budgets list --billing-account=YOUR_BILLING_ACCOUNT
```