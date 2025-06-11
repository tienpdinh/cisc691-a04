# Kubernetes Deployment for RAG API

This directory contains Kubernetes manifests for deploying the RAG API to Google Kubernetes Engine (GKE).

## Architecture

The deployment consists of:
- **FastAPI Application**: Containerized REST API server
- **ChromaDB**: Vector database service with persistent storage
- **Load Balancer**: Google Cloud Load Balancer with SSL termination
- **GCS Storage**: Google Cloud Storage buckets for document storage
- **Workload Identity**: Secure authentication without service account keys
- **Auto-scaling**: Horizontal Pod Autoscaler based on CPU/memory

## Prerequisites

1. **GKE Cluster**: Created via Terraform in `../terraform/`
2. **GCS Buckets**: Created via Terraform with proper IAM permissions
3. **Container Image**: Built and pushed to Artifact Registry
4. **Domain Name**: (Optional) For custom domain with SSL

## Quick Deployment

### 1. Apply Terraform Infrastructure
```bash
cd ../terraform
terraform init
terraform plan
terraform apply
```

### 2. Configure kubectl
```bash
# Get cluster credentials (use output from terraform)
gcloud container clusters get-credentials rag-api-cluster --region us-central1 --project cisc691-a04
```

### 3. Build and Push Container Image
```bash
# Build image
docker build -t gcr.io/cisc691-a04/rag-api:latest .

# Push to Artifact Registry
docker push gcr.io/cisc691-a04/rag-api:latest
```

### 4. Verify Configuration
Check that the following files have correct project details:

**configmap.yaml** - Application configuration:
```yaml
"project_id": "cisc691-a04"
"raw_input_bucket": "cisc691-a04-rag-raw-input"
"cleaned_text_bucket": "cisc691-a04-rag-cleaned-text"
"embeddings_bucket": "cisc691-a04-rag-embeddings"
```

**service-account.yaml** - Workload Identity binding:
```yaml
iam.gke.io/gcp-service-account: rag-api@cisc691-a04.iam.gserviceaccount.com
```

**deployment.yaml** - Container image:
```yaml
image: gcr.io/cisc691-a04/rag-api:latest
```

**ingress.yaml** - Domain configuration:
```yaml
host: rag-api.tienpdinh.com
```

### 5. Deploy to Kubernetes
```bash
# Create namespace
kubectl create namespace rag-api

# Apply all manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n rag-api
kubectl get services -n rag-api
kubectl get ingress -n rag-api

# Verify GCS access (optional)
kubectl logs deployment/rag-api -n rag-api | grep -i "gcs\|bucket"
```

## API Endpoints

Once deployed, the API will be available at:
- **Production URL**: `https://rag-api.tienpdinh.com`
- **With IP**: `https://LOAD_BALANCER_IP`

Available endpoints:
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation
- `POST /upload-document` - Upload and process documents
- `POST /query` - Query documents with RAG
- `POST /retrieve` - Retrieve relevant document chunks

## Monitoring and Debugging

### Check Pod Status
```bash
kubectl get pods -n rag-api
kubectl describe pod POD_NAME -n rag-api
kubectl logs POD_NAME -n rag-api
```

### Check Services
```bash
kubectl get services -n rag-api
kubectl describe service rag-api-service -n rag-api
```

### Check Ingress
```bash
kubectl get ingress -n rag-api
kubectl describe ingress rag-api-ingress -n rag-api
```

### Port Forward for Local Testing
```bash
kubectl port-forward service/rag-api-service 8000:80 -n rag-api
# API available at http://localhost:8000
```

## Scaling

### Manual Scaling
```bash
kubectl scale deployment rag-api --replicas=3 -n rag-api
```

### Auto-scaling (Optional)
```bash
kubectl autoscale deployment rag-api --cpu-percent=70 --min=2 --max=10 -n rag-api
```

## SSL Certificate

The deployment uses Google-managed SSL certificates. It may take 15-60 minutes for the certificate to be provisioned after applying the ingress.

Check certificate status:
```bash
kubectl describe managedcertificate rag-api-ssl-cert -n rag-api
```

## Storage

### ChromaDB Storage
- **Persistent Volume**: 5GB storage for vector database
- **Storage Class**: `standard-rwo` (Regional persistent disk)
- **Mount Path**: `/chroma/chroma` inside ChromaDB container

### Document Storage
- **GCS Buckets**: Separate buckets for different data types
  - `cisc691-a04-rag-raw-input`: Original uploaded documents
  - `cisc691-a04-rag-cleaned-text`: Processed text files
  - `cisc691-a04-rag-embeddings`: Vector embeddings
- **Authentication**: Via Workload Identity (no service account keys)

## Configuration

All configuration is managed via ConfigMap. To update:

1. Edit `configmap.yaml`
2. Apply changes: `kubectl apply -f k8s/configmap.yaml`
3. Restart pods: `kubectl rollout restart deployment/rag-api -n rag-api`

## Troubleshooting

### Common Issues

**Pods not starting**:
- Check image exists: `gcloud container images list --repository=gcr.io/cisc691-a04`
- Check service account permissions
- Check resource limits

**Ingress not working**:
- Verify static IP exists: `gcloud compute addresses list`
- Check DNS configuration
- Wait for SSL certificate provisioning

**API errors**:
- Check logs: `kubectl logs deployment/rag-api -n rag-api`
- Verify Vertex AI permissions
- Check GCS bucket permissions and Workload Identity
- Check config.json values

**Storage errors**:
- Verify GCS buckets exist: `gsutil ls gs://cisc691-a04-rag-*`
- Check Workload Identity binding: `kubectl describe serviceaccount rag-api-sa -n rag-api`
- Test GCS access from pod: `kubectl exec -it POD_NAME -n rag-api -- gsutil ls`

### Cleanup

```bash
# Delete Kubernetes resources
kubectl delete namespace rag-api

# Delete Terraform infrastructure
cd ../terraform
terraform destroy
```

## Security

- **Workload Identity**: Secure GCP authentication without service account keys
- **IAM Roles**: Minimal permissions (Vertex AI, GCS, Artifact Registry)
- **SSL/TLS**: HTTPS enforced with Google-managed certificates
- **RBAC**: Minimal Kubernetes permissions for service accounts
- **Network Policies**: Restrict pod-to-pod communication
- **Bucket Security**: Uniform bucket-level access with versioning

## Manifest Files

| File | Purpose |
|------|---------|
| `namespace.yaml` | Creates rag-api namespace |
| `service-account.yaml` | Kubernetes SA with Workload Identity |
| `configmap.yaml` | Application configuration with GCS buckets |
| `deployment.yaml` | RAG API pods (no PVC mounts) |
| `service.yaml` | Internal service for load balancing |
| `chromadb-deployment.yaml` | ChromaDB with persistent storage |
| `chromadb-service.yaml` | ChromaDB internal service |
| `chromadb-pvc.yaml` | Persistent volume for ChromaDB |
| `ingress.yaml` | External load balancer with SSL |