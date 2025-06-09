# Kubernetes Deployment for RAG API

This directory contains Kubernetes manifests for deploying the RAG API to Google Kubernetes Engine (GKE).

## Architecture

The deployment consists of:
- **FastAPI Application**: Containerized REST API server
- **Load Balancer**: Google Cloud Load Balancer with SSL termination
- **Persistent Storage**: For vector database and uploaded documents
- **Auto-scaling**: Horizontal Pod Autoscaler based on CPU/memory

## Prerequisites

1. **GKE Cluster**: Created via Terraform in `../terraform/`
2. **Container Image**: Built and pushed to Artifact Registry
3. **Domain Name**: (Optional) For custom domain with SSL

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
gcloud container clusters get-credentials rag-pipeline-cluster --region us-central1 --project cisc691-a04
```

### 3. Build and Push Container Image
```bash
# Build image
docker build -t gcr.io/cisc691-a04/rag-api:latest .

# Push to Artifact Registry
docker push gcr.io/cisc691-a04/rag-api:latest
```

### 4. Update Configuration
Edit the following files with your project details:

**configmap.yaml**:
```yaml
"project_id": "cisc691-a04"
```

**service-account.yaml**:
```yaml
rag-pipeline@cisc691-a04.iam.gserviceaccount.com
```

**deployment.yaml**:
```yaml
image: gcr.io/cisc691-a04/rag-api:latest
```

**ingress.yaml**:
```yaml
host: rag-api.your-domain.com
```

### 5. Deploy to Kubernetes
```bash
# Apply all manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n rag-pipeline
kubectl get services -n rag-pipeline
kubectl get ingress -n rag-pipeline
```

## API Endpoints

Once deployed, the API will be available at:
- **With custom domain**: `https://rag-api.your-domain.com`
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
kubectl get pods -n rag-pipeline
kubectl describe pod POD_NAME -n rag-pipeline
kubectl logs POD_NAME -n rag-pipeline
```

### Check Services
```bash
kubectl get services -n rag-pipeline
kubectl describe service rag-api-service -n rag-pipeline
```

### Check Ingress
```bash
kubectl get ingress -n rag-pipeline
kubectl describe ingress rag-api-ingress -n rag-pipeline
```

### Port Forward for Local Testing
```bash
kubectl port-forward service/rag-api-service 8000:80 -n rag-pipeline
# API available at http://localhost:8000
```

## Scaling

### Manual Scaling
```bash
kubectl scale deployment rag-api --replicas=3 -n rag-pipeline
```

### Auto-scaling (Optional)
```bash
kubectl autoscale deployment rag-api --cpu-percent=70 --min=2 --max=10 -n rag-pipeline
```

## SSL Certificate

The deployment uses Google-managed SSL certificates. It may take 15-60 minutes for the certificate to be provisioned after applying the ingress.

Check certificate status:
```bash
kubectl describe managedcertificate rag-api-ssl-cert -n rag-pipeline
```

## Storage

- **Persistent Volume**: 10GB storage for vector database and documents
- **Storage Class**: `standard-rwo` (Regional persistent disk)
- **Mount Path**: `/app/data` inside containers

## Configuration

All configuration is managed via ConfigMap. To update:

1. Edit `configmap.yaml`
2. Apply changes: `kubectl apply -f k8s/configmap.yaml`
3. Restart pods: `kubectl rollout restart deployment/rag-api -n rag-pipeline`

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
- Check logs: `kubectl logs deployment/rag-api -n rag-pipeline`
- Verify Vertex AI permissions
- Check config.json values

### Cleanup

```bash
# Delete Kubernetes resources
kubectl delete namespace rag-pipeline

# Delete Terraform infrastructure
cd ../terraform
terraform destroy
```

## Security

- **Service Account**: Uses Workload Identity for secure GCP access
- **Network Policies**: Restrict pod-to-pod communication
- **SSL/TLS**: HTTPS enforced with Google-managed certificates
- **RBAC**: Minimal permissions for service accounts