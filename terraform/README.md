# Terraform Infrastructure for RAG Pipeline

This Terraform configuration provisions the GCP infrastructure needed for the RAG pipeline project.

## Resources Created

- **GKE Cluster**: Kubernetes cluster for running the RAG pipeline
- **Node Pool**: Auto-scaling node pool with configurable machine types
- **Service Account**: IAM service account with necessary permissions for Vertex AI and ML services
- **API Enablement**: Required GCP APIs (Container, AI Platform, ML, Compute)
- **Artifact Registry**: Container repository for custom images
- **IAM Bindings**: Workload Identity configuration for secure access

## Prerequisites

1. **GCP Project**: Ensure you have a GCP project with billing enabled
2. **Terraform**: Install Terraform >= 1.0
3. **gcloud CLI**: Install and authenticate with `gcloud auth application-default login`
4. **Permissions**: Your account needs the following roles:
   - Project Editor or Owner
   - Kubernetes Engine Admin
   - Service Account Admin

## Usage

### 1. Initialize Terraform

```bash
cd terraform
terraform init
```

### 2. Configure Variables

```bash
# Copy the example variables file
cp terraform.tfvars.example terraform.tfvars

# Edit the variables to match your project
vim terraform.tfvars
```

### 3. Plan and Apply

```bash
# Review the planned changes
terraform plan

# Apply the configuration
terraform apply
```

### 4. Configure kubectl

After applying, configure kubectl to connect to your new cluster:

```bash
# Get the command from terraform output
terraform output kubectl_config_command

# Or run directly:
gcloud container clusters get-credentials rag-pipeline-cluster --region us-central1 --project cisc691-a04
```

### 5. Deploy Kubernetes Resources

Once the infrastructure is ready, deploy the RAG pipeline:

```bash
# Switch to gitops branch and apply k8s manifests
git checkout gitops
kubectl apply -f k8s/
```

## Configuration Options

### Machine Types

- `e2-standard-2`: 2 vCPUs, 8GB RAM (minimal)
- `e2-standard-4`: 4 vCPUs, 16GB RAM (recommended)
- `e2-highmem-4`: 4 vCPUs, 32GB RAM (for memory-intensive workloads)

### Cost Optimization

- Set `use_preemptible_nodes = true` to use preemptible instances (60-91% cheaper)
- Adjust `min_node_count` and `max_node_count` in the node pool autoscaling configuration
- Use smaller machine types if your workload allows

## Cleanup

To destroy all resources:

```bash
terraform destroy
```

**Warning**: This will delete all resources including persistent data. Make sure to backup any important data first.

## Estimated Costs

With default settings (preemptible e2-standard-4 nodes):
- **GKE Cluster**: ~$73/month (management fee)
- **Nodes**: ~$50-150/month (depending on usage)
- **Persistent Disks**: ~$5-20/month
- **Vertex AI**: Pay per request

Total estimated cost: **$130-250/month** for development workloads.

## Troubleshooting

### Common Issues

1. **API not enabled**: Run `gcloud services enable container.googleapis.com`
2. **Insufficient permissions**: Ensure your account has the required IAM roles
3. **Quota exceeded**: Check GCP quotas in the console
4. **Region not available**: Try a different region in `variables.tf`

### Debugging

```bash
# Check terraform state
terraform show

# View detailed logs
export TF_LOG=DEBUG
terraform apply
```