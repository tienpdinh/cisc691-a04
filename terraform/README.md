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

## Quick Start

### 1. Initialize Terraform

```bash
terraform init
```

### 2. Configure Variables

```bash
# Copy the example variables file
cp terraform.tfvars.example terraform.tfvars

# Edit the variables to match your project
vim terraform.tfvars
```

### 3. Apply Infrastructure

```bash
# Review the planned changes
terraform plan

# Apply the configuration
terraform apply
```

When prompted, type `yes` to confirm the infrastructure creation.

### 4. Configure kubectl

After applying, configure kubectl to connect to your new cluster:

```bash
# Get the command from terraform output
terraform output kubectl_config_command

# Or run directly:
gcloud container clusters get-credentials rag-pipeline-cluster --region us-central1 --project cisc691-a04
```

### 5. Deploy Application

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
- Adjust node pool autoscaling in `main.tf`
- Use smaller machine types if your workload allows

## Spinning Down Resources

To destroy all infrastructure and stop billing:

```bash
# Preview what will be destroyed
terraform plan -destroy

# Destroy all resources
terraform destroy
```

When prompted, type `yes` to confirm the destruction.

**⚠️ WARNING**: This will permanently delete:
- The entire GKE cluster and all workloads
- All persistent volumes and data
- Service accounts and IAM bindings
- Container images in Artifact Registry

**Before destroying, make sure to:**
1. Backup any important data from persistent volumes
2. Export any container images you want to keep
3. Save any configuration or secrets you need

## Estimated Costs

With default settings (preemptible e2-standard-4 nodes):
- **GKE Cluster**: ~$73/month (management fee)
- **Nodes**: ~$50-150/month (depending on usage)
- **Persistent Disks**: ~$5-20/month
- **Vertex AI**: Pay per request

**Total estimated cost**: $130-250/month for development workloads.

## Troubleshooting

### Common Issues

1. **API not enabled**: 
   ```bash
   gcloud services enable container.googleapis.com aiplatform.googleapis.com
   ```

2. **Insufficient permissions**: Ensure your account has the required IAM roles

3. **Quota exceeded**: Check GCP quotas in the console under IAM & Admin > Quotas

4. **Region not available**: Try a different region in `variables.tf`

5. **Terraform state conflicts**: If working in a team, consider using remote state:
   ```bash
   # Configure remote backend in main.tf
   terraform {
     backend "gcs" {
       bucket = "your-terraform-state-bucket"
       prefix = "terraform/state"
     }
   }
   ```

### Debugging

```bash
# Check terraform state
terraform show

# View detailed logs
export TF_LOG=DEBUG
terraform apply

# Validate configuration
terraform validate

# Format code
terraform fmt
```

### Emergency Cleanup

If `terraform destroy` fails, you can manually clean up resources:

```bash
# Delete GKE cluster
gcloud container clusters delete rag-pipeline-cluster --region us-central1

# Delete service account
gcloud iam service-accounts delete rag-pipeline@PROJECT_ID.iam.gserviceaccount.com

# Delete Artifact Registry repository
gcloud artifacts repositories delete rag-pipeline --location us-central1
```

## Security Notes

- Never commit `terraform.tfvars` to version control
- Use least-privilege IAM permissions
- Enable audit logging in production
- Consider using private GKE clusters for production workloads
- Regularly rotate service account keys

## Support

For issues:
1. Check the troubleshooting section above
2. Review Terraform and GCP documentation
3. Check GCP status page for service outages
4. Contact your GCP support team for billing or quota issues