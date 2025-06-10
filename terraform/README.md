# Terraform Infrastructure for RAG API

This Terraform configuration provisions the GCP infrastructure needed for the RAG API project.

## Resources Created

- **GKE Cluster**: Kubernetes cluster for running the FastAPI-based RAG service
- **Node Pool**: Auto-scaling node pool with configurable machine types
- **Service Account**: IAM service account with necessary permissions for Vertex AI and ML services
- **Load Balancer**: Global static IP address for the API
- **DNS (Optional)**: Cloud DNS zone and records for custom domain
- **API Enablement**: Required GCP APIs (Container, AI Platform, ML, Compute)
- **Artifact Registry**: Container repository for RAG API images
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
gcloud container clusters get-credentials rag-api-cluster --region us-central1 --project cisc691-a04
```

### 5. Deploy API

Once the infrastructure is ready, deploy the RAG API:

```bash
# Build and push container image
docker build -t gcr.io/cisc691-a04/rag-api:latest .
docker push gcr.io/cisc691-a04/rag-api:latest

# Deploy to Kubernetes
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n rag-api
kubectl get ingress -n rag-api
```

### 6. Set Up DNS (Optional)

If using a custom domain, get the load balancer IP and create DNS records:

```bash
# Get the static IP
terraform output load_balancer_ip

# Create DNS A record:
# rag-api.tienpdinh.com → [LOAD_BALANCER_IP]
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

### DNS Configuration

- Set `create_dns_zone = true` to manage DNS in GCP Cloud DNS
- Set `domain_name = "yourdomain.com"` to use your custom domain
- Default configuration uses `tienpdinh.com`

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
- Static IP address and DNS records
- All API data and uploaded documents

**Before destroying, make sure to:**
1. Backup any important data from persistent volumes
2. Export any container images you want to keep
3. Save any configuration or secrets you need

## Estimated Costs

With default settings (preemptible e2-standard-4 nodes):
- **GKE Cluster**: ~$73/month (management fee)
- **Nodes**: ~$50-150/month (depending on usage)
- **Persistent Disks**: ~$5-20/month
- **Load Balancer**: ~$18/month (global)
- **Static IP**: ~$3/month (if not in use)
- **Vertex AI**: Pay per API request
- **Cloud DNS**: ~$0.50/month (if enabled)

**Total estimated cost**: $150-270/month for development workloads.

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
gcloud container clusters delete rag-api-cluster --region us-central1

# Delete service account
gcloud iam service-accounts delete rag-api@PROJECT_ID.iam.gserviceaccount.com

# Delete Artifact Registry repository
gcloud artifacts repositories delete rag-api --location us-central1

# Delete static IP
gcloud compute addresses delete rag-api-ip --global

# Delete DNS zone (if created)
gcloud dns managed-zones delete rag-api-zone
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