output "cluster_name" {
  description = "GKE cluster name"
  value       = google_container_cluster.rag_cluster.name
}

output "cluster_endpoint" {
  description = "GKE cluster endpoint"
  value       = google_container_cluster.rag_cluster.endpoint
  sensitive   = true
}

output "cluster_ca_certificate" {
  description = "GKE cluster CA certificate"
  value       = google_container_cluster.rag_cluster.master_auth[0].cluster_ca_certificate
  sensitive   = true
}

output "service_account_email" {
  description = "Service account email for RAG pipeline"
  value       = google_service_account.rag_pipeline.email
}

output "artifact_registry_url" {
  description = "Artifact Registry repository URL"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.rag_pipeline.repository_id}"
}

output "kubectl_config_command" {
  description = "Command to configure kubectl"
  value       = "gcloud container clusters get-credentials ${google_container_cluster.rag_cluster.name} --region ${var.region} --project ${var.project_id}"
}

output "load_balancer_ip" {
  description = "Static IP address for the load balancer"
  value       = google_compute_global_address.rag_api_ip.address
}

output "api_url" {
  description = "URL for the RAG API"
  value       = var.create_dns_zone ? "https://rag-api.${var.domain_name}" : "https://${google_compute_global_address.rag_api_ip.address}"
}

output "dns_zone_name_servers" {
  description = "Name servers for the DNS zone (if created)"
  value       = var.create_dns_zone ? google_dns_managed_zone.rag_api_zone[0].name_servers : []
}

output "storage_buckets" {
  description = "GCS bucket names for RAG data storage"
  value = {
    raw_input    = google_storage_bucket.rag_buckets["rag-raw-input"].name
    cleaned_text = google_storage_bucket.rag_buckets["rag-cleaned-text"].name
    embeddings   = google_storage_bucket.rag_buckets["rag-embeddings"].name
  }
}