terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Enable required APIs
resource "google_project_service" "apis" {
  for_each = toset([
    "container.googleapis.com",
    "aiplatform.googleapis.com",
    "ml.googleapis.com",
    "compute.googleapis.com"
  ])
  
  project = var.project_id
  service = each.value
  
  disable_on_destroy = false
}

# GKE Cluster
resource "google_container_cluster" "rag_cluster" {
  name     = "rag-pipeline-cluster"
  location = var.region
  
  # We can't create a cluster with no node pool defined, but we want to only use
  # separately managed node pools. So we create the smallest possible default
  # node pool and immediately delete it.
  remove_default_node_pool = true
  initial_node_count       = 1
  
  # Allow deletion for easier cleanup
  deletion_protection = false
  
  # Workload Identity
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }
  
  depends_on = [google_project_service.apis]
}

# Node Pool
resource "google_container_node_pool" "rag_nodes" {
  name       = "rag-node-pool"
  location   = var.region
  cluster    = google_container_cluster.rag_cluster.name
  
  initial_node_count = 1
  
  autoscaling {
    min_node_count = 1
    max_node_count = 3
  }
  
  node_config {
    preemptible  = var.use_preemptible_nodes
    machine_type = var.node_machine_type
    
    service_account = google_service_account.rag_pipeline.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    
    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }
}

# Service Account for RAG Pipeline
resource "google_service_account" "rag_pipeline" {
  account_id   = "rag-pipeline"
  display_name = "RAG Pipeline Service Account"
  description  = "Service account for RAG pipeline workloads"
}

# IAM bindings for the service account
resource "google_project_iam_member" "rag_pipeline_roles" {
  for_each = toset([
    "roles/aiplatform.user",
    "roles/ml.developer",
    "roles/storage.objectViewer"
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.rag_pipeline.email}"
}

# Kubernetes Service Account binding
resource "google_service_account_iam_binding" "workload_identity" {
  service_account_id = google_service_account.rag_pipeline.name
  role               = "roles/iam.workloadIdentityUser"
  
  members = [
    "serviceAccount:${var.project_id}.svc.id.goog[rag-pipeline/rag-pipeline-sa]"
  ]
  
  depends_on = [
    google_container_cluster.rag_cluster,
    google_container_node_pool.rag_nodes
  ]
}

# Container Registry (optional - for storing custom images)
resource "google_artifact_registry_repository" "rag_pipeline" {
  location      = var.region
  repository_id = "rag-pipeline"
  description   = "Docker repository for RAG pipeline images"
  format        = "DOCKER"
  
  depends_on = [google_project_service.apis]
}

# Static IP for the Load Balancer
resource "google_compute_global_address" "rag_api_ip" {
  name        = "rag-api-ip"
  description = "Static IP for RAG API load balancer"
}

# DNS Zone (optional - if you want to manage DNS in GCP)
resource "google_dns_managed_zone" "rag_api_zone" {
  count       = var.create_dns_zone ? 1 : 0
  name        = "rag-api-zone"
  dns_name    = "${var.domain_name}."
  description = "DNS zone for RAG API"
}

# DNS Record for the API
resource "google_dns_record_set" "rag_api_record" {
  count        = var.create_dns_zone ? 1 : 0
  name         = "rag-api.${var.domain_name}."
  type         = "A"
  ttl          = 300
  managed_zone = google_dns_managed_zone.rag_api_zone[0].name
  rrdatas      = [google_compute_global_address.rag_api_ip.address]
}