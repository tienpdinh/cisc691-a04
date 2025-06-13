"""
Google Cloud Storage abstraction layer for RAG API.
Supports both real GCS and fake-gcs-server for local development.
"""

import os
import logging
from typing import BinaryIO, Optional
from pathlib import Path
from google.cloud import storage
from google.cloud.exceptions import NotFound


class GCSStorage:
    """Abstraction layer for Google Cloud Storage operations."""
    
    def __init__(self, project_id: Optional[str] = None):
        """
        Initialize GCS client.
        
        Args:
            project_id: GCP project ID. If None, uses default from environment.
        """
        self.project_id = project_id
        
        # Check if using fake-gcs-server (local development)
        storage_emulator_host = os.getenv('STORAGE_EMULATOR_HOST')
        if storage_emulator_host:
            # For fake-gcs-server, we need to set the client to use HTTP
            self.client = storage.Client(project=project_id or 'fake-project')
            self.client._http.adapters['http://'].poolmanager.connection_pool_kw['server_hostname'] = None
            logging.info(f"Using fake GCS server at {storage_emulator_host}")
        else:
            # Production GCS
            self.client = storage.Client(project=project_id)
            logging.info("Using production Google Cloud Storage")
    
    def create_bucket_if_not_exists(self, bucket_name: str) -> None:
        """Create bucket if it doesn't exist."""
        try:
            bucket = self.client.bucket(bucket_name)
            if not bucket.exists():
                bucket.create()
                logging.info(f"Created bucket: {bucket_name}")
            else:
                logging.debug(f"Bucket already exists: {bucket_name}")
        except Exception as e:
            logging.error(f"Error creating bucket {bucket_name}: {e}")
            raise
    
    def upload_file(self, bucket_name: str, file_path: str, file_obj: BinaryIO) -> str:
        """
        Upload file to GCS bucket.
        
        Args:
            bucket_name: Name of the GCS bucket
            file_path: Path within bucket (e.g., 'documents/file.pdf')
            file_obj: File-like object to upload
            
        Returns:
            GCS object path
        """
        try:
            self.create_bucket_if_not_exists(bucket_name)
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(file_path)
            
            file_obj.seek(0)  # Reset file pointer
            blob.upload_from_file(file_obj)
            
            gcs_path = f"gs://{bucket_name}/{file_path}"
            logging.info(f"Uploaded file to {gcs_path}")
            return gcs_path
            
        except Exception as e:
            logging.error(f"Error uploading file to {bucket_name}/{file_path}: {e}")
            raise
    
    def upload_from_string(self, bucket_name: str, file_path: str, content: str) -> str:
        """
        Upload string content to GCS bucket.
        
        Args:
            bucket_name: Name of the GCS bucket
            file_path: Path within bucket
            content: String content to upload
            
        Returns:
            GCS object path
        """
        try:
            self.create_bucket_if_not_exists(bucket_name)
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(file_path)
            
            blob.upload_from_string(content)
            
            gcs_path = f"gs://{bucket_name}/{file_path}"
            logging.info(f"Uploaded content to {gcs_path}")
            return gcs_path
            
        except Exception as e:
            logging.error(f"Error uploading content to {bucket_name}/{file_path}: {e}")
            raise
    
    def download_to_file(self, bucket_name: str, file_path: str, local_path: str) -> None:
        """Download file from GCS to local path."""
        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(file_path)
            
            # Create parent directories if they don't exist
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            
            blob.download_to_filename(local_path)
            logging.info(f"Downloaded gs://{bucket_name}/{file_path} to {local_path}")
            
        except NotFound:
            logging.error(f"File not found: gs://{bucket_name}/{file_path}")
            raise
        except Exception as e:
            logging.error(f"Error downloading file from {bucket_name}/{file_path}: {e}")
            raise
    
    def download_as_text(self, bucket_name: str, file_path: str) -> str:
        """Download file content as text string."""
        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(file_path)
            
            content = blob.download_as_text()
            logging.debug(f"Downloaded text content from gs://{bucket_name}/{file_path}")
            return content
            
        except NotFound:
            logging.error(f"File not found: gs://{bucket_name}/{file_path}")
            raise
        except Exception as e:
            logging.error(f"Error downloading text from {bucket_name}/{file_path}: {e}")
            raise
    
    def list_files(self, bucket_name: str, prefix: str = "") -> list[str]:
        """List files in bucket with optional prefix filter."""
        try:
            bucket = self.client.bucket(bucket_name)
            blobs = bucket.list_blobs(prefix=prefix)
            
            file_paths = [blob.name for blob in blobs]
            logging.debug(f"Listed {len(file_paths)} files in gs://{bucket_name}/{prefix}")
            return file_paths
            
        except Exception as e:
            logging.error(f"Error listing files in {bucket_name}: {e}")
            raise
    
    def delete_file(self, bucket_name: str, file_path: str) -> None:
        """Delete file from GCS bucket."""
        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(file_path)
            blob.delete()
            
            logging.info(f"Deleted gs://{bucket_name}/{file_path}")
            
        except NotFound:
            logging.warning(f"File not found for deletion: gs://{bucket_name}/{file_path}")
        except Exception as e:
            logging.error(f"Error deleting file {bucket_name}/{file_path}: {e}")
            raise
    
    def file_exists(self, bucket_name: str, file_path: str) -> bool:
        """Check if file exists in GCS bucket."""
        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(file_path)
            return blob.exists()
            
        except Exception as e:
            logging.error(f"Error checking file existence {bucket_name}/{file_path}: {e}")
            return False