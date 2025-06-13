import pytest
import tempfile
import io
from unittest.mock import MagicMock, patch, Mock
from pathlib import Path

from src.gcs_storage import GCSStorage


class TestGCSStorage:
    """Test cases for GCS storage abstraction layer."""

    @patch('src.gcs_storage.storage.Client')
    def test_init_production(self, mock_client_class):
        """Test GCS client initialization for production."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        storage = GCSStorage(project_id="test-project")
        
        assert storage.project_id == "test-project"
        assert storage.client == mock_client
        mock_client_class.assert_called_once_with(project="test-project")

    @patch.dict('os.environ', {'STORAGE_EMULATOR_HOST': 'http://fake-gcs:4443'})
    @patch('src.gcs_storage.storage.Client')
    def test_init_fake_gcs(self, mock_client_class):
        """Test GCS client initialization for fake-gcs-server."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        storage = GCSStorage(project_id="test-project")
        
        assert storage.project_id == "test-project"
        mock_client_class.assert_called_once_with(project="test-project")

    @patch('src.gcs_storage.storage.Client')
    def test_create_bucket_if_not_exists_new_bucket(self, mock_client_class):
        """Test creating a new bucket."""
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_bucket.exists.return_value = False
        mock_client.bucket.return_value = mock_bucket
        mock_client_class.return_value = mock_client
        
        storage = GCSStorage()
        storage.create_bucket_if_not_exists("test-bucket")
        
        mock_client.bucket.assert_called_once_with("test-bucket")
        mock_bucket.exists.assert_called_once()
        mock_bucket.create.assert_called_once()

    @patch('src.gcs_storage.storage.Client')
    def test_create_bucket_if_not_exists_existing_bucket(self, mock_client_class):
        """Test handling existing bucket."""
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_bucket.exists.return_value = True
        mock_client.bucket.return_value = mock_bucket
        mock_client_class.return_value = mock_client
        
        storage = GCSStorage()
        storage.create_bucket_if_not_exists("test-bucket")
        
        mock_client.bucket.assert_called_once_with("test-bucket")
        mock_bucket.exists.assert_called_once()
        mock_bucket.create.assert_not_called()

    @patch('src.gcs_storage.storage.Client')
    def test_upload_file_success(self, mock_client_class):
        """Test successful file upload."""
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_bucket.exists.return_value = True
        mock_bucket.blob.return_value = mock_blob
        mock_client.bucket.return_value = mock_bucket
        mock_client_class.return_value = mock_client
        
        storage = GCSStorage()
        
        file_obj = io.BytesIO(b"test content")
        result = storage.upload_file("test-bucket", "test-file.txt", file_obj)
        
        assert result == "gs://test-bucket/test-file.txt"
        mock_bucket.blob.assert_called_once_with("test-file.txt")
        mock_blob.upload_from_file.assert_called_once_with(file_obj)

    @patch('src.gcs_storage.storage.Client')
    def test_upload_from_string_success(self, mock_client_class):
        """Test successful string upload."""
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_bucket.exists.return_value = True
        mock_bucket.blob.return_value = mock_blob
        mock_client.bucket.return_value = mock_bucket
        mock_client_class.return_value = mock_client
        
        storage = GCSStorage()
        
        result = storage.upload_from_string("test-bucket", "test-file.txt", "test content")
        
        assert result == "gs://test-bucket/test-file.txt"
        mock_bucket.blob.assert_called_once_with("test-file.txt")
        mock_blob.upload_from_string.assert_called_once_with("test content")

    @patch('src.gcs_storage.storage.Client')
    def test_download_to_file_success(self, mock_client_class):
        """Test successful file download."""
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client.bucket.return_value = mock_bucket
        mock_client_class.return_value = mock_client
        
        storage = GCSStorage()
        
        with tempfile.NamedTemporaryFile() as temp_file:
            storage.download_to_file("test-bucket", "test-file.txt", temp_file.name)
            
            mock_bucket.blob.assert_called_once_with("test-file.txt")
            mock_blob.download_to_filename.assert_called_once_with(temp_file.name)

    @patch('src.gcs_storage.storage.Client')
    def test_download_as_text_success(self, mock_client_class):
        """Test successful text download."""
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.download_as_text.return_value = "test content"
        mock_bucket.blob.return_value = mock_blob
        mock_client.bucket.return_value = mock_bucket
        mock_client_class.return_value = mock_client
        
        storage = GCSStorage()
        
        result = storage.download_as_text("test-bucket", "test-file.txt")
        
        assert result == "test content"
        mock_bucket.blob.assert_called_once_with("test-file.txt")
        mock_blob.download_as_text.assert_called_once()

    @patch('src.gcs_storage.storage.Client')
    def test_list_files_success(self, mock_client_class):
        """Test successful file listing."""
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob1 = MagicMock()
        mock_blob1.name = "file1.txt"
        mock_blob2 = MagicMock()
        mock_blob2.name = "file2.txt"
        mock_bucket.list_blobs.return_value = [mock_blob1, mock_blob2]
        mock_client.bucket.return_value = mock_bucket
        mock_client_class.return_value = mock_client
        
        storage = GCSStorage()
        
        result = storage.list_files("test-bucket", "prefix/")
        
        assert result == ["file1.txt", "file2.txt"]
        mock_bucket.list_blobs.assert_called_once_with(prefix="prefix/")

    @patch('src.gcs_storage.storage.Client')
    def test_delete_file_success(self, mock_client_class):
        """Test successful file deletion."""
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client.bucket.return_value = mock_bucket
        mock_client_class.return_value = mock_client
        
        storage = GCSStorage()
        
        storage.delete_file("test-bucket", "test-file.txt")
        
        mock_bucket.blob.assert_called_once_with("test-file.txt")
        mock_blob.delete.assert_called_once()

    @patch('src.gcs_storage.storage.Client')
    def test_file_exists_true(self, mock_client_class):
        """Test file existence check when file exists."""
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_bucket.blob.return_value = mock_blob
        mock_client.bucket.return_value = mock_bucket
        mock_client_class.return_value = mock_client
        
        storage = GCSStorage()
        
        result = storage.file_exists("test-bucket", "test-file.txt")
        
        assert result is True
        mock_bucket.blob.assert_called_once_with("test-file.txt")
        mock_blob.exists.assert_called_once()

    @patch('src.gcs_storage.storage.Client')
    def test_file_exists_false(self, mock_client_class):
        """Test file existence check when file doesn't exist."""
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.exists.return_value = False
        mock_bucket.blob.return_value = mock_blob
        mock_client.bucket.return_value = mock_bucket
        mock_client_class.return_value = mock_client
        
        storage = GCSStorage()
        
        result = storage.file_exists("test-bucket", "test-file.txt")
        
        assert result is False
        mock_bucket.blob.assert_called_once_with("test-file.txt")
        mock_blob.exists.assert_called_once()