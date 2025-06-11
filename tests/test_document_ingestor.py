import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from src.document_ingestor import DocumentIngestor


class TestDocumentIngestor:
    
    @patch('src.document_ingestor.GCSStorage')
    @patch('src.document_ingestor.AutoTokenizer')
    def test_init(self, mock_tokenizer_class, mock_gcs_storage_class):
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_gcs_storage_class.return_value = Mock()
        mock_gcs_storage_class.return_value = Mock()
        
        ingestor = DocumentIngestor(
            file_list=["test.txt"],
            input_bucket="test-input-bucket",
            output_bucket="test-output-bucket",
            project_id="test-project",
            embedding_model_name="test-model"
        )
        
        assert ingestor.file_list == ["test.txt"]
        assert ingestor.input_bucket == "test-input-bucket"
        assert ingestor.output_bucket == "test-output-bucket"
        mock_tokenizer_class.from_pretrained.assert_called_once_with("test-model")
        mock_gcs_storage_class.assert_called_once_with(project_id="test-project")
    
    @patch('src.document_ingestor.GCSStorage')
    @patch('src.document_ingestor.AutoTokenizer')
    def test_extract_text_from_txt_success(self, mock_tokenizer_class, mock_gcs_storage_class):
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_gcs_storage_class.return_value = Mock()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            input_dir.mkdir()
            
            test_file = input_dir / "test.txt"
            test_content = "This is test content"
            test_file.write_text(test_content, encoding="utf-8")
            
            ingestor = DocumentIngestor(
                file_list=["test.txt"],
                input_bucket="test-input-bucket",
                output_bucket="test-output-bucket",
                project_id="test-project",
                embedding_model_name="test-model"
            )
            
            result = ingestor._extract_text_from_txt(test_file)
            assert result == test_content
    
    @patch('src.document_ingestor.GCSStorage')
    @patch('src.document_ingestor.AutoTokenizer')
    def test_extract_text_from_txt_file_not_found(self, mock_tokenizer_class, mock_gcs_storage_class):
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_gcs_storage_class.return_value = Mock()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            
            ingestor = DocumentIngestor(
                file_list=[],
                input_bucket="test-input-bucket",
                output_bucket="test-output-bucket",
                project_id="test-project",
                embedding_model_name="test-model"
            )
            
            result = ingestor._extract_text_from_txt("nonexistent.txt")
            assert result is None
    
    @patch('src.document_ingestor.pdfplumber')
    @patch('src.document_ingestor.GCSStorage')
    @patch('src.document_ingestor.AutoTokenizer')
    def test_extract_text_from_pdf_success(self, mock_tokenizer_class, mock_gcs_storage_class, mock_pdfplumber):
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_gcs_storage_class.return_value = Mock()
        
        mock_page = Mock()
        mock_page.extract_text.return_value = "PDF content"
        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            
            ingestor = DocumentIngestor(
                file_list=[],
                input_bucket="test-input-bucket",
                output_bucket="test-output-bucket",
                project_id="test-project",
                embedding_model_name="test-model"
            )
            
            result = ingestor._extract_text_from_pdf("test.pdf")
            assert result == "PDF content"
    
    @patch('src.document_ingestor.GCSStorage')
    @patch('src.document_ingestor.AutoTokenizer')
    def test_clean_text(self, mock_tokenizer_class, mock_gcs_storage_class):
        mock_tokenizer = Mock()
        mock_tokenizer.tokenize.return_value = ["token1", "token2"]
        mock_tokenizer.convert_tokens_to_string.return_value = "cleaned text"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_gcs_storage_class.return_value = Mock()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            
            ingestor = DocumentIngestor(
                file_list=[],
                input_bucket="test-input-bucket",
                output_bucket="test-output-bucket",
                project_id="test-project",
                embedding_model_name="test-model"
            )
            
            result = ingestor._clean_text("text\nwith\nnewlines  ")
            assert result == "cleaned text"
            mock_tokenizer.tokenize.assert_called_once_with("text with newlines")
    
    @patch('src.document_ingestor.GCSStorage')
    @patch('src.document_ingestor.AutoTokenizer')
    def test_clean_text_empty_input(self, mock_tokenizer_class, mock_gcs_storage_class):
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_gcs_storage_class.return_value = Mock()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            
            ingestor = DocumentIngestor(
                file_list=[],
                input_bucket="test-input-bucket",
                output_bucket="test-output-bucket",
                project_id="test-project",
                embedding_model_name="test-model"
            )
            
            assert ingestor._clean_text("") is None
            assert ingestor._clean_text(None) is None
    
    @patch('src.document_ingestor.Path.unlink')
    @patch('src.document_ingestor.tempfile.NamedTemporaryFile')
    @patch('src.document_ingestor.GCSStorage')
    @patch('src.document_ingestor.AutoTokenizer')
    def test_process_files_txt_success(self, mock_tokenizer_class, mock_gcs_storage_class, mock_tempfile, mock_unlink):
        mock_tokenizer = Mock()
        mock_tokenizer.tokenize.return_value = ["test", "tokens"]
        mock_tokenizer.convert_tokens_to_string.return_value = "cleaned content"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock GCS storage
        mock_gcs_storage = Mock()
        mock_gcs_storage.upload_from_string.return_value = "gs://test-output-bucket/test_cleaned.txt"
        mock_gcs_storage_class.return_value = mock_gcs_storage
        
        # Create a real temp file for proper path handling
        temp_path = "/tmp/test_temp_file.txt"
        
        # Mock tempfile behavior
        mock_temp_file = Mock()
        mock_temp_file.name = temp_path
        mock_temp_file.suffix = '.txt'
        mock_temp_context = Mock()
        mock_temp_context.__enter__ = Mock(return_value=mock_temp_file)
        mock_temp_context.__exit__ = Mock(return_value=None)
        mock_tempfile.return_value = mock_temp_context
        
        ingestor = DocumentIngestor(
            file_list=["test.txt"],
            input_bucket="test-input-bucket", 
            output_bucket="test-output-bucket",
            project_id="test-project",
            embedding_model_name="test-model"
        )
        
        # Mock the file extraction method to return test content
        with patch.object(ingestor, '_extract_text_from_txt', return_value="Original content"):
            ingestor.process_files()
        
        # Verify GCS operations
        mock_gcs_storage.download_to_file.assert_called_once_with(
            bucket_name="test-input-bucket",
            file_path="test.txt", 
            local_path=temp_path
        )
        mock_gcs_storage.upload_from_string.assert_called_once_with(
            bucket_name="test-output-bucket",
            file_path="test_cleaned.txt",
            content="cleaned content"
        )
    
    @patch('src.document_ingestor.tempfile.NamedTemporaryFile')
    @patch('src.document_ingestor.GCSStorage')
    @patch('src.document_ingestor.AutoTokenizer')
    def test_process_files_unsupported_format(self, mock_tokenizer_class, mock_gcs_storage_class, mock_tempfile):
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock GCS storage
        mock_gcs_storage = Mock()
        mock_gcs_storage_class.return_value = mock_gcs_storage
        
        # Mock temporary file
        temp_path = "/tmp/test_temp_file.doc"
        
        mock_temp_file = Mock()
        mock_temp_file.name = temp_path
        mock_temp_file.suffix = '.doc'
        mock_temp_context = Mock()
        mock_temp_context.__enter__ = Mock(return_value=mock_temp_file)
        mock_temp_context.__exit__ = Mock(return_value=None)
        mock_tempfile.return_value = mock_temp_context
        
        ingestor = DocumentIngestor(
            file_list=["test.doc"],
            input_bucket="test-input-bucket",
            output_bucket="test-output-bucket", 
            project_id="test-project",
            embedding_model_name="test-model"
        )
        
        ingestor.process_files()
        
        # Verify download was attempted but no upload due to unsupported format
        mock_gcs_storage.download_to_file.assert_called_once()
        mock_gcs_storage.upload_from_string.assert_not_called()