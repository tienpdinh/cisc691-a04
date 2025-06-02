"""Tests for DocumentIngestor class."""
import pytest
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path
from classes.document_ingestor import DocumentIngestor

class TestDocumentIngestor:
    """Test cases for DocumentIngestor."""
    
    @pytest.fixture
    def ingestor(self, temp_dir, mock_tokenizer):
        """Create DocumentIngestor instance for testing."""
        input_dir = temp_dir / "input"
        output_dir = temp_dir / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        
        with patch('classes.document_ingestor.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):
            return DocumentIngestor(
                file_list=["test.txt", "test.pdf"],
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                embedding_model_name="test-model"
            )
    
    def test_init(self, ingestor, temp_dir, mock_tokenizer):
        """Test DocumentIngestor initialization."""
        assert ingestor.file_list == ["test.txt", "test.pdf"]
        assert ingestor.input_dir == temp_dir / "input"
        assert ingestor.output_dir == temp_dir / "output"
        assert ingestor.tokenizer == mock_tokenizer
    
    def test_extract_text_from_txt_success(self, ingestor, sample_text):
        """Test successful text extraction from TXT file."""
        with patch("builtins.open", mock_open(read_data=sample_text)):
            result = ingestor._extract_text_from_txt("test.txt")
            assert result == sample_text
    
    def test_extract_text_from_txt_error(self, ingestor):
        """Test text extraction error handling."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            result = ingestor._extract_text_from_txt("missing.txt")
            assert result is None
    
    @patch('classes.document_ingestor.pdfplumber.open')
    def test_extract_text_from_pdf_success(self, mock_pdf_open, ingestor, sample_pdf_content):
        """Test successful text extraction from PDF file."""
        # Mock PDF structure
        mock_page = MagicMock()
        mock_page.extract_text.return_value = sample_pdf_content
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf_open.return_value.__enter__.return_value = mock_pdf
        
        result = ingestor._extract_text_from_pdf("test.pdf")
        assert result == sample_pdf_content
    
    @patch('classes.document_ingestor.pdfplumber.open')
    def test_extract_text_from_pdf_error(self, mock_pdf_open, ingestor):
        """Test PDF extraction error handling."""
        mock_pdf_open.side_effect = Exception("PDF error")
        result = ingestor._extract_text_from_pdf("test.pdf")
        assert result is None
    
    def test_clean_text_success(self, ingestor, mock_tokenizer):
        """Test successful text cleaning."""
        text = "This is\na test\n\nwith newlines."
        mock_tokenizer.tokenize.return_value = ["This", "is", "a", "test"]
        mock_tokenizer.convert_tokens_to_string.return_value = "This is a test"
        
        result = ingestor._clean_text(text)
        assert result == "This is a test"
        mock_tokenizer.tokenize.assert_called_once_with("This is a test with newlines.")
    
    def test_clean_text_empty_input(self, ingestor):
        """Test text cleaning with empty input."""
        assert ingestor._clean_text("") is None
        assert ingestor._clean_text(None) is None
    
    @patch.object(Path, 'exists')
    def test_process_files_txt_success(self, mock_exists, ingestor, sample_text, mock_tokenizer):
        """Test successful processing of TXT files."""
        mock_exists.return_value = True
        mock_tokenizer.tokenize.return_value = ["sample", "text"]
        mock_tokenizer.convert_tokens_to_string.return_value = "sample text"
        
        with patch("builtins.open", mock_open(read_data=sample_text)) as mock_file:
            ingestor.process_files()
            
            # Check that files were opened for reading and writing
            assert mock_file.call_count >= 2
    
    @patch.object(Path, 'exists')
    def test_process_files_missing_file(self, mock_exists, ingestor, caplog):
        """Test processing with missing files."""
        mock_exists.return_value = False
        
        ingestor.process_files()
        
        assert "File not found" in caplog.text
    
    @patch.object(Path, 'exists')
    def test_process_files_unsupported_format(self, mock_exists, ingestor, caplog):
        """Test processing unsupported file format."""
        mock_exists.return_value = True
        ingestor.file_list = ["test.doc"]  # Unsupported format
        
        ingestor.process_files()
        
        assert "Unsupported file type" in caplog.text