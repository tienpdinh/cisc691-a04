import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from src.document_ingestor import DocumentIngestor


class TestDocumentIngestor:
    
    def test_init(self):
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            
            ingestor = DocumentIngestor(
                file_list=["test.txt"],
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                embedding_model_name="test-model"
            )
            
            assert ingestor.file_list == ["test.txt"]
            assert ingestor.input_dir == input_dir
            assert ingestor.output_dir == output_dir
            assert output_dir.exists()
            # AutoTokenizer no longer used for performance
    
    def test_extract_text_from_txt_success(self):
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            input_dir.mkdir()
            
            test_file = input_dir / "test.txt"
            test_content = "This is test content"
            test_file.write_text(test_content, encoding="utf-8")
            
            ingestor = DocumentIngestor(
                file_list=["test.txt"],
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                embedding_model_name="test-model"
            )
            
            result = ingestor._extract_text_from_txt(test_file)
            assert result == test_content
    
    
    def test_extract_text_from_txt_file_not_found(self):
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            
            ingestor = DocumentIngestor(
                file_list=[],
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                embedding_model_name="test-model"
            )
            
            result = ingestor._extract_text_from_txt("nonexistent.txt")
            assert result is None
    
    @patch('src.document_ingestor.pdfplumber')
    
    def test_extract_text_from_pdf_success(self, mock_pdfplumber):
        
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
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                embedding_model_name="test-model"
            )
            
            result = ingestor._extract_text_from_pdf("test.pdf")
            assert result == "PDF content"
    
    
    def test_clean_text(self):
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            
            ingestor = DocumentIngestor(
                file_list=[],
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                embedding_model_name="test-model"
            )
            
            result = ingestor._clean_text("text\nwith\nnewlines  ")
            assert result == "text with newlines"
    
    
    def test_clean_text_empty_input(self):
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            
            ingestor = DocumentIngestor(
                file_list=[],
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                embedding_model_name="test-model"
            )
            
            assert ingestor._clean_text("") is None
            assert ingestor._clean_text(None) is None
    
    
    def test_process_files_txt_success(self):
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            input_dir.mkdir()
            
            test_file = input_dir / "test.txt"
            test_file.write_text("Original content", encoding="utf-8")
            
            ingestor = DocumentIngestor(
                file_list=["test.txt"],
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                embedding_model_name="test-model"
            )
            
            ingestor.process_files()
            
            output_file = output_dir / "test_cleaned.txt"
            assert output_file.exists()
            content = output_file.read_text(encoding="utf-8")
            assert "Original content" in content
    
    
    def test_process_files_unsupported_format(self):
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            input_dir.mkdir()
            
            test_file = input_dir / "test.doc"
            test_file.write_text("Content", encoding="utf-8")
            
            ingestor = DocumentIngestor(
                file_list=["test.doc"],
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                embedding_model_name="test-model"
            )
            
            ingestor.process_files()
            
            output_file = output_dir / "test_cleaned.txt"
            assert not output_file.exists()