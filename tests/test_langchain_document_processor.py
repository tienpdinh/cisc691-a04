import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from langchain.schema import Document

from src.langchain_document_processor import LangChainDocumentProcessor


class TestLangChainDocumentProcessor:
    """Test cases for LangChainDocumentProcessor."""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        input_dir = tempfile.mkdtemp()
        output_dir = tempfile.mkdtemp()
        yield input_dir, output_dir
        shutil.rmtree(input_dir)
        shutil.rmtree(output_dir)
    
    @pytest.fixture
    def sample_files(self, temp_dirs):
        """Create sample files for testing."""
        input_dir, _ = temp_dirs
        
        # Create sample text file
        txt_file = Path(input_dir) / "sample.txt"
        txt_file.write_text("This is a sample text file for testing.", encoding="utf-8")
        
        # Create sample PDF file (mock)
        pdf_file = Path(input_dir) / "sample.pdf"
        pdf_file.touch()
        
        # Create sample DOCX file (mock)
        docx_file = Path(input_dir) / "sample.docx"
        docx_file.touch()
        
        return ["sample.txt", "sample.pdf", "sample.docx"]
    
    @patch('src.langchain_document_processor.RecursiveCharacterTextSplitter')
    def test_init(self, mock_splitter, temp_dirs):
        """Test processor initialization."""
        input_dir, output_dir = temp_dirs
        file_list = ["test.txt"]
        
        processor = LangChainDocumentProcessor(
            file_list=file_list,
            input_dir=input_dir,
            output_dir=output_dir,
            chunk_size=500,
            chunk_overlap=100
        )
        
        assert processor.file_list == file_list
        assert processor.input_dir == Path(input_dir)
        assert processor.output_dir == Path(output_dir)
        
        # Verify text splitter was initialized correctly
        mock_splitter.assert_called_once_with(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def test_get_document_loader_txt(self, temp_dirs):
        """Test getting TextLoader for TXT files."""
        input_dir, output_dir = temp_dirs
        processor = LangChainDocumentProcessor(["test.txt"], input_dir, output_dir)
        
        loader = processor._get_document_loader(Path("test.txt"))
        assert loader.__class__.__name__ == "TextLoader"
    
    @patch('src.langchain_document_processor.PyPDFLoader')
    def test_get_document_loader_pdf(self, mock_pdf_loader, temp_dirs):
        """Test getting PyPDFLoader for PDF files."""
        input_dir, output_dir = temp_dirs
        processor = LangChainDocumentProcessor(["test.pdf"], input_dir, output_dir)
        
        loader = processor._get_document_loader(Path("test.pdf"))
        mock_pdf_loader.assert_called_once_with("test.pdf")
    
    @patch('src.langchain_document_processor.Docx2txtLoader')
    def test_get_document_loader_docx(self, mock_docx_loader, temp_dirs):
        """Test getting Docx2txtLoader for DOCX files."""
        input_dir, output_dir = temp_dirs
        processor = LangChainDocumentProcessor(["test.docx"], input_dir, output_dir)
        
        loader = processor._get_document_loader(Path("test.docx"))
        mock_docx_loader.assert_called_once_with("test.docx")
    
    def test_get_document_loader_unsupported(self, temp_dirs):
        """Test error for unsupported file types."""
        input_dir, output_dir = temp_dirs
        processor = LangChainDocumentProcessor(["test.xyz"], input_dir, output_dir)
        
        with pytest.raises(ValueError, match="Unsupported file type"):
            processor._get_document_loader(Path("test.xyz"))
    
    @patch('src.langchain_document_processor.TextLoader')
    def test_load_and_split_document_success(self, mock_loader_class, temp_dirs):
        """Test successful document loading and splitting."""
        input_dir, output_dir = temp_dirs
        processor = LangChainDocumentProcessor(["test.txt"], input_dir, output_dir)
        
        # Mock loader and documents
        mock_loader = Mock()
        mock_loader.load.return_value = [
            Document(page_content="This is a long text that will be split into chunks."),
            Document(page_content="This is another page of content.")
        ]
        mock_loader_class.return_value = mock_loader
        
        # Mock text splitter
        with patch.object(processor.text_splitter, 'split_documents') as mock_split:
            mock_split.return_value = [
                Document(page_content="This is a long text"),
                Document(page_content="that will be split"),
                Document(page_content="into chunks.")
            ]
            
            result = processor._load_and_split_document(Path("test.txt"))
            
            assert len(result) == 3
            assert all(isinstance(doc, Document) for doc in result)
            mock_loader.load.assert_called_once()
            mock_split.assert_called_once()
    
    @patch('src.langchain_document_processor.TextLoader')
    def test_load_and_split_document_error(self, mock_loader_class, temp_dirs):
        """Test error handling in document loading."""
        input_dir, output_dir = temp_dirs
        processor = LangChainDocumentProcessor(["test.txt"], input_dir, output_dir)
        
        # Mock loader to raise exception
        mock_loader = Mock()
        mock_loader.load.side_effect = Exception("Loading failed")
        mock_loader_class.return_value = mock_loader
        
        result = processor._load_and_split_document(Path("test.txt"))
        
        assert result == []
    
    def test_save_processed_document(self, temp_dirs):
        """Test saving processed documents."""
        input_dir, output_dir = temp_dirs
        processor = LangChainDocumentProcessor(["test.txt"], input_dir, output_dir)
        
        documents = [
            Document(page_content="First chunk", metadata={"source": "test.txt"}),
            Document(page_content="Second chunk", metadata={"source": "test.txt"})
        ]
        
        processor._save_processed_document(Path("test.txt"), documents)
        
        # Check main output file
        output_file = Path(output_dir) / "test_cleaned.txt"
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert "First chunk" in content
        assert "Second chunk" in content
        
        # Check chunks directory
        chunks_dir = Path(output_dir) / "chunks"
        assert chunks_dir.exists()
        chunk_files = list(chunks_dir.glob("test_chunk_*.txt"))
        assert len(chunk_files) == 2
    
    @patch.object(LangChainDocumentProcessor, '_load_and_split_document')
    @patch.object(LangChainDocumentProcessor, '_save_processed_document')
    def test_process_files_success(self, mock_save, mock_load, temp_dirs, sample_files):
        """Test successful file processing."""
        input_dir, output_dir = temp_dirs
        processor = LangChainDocumentProcessor(sample_files, input_dir, output_dir)
        
        # Mock document loading
        mock_documents = [Document(page_content="Test content")]
        mock_load.return_value = mock_documents
        
        result = processor.process_files()
        
        # Should process 3 files
        assert mock_load.call_count == 3
        assert mock_save.call_count == 3
        assert len(result) == 3  # 3 files * 1 document each
    
    @patch.object(LangChainDocumentProcessor, '_load_and_split_document')
    def test_process_files_missing_file(self, mock_load, temp_dirs):
        """Test processing with missing files."""
        input_dir, output_dir = temp_dirs
        processor = LangChainDocumentProcessor(["missing.txt"], input_dir, output_dir)
        
        result = processor.process_files()
        
        # Should not call load for missing file
        mock_load.assert_not_called()
        assert result == []
    
    @patch.object(LangChainDocumentProcessor, '_load_and_split_document')
    def test_process_files_load_failure(self, mock_load, temp_dirs, sample_files):
        """Test processing with document loading failure."""
        input_dir, output_dir = temp_dirs
        processor = LangChainDocumentProcessor(sample_files[:1], input_dir, output_dir)  # Only process first file
        
        # Mock loading failure
        mock_load.return_value = []
        
        result = processor.process_files()
        
        mock_load.assert_called_once()
        assert result == []