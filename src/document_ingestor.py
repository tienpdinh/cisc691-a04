import logging
from pathlib import Path
import pdfplumber
# from transformers import AutoTokenizer  # Removed for performance
from docx import Document
import re


class DocumentIngestor:
    def __init__(self,
                 file_list,
                 input_dir,
                 output_dir,
                 embedding_model_name):
        """
        Initializes the document ingestor.

        :param file_list: List of file paths to process.
        :param output_dir: Directory to save cleaned text files.
        :param model_name: Hugging Face tokenizer model for preprocessing.
        """
        self.file_list = file_list
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Remove tokenizer - not needed for basic text cleaning
        # self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized DocumentIngestor: input_dir: {self.input_dir}, "
                         f"output_dir: {self.output_dir}")

    def _extract_text_from_pdf(self, file_path):
        """Extracts text from a PDF file using pdfplumber with enhanced error handling."""
        try:
            self.logger.info(f"Starting PDF extraction: {file_path}")
            text_parts = []
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                self.logger.info(f"PDF has {total_pages} pages")
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                        else:
                            self.logger.warning(f"Empty text on page {page_num}")
                        
                        # Log progress every 10 pages for large PDFs
                        if page_num % 10 == 0:
                            self.logger.info(f"Processed {page_num}/{total_pages} pages")
                            
                    except Exception as e:
                        self.logger.error(f"Error on page {page_num}: {e}")
                        continue
            
            if not text_parts:
                self.logger.error("No text extracted from PDF")
                return None
                
            extracted_text = "\n".join(text_parts)
            self.logger.info(f"PDF extraction completed. Total text length: {len(extracted_text)} characters")
            return extracted_text
        except Exception as e:
            self.logger.error(f"PDF extraction failed: {e}")
            return None

    def _extract_text_from_docx(self, file_path):
        """Extracts text from a DOCX file."""
        try:
            self.logger.debug(f"Starting DOCX extraction: {file_path}")
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])
            return text
        except Exception as e:
            self.logger.error(f"DOCX extraction failed: {e}")
            return None

    def _extract_text_from_txt(self, file_path):
        """Extracts text from a TXT file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Error reading TXT {file_path}: {e}")
            return None

    def _clean_text(self, text):
        """Enhanced text cleaning with better preprocessing."""
        if not text:
            return None

        # Enhanced cleaning steps
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
        text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
        text = re.sub(r'[\n\r\t]+', ' ', text)  # Replace newlines and tabs
        text = text.strip()
        
        self.logger.debug(f"Cleaned text length: {len(text)}")
        return text

    def process_files(self):
        """Processes the list of files with enhanced support for PDF and DOCX."""
        for file_path in self.file_list:
            file_path = Path(self.input_dir/file_path)
            if not file_path.exists():
                self.logger.warning(f"File not found: {file_path}")
                continue

            self.logger.info(f"Processing file: {file_path}")
            
            # Add progress logging for large files
            if file_path.suffix.lower() == ".pdf":
                self.logger.info(f"Starting PDF processing for {file_path.name}")
            text = None

            if file_path.suffix.lower() == ".pdf":
                text = self._extract_text_from_pdf(file_path)
            elif file_path.suffix.lower() == ".txt":
                text = self._extract_text_from_txt(file_path)
            elif file_path.suffix.lower() == ".docx":
                text = self._extract_text_from_docx(file_path)
            else:
                self.logger.warning(f"Unsupported file type: {file_path.suffix}")
                continue

            if text:
                cleaned_text = self._clean_text(text)
                if cleaned_text:
                    output_file = self.output_dir / f"{file_path.stem}_cleaned.txt"
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(cleaned_text)
                    self.logger.info(f"Saved cleaned text to {output_file}")
                else:
                    self.logger.error(f"Text cleaning failed for {file_path}")
            else:
                self.logger.error(f"Text extraction failed for {file_path}")