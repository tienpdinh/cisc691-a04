import logging
import tempfile
from pathlib import Path
import pdfplumber
from transformers import AutoTokenizer
from .gcs_storage import GCSStorage


class DocumentIngestor:
    def __init__(self,
                 file_list,
                 input_bucket,
                 output_bucket,
                 embedding_model_name,
                 project_id=None):
        """
        Initializes the document ingestor.

        :param file_list: List of file names to process.
        :param input_bucket: GCS bucket name for input files.
        :param output_bucket: GCS bucket name for cleaned text files.
        :param embedding_model_name: Hugging Face tokenizer model for preprocessing.
        :param project_id: GCP project ID (optional).
        """
        self.file_list = file_list
        self.input_bucket = input_bucket
        self.output_bucket = output_bucket
        self.gcs_storage = GCSStorage(project_id=project_id)
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized DocumentIngestor: input_bucket: {self.input_bucket}, "
                         f"output_bucket: {self.output_bucket}, embedding_model_name: {embedding_model_name}")

    def _extract_text_from_pdf(self, file_path):
        """Extracts text from a PDF file using pdfplumber."""
        try:
            save_log_level = logging.getLogger().getEffectiveLevel()
            logging.getLogger().setLevel(logging.INFO)
            with pdfplumber.open(file_path) as pdf:
                text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
            logging.getLogger().setLevel(save_log_level)
            return text
        except Exception as e:
            self.logger.error(f"Error reading PDF {file_path}: {e}")
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
        """Cleans and tokenizes text for better embedding preparation."""
        if not text:
            return None

        text = text.replace("\n", " ").strip()  # Remove excessive newlines and trim
        tokens = self.tokenizer.tokenize(text)
        return self.tokenizer.convert_tokens_to_string(tokens)

    def process_files(self):
        """Processes the list of files from GCS and saves the cleaned text."""
        for file_name in self.file_list:
            self.logger.info(f"Processing file: {file_name}")

            try:
                # Download file from GCS to temporary location
                with tempfile.NamedTemporaryFile(suffix=Path(file_name).suffix, delete=False) as temp_file:
                    self.gcs_storage.download_to_file(
                        bucket_name=self.input_bucket,
                        file_path=file_name,
                        local_path=temp_file.name
                    )
                    temp_path = Path(temp_file.name)

                # Extract text based on file type
                if temp_path.suffix.lower() == ".pdf":
                    text = self._extract_text_from_pdf(temp_path)
                elif temp_path.suffix.lower() == ".txt":
                    text = self._extract_text_from_txt(temp_path)
                else:
                    self.logger.warning(f"Unsupported file type: {temp_path.suffix}")
                    temp_path.unlink()  # Clean up temp file
                    continue

                # Clean up temp file
                temp_path.unlink()

                # Clean and save text
                cleaned_text = self._clean_text(text)
                if cleaned_text:
                    output_file_name = f"{Path(file_name).stem}_cleaned.txt"
                    gcs_path = self.gcs_storage.upload_from_string(
                        bucket_name=self.output_bucket,
                        file_path=output_file_name,
                        content=cleaned_text
                    )
                    self.logger.info(f"Saved cleaned text to {gcs_path}")
                else:
                    self.logger.warning(f"Skipping {file_name} due to extraction failure.")

            except Exception as e:
                self.logger.error(f"Error processing file {file_name}: {e}")
                continue