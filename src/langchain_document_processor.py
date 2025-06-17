import logging
from pathlib import Path
from typing import List
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document


class LangChainDocumentProcessor:
    """
    LangChain-based document processor that replaces custom document ingestion.
    Handles PDF, TXT, and DOCX files using LangChain document loaders.
    """
    
    def __init__(self,
                 file_list: List[str],
                 input_dir: str,
                 output_dir: str,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize the LangChain document processor.
        
        :param file_list: List of file paths to process
        :param input_dir: Directory containing input files
        :param output_dir: Directory to save processed text
        :param chunk_size: Size of text chunks for splitting
        :param chunk_overlap: Overlap between chunks
        """
        self.file_list = file_list
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize LangChain text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized LangChainDocumentProcessor: input_dir: {self.input_dir}, "
                         f"output_dir: {self.output_dir}, chunk_size: {chunk_size}")

    def _get_document_loader(self, file_path: Path):
        """
        Get appropriate LangChain document loader based on file extension.
        
        :param file_path: Path to the document
        :return: LangChain document loader instance
        """
        suffix = file_path.suffix.lower()
        
        if suffix == ".pdf":
            return PyPDFLoader(str(file_path))
        elif suffix == ".txt":
            return TextLoader(str(file_path), encoding="utf-8")
        elif suffix == ".docx":
            return Docx2txtLoader(str(file_path))
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def _load_and_split_document(self, file_path: Path) -> List[Document]:
        """
        Load and split a document using LangChain.
        
        :param file_path: Path to the document
        :return: List of LangChain Document objects
        """
        try:
            self.logger.info(f"Loading document: {file_path}")
            
            # Get appropriate loader
            loader = self._get_document_loader(file_path)
            
            # Load document
            documents = loader.load()
            self.logger.info(f"Loaded {len(documents)} pages from {file_path}")
            
            # Split documents into chunks
            split_docs = self.text_splitter.split_documents(documents)
            self.logger.info(f"Split into {len(split_docs)} chunks")
            
            return split_docs
            
        except Exception as e:
            self.logger.error(f"Error loading document {file_path}: {e}")
            return []

    def _save_processed_document(self, file_path: Path, documents: List[Document]):
        """
        Save processed document chunks to output directory.
        
        :param file_path: Original file path
        :param documents: List of processed document chunks
        """
        try:
            # Combine all chunks into a single text
            combined_text = "\n\n".join([doc.page_content for doc in documents])
            
            # Save to output file
            output_file = self.output_dir / f"{file_path.stem}_cleaned.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(combined_text)
            
            self.logger.info(f"Saved processed document to {output_file}")
            
            # Also save chunks separately for advanced processing
            chunks_dir = self.output_dir / "chunks"
            chunks_dir.mkdir(exist_ok=True)
            
            for i, doc in enumerate(documents):
                chunk_file = chunks_dir / f"{file_path.stem}_chunk_{i:03d}.txt"
                with open(chunk_file, "w", encoding="utf-8") as f:
                    f.write(doc.page_content)
                    # Add metadata as comments
                    f.write(f"\n\n# Metadata: {doc.metadata}")
            
            self.logger.info(f"Saved {len(documents)} chunks to {chunks_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving processed document {file_path}: {e}")

    def process_files(self) -> List[Document]:
        """
        Process all files using LangChain document loaders and text splitters.
        
        :return: List of all processed document chunks
        """
        all_documents = []
        
        for file_name in self.file_list:
            file_path = self.input_dir / file_name
            
            if not file_path.exists():
                self.logger.warning(f"File not found: {file_path}")
                continue
            
            self.logger.info(f"Processing file: {file_path}")
            
            # Load and split document
            documents = self._load_and_split_document(file_path)
            
            if documents:
                # Save processed document
                self._save_processed_document(file_path, documents)
                all_documents.extend(documents)
            else:
                self.logger.error(f"Failed to process {file_path}")
        
        self.logger.info(f"Processed {len(self.file_list)} files, "
                         f"generated {len(all_documents)} document chunks")
        
        return all_documents