"""
Document Processor Error Monitor

Specialized monitoring for document processing operations including
PDF extraction, text chunking, and preprocessing errors.
"""

from .base_monitor import BaseErrorMonitor


class DocumentProcessorMonitor(BaseErrorMonitor):
    """Specialized monitor for document processing operations."""
    
    def __init__(self, error_classifier, error_logger, failure_tracker):
        super().__init__("document_processor", error_classifier, error_logger, failure_tracker)