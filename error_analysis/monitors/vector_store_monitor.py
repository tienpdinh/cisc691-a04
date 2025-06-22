"""
Vector Store Error Monitor

Specialized monitoring for ChromaDB vector store operations including
embedding generation, similarity search, and index management.
"""

from .base_monitor import BaseErrorMonitor


class VectorStoreMonitor(BaseErrorMonitor):
    """Specialized monitor for vector store operations."""
    
    def __init__(self, error_classifier, error_logger, failure_tracker):
        super().__init__("vector_store", error_classifier, error_logger, failure_tracker)