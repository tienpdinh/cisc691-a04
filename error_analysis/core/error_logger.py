"""
Error Logging System for RAG Pipeline

Provides structured logging, error aggregation, and persistent storage
for all error events in the RAG system.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta
import sqlite3
from dataclasses import asdict
import threading
from collections import defaultdict, deque

from .error_classifier import ErrorInfo, ErrorSeverity, ErrorCategory


class ErrorLogger:
    """
    Comprehensive error logging system with structured storage and aggregation.
    
    Provides persistent storage, real-time aggregation, and query capabilities
    for all error events in the RAG pipeline.
    """
    
    def __init__(self, log_directory: str = "logs/errors", max_memory_entries: int = 1000):
        """
        Initialize error logger.
        
        Args:
            log_directory: Directory for error log files
            max_memory_entries: Maximum entries to keep in memory
        """
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        self.max_memory_entries = max_memory_entries
        self.memory_buffer = deque(maxlen=max_memory_entries)
        
        # Thread-safe aggregation
        self._lock = threading.Lock()
        self._aggregations = defaultdict(int)
        self._component_errors = defaultdict(int)
        self._hourly_counts = defaultdict(int)
        
        # Initialize database
        self.db_path = self.log_directory / "errors.db"
        self._init_database()
        
        # Initialize file logger
        self.logger = logging.getLogger(__name__)
        self._setup_file_logger()
    
    def _init_database(self):
        """Initialize SQLite database for error storage."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS errors (
                    error_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    category TEXT NOT NULL,
                    component TEXT NOT NULL,
                    error_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    details TEXT,
                    stack_trace TEXT,
                    context TEXT,
                    recovery_suggested TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolution_notes TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON errors (timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_severity ON errors (severity)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_component ON errors (component)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_category ON errors (category)
            """)
    
    def _setup_file_logger(self):
        """Setup structured file logging."""
        # Create formatter for JSON logs
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler for error logs
        log_file = self.log_directory / f"errors_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.ERROR)
        
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.ERROR)
    
    def log_error(self, error_info: ErrorInfo):
        """
        Log error information to all storage systems.
        
        Args:
            error_info: ErrorInfo object to log
        """
        with self._lock:
            # Add to memory buffer
            self.memory_buffer.append(error_info)
            
            # Update aggregations
            self._update_aggregations(error_info)
            
            # Store in database
            self._store_in_database(error_info)
            
            # Log to file
            self._log_to_file(error_info)
    
    def _update_aggregations(self, error_info: ErrorInfo):
        """Update real-time aggregation counters."""
        # Overall error counts by severity and category
        severity_key = f"severity_{error_info.severity.value}"
        category_key = f"category_{error_info.category.value}"
        
        self._aggregations[severity_key] += 1
        self._aggregations[category_key] += 1
        self._component_errors[error_info.component] += 1
        
        # Hourly counts for trend analysis
        hour_key = error_info.timestamp.strftime("%Y%m%d_%H")
        self._hourly_counts[hour_key] += 1
    
    def _store_in_database(self, error_info: ErrorInfo):
        """Store error in SQLite database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO errors (
                        error_id, timestamp, severity, category, component,
                        error_type, message, details, stack_trace, context,
                        recovery_suggested
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    error_info.error_id,
                    error_info.timestamp.isoformat(),
                    error_info.severity.value,
                    error_info.category.value,
                    error_info.component,
                    error_info.error_type,
                    error_info.message,
                    json.dumps(error_info.details),
                    error_info.stack_trace,
                    json.dumps(error_info.context) if error_info.context else None,
                    error_info.recovery_suggested
                ))
        except Exception as e:
            self.logger.error(f"Failed to store error in database: {e}")
    
    def _log_to_file(self, error_info: ErrorInfo):
        """Log error to structured file."""
        log_entry = {
            'error_id': error_info.error_id,
            'timestamp': error_info.timestamp.isoformat(),
            'severity': error_info.severity.value,
            'category': error_info.category.value,
            'component': error_info.component,
            'error_type': error_info.error_type,
            'message': error_info.message,
            'details': error_info.details,
            'recovery_suggested': error_info.recovery_suggested
        }
        
        # Log based on severity
        if error_info.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(json.dumps(log_entry))
        elif error_info.severity == ErrorSeverity.HIGH:
            self.logger.error(json.dumps(log_entry))
        elif error_info.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(json.dumps(log_entry))
        else:
            self.logger.info(json.dumps(log_entry))
    
    def get_recent_errors(self, 
                         limit: int = 50,
                         severity: Optional[ErrorSeverity] = None,
                         component: Optional[str] = None,
                         hours: int = 24) -> List[ErrorInfo]:
        """
        Get recent errors from memory buffer or database.
        
        Args:
            limit: Maximum number of errors to return
            severity: Filter by severity level
            component: Filter by component
            hours: Look back this many hours
            
        Returns:
            List of ErrorInfo objects
        """
        # First try memory buffer for recent errors
        recent_errors = []
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        for error in reversed(self.memory_buffer):
            if error.timestamp < cutoff_time:
                break
            
            # Apply filters
            if severity and error.severity != severity:
                continue
            if component and error.component != component:
                continue
            
            recent_errors.append(error)
            
            if len(recent_errors) >= limit:
                break
        
        # If not enough from memory, query database
        if len(recent_errors) < limit:
            db_errors = self._query_database(
                limit=limit - len(recent_errors),
                severity=severity,
                component=component,
                hours=hours
            )
            recent_errors.extend(db_errors)
        
        return recent_errors[:limit]
    
    def _query_database(self,
                       limit: int,
                       severity: Optional[ErrorSeverity] = None,
                       component: Optional[str] = None,
                       hours: int = 24) -> List[ErrorInfo]:
        """Query errors from database."""
        try:
            cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            query = """
                SELECT * FROM errors 
                WHERE timestamp >= ?
            """
            params = [cutoff_time]
            
            if severity:
                query += " AND severity = ?"
                params.append(severity.value)
            
            if component:
                query += " AND component = ?"
                params.append(component)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                
                errors = []
                for row in cursor.fetchall():
                    error_info = ErrorInfo(
                        error_id=row['error_id'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        severity=ErrorSeverity(row['severity']),
                        category=ErrorCategory(row['category']),
                        component=row['component'],
                        error_type=row['error_type'],
                        message=row['message'],
                        details=json.loads(row['details']) if row['details'] else {},
                        stack_trace=row['stack_trace'],
                        context=json.loads(row['context']) if row['context'] else None,
                        recovery_suggested=row['recovery_suggested']
                    )
                    errors.append(error_info)
                
                return errors
                
        except Exception as e:
            self.logger.error(f"Failed to query database: {e}")
            return []
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get summary statistics for errors in the specified time period.
        
        Args:
            hours: Time period to analyze
            
        Returns:
            Dictionary with error statistics
        """
        with self._lock:
            # Recent error counts
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_errors = [
                error for error in self.memory_buffer 
                if error.timestamp >= cutoff_time
            ]
            
            # Count by severity
            severity_counts = defaultdict(int)
            category_counts = defaultdict(int)
            component_counts = defaultdict(int)
            
            for error in recent_errors:
                severity_counts[error.severity.value] += 1
                category_counts[error.category.value] += 1
                component_counts[error.component] += 1
            
            # Calculate error rate
            total_errors = len(recent_errors)
            error_rate = total_errors / hours if hours > 0 else 0
            
            return {
                'time_period_hours': hours,
                'total_errors': total_errors,
                'error_rate_per_hour': round(error_rate, 2),
                'severity_breakdown': dict(severity_counts),
                'category_breakdown': dict(category_counts),
                'component_breakdown': dict(component_counts),
                'most_frequent_component': max(component_counts, key=component_counts.get) if component_counts else None,
                'critical_errors': severity_counts.get('critical', 0),
                'memory_buffer_size': len(self.memory_buffer),
                'aggregation_stats': dict(self._aggregations)
            }
    
    def mark_error_resolved(self, error_id: str, resolution_notes: str = ""):
        """
        Mark an error as resolved with optional notes.
        
        Args:
            error_id: ID of the error to mark as resolved
            resolution_notes: Optional notes about the resolution
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE errors 
                    SET resolved = TRUE, resolution_notes = ?
                    WHERE error_id = ?
                """, (resolution_notes, error_id))
                
                self.logger.info(f"Marked error {error_id} as resolved: {resolution_notes}")
        except Exception as e:
            self.logger.error(f"Failed to mark error as resolved: {e}")
    
    def export_errors(self, 
                     output_file: Path,
                     hours: int = 24,
                     format: str = "json") -> bool:
        """
        Export errors to file.
        
        Args:
            output_file: Output file path
            hours: Time period to export
            format: Export format ('json' or 'csv')
            
        Returns:
            True if export successful
        """
        try:
            errors = self.get_recent_errors(limit=10000, hours=hours)
            
            if format == "json":
                export_data = [asdict(error) for error in errors]
                # Convert datetime objects to strings
                for item in export_data:
                    item['timestamp'] = item['timestamp'].isoformat()
                    item['severity'] = item['severity'].value
                    item['category'] = item['category'].value
                
                with open(output_file, 'w') as f:
                    json.dump(export_data, f, indent=2)
            
            elif format == "csv":
                import csv
                with open(output_file, 'w', newline='') as f:
                    if errors:
                        writer = csv.DictWriter(f, fieldnames=[
                            'error_id', 'timestamp', 'severity', 'category',
                            'component', 'error_type', 'message', 'recovery_suggested'
                        ])
                        writer.writeheader()
                        
                        for error in errors:
                            writer.writerow({
                                'error_id': error.error_id,
                                'timestamp': error.timestamp.isoformat(),
                                'severity': error.severity.value,
                                'category': error.category.value,
                                'component': error.component,
                                'error_type': error.error_type,
                                'message': error.message,
                                'recovery_suggested': error.recovery_suggested
                            })
            
            self.logger.info(f"Exported {len(errors)} errors to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export errors: {e}")
            return False
    
    def cleanup_old_errors(self, days: int = 30):
        """
        Clean up errors older than specified days.
        
        Args:
            days: Remove errors older than this many days
        """
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("DELETE FROM errors WHERE timestamp < ?", (cutoff_date,))
                deleted_count = cursor.rowcount
                
                self.logger.info(f"Cleaned up {deleted_count} errors older than {days} days")
        except Exception as e:
            self.logger.error(f"Failed to cleanup old errors: {e}")
    
    def get_error_trends(self, hours: int = 168) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get error trends over time for analysis.
        
        Args:
            hours: Time period to analyze (default: 1 week)
            
        Returns:
            Dictionary with trend data
        """
        try:
            cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                # Hourly error counts
                cursor = conn.execute("""
                    SELECT 
                        strftime('%Y-%m-%d %H:00:00', timestamp) as hour,
                        COUNT(*) as error_count,
                        severity
                    FROM errors 
                    WHERE timestamp >= ?
                    GROUP BY hour, severity
                    ORDER BY hour
                """, (cutoff_time,))
                
                hourly_trends = []
                for row in cursor.fetchall():
                    hourly_trends.append({
                        'hour': row[0],
                        'error_count': row[1],
                        'severity': row[2]
                    })
                
                # Component error trends
                cursor = conn.execute("""
                    SELECT 
                        component,
                        COUNT(*) as error_count,
                        strftime('%Y-%m-%d', timestamp) as date
                    FROM errors 
                    WHERE timestamp >= ?
                    GROUP BY component, date
                    ORDER BY date, error_count DESC
                """, (cutoff_time,))
                
                component_trends = []
                for row in cursor.fetchall():
                    component_trends.append({
                        'component': row[0],
                        'error_count': row[1],
                        'date': row[2]
                    })
                
                return {
                    'hourly_trends': hourly_trends,
                    'component_trends': component_trends,
                    'analysis_period_hours': hours
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get error trends: {e}")
            return {'hourly_trends': [], 'component_trends': [], 'analysis_period_hours': hours}