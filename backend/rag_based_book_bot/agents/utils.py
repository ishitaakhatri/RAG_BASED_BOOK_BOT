"""
Utility functions for the RAG-based Book Bot
Essential validation and logging utilities
"""
import os
import logging
from typing import Optional
from pathlib import Path


# =============================================================================
# LOGGING UTILITIES
# =============================================================================

def setup_logger(
    name: str = "rag_bot",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup and configure logger for the application
    
    Args:
        name: Logger name
        log_file: Optional log file path (if None, logs to console only)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
    
    Returns:
        Configured logger instance
    
    Example:
        >>> logger = setup_logger("my_app", "logs/app.log")
        >>> logger.info("Application started")
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file specified)
    if log_file:
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def validate_pdf_path(pdf_path: str) -> tuple[bool, Optional[str]]:
    """
    Validate if the given path points to a valid PDF file
    
    Args:
        pdf_path: Path to the PDF file
    
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if path is valid, False otherwise
        - error_message: None if valid, error description if invalid
    
    Example:
        >>> is_valid, error = validate_pdf_path("book.pdf")
        >>> if not is_valid:
        ...     print(f"Error: {error}")
    """
    # Check if path is provided
    if not pdf_path:
        return False, "PDF path is empty or None"
    
    # Check if path is a string
    if not isinstance(pdf_path, (str, Path)):
        return False, f"PDF path must be a string or Path, got {type(pdf_path).__name__}"
    
    # Convert to Path object for easier handling
    path = Path(pdf_path)
    
    # Check if file exists
    if not path.exists():
        return False, f"File does not exist: {pdf_path}"
    
    # Check if it's a file (not a directory)
    if not path.is_file():
        return False, f"Path is not a file: {pdf_path}"
    
    # Check if it has .pdf extension
    if path.suffix.lower() != '.pdf':
        return False, f"File is not a PDF (extension: {path.suffix})"
    
    # Check if file is readable
    if not os.access(path, os.R_OK):
        return False, f"File is not readable: {pdf_path}"
    
    # Check if file is not empty
    if path.stat().st_size == 0:
        return False, f"PDF file is empty: {pdf_path}"
    
    # All checks passed
    return True, None


def validate_query(query: str, min_length: int = 3, max_length: int = 1000) -> tuple[bool, Optional[str]]:
    """
    Validate user query for RAG system
    
    Args:
        query: User query string
        min_length: Minimum query length (default: 3)
        max_length: Maximum query length (default: 1000)
    
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if query is valid, False otherwise
        - error_message: None if valid, error description if invalid
    
    Example:
        >>> is_valid, error = validate_query("How to build a CNN?")
        >>> if not is_valid:
        ...     print(f"Error: {error}")
    """
    # Check if query is provided
    if query is None:
        return False, "Query is None"
    
    # Check if query is a string
    if not isinstance(query, str):
        return False, f"Query must be a string, got {type(query).__name__}"
    
    # Strip whitespace
    query = query.strip()
    
    # Check if query is empty
    if not query:
        return False, "Query is empty"
    
    # Check minimum length
    if len(query) < min_length:
        return False, f"Query too short (min {min_length} characters, got {len(query)})"
    
    # Check maximum length
    if len(query) > max_length:
        return False, f"Query too long (max {max_length} characters, got {len(query)})"
    
    # Check if query contains only whitespace or special characters
    if not any(c.isalnum() for c in query):
        return False, "Query must contain at least one alphanumeric character"
    
    # All checks passed
    return True, None


# =============================================================================
# HELPER FUNCTION FOR CLEAN ERROR MESSAGES
# =============================================================================

def get_error_message(validation_result: tuple[bool, Optional[str]], default: str = "Validation failed") -> str:
    """
    Extract error message from validation result
    
    Args:
        validation_result: Tuple from validate_* functions
        default: Default message if no specific error
    
    Returns:
        Error message string
    
    Example:
        >>> result = validate_query("")
        >>> if not result[0]:
        ...     error_msg = get_error_message(result)
        ...     print(error_msg)
    """
    is_valid, error_message = validation_result
    
    if is_valid:
        return ""
    
    return error_message or default
