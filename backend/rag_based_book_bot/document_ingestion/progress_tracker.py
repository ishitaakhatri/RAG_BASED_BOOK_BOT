#   progress_tracker.py
"""
Progress Tracker for Document Ingestion

Tracks ingestion progress in real-time and provides:
- Current page/batch processing status
- Percentage calculation
- Real-time progress updates via callback
- Progress state management for WebSocket streaming
- Live log capture from Python logging
"""

import logging
import time
from typing import Callable, Optional, Dict, Any, Coroutine, List
from dataclasses import dataclass, field
from threading import Lock
from datetime import datetime
import asyncio
import inspect


logger = logging.getLogger("progress_tracker")


@dataclass
class ProgressState:
    """Current progress state"""
    total_pages: int = 0
    current_page: int = 0
    current_batch: int = 0
    total_batches: int = 0
    percentage: float = 0.0
    status: str = "initializing"
    current_task: str = ""
    chunks_created: int = 0
    embeddings_generated: int = 0
    vectors_upserted: int = 0
    start_time: float = field(default_factory=time.time)
    elapsed_time: float = 0.0
    estimated_time_remaining: float = 0.0
    speed_pages_per_sec: float = 0.0
    book_title: str = ""
    author: str = ""
    errors: list = field(default_factory=list)
    logs: List[str] = field(default_factory=list)  # NEW: Live logs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "total_pages": self.total_pages,
            "current_page": self.current_page,
            "current_batch": self.current_batch,
            "total_batches": self.total_batches,
            "percentage": round(self.percentage, 2),
            "status": self.status,
            "current_task": self.current_task,
            "chunks_created": self.chunks_created,
            "embeddings_generated": self.embeddings_generated,
            "vectors_upserted": self.vectors_upserted,
            "elapsed_time": round(self.elapsed_time, 2),
            "estimated_time_remaining": round(self.estimated_time_remaining, 2),
            "speed_pages_per_sec": round(self.speed_pages_per_sec, 2),
            "book_title": self.book_title,
            "author": self.author,
            "errors": self.errors,
            "logs": self.logs  # NEW: Include logs in state
        }


class ProgressLogHandler(logging.Handler):
    """
    Custom logging handler that forwards logs to ProgressTracker
    
    This captures Python logging output and sends it to the frontend
    via WebSocket in real-time.
    """
    
    def __init__(self, tracker: 'ProgressTracker'):
        super().__init__()
        self.tracker = tracker
        self.setFormatter(logging.Formatter('%(name)s: %(message)s'))
    
    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            self.tracker.add_log(msg, record.levelname)
        except Exception:
            self.handleError(record)


class ProgressTracker:
    """
    Tracks ingestion progress and broadcasts updates
    
    Usage:
        tracker = ProgressTracker()
        tracker.on_progress(lambda state: print(f"{state.percentage}%"))
        
        tracker.start_ingestion(pdf_path, total_pages=278)
        tracker.update_batch(batch_num=1, current_page=20)
        tracker.update_chunks(chunks_count=100)
        tracker.update_embeddings(count=100)
        tracker.update_upsert(count=100)
        tracker.finish()
    """
    
    def __init__(self):
        self.state = ProgressState()
        self.lock = Lock()
        self.callbacks: List[Callable[[ProgressState], Coroutine | None]] = []
        self.log_history: list[Dict[str, Any]] = []
        self._loop = None
        self._main_loop = None  # Reference to the main event loop
        self._log_handlers: List[ProgressLogHandler] = []  # Track handlers for cleanup
        
        # ✅ FIX: Throttling variables
        self.last_emit_time = 0.0
        self.emit_interval = 0.1  # Max 10 updates per second
        self._last_emitted_status = "" # Track last emitted status to force updates on change

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the main application event loop for scheduling async callbacks"""
        self._main_loop = loop
    
    def on_progress(self, callback: Callable[[ProgressState], Coroutine | None]) -> None:
        """Register a callback to receive progress updates"""
        with self.lock:
            self.callbacks.append(callback)

    def remove_callback(self, callback: Callable[[ProgressState], Coroutine | None]) -> None:
        """Safely remove a callback"""
        with self.lock:
            if callback in self.callbacks:
                self.callbacks.remove(callback)
    
    def _notify_callbacks(self) -> None:
        """Notify all registered callbacks - Thread Safe Version with Smart Throttling"""
        
        current_time = time.time()
        
        # ✅ FIX: Smart Throttling Logic
        # 1. Always emit if we are in a terminal state (completed/failed)
        is_terminal = self.state.status in ["completed", "failed"]
        
        # 2. Always emit if the status string has changed (e.g. "parsing_pdf" -> "chunking")
        # This ensures the UI doesn't get stuck on an old step even if logs are spamming
        status_changed = self.state.status != self._last_emitted_status
        
        # 3. Otherwise, check time interval
        time_elapsed = (current_time - self.last_emit_time) >= self.emit_interval
        
        should_emit = is_terminal or status_changed or time_elapsed
        
        if not should_emit:
            return

        # Snapshot callbacks under lock to prevent modification during iteration
        with self.lock:
            if not self.callbacks:
                return
            callbacks_snapshot = self.callbacks[:]
            self.last_emit_time = current_time
            self._last_emitted_status = self.state.status

        for callback in callbacks_snapshot:
            try:
                result = callback(self.state)
                
                # If it's a coroutine, we must schedule it safely
                if inspect.iscoroutine(result):
                    try:
                        # 1. Try to schedule on the stored main loop (Best for worker threads)
                        if self._main_loop and not self._main_loop.is_closed():
                            asyncio.run_coroutine_threadsafe(result, self._main_loop)
                        
                        # 2. Fallback: Try getting the running loop (Works if called from main thread)
                        else:
                            try:
                                loop = asyncio.get_running_loop()
                                loop.create_task(result)
                            except RuntimeError:
                                # No running loop available
                                pass
                    except Exception as e:
                        logger.debug(f"Could not schedule async callback: {e}")
            except Exception as e:
                logger.exception(f"Progress callback failed: {e}")

    def add_log(self, message: str, level: str = "INFO") -> None:
        """
        Add a log message to the progress state
        
        Args:
            message: Log message
            level: Log level (INFO, WARNING, ERROR, DEBUG)
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_log = f"[{timestamp}] {level}: {message}"
        
        with self.lock:
            self.state.logs.append(formatted_log)
            # Keep only last 200 logs to prevent memory issues
            if len(self.state.logs) > 200:
                self.state.logs = self.state.logs[-200:]
        
        self._notify_callbacks()

    def update_total_pages(self, total_pages: int) -> None:
        """Safely update total pages and notify listeners"""
        with self.lock:
            self.state.total_pages = total_pages
        self._notify_callbacks()
    
    def _calculate_eta(self) -> None:
        """Calculate estimated time remaining"""
        if self.state.current_page == 0 or self.state.current_page == self.state.total_pages:
            self.state.estimated_time_remaining = 0.0
            return
        
        self.state.elapsed_time = time.time() - self.state.start_time
        
        if self.state.elapsed_time > 0:
            self.state.speed_pages_per_sec = self.state.current_page / self.state.elapsed_time
            
            if self.state.speed_pages_per_sec > 0:
                remaining_pages = self.state.total_pages - self.state.current_page
                self.state.estimated_time_remaining = remaining_pages / self.state.speed_pages_per_sec
    
    def start_ingestion(
        self, 
        pdf_path: str, 
        total_pages: int,
        book_title: str = "",
        author: str = ""
    ) -> None:
        """Initialize ingestion tracker"""
        with self.lock:
            self.state.total_pages = total_pages
            self.state.book_title = book_title
            self.state.author = author
            self.state.status = "parsing_pdf"
            self.state.current_task = f"Loading PDF: {pdf_path}"
            self.state.start_time = time.time()
            self.state.percentage = 0.0
            
            self._log_event("ingestion_started", {
                "total_pages": total_pages,
                "book_title": book_title,
                "pdf_path": pdf_path
            })
        self._notify_callbacks()
    
    def update_batch(
        self, 
        batch_num: int, 
        total_batches: int,
        current_page: int
    ) -> None:
        """Update batch processing progress"""
        with self.lock:
            self.state.current_batch = batch_num
            self.state.total_batches = total_batches
            self.state.current_page = current_page
            self.state.status = "chunking"
            self.state.current_task = f"Processing pages {current_page - 19} - {current_page} (Batch {batch_num}/{total_batches})"
            
            # Calculate percentage based on pages processed
            if self.state.total_pages > 0:
                self.state.percentage = (self.state.current_page / self.state.total_pages) * 100
            
            self._calculate_eta()
        self._notify_callbacks()
    
    def start_chunking(self) -> None:
        """Mark start of chunking phase"""
        with self.lock:
            self.state.status = "chunking"
            self.state.current_task = "Applying semantic chunking..."
            self.state.percentage = 45.0
            self._log_event("chunking_started", {})

        self._notify_callbacks()
    
    def update_chunks(self, chunks_count: int) -> None:
        """Update number of chunks created"""
        with self.lock:
            self.state.chunks_created = chunks_count
            self.state.current_task = f"Created {chunks_count} semantic chunks"
            self.state.percentage = 50.0

        self._notify_callbacks()
    
    def start_embedding(self) -> None:
        with self.lock:
            self.state.status = "embedding"
            self.state.current_task = "Generating embeddings..."
            self.state.percentage = 55.0
            self._log_event("embedding_started", {
                "chunks_count": self.state.chunks_created
            })

        self._notify_callbacks()
    
    def update_embeddings(self, count: int) -> None:
        with self.lock:
            self.state.embeddings_generated = count
            if self.state.chunks_created > 0:
                embed_progress = (count / self.state.chunks_created) * 30
                self.state.percentage = 55.0 + embed_progress
            self.state.current_task = f"Generated {count}/{self.state.chunks_created} embeddings"

        self._notify_callbacks()
    
    def start_upsert(self) -> None:
        """Mark start of Pinecone upsert phase"""
        with self.lock:
            self.state.status = "upserting"
            self.state.current_task = "Upserting vectors to Pinecone..."
            self.state.percentage = 85.0
            self._log_event("upsert_started", {
                "embeddings_count": self.state.embeddings_generated
            })

        self._notify_callbacks()
    
    def update_upsert(self, count: int) -> None:
        """Update number of vectors upserted"""
        with self.lock:
            self.state.vectors_upserted = count
            if self.state.embeddings_generated > 0:
                upsert_progress = (count / self.state.embeddings_generated) * 10  # 10% of total
                self.state.percentage = 85.0 + upsert_progress
            self.state.current_task = f"Upserted {count}/{self.state.embeddings_generated} vectors"
        self._notify_callbacks()
    
    def add_error(self, error_msg: str) -> None:
        """Log an error during ingestion"""
        with self.lock:
            self.state.errors.append(error_msg)
            logger.error(f"Ingestion error: {error_msg}")
            self._log_event("error_occurred", {"error": error_msg})
        self._notify_callbacks()
    
    def finish(self, success: bool = True) -> None:
        """Mark ingestion as complete"""
        with self.lock:
            if success:
                self.state.status = "completed"
                self.state.percentage = 100.0
                self.state.current_task = "Ingestion completed successfully"
                self._log_event("ingestion_completed", {
                    "chunks_created": self.state.chunks_created,
                    "embeddings_generated": self.state.embeddings_generated,
                    "vectors_upserted": self.state.vectors_upserted,
                    "total_time": round(time.time() - self.state.start_time, 2)
                })
            else:
                self.state.status = "failed"
                self.state.current_task = "Ingestion failed"
                self._log_event("ingestion_failed", {
                    "errors": self.state.errors
                })
            
            self.state.elapsed_time = time.time() - self.state.start_time
        self._notify_callbacks()
    
    def get_state(self) -> Dict[str, Any]:
        """Get current progress state as dictionary"""
        with self.lock:
            return self.state.to_dict()
    
    def get_log_history(self) -> list[Dict[str, Any]]:
        """Get all logged events"""
        with self.lock:
            return self.log_history.copy()
    
    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log an event internally"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data
        }
        self.log_history.append(event)
    
    def setup_logging_handlers(self) -> None:
        """
        Setup custom logging handlers to capture logs from various loggers
        
        Call this once during initialization to start capturing logs
        """
        # Remove old handlers if they exist
        self.cleanup_logging_handlers()
        
        # Create new handler
        handler = ProgressLogHandler(self)
        handler.setLevel(logging.INFO)
        
        # ✅ FIX: Updated list of loggers to catch everything
        loggers_to_track = [
            "main",                # Main application logs (initialization, cleanup)
            "rag_based_book_bot",  # Root package logger (catches all submodules)
            "uvicorn",             # Server logs
            "uvicorn.error",       # Server errors
            # Redundant but kept for safety if they are initialized independently:
            "enhanced_ingestion",
            "semantic_chunker",
            "grobid_parser",
            "sentence_transformers",
        ]
        
        for logger_name in loggers_to_track:
            target_logger = logging.getLogger(logger_name)
            target_logger.addHandler(handler)
            target_logger.setLevel(logging.INFO)
        
        self._log_handlers.append(handler)
        logger.info("✅ Logging handlers attached to progress tracker")
    
    def cleanup_logging_handlers(self) -> None:
        """Remove all logging handlers"""
        for handler in self._log_handlers:
            # Remove from all loggers
            for logger_name in logging.Logger.manager.loggerDict:
                target_logger = logging.getLogger(logger_name)
                if handler in target_logger.handlers:
                    target_logger.removeHandler(handler)
        
        self._log_handlers.clear()
    
    def reset(self) -> None:
        """Reset tracker for new ingestion"""
        with self.lock:
            self.state = ProgressState()
            self.log_history = []
        # Notify listeners that state has been reset
        self._notify_callbacks()


# Global tracker instance (shared across all requests)
_global_tracker = ProgressTracker()


def get_progress_tracker() -> ProgressTracker:
    """Get the global progress tracker instance and ensure logging is set up"""
    # Set up logging handlers on first access
    if not _global_tracker._log_handlers:
        _global_tracker.setup_logging_handlers()
    return _global_tracker


def reset_progress_tracker() -> None:
    """Reset the global tracker"""
    _global_tracker.reset()