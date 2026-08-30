#   progress_tracker.py
"""
Progress Tracker for Document Ingestion - Distributed Version
Uses Redis to store state accessible by both API and Worker nodes.
"""

import logging
import json
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from redis import Redis
from app_config import get_settings

logger = logging.getLogger("progress_tracker")
settings = get_settings()

# Initialize Redis connection
import ssl as _ssl
try:
    _redis_kwargs = {"decode_responses": True}
    if settings.REDIS_URL.startswith("rediss://"):
        _redis_kwargs["ssl_cert_reqs"] = _ssl.CERT_REQUIRED
    redis_client = Redis.from_url(settings.REDIS_URL, **_redis_kwargs)
except Exception as e:
    logger.error(f"Failed to connect to Redis: {e}")
    redis_client = None

EXPIRY_TIME = 3600  # 1 hour

@dataclass
class ProgressState:
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
    error: Optional[str] = None
    logs: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

class ProgressTracker:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.redis_key = f"task:{task_id}"
        # Always load fresh state from Redis
        self.state = self._load_state()

    def _load_state(self) -> ProgressState:
        if not redis_client:
            return ProgressState()
        
        raw = redis_client.get(self.redis_key)
        if raw:
            try:
                return ProgressState(**json.loads(raw))
            except Exception:
                pass
        return ProgressState()

    def _save(self):
        if not redis_client:
            return
            
        # Calculate timing stats before saving
        self.state.elapsed_time = time.time() - self.state.start_time
        
        # Calculate ETA if processing
        if self.state.total_pages > 0 and self.state.current_page > 0:
             self.state.speed_pages_per_sec = self.state.current_page / max(0.1, self.state.elapsed_time)
             remaining = self.state.total_pages - self.state.current_page
             if self.state.speed_pages_per_sec > 0:
                 self.state.estimated_time_remaining = remaining / self.state.speed_pages_per_sec

        redis_client.setex(
            self.redis_key, 
            EXPIRY_TIME, 
            json.dumps(self.state.to_dict())
        )

    # --- Ingestion Lifecycle Methods ---

    def start_ingestion(self, pdf_path: str, total_pages: int, book_title: str, author: str):
        self.state.status = "parsing_pdf"
        self.state.total_pages = total_pages
        self.state.book_title = book_title
        self.state.author = author
        self.state.current_task = f"Processing {book_title}"
        self.state.start_time = time.time()
        self.state.percentage = 0.0
        self.add_log(f"Started ingestion for {book_title} ({total_pages} pages)")
        self._save()

    def update_total_pages(self, total_pages: int):
        self.state.total_pages = total_pages
        self._save()

    def update_batch(self, batch_num: int, total_batches: int, current_page: int):
        self.state.status = "chunking"
        self.state.current_batch = batch_num
        self.state.total_batches = total_batches
        self.state.current_page = current_page
        
        # Map chunking phase to 0-45% of total progress
        progress_ratio = current_page / max(1, self.state.total_pages)
        self.state.percentage = progress_ratio * 45.0
        
        self.state.current_task = f"Chunking batch {batch_num}/{total_batches}"
        self._save()

    def start_chunking(self):
        self.state.status = "chunking"
        self.state.current_task = "Starting semantic chunking..."
        self._save()

    def update_chunks(self, count: int):
        self.state.chunks_created = count
        self.state.status = "embedding"
        self.state.percentage = 45.0
        self.state.current_task = f"Generated {count} chunks"
        self._save()

    def start_embedding(self):
        self.state.status = "embedding"
        self.state.current_task = "Generating embeddings..."
        self.state.percentage = 50.0
        self._save()

    def update_embeddings(self, count: int):
        self.state.embeddings_generated = count
        # Map embedding phase to 50-85%
        # We approximate progress here since we stream batches
        if self.state.percentage < 85.0:
            self.state.percentage += 1.0
        self.state.current_task = f"Generated {count} embeddings"
        self._save()

    def start_upsert(self):
        self.state.status = "upserting"
        self.state.percentage = 85.0
        self.state.current_task = "Upserting to Vector DB..."
        self._save()

    def update_upsert(self, count: int):
        self.state.vectors_upserted = count
        # Map upsert phase to 85-95%
        if self.state.percentage < 95.0:
             self.state.percentage += 0.5
        self.state.current_task = f"Upserted {count} vectors"
        self._save()

    def finish(self, success: bool = True):
        self.state.status = "completed" if success else "failed"
        self.state.percentage = 100.0 if success else self.state.percentage
        self.state.current_task = "Ingestion Complete" if success else "Ingestion Failed"
        self.state.estimated_time_remaining = 0.0
        self._save()

    def add_error(self, msg: str):
        self.state.error = msg
        self.add_log(f"ERROR: {msg}", "ERROR")
        self._save()

    def add_log(self, message: str, level: str = "INFO"):
        ts = datetime.now().strftime("%H:%M:%S")
        self.state.logs.append(f"[{ts}] {level}: {message}")
        # Keep log size manageable in Redis
        if len(self.state.logs) > 50:
            self.state.logs = self.state.logs[-50:]
        self._save()

    # --- Backward Compatibility Stubs ---
    # These are needed so enhanced_ingestion.py doesn't break if it calls them
    
    def on_progress(self, callback):
        pass # Not used in distributed mode

    def remove_callback(self, callback):
        pass
    
    def set_loop(self, loop):
        pass

# Factory functions
def get_tracker(task_id: str):
    return ProgressTracker(task_id)

def create_tracker(task_id: str):
    return ProgressTracker(task_id)

def remove_tracker(task_id: str):
    pass # Redis expiry handles cleanup