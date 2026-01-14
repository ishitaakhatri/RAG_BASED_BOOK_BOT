import os
import boto3
import time
import hashlib
from celery import Task
from celery_app import celery_app
from app_config import get_settings
from rag_based_book_bot.document_ingestion.enhanced_ingestion import EnhancedBookIngestorPaddle, IngestorConfig
from rag_based_book_bot.document_ingestion.progress_tracker import get_tracker
from pinecone import Pinecone

settings = get_settings()

def store_book_metadata(book_title: str, author: str, total_chunks: int, code_chunks: int = 0):
    """
    Registers the book in the Pinecone metadata namespace so it appears in the frontend list.
    """
    try:
        # Re-initialize Pinecone inside the worker process to avoid thread-safety issues
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        index = pc.Index(settings.PINECONE_INDEX_NAME)
        
        # Determine metadata namespace (fallback to 'books_metadata' if not set clearly)
        metadata_namespace = settings.PINECONE_NAMESPACE.replace("_rag", "_metadata")
        if not metadata_namespace.endswith("metadata"):
             metadata_namespace = "books_metadata"

        # Create a unique ID for the book metadata entry
        book_id = hashlib.md5(book_title.encode()).hexdigest()
        
        print(f"📝 Registering book metadata for '{book_title}' in {metadata_namespace}...")
        
        # Upsert the "Book Card"
        # We use a dummy vector of all 1.0s because we only query this by metadata or list all
        dummy_vector = [1.0] * settings.vector_db.dimension
        
        index.upsert(
            vectors=[{
                "id": book_id,
                "values": dummy_vector,
                "metadata": {
                    "book_title": book_title,
                    "author": author,
                    "total_chunks": total_chunks,
                    "code_chunks": code_chunks,
                    "text_chunks": total_chunks - code_chunks,
                    "indexed_at": time.time()
                }
            }],
            namespace=metadata_namespace
        )
        print("✅ Metadata registered successfully.")
    except Exception as e:
        print(f"⚠️ Failed to store metadata: {e}")

class IngestionTask(Task):
    """Base Task class to handle global error logging for ingestion"""
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        tracker = get_tracker(task_id)
        if tracker:
            tracker.add_error(f"Worker failure: {str(exc)}")
            tracker.finish(success=False)

@celery_app.task(bind=True, base=IngestionTask, name="ingest_book_task")
def ingest_book_task(self, task_id: str, s3_key: str, book_title: str, author: str):
    """
    Celery task that downloads a PDF from S3 and ingests it.
    """
    tracker = get_tracker(task_id)
    local_path = f"/tmp/{task_id}.pdf"
    
    # --- THREAD SAFETY FIX ---
    # Initialize S3 Client INSIDE the task. Boto3 clients are not thread-safe.
    # This ensures each worker process gets its own clean connection.
    s3_client = boto3.client(
        's3',
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_DEFAULT_REGION
    )
    
    try:
        tracker.add_log(f"📥 Downloading {s3_key} from S3...")
        
        # Download from S3 to local worker temp storage
        s3_client.download_file(settings.S3_BUCKET_NAME, s3_key, local_path)
        
        tracker.add_log("⚙️ Initializing AI Ingestor...")
        
        # Ensure Grobid URL points to the Grobid service (usually on the same worker node or network)
        config = IngestorConfig(
            use_grobid=True, 
        )
        
        ingestor = EnhancedBookIngestorPaddle(config=config)
        
        # Run Ingestion
        # This will update Redis internally via the tracker
        result = ingestor.ingest_book(
            pdf_path=local_path,
            book_title=book_title,
            author=author,
            task_id=task_id 
        )
        
        # --- CRITICAL FIX: Store Metadata ---
        tracker.add_log("📝 Registering book in library catalog...")
        store_book_metadata(
            book_title=book_title, 
            author=author, 
            total_chunks=result.get("chunks", 0),
            code_chunks=result.get("code_chunks", 0)
        )
        # ------------------------------------
        
        tracker.finish(success=True)
        return result

    except Exception as e:
        # Detailed logging before re-raising
        if tracker:
            tracker.add_error(f"Task Exception: {str(e)}")
        # We re-raise so Celery marks the task as FAILED in its own internal backend
        raise e
    finally:
        # Cleanup temporary file to save disk space
        if os.path.exists(local_path):
            os.remove(local_path)