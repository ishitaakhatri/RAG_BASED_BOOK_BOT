import os
import boto3
from celery import Task
from celery_app import celery_app
from app_config import get_settings
from rag_based_book_bot.document_ingestion.enhanced_ingestion import EnhancedBookIngestorPaddle, IngestorConfig
from rag_based_book_bot.document_ingestion.progress_tracker import get_tracker

settings = get_settings()

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
            grobid_url=settings.GROBID_URL 
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