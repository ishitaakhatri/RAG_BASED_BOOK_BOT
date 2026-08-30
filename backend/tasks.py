import os
from azure.storage.blob import BlobServiceClient
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
        dummy_vector = [1.0] * 1024  # Match embedding dimension
        
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
def ingest_book_task(self, task_id: str, blob_name: str, book_title: str, author: str):
    """
    Celery task that downloads a PDF from Azure Blob Storage and ingests it.
    """
    tracker = get_tracker(task_id)
    local_path = f"/tmp/{task_id}.pdf"
    
    # Initialize Azure Blob client INSIDE the task for thread safety
    blob_service = BlobServiceClient.from_connection_string(settings.AZURE_STORAGE_CONNECTION_STRING)
    blob_container = blob_service.get_container_client(settings.AZURE_STORAGE_CONTAINER)
    
    try:
        tracker.add_log(f"📥 Downloading {blob_name} from Azure Blob Storage...")
        
        # Download from Azure Blob to local worker temp storage
        blob_client = blob_container.get_blob_client(blob_name)
        with open(local_path, "wb") as download_file:
            blob_data = blob_client.download_blob()
            blob_data.readinto(download_file)
        
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