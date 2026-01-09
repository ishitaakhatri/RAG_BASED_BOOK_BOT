from celery import Celery
import os

# Celery configuration: use Redis as broker and backend
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

celery_app = Celery('rag_book_bot', broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)

# Example task (replace with real ingestion logic)
@celery_app.task(bind=True)
def ingest_book_task(self, book_data):
    # Simulate ingestion work
    import time
    time.sleep(5)  # Replace with actual ingestion logic
    return {'status': 'completed', 'book': book_data}
