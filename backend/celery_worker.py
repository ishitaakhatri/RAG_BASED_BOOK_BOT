import os
import logging
from celery_app import celery_app
# CRITICAL: Import tasks so they are registered with the worker
import tasks

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    redis_url = os.getenv("REDIS_URL", "UNKNOWN")
    logger.info(f"🚀 Celery Worker starting... connecting to broker: {redis_url}")
    
    # This allows running the worker directly with python celery_worker.py if needed
    celery_app.start()