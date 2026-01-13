from celery import Celery
from app_config import get_settings

settings = get_settings()

celery_app = Celery("rag_book_bot")

celery_app.conf.update(
    broker_url=settings.REDIS_URL,
    result_backend=settings.REDIS_URL,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)