# Celery worker for async document processing.
# Not required for alpha; kept as skeleton for production upgrade.
from celery import Celery
import os
import asyncio

celery_app = Celery('tasks', broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"))

@celery_app.task
def process_document_task(file_path: str, filename: str, workspace_id: str, document_id: str):
    # Import here to avoid circular imports
    from main import process_document_async
    asyncio.run(process_document_async(file_path, filename, workspace_id, document_id))