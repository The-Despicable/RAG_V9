import os
import uuid
import json
import time
import tempfile
import magic
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query, BackgroundTasks
from pydantic import BaseModel, Field
from fastapi.responses import Response

from app.main import db_pool, redis_client, rebuild_bm25_for_workspace
from app.core.security import get_current_user
from app.config import get_settings
from app.middleware.request_id import get_request_id
from app.middleware.logging import logger, log_event
from app.ingestion import is_supported_type, get_file_extension
from app.utils.metrics import metrics
from app.services.pinecone import pinecone_service
from app.workers.tasks import process_document

router = APIRouter(prefix="/documents", tags=["documents"])
settings = get_settings()


class DocumentResponse(BaseModel):
    id: str
    filename: str
    status: str
    mime_type: str = None
    file_size_bytes: int = None
    created_at: datetime = None
    processed_at: datetime = None


class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int
    page: int
    page_size: int


class DocumentDetailResponse(BaseModel):
    id: str
    filename: str
    status: str
    mime_type: str
    file_size_bytes: int
    created_at: datetime
    processed_at: datetime = None
    chunks: dict = None


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user)
):
    await log_event("DOC_UPLOAD", user["user_id"], f"filename={file.filename}")
    
    total = 0
    first_chunk = None
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > settings.MAX_FILE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large, max {settings.MAX_FILE_MB}MB"
            )
        if first_chunk is None:
            first_chunk = chunk
            mime = magic.from_buffer(first_chunk, mime=True)
            if not is_supported_type(mime):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported file type: {mime}"
                )
            file_type = mime
    
    await file.seek(0)
    ext = file.filename.split(".")[-1].lower()
    document_id = str(uuid.uuid4())
    
    async with db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO documents (id, workspace_id, filename, mime_type, file_size_bytes, status) VALUES ($1, $2, $3, $4, $5, 'processing')",
            document_id, user["workspace_id"], file.filename, file_type, total
        )
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    
    task = process_document.apply_async(
        args=[tmp_path, file.filename, user["workspace_id"], document_id],
        task_id=document_id
    )
    
    metrics.document_processing.labels(status='queued').inc()
    
    return {
        "success": True,
        "data": {
            "document_id": document_id,
            "filename": file.filename,
            "status": "uploading",
            "task_id": task.id,
            "created_at": datetime.utcnow().isoformat() + "Z"
        },
        "request_id": get_request_id()
    }


@router.get("")
async def list_documents(
    user: dict = Depends(get_current_user),
    status_filter: Optional[str] = Query(None, alias="status"),
    sort: str = Query("created_at", regex="^(created_at|filename)$"),
    order: str = Query("desc", regex="^(asc|desc)$"),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    search: Optional[str] = Query(None)
):
    workspace_id = user["workspace_id"]
    
    where_clause = "WHERE workspace_id = $1"
    params = [workspace_id]
    
    if status_filter:
        where_clause += " AND status = $2"
        params.append(status_filter)
    
    if search:
        where_clause += f" AND filename ILIKE ${len(params) + 1}"
        params.append(f"%{search}%")
    
    offset = (page - 1) * limit
    order = "DESC" if order == "desc" else "ASC"
    
    async with db_pool.acquire() as conn:
        total = await conn.fetchval(
            f"SELECT COUNT(*) FROM documents {where_clause}",
            *params
        )
        
        rows = await conn.fetch(
            f"""SELECT d.id, d.filename, d.status, d.mime_type, d.file_size_bytes, d.created_at, d.processed_at,
                      dm.total_pages, dm.chunk_count, dm.vector_count
               FROM documents d
               LEFT JOIN document_metadata dm ON d.id = dm.document_id
               {where_clause}
               ORDER BY d.{sort} {order}
               LIMIT $${len(params) + 1} OFFSET $${len(params) + 2}""",
            *params, limit, offset
        )
    
    documents = []
    for r in rows:
        documents.append({
            "id": str(r["id"]),
            "filename": r["filename"],
            "status": r["status"],
            "mime_type": r["mime_type"],
            "file_size_bytes": r["file_size_bytes"],
            "created_at": r["created_at"].isoformat() + "Z" if r["created_at"] else None,
            "processed_at": r["processed_at"].isoformat() + "Z" if r["processed_at"] else None,
            "metadata": {
                "total_pages": r["total_pages"],
                "chunk_count": r["chunk_count"],
                "vector_count": r["vector_count"]
            } if r["chunk_count"] else None
        })
    
    return {
        "success": True,
        "data": {
            "documents": documents,
            "total": total,
            "page": page,
            "page_size": limit
        },
        "request_id": get_request_id()
    }


@router.get("/{document_id}")
async def get_document(document_id: str, user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        doc = await conn.fetchrow(
            """SELECT d.id, d.filename, d.status, d.mime_type, d.file_size_bytes, d.created_at, d.processed_at, d.error_message,
                      dm.total_pages, dm.chunk_count, dm.vector_count
               FROM documents d
               LEFT JOIN document_metadata dm ON d.id = dm.document_id
               WHERE d.id = $1 AND d.workspace_id = $2""",
            document_id, user["workspace_id"]
        )
        
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        chunks = await conn.fetch(
            "SELECT id, text, metadata FROM chunks WHERE document_id = $1 LIMIT 5",
            document_id
        )
    
    return {
        "success": True,
        "data": {
            "id": str(doc["id"]),
            "filename": doc["filename"],
            "status": doc["status"],
            "mime_type": doc["mime_type"],
            "file_size_bytes": doc["file_size_bytes"],
            "created_at": doc["created_at"].isoformat() + "Z" if doc["created_at"] else None,
            "processed_at": doc["processed_at"].isoformat() + "Z" if doc["processed_at"] else None,
            "error_message": doc["error_message"],
            "chunks": {
                "total": len(chunks),
                "preview": [
                    {
                        "id": str(c["id"]),
                        "text": c["text"][:200] if c["text"] else "",
                        "metadata": c["metadata"]
                    }
                    for c in chunks
                ]
            }
        },
        "request_id": get_request_id()
    }


@router.delete("/{document_id}")
async def delete_document(document_id: str, user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        doc = await conn.fetchrow(
            "SELECT id FROM documents WHERE id = $1 AND workspace_id = $2",
            document_id, user["workspace_id"]
        )
        
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        await conn.execute("DELETE FROM chunks WHERE document_id = $1", document_id)
        await conn.execute("DELETE FROM documents WHERE id = $1", document_id)
    
    try:
        await pinecone_service.delete(
            filter={"document_id": document_id},
            namespace=f"ws_{user['workspace_id']}"
        )
    except Exception as e:
        logger.warning(f"Failed to delete from Pinecone: {e}")
    
    await rebuild_bm25_for_workspace(user["workspace_id"])
    await log_event("DOC_DELETE", user["user_id"], f"document_id={document_id}")
    
    return {
        "success": True,
        "data": {
            "document_id": document_id,
            "status": "deleted"
        },
        "request_id": get_request_id()
    }


@router.get("/{document_id}/preview")
async def get_document_preview(document_id: str, user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        doc = await conn.fetchrow(
            "SELECT filename, mime_type FROM documents WHERE id = $1 AND workspace_id = $2",
            document_id, user["workspace_id"]
        )
        
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
    
    if not doc["mime_type"].startswith(("image/", "application/pdf")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Preview not available for this file type"
        )
    
    return {
        "success": True,
        "data": {
            "message": "Preview endpoint - implement file storage to serve binary",
            "filename": doc["filename"],
            "mime_type": doc["mime_type"]
        },
        "request_id": get_request_id()
    }