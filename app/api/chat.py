import uuid
import json
import time
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse

from app.main import db_pool, hybrid_search, stream_llm_sse
from app.config import get_settings
from app.middleware.request_id import get_request_id
from app.middleware.logging import logger, log_event
from app.utils.metrics import metrics

router = APIRouter(prefix="/chat", tags=["chat"])
settings = get_settings()


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    document_ids: List[str] = Field(default_factory=list)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_k: int = Field(default=10, ge=1, le=50)
    max_tokens: int = Field(default=2000, ge=1, le=4000)
    metadata_filters: dict = None


class RegenerateRequest(BaseModel):
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)


class FeedbackRequest(BaseModel):
    rating: int = Field(..., ge=1, le=5)
    feedback: Optional[str] = None
    was_helpful: bool = True


@router.post("")
async def chat_query(request: QueryRequest, user: dict = Depends(lambda: {"user_id": "test", "workspace_id": "test"})):
    start_time = time.time()
    message_id = str(uuid.uuid4())
    
    log_event("CHAT_QUERY", user["user_id"], f"query={request.query[:50]}")
    
    retrieved = await hybrid_search(request.query, user["workspace_id"], request.document_ids)
    
    if not retrieved:
        return StreamingResponse(
            iter([f"data: {json.dumps({'type': 'error', 'message': 'No relevant documents found'})}\n\n"]),
            media_type="text/event-stream"
        )
    
    context = ""
    citations = []
    seen = set()
    for i, chunk in enumerate(retrieved):
        doc_id = chunk["metadata"].get("document_id", "unknown")
        if doc_id not in seen:
            citations.append({
                "document_id": doc_id,
                "snippet": chunk["text"][:200]
            })
            seen.add(doc_id)
        context += f"[Source {i+1}] {chunk['text']}\n\n"
    
    async def generate():
        nonlocal start_time
        agen = stream_llm_sse(request.query, context, citations, message_id)
        try:
            async for sse in agen:
                yield sse
        finally:
            await agen.aclose()
        
        response_time = (time.time() - start_time) * 1000
        confidence = sum(c["score"] for c in retrieved) / len(retrieved) if retrieved else 0
        
        async with db_pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO query_logs (id, workspace_id, user_id, query, retrieved_count, response_time_ms, confidence_score, provider_used)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, 'ollama')""",
                str(uuid.uuid4()), user["workspace_id"], user["user_id"], request.query, len(retrieved), response_time, confidence
            )
        
        yield f"data: {json.dumps({'type': 'citations', 'data': citations})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'message_id': message_id, 'confidence': confidence})}\n\n"
        
        metrics.track_latency("chat", request.query[:30], response_time / 1000)
    
    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/{message_id}/regenerate")
async def regenerate_response(message_id: str, request: RegenerateRequest, user: dict = Depends(lambda: {"user_id": "test", "workspace_id": "test"})):
    async with db_pool.acquire() as conn:
        last_query = await conn.fetchrow(
            "SELECT query FROM query_logs WHERE workspace_id = $1 ORDER BY created_at DESC LIMIT 1",
            user["workspace_id"]
        )
        
        if not last_query:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No previous query found to regenerate"
            )
    
    new_request = QueryRequest(
        query=last_query["query"],
        temperature=request.temperature
    )
    
    return await chat_query(new_request, user)


@router.post("/{message_id}/feedback")
async def give_feedback(message_id: str, request: FeedbackRequest, user: dict = Depends(lambda: {"user_id": "test"})):
    async with db_pool.acquire() as conn:
        await conn.execute(
            """UPDATE query_logs SET feedback = $1 
               WHERE id = $2 AND workspace_id = $3""",
            json.dumps({"rating": request.rating, "feedback": request.feedback, "was_helpful": request.was_helpful}),
            message_id, user["workspace_id"]
        )
    
    return {
        "success": True,
        "data": {
            "message_id": message_id,
            "feedback_recorded": True
        },
        "request_id": get_request_id()
    }


@router.get("/history")
async def get_chat_history(
    user: dict = Depends(lambda: {"workspace_id": "test"}),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    document_id: Optional[str] = Query(None)
):
    async with db_pool.acquire() as conn:
        if document_id:
            rows = await conn.fetch(
                """SELECT ql.id, ql.query, ql.retrieved_count, ql.confidence_score, ql.created_at
                   FROM query_logs ql
                   JOIN chunks c ON c.document_id = $3
                   WHERE ql.workspace_id = $1
                   ORDER BY ql.created_at DESC
                   LIMIT $2 OFFSET $3""",
                user["workspace_id"], limit, document_id
            )
        else:
            rows = await conn.fetch(
                """SELECT id, query, retrieved_count, confidence_score, created_at
                   FROM query_logs
                   WHERE workspace_id = $1
                   ORDER BY created_at DESC
                   LIMIT $2 OFFSET $3""",
                user["workspace_id"], limit, offset
            )
        
        total = await conn.fetchval(
            "SELECT COUNT(*) FROM query_logs WHERE workspace_id = $1",
            user["workspace_id"]
        )
    
    messages = []
    for r in rows:
        messages.append({
            "id": str(r["id"]),
            "role": "user",
            "content": r["query"],
            "created_at": r["created_at"].isoformat() + "Z" if r["created_at"] else None
        })
    
    return {
        "success": True,
        "data": {
            "messages": messages,
            "total": total
        },
        "request_id": get_request_id()
    }