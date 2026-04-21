import uuid
import json
import time
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse

from app.main import db_pool, hybrid_search, stream_llm_sse
from app.core.security import get_current_user
from app.config import get_settings
from app.middleware.request_id import get_request_id
from app.middleware.logging import logger, log_event
from app.utils.metrics import metrics
from app.services.tokens import estimate_tokens, deduct_tokens, get_token_balance

router = APIRouter(prefix="/chat", tags=["chat"])
settings = get_settings()


class CreateConversationRequest(BaseModel):
    title: Optional[str] = None


class ConversationResponse(BaseModel):
    id: str
    title: Optional[str]
    status: str
    created_at: datetime
    updated_at: datetime


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    document_ids: List[str] = Field(default_factory=list)
    conversation_id: Optional[str] = None
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
async def chat_query(request: QueryRequest, user: dict = Depends(get_current_user)):
    start_time = time.time()
    message_id = str(uuid.uuid4())
    
    log_event("CHAT_QUERY", user["user_id"], f"query={request.query[:50]}")
    
    conversation_context = ""
    if request.conversation_id:
        async with db_pool.acquire() as conn:
            recent_messages = await conn.fetch(
                """SELECT role, content FROM messages
                   WHERE conversation_id = $1
                   ORDER BY created_at DESC
                   LIMIT 10""",
                request.conversation_id
            )
            if recent_messages:
                messages_text = "\n".join(
                    f"{m['role'].upper()}: {m['content']}" 
                    for m in reversed(recent_messages)
                )
                conversation_context = f"Previous conversation:\n{messages_text}\n\n"
    
    retrieved = await hybrid_search(request.query, user["workspace_id"], request.document_ids)
    
    if not retrieved:
        return StreamingResponse(
            iter([f"data: {json.dumps({'type': 'error', 'message': 'No relevant documents found'})}\n\n"]),
            media_type="text/event-stream"
        )
    
    input_tokens = await estimate_tokens(request.query + context)
    
    success, balance, token_msg = await deduct_tokens(
        db_pool, 
        user["workspace_id"], 
        message_id,
        input_tokens,
        0,
        "chat_query"
    )
    
    if not success:
        return StreamingResponse(
            iter([f"data: {json.dumps({'type': 'error', 'message': 'Token balance exceeded: {token_msg}'})}\n\n"]),
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
        
        if request.conversation_id:
            async with db_pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO messages (id, conversation_id, role, content)
                       VALUES ($1, $2, 'user', $3)""",
                    str(uuid.uuid4()), request.conversation_id, request.query
                )
        
        agen = stream_llm_sse(request.query, context, citations, message_id)
        response_text = ""
        try:
            async for sse in agen:
                yield sse
                if "chunk" in sse:
                    try:
                        data = json.loads(sse.replace("data: ", ""))
                        if data.get("type") == "chunk":
                            response_text += data.get("text", "")
                    except:
                        pass
        finally:
            await agen.aclose()
        
        response_time = (time.time() - start_time) * 1000
        confidence = sum(c["score"] for c in retrieved) / len(retrieved) if retrieved else 0
        
        output_tokens = await estimate_tokens(response_text)
        
        success, _, _ = await deduct_tokens(
            db_pool,
            user["workspace_id"],
            message_id,
            0,
            output_tokens,
            "chat_output"
        )
        
        async with db_pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO query_logs (id, workspace_id, user_id, query, retrieved_count, response_time_ms, confidence_score, provider_used, input_tokens, output_tokens)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, 'ollama', $8, $9)""",
                str(uuid.uuid4()), user["workspace_id"], user["user_id"], request.query, len(retrieved), response_time, confidence, input_tokens, output_tokens
            )
            
            if request.conversation_id and response_text:
                await conn.execute(
                    """INSERT INTO messages (id, conversation_id, role, content)
                       VALUES ($1, $2, 'assistant', $3)""",
                    str(uuid.uuid4()), request.conversation_id, response_text
                )
                await conn.execute(
                    "UPDATE conversations SET updated_at = NOW() WHERE id = $1",
                    request.conversation_id
                )
        
        yield f"data: {json.dumps({'type': 'citations', 'data': citations})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'message_id': message_id, 'confidence': confidence})}\n\n"
        
        metrics.track_latency("chat", request.query[:30], response_time / 1000)
    
    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/{message_id}/regenerate")
async def regenerate_response(message_id: str, request: RegenerateRequest, user: dict = Depends(get_current_user)):
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
async def give_feedback(message_id: str, request: FeedbackRequest, user: dict = Depends(get_current_user)):
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
    user: dict = Depends(get_current_user),
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


@router.post("/session")
async def create_conversation(
    request: CreateConversationRequest,
    user: dict = Depends(get_current_user)
):
    conversation_id = str(uuid.uuid4())
    title = request.title or f"Chat {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
    
    async with db_pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO conversations (id, workspace_id, user_id, title)
               VALUES ($1, $2, $3, $4)""",
            conversation_id, user["workspace_id"], user["user_id"], title
        )
    
    return {
        "success": True,
        "data": {
            "id": conversation_id,
            "title": title,
            "status": "active",
            "created_at": datetime.utcnow().isoformat() + "Z"
        },
        "request_id": get_request_id()
    }


@router.get("/session/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    user: dict = Depends(get_current_user)
):
    async with db_pool.acquire() as conn:
        conv = await conn.fetchrow(
            """SELECT id, title, status, created_at, updated_at
               FROM conversations 
               WHERE id = $1 AND workspace_id = $2""",
            conversation_id, user["workspace_id"]
        )
        
        if not conv:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        messages = await conn.fetch(
            """SELECT id, role, content, token_count, created_at
               FROM messages
               WHERE conversation_id = $1
               ORDER BY created_at ASC""",
            conversation_id
        )
    
    return {
        "success": True,
        "data": {
            "id": str(conv["id"]),
            "title": conv["title"],
            "status": conv["status"],
            "created_at": conv["created_at"].isoformat() + "Z" if conv["created_at"] else None,
            "updated_at": conv["updated_at"].isoformat() + "Z" if conv["updated_at"] else None,
            "messages": [
                {
                    "id": str(m["id"]),
                    "role": m["role"],
                    "content": m["content"],
                    "token_count": m["token_count"],
                    "created_at": m["created_at"].isoformat() + "Z" if m["created_at"] else None
                }
                for m in messages
            ]
        },
        "request_id": get_request_id()
    }


@router.get("/sessions")
async def list_conversations(
    user: dict = Depends(get_current_user),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT id, title, status, created_at, updated_at
               FROM conversations
               WHERE workspace_id = $1
               ORDER BY updated_at DESC
               LIMIT $2 OFFSET $3""",
            user["workspace_id"], limit, offset
        )
        
        total = await conn.fetchval(
            "SELECT COUNT(*) FROM conversations WHERE workspace_id = $1",
            user["workspace_id"]
        )
    
    return {
        "success": True,
        "data": {
            "conversations": [
                {
                    "id": str(r["id"]),
                    "title": r["title"],
                    "status": r["status"],
                    "created_at": r["created_at"].isoformat() + "Z" if r["created_at"] else None,
                    "updated_at": r["updated_at"].isoformat() + "Z" if r["updated_at"] else None
                }
                for r in rows
            ],
            "total": total
        },
        "request_id": get_request_id()
    }


@router.delete("/session/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    user: dict = Depends(get_current_user)
):
    async with db_pool.acquire() as conn:
        result = await conn.execute(
            """DELETE FROM conversations 
               WHERE id = $1 AND workspace_id = $2""",
            conversation_id, user["workspace_id"]
        )
        
        if result == "DELETE 0":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
    
    return {
        "success": True,
        "data": {"status": "deleted"},
        "request_id": get_request_id()
    }