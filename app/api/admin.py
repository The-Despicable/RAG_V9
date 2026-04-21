import time
import json
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, Response
from fastapi.responses import JSONResponse
import httpx

from app.main import db_pool, redis_client
from app.config import get_settings
from app.core.security import get_current_user
from app.middleware.request_id import get_request_id
from app.utils.metrics import metrics
from app.services.pinecone import pinecone_service

router = APIRouter(prefix="/admin", tags=["admin"])
settings = get_settings()


@router.get("/health")
async def health_check():
    checks = {}
    
    try:
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"error: {str(e)}"
    
    try:
        await redis_client.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {str(e)}"
    
    try:
        await pinecone_service.describe_stats()
        checks["pinecone"] = "ok"
    except Exception as e:
        checks["pinecone"] = f"error: {str(e)}"
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(
                f"{settings.OLLAMA_EMBEDDING_URL}/api/embeddings",
                json={"model": settings.OLLAMA_EMBEDDING_MODEL, "prompt": "test"},
                timeout=5.0
            )
        checks["ollama"] = "ok"
    except Exception as e:
        checks["ollama"] = f"error: {str(e)}"
    
    all_ok = all("ok" in v for v in checks.values())
    status_code = 200 if all_ok else 503
    
    return JSONResponse(
        status_code=status_code,
        content={
            "success": True,
            "data": {
                "status": "ok" if all_ok else "degraded",
                "checks": checks,
                "uptime_seconds": int(time.time() - 1700000000),
                "version": "1.0.0"
            },
            "request_id": get_request_id()
        }
    )


@router.get("/metrics")
async def get_metrics():
    return Response(
        content=metrics.generate_metrics(),
        media_type="text/plain"
    )


@router.get("/dashboard")
async def get_dashboard(user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        active_users_24h = await conn.fetchval(
            "SELECT COUNT(DISTINCT workspace_id) FROM query_logs WHERE created_at > NOW() - INTERVAL '24 hours'"
        )
        
        queries_24h = await conn.fetchval(
            "SELECT COUNT(*) FROM query_logs WHERE created_at > NOW() - INTERVAL '24 hours'"
        )
        
        tokens_24h = await conn.fetchval(
            "SELECT COALESCE(SUM(tokens_deducted), 0) FROM token_ledger WHERE created_at > NOW() - INTERVAL '24 hours'"
        )
        
        provider_stats = await conn.fetchrow(
            """SELECT provider_used, COUNT(*) as count, AVG(response_time_ms) as avg_latency
               FROM query_logs 
               WHERE created_at > NOW() - INTERVAL '7 days'
               GROUP BY provider_used"""
        )
        
        total_docs = await conn.fetchval("SELECT COUNT(*) FROM documents")
        total_chunks = await conn.fetchval("SELECT COUNT(*) FROM chunks")
        
        error_rate_query = await conn.fetchval(
            """SELECT COUNT(*)::float / NULLIF(COUNT(*), 0)
               FROM query_logs WHERE created_at > NOW() - INTERVAL '24 hours'"""
        )
    
    return {
        "success": True,
        "data": {
            "active_users_24h": active_users_24h or 0,
            "queries_24h": queries_24h or 0,
            "tokens_burned_24h": tokens_24h or 0,
            "provider_stats": {
                provider_stats["provider_used"]: {
                    "count": provider_stats["count"],
                    "avg_latency_ms": int(provider_stats["avg_latency"] or 0)
                }
            } if provider_stats else {},
            "error_rate": round(error_rate_query or 0, 4),
            "system_health": "ok",
            "total_documents": total_docs or 0,
            "total_chunks": total_chunks or 0
        },
        "request_id": get_request_id()
    }


@router.get("/logs")
async def get_logs(
    limit: int = 100,
    offset: int = 0,
    level: Optional[str] = None
):
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT id, endpoint, method, status_code, duration_ms, error_type, created_at
               FROM api_calls
               ORDER BY created_at DESC
               LIMIT $1 OFFSET $2""",
            limit, offset
        )
    
    return {
        "success": True,
        "data": {
            "logs": [
                {
                    "id": str(r["id"]),
                    "endpoint": r["endpoint"],
                    "method": r["method"],
                    "status_code": r["status_code"],
                    "duration_ms": r["duration_ms"],
                    "error_type": r["error_type"],
                    "created_at": r["created_at"].isoformat() + "Z" if r["created_at"] else None
                }
                for r in rows
            ]
        },
        "request_id": get_request_id()
    }


@router.get("/providers")
async def get_provider_health():
    providers = {}
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{settings.OLLAMA_EMBEDDING_URL}/api/embeddings",
                json={"model": settings.OLLAMA_EMBEDDING_MODEL, "prompt": "health"},
                timeout=5.0
            )
            providers["ollama"] = "ok" if resp.status_code == 200 else "error"
    except Exception as e:
        providers["ollama"] = f"error: {str(e)}"
    
    if settings.OPENROUTER_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    "https://openrouter.ai/api/v1/models",
                    headers={"Authorization": f"Bearer {settings.OPENROUTER_API_KEY}"},
                    timeout=5.0
                )
                providers["openrouter"] = "ok" if resp.status_code == 200 else "error"
        except Exception as e:
            providers["openrouter"] = f"error: {str(e)}"
    
    if settings.OPENAI_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
                    timeout=5.0
                )
                providers["openai"] = "ok" if resp.status_code == 200 else "error"
        except Exception as e:
            providers["openai"] = f"error: {str(e)}"
    
    return {
        "success": True,
        "data": {"providers": providers},
        "request_id": get_request_id()
    }