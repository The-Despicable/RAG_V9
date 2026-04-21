import os
import uuid
import json
import asyncio
import tempfile
import time
import pickle
import logging
from datetime import datetime
from typing import List, Tuple, Dict, Optional, AsyncGenerator
from contextlib import asynccontextmanager

import jwt
import bcrypt
import magic
import redis.asyncio as redis

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rag")

# ========== AUDIT LOGGING ==========
async def log_event(event_type: str, user_id: str, details: str = ""):
    """Simple event logging (no database overhead yet)."""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "type": event_type,
        "user": user_id,
        "details": details
    }
    logger.info(f"EVENT: {json.dumps(log_entry)}")

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, HTMLResponse, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import asyncpg
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
import numpy as np
import httpx
# from sentence_transformers import CrossEncoder
import tiktoken
from prometheus_client import Counter, Histogram, generate_latest, REGISTRY

# Local provider for LLM (Ollama primary, OpenAI fallback)
from app.providers import provider_stream, get_embedding

load_dotenv()

# ========== CONFIG ==========
DATABASE_URL = os.getenv("DATABASE_URL")
REDIS_URL = os.getenv("REDIS_URL")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "rag-index")
JWT_SECRET = os.getenv("JWT_SECRET")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))

OLLAMA_URL = os.getenv("OLLAMA_URL", "https://ollama.com")
OLLAMA_EMBEDDING_URL = os.getenv("OLLAMA_EMBEDDING_URL", "http://host.docker.internal:11434")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

if not JWT_SECRET or JWT_SECRET == "change-me-in-production":
    raise ValueError("JWT_SECRET must be set to a secure value")
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K_DENSE = 20
TOP_K_SPARSE = 20
TOP_K_FUSED = 10
TOP_K_FINAL = 5
MAX_FILE_MB = 10
RATE_LIMIT_REQUESTS = 30
RATE_LIMIT_WINDOW = 60

# ========== QUOTAS ==========
MAX_DOCUMENTS_PER_WORKSPACE = 1000
MAX_STORAGE_GB_PER_WORKSPACE = 100.0

# ========== CLIENTS ==========
pc = Pinecone(api_key=PINECONE_API_KEY)
redis_client = None
db_pool = None

# ========== BM25 CACHE (Redis-backed) ==========
# bm25_cache removed, using Redis for multi-worker support

# ========== SPLITTER & RERANKER ==========
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=lambda t: len(tiktoken.get_encoding("cl100k_base").encode(t)),
)
# reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ========== METRICS ==========
REQUESTS = Counter('http_requests_total', 'Total requests', ['method', 'endpoint'])
LATENCY = Histogram('http_request_duration_seconds', 'Request latency')

# ========== DB INIT ==========
async def init_db():
    global db_pool
    db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
    await db_pool.execute("""
        CREATE TABLE IF NOT EXISTS workspaces (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    await db_pool.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            workspace_id UUID NOT NULL REFERENCES workspaces(id),
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    await db_pool.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            workspace_id UUID NOT NULL REFERENCES workspaces(id),
            filename TEXT NOT NULL,
            status TEXT DEFAULT 'processing',
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    await db_pool.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            document_id UUID NOT NULL REFERENCES documents(id),
            text TEXT NOT NULL,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    await db_pool.execute("""
        CREATE TABLE IF NOT EXISTS query_logs (
            id UUID PRIMARY KEY,
            workspace_id UUID NOT NULL,
            query TEXT NOT NULL,
            retrieved_count INT,
            response_tokens INT,
            response_time_ms INT,
            confidence_score FLOAT,
            feedback TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    # Indexes (safe; run once)
    await db_pool.execute("CREATE INDEX IF NOT EXISTS idx_docs_workspace ON documents(workspace_id)")
    await db_pool.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id)")
    await db_pool.execute("CREATE INDEX IF NOT EXISTS idx_logs_workspace_created ON query_logs(workspace_id, created_at DESC)")
    await db_pool.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email)")

    if PINECONE_INDEX not in pc.list_indexes().names():
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    logger.info("DB ready")

# ========== REDIS ==========
async def init_redis():
    global redis_client
    redis_client = redis.from_url(REDIS_URL)
    await redis_client.ping()
    logger.info("Redis ready")

# ========== AUTH ==========
security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return {
            "user_id": payload["sub"],
            "email": payload["email"],
            "workspace_id": payload["workspace_id"]
        }
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ========== RATE LIMITING ==========
async def rate_limit_check(user_id: str):
    key = f"rl:{user_id}"
    current = await redis_client.incr(key)
    if current == 1:
        await redis_client.expire(key, RATE_LIMIT_WINDOW)
    if current > RATE_LIMIT_REQUESTS:
        raise HTTPException(429, f"Rate limit exceeded: max {RATE_LIMIT_REQUESTS} per {RATE_LIMIT_WINDOW}s")

async def check_workspace_quotas(workspace_id: str):
    """Check if workspace has hit document/storage quotas."""
    doc_count = await db_pool.fetchval(
        "SELECT COUNT(*) FROM documents WHERE workspace_id = $1",
        workspace_id
    )
    if doc_count >= MAX_DOCUMENTS_PER_WORKSPACE:
        raise HTTPException(
            507,
            f"Document quota exceeded. Maximum {MAX_DOCUMENTS_PER_WORKSPACE} documents per workspace."
        )
    logger.debug(f"Workspace {workspace_id}: {doc_count}/{MAX_DOCUMENTS_PER_WORKSPACE} documents")

# ========== REFRESH TOKEN (Proper Rotation) ==========
async def create_refresh_token(user_id: str) -> str:
    """Create a single-use refresh token with rotation tracking."""
    token = str(uuid.uuid4())
    payload = {
        "user_id": user_id,
        "issued_at": int(time.time()),
        "rotation_count": 0
    }
    # 7-day TTL
    await redis_client.setex(f"refresh:{token}", 86400 * 7, json.dumps(payload))
    return token

async def validate_refresh_token(token: str) -> Optional[dict]:
    """Validate refresh token and return payload. Does NOT consume the token."""
    try:
        data = await redis_client.get(f"refresh:{token}")
        if not data:
            return None
        return json.loads(data)
    except Exception as e:
        logger.error(f"Token validation failed: {e}")
        return None

async def consume_refresh_token(token: str) -> Optional[str]:
    """Consume (invalidate) old token. Returns user_id if valid, None otherwise."""
    payload = await validate_refresh_token(token)
    if not payload:
        return None

    user_id = payload["user_id"]
    rotations = payload.get("rotation_count", 0)

    # Check for token reuse attack (>100 rotations in 7 days)
    if rotations > 100:
        logger.warning(f"Token reuse attack detected: {user_id} after {rotations} rotations")
        await redis_client.delete(f"refresh:{token}")
        return None

    # Consume the token (one-time use)
    await redis_client.delete(f"refresh:{token}")
    return user_id

async def get_user_from_refresh(token: str) -> Optional[str]:
    """Legacy compatibility - use consume_refresh_token instead."""
    return await consume_refresh_token(token)

# ========== UTILITIES ==========
async def rebuild_bm25_for_workspace(workspace_id: str):
    rows = await db_pool.fetch("""
        SELECT c.id, c.text, c.metadata
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE d.workspace_id = $1
    """, workspace_id)
    if not rows:
        await redis_client.delete(f"bm25:{workspace_id}")
        return
    texts = [r["text"] for r in rows]
    ids = [str(r["id"]) for r in rows]  # Convert UUID to string for JSON serialization
    metas = [json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"] for r in rows]
    tokenized = [t.lower().split() for t in texts]
    text_map = {ids[i]: texts[i] for i in range(len(ids))}
    meta_map = {ids[i]: metas[i] for i in range(len(ids))}
    # Store tokenized data for BM25 reconstruction (JSON-safe, no pickle)
    payload = {
        "tokenized": tokenized,
        "text_map": text_map,
        "meta_map": meta_map
    }
    await redis_client.setex(f"bm25:{workspace_id}", 86400 * 7, json.dumps(payload))
    logger.info(f"BM25 index rebuilt for workspace {workspace_id} with {len(tokenized)} chunks")

async def get_bm25_for_workspace(workspace_id: str):
    """Retrieve BM25 index from Redis cache, rebuild if not found."""
    try:
        data = await redis_client.get(f"bm25:{workspace_id}")
        if data:
            payload = json.loads(data)
            tokenized = payload["tokenized"]
            text_map = payload["text_map"]
            meta_map = payload["meta_map"]
            bm25 = BM25Okapi(tokenized)
            return (bm25, text_map, meta_map)
    except Exception as e:
        logger.warning(f"Failed to load BM25 from cache: {e}")
    # Rebuild if not in cache or on error
    await rebuild_bm25_for_workspace(workspace_id)
    data = await redis_client.get(f"bm25:{workspace_id}")
    if data:
        payload = json.loads(data)
        tokenized = payload["tokenized"]
        text_map = payload["text_map"]
        meta_map = payload["meta_map"]
        bm25 = BM25Okapi(tokenized)
        return (bm25, text_map, meta_map)
    return None

# ========== HYBRID SEARCH ==========
async def dense_search(query: str, workspace_id: str, doc_ids: List[str], top_k: int) -> List[Tuple[str, float]]:
    emb = await get_embedding(query)
    namespace = f"ws_{workspace_id}"
    filter_dict = {"document_id": {"$in": doc_ids}} if doc_ids else {}
    resp = pc.Index(PINECONE_INDEX).query(
        namespace=namespace,
        vector=emb,
        top_k=top_k,
        include_metadata=True,
        filter=filter_dict
    )
    return [(m.metadata["chunk_id"], m.score) for m in resp.matches if m.metadata.get("chunk_id")]

async def sparse_search(query: str, workspace_id: str, doc_ids: List[str], top_k: int) -> List[Tuple[str, float]]:
    """Sparse search using BM25 keyword matching."""
    bm25_data = await get_bm25_for_workspace(workspace_id)
    if bm25_data is None:
        return []
    bm25, text_map, meta_map = bm25_data
    if bm25 is None:
        return []
    tokens = query.lower().split()
    if not tokens:
        return []
    scores = bm25.get_scores(tokens)
    # Get top indices, but ensure we don't exceed array bounds
    top_indices = np.argsort(scores)[-top_k*2:][::-1]
    chunk_ids = list(text_map.keys())
    results = []
    for i in top_indices:
        # Bounds check before accessing array
        if i < 0 or i >= len(chunk_ids):
            logger.warning(f"Index out of bounds: {i} >= {len(chunk_ids)}")
            continue
        cid = chunk_ids[i]
        score = scores[i]
        if score > 0:
            # Filter by document if specified
            if not doc_ids or meta_map.get(cid, {}).get("document_id") in doc_ids:
                results.append((cid, float(score)))
            # Stop when we have enough results
            if len(results) >= top_k:
                break
    return results

def rrf_fusion(dense: List[Tuple[str, float]], sparse: List[Tuple[str, float]], k=60) -> List[Tuple[str, float]]:
    scores = {}
    for rank, (cid, _) in enumerate(dense, 1):
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank)
    for rank, (cid, _) in enumerate(sparse, 1):
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


async def rerank_chunks(query: str, chunk_ids: List[str], workspace_id: str) -> List[Tuple[str, float]]:
    """Placeholder reranking - returns original order with dummy score 1.0"""
    return [(cid, 1.0) for cid in chunk_ids]


async def hybrid_search(query: str, workspace_id: str, doc_ids: List[str]) -> List[Dict]:
    dense = await dense_search(query, workspace_id, doc_ids, TOP_K_DENSE)
    sparse = await sparse_search(query, workspace_id, doc_ids, TOP_K_SPARSE)
    fused = rrf_fusion(dense, sparse)[:TOP_K_FUSED]
    reranked = await rerank_chunks(query, [cid for cid, _ in fused], workspace_id)
    final = reranked[:TOP_K_FINAL]
    bm25_data = await get_bm25_for_workspace(workspace_id)
    _, text_map, meta_map = bm25_data if bm25_data else (None, {}, {})
    return [{"id": cid, "text": text_map.get(cid, ""), "score": score, "metadata": meta_map.get(cid, {})} for cid, score in final]

# ========== DOCUMENT PROCESSING ==========
SUPPORTED_MIME_TYPES = [
    "text/plain", "application/pdf", "text/markdown", "text/x-script.python", "text/csv",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "text/html", "application/epub+zip",
    "audio/mpeg", "audio/wav", "audio/mp4", "audio/flac",
    "image/png", "image/jpeg", "image/gif", "image/bmp",
]

from app.ingestion import (
    extract_text_from_file, extract_images_from_pdf, describe_image,
    transcribe_audio, get_file_extension, is_supported_type,
    extract_text_from_image_bytes, extract_text_from_image_path,
    ENABLE_VISION, ENABLE_WHISPER
)

async def process_document_async(file_path: str, filename: str, workspace_id: str, document_id: str):
    try:
        mime = magic.from_file(file_path, mime=True)
        ext = get_file_extension(mime, filename)
        if not is_supported_type(mime):
            os.unlink(file_path)
            raise ValueError(f"Unsupported file type: {mime}")
        async with db_pool.acquire() as conn:
            await conn.execute("UPDATE documents SET status = $1 WHERE id = $2", 'extracting_text', document_id)
        all_chunks = []
        all_metadata = []
        
        if mime in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
            from app.document_handlers import TabularDocumentHandler
            text_chunks = TabularDocumentHandler.extract_from_excel(file_path)
            for i, chunk_text in enumerate(text_chunks):
                all_chunks.append(chunk_text)
                all_metadata.append({"index": i, "document_id": document_id, "type": "excel"})
        elif mime == "text/csv":
            from app.document_handlers import TabularDocumentHandler
            text_chunks = TabularDocumentHandler.extract_from_csv(file_path)
            for i, chunk_text in enumerate(text_chunks):
                all_chunks.append(chunk_text)
                all_metadata.append({"index": i, "document_id": document_id, "type": "csv"})
        else:
            text = extract_text_from_file(file_path, mime, ext)
            if text and "[Audio - transcription disabled]" not in text:
                text_chunks = splitter.split_text(text)
                for i, chunk_text in enumerate(text_chunks):
                    all_chunks.append(chunk_text)
                    all_metadata.append({"index": i, "document_id": document_id, "type": "text"})
        if mime == "application/pdf" and ENABLE_VISION:
            images = await extract_images_from_pdf(file_path)
            for img_idx, (page_num, img_bytes, img_ext) in enumerate(images):
                ocr_text = await extract_text_from_image_bytes(img_bytes, img_ext)
                description = await describe_image(img_bytes, img_ext)
                desc_chunk = f"[Page {page_num} Image {img_idx+1} OCR]\n{ocr_text}\n\n[Description]\n{description}"
                all_chunks.append(desc_chunk)
                all_metadata.append({"index": len(all_chunks)-1, "document_id": document_id, "type": "image_description", "page": page_num})
        if mime.startswith("audio/") and ENABLE_WHISPER:
            transcript = await transcribe_audio(file_path)
            if transcript and "[Audio - transcription failed]" not in transcript:
                audio_chunks = splitter.split_text(transcript)
                base_idx = len(all_chunks)
                for i, chunk_text in enumerate(audio_chunks):
                    all_chunks.append(chunk_text)
                    all_metadata.append({"index": base_idx + i, "document_id": document_id, "type": "audio_transcript"})
        if mime.startswith("image/") and ENABLE_VISION:
            with open(file_path, "rb") as f:
                img_bytes = f.read()
            ocr_text = extract_text_from_image_path(file_path)
            description = await describe_image(img_bytes, ext)
            img_chunk = f"[OCR TEXT]\n{ocr_text}\n\n[Image Description]\n{description}"
            all_chunks.append(img_chunk)
            all_metadata.append({"index": len(all_chunks)-1, "document_id": document_id, "type": "image_description"})
        if not all_chunks:
            all_chunks = ["[Empty document]"]
            all_metadata = [{"index": 0, "document_id": document_id, "type": "fallback"}]
        async with db_pool.acquire() as conn:
            chunk_ids = []
            for i, chunk_text in enumerate(all_chunks):
                chunk_id = str(uuid.uuid4())
                chunk_ids.append(chunk_id)
                await conn.execute("INSERT INTO chunks (id, document_id, text, metadata) VALUES ($1, $2, $3, $4)",
                                   chunk_id, document_id, chunk_text, json.dumps(all_metadata[i]))
            namespace = f"ws_{workspace_id}"
            vectors = []
            embeddings = await asyncio.gather(*[get_embedding(chunk) for chunk in all_chunks])
            for i, chunk_id in enumerate(chunk_ids):
                vectors.append({"id": chunk_id, "values": embeddings[i], "metadata": {"chunk_id": chunk_id, "document_id": document_id, "text": all_chunks[i][:500]}})
            index = pc.Index(PINECONE_INDEX)
            for i in range(0, len(vectors), 100):
                index.upsert(vectors=vectors[i:i+100], namespace=namespace)
        await rebuild_bm25_for_workspace(workspace_id)
        async with db_pool.acquire() as conn:
            await conn.execute("UPDATE documents SET status = $1 WHERE id = $2", 'completed', document_id)
        logger.info(f"Document {document_id} processed successfully with {len(all_chunks)} chunks")
    except Exception as e:
        import traceback
        async with db_pool.acquire() as conn:
            await conn.execute("UPDATE documents SET status = $1 WHERE id = $2", 'failed', document_id)
        logger.error(f"Document processing failed for {document_id}: {e}\n{traceback.format_exc()}")
        raise
    finally:
        if os.path.exists(file_path):
            os.unlink(file_path)

# ========== LLM STREAMING (Ollama only) ==========
# Wrapper that uses provider_stream and formats SSE
async def stream_llm_sse(query: str, context: str, citations: list, request_id: str):
    try:
        async for token in provider_stream(query, context, citations):
            yield f"data: {json.dumps({'type': 'chunk', 'text': token})}\n\n"
    except asyncio.CancelledError:
        yield f"data: {json.dumps({'type': 'error', 'message': 'Stream cancelled'})}\n\n"
        raise
    except Exception as e:
        logger.error(f"LLM stream failed: {e}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'message': 'Failed to generate response'})}\n\n"

# ========== FASTAPI APP ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await init_redis()
    yield
    await db_pool.close()
    await redis_client.close()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=[os.getenv("FRONTEND_URL", "http://localhost:3000")], allow_methods=["*"], allow_headers=["*"])

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    REQUESTS.labels(method=request.method, endpoint=request.url.path).inc()
    start = time.time()
    response = await call_next(request)
    LATENCY.observe(time.time() - start)
    return response

# ========== API ROUTES ==========
class RegisterRequest(BaseModel): email: str; password: str; workspace_name: str
class LoginRequest(BaseModel): email: str; password: str
class RefreshRequest(BaseModel): refresh_token: str
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Search query (1-1000 characters)")
    document_ids: List[str] = Field(default_factory=list, description="Filter results to specific documents")
class FeedbackRequest(BaseModel): feedback: str

@app.post("/auth/register")
async def register(data: RegisterRequest):
    if len(data.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    hashed = bcrypt.hashpw(data.password.encode(), bcrypt.gensalt()).decode()
    try:
        async with db_pool.acquire() as conn:
            async with conn.transaction():
                ws_id = await conn.fetchval("INSERT INTO workspaces (name) VALUES ($1) RETURNING id", data.workspace_name)
                await conn.execute("INSERT INTO users (email, password_hash, workspace_id) VALUES ($1, $2, $3)",
                                   data.email, hashed, ws_id)
                access_token = jwt.encode({"sub": data.email, "email": data.email, "workspace_id": str(ws_id), "exp": int(time.time()) + 900},
                                          JWT_SECRET, algorithm="HS256")
                refresh_token = await create_refresh_token(data.email)
        return {"access_token": access_token, "refresh_token": refresh_token, "workspace_id": str(ws_id)}
    except asyncpg.UniqueViolationError:
        raise HTTPException(409, "Email already exists")

@app.post("/auth/login")
async def login(data: LoginRequest):
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT password_hash, workspace_id FROM users WHERE email = $1", data.email)
        if not row or not bcrypt.checkpw(data.password.encode(), row["password_hash"].encode()):
            await log_event("AUTH_FAIL", data.email, "Invalid credentials")
            raise HTTPException(401, "Invalid credentials")
        access_token = jwt.encode({"sub": data.email, "email": data.email, "workspace_id": str(row["workspace_id"]), "exp": int(time.time()) + 900},
                                  JWT_SECRET, algorithm="HS256")
        refresh_token = await create_refresh_token(data.email)
        await log_event("AUTH_SUCCESS", data.email, "Login successful")
    return {"access_token": access_token, "refresh_token": refresh_token, "workspace_id": str(row["workspace_id"])}

@app.post("/auth/refresh")
async def refresh(data: RefreshRequest):
    """Exchange old refresh token for new access + refresh tokens (one-time use)."""
    # Step 1: Validate and consume old token
    user_id = await consume_refresh_token(data.refresh_token)
    if not user_id:
        logger.warning("Refresh token rejected: invalid or reused")
        raise HTTPException(401, "Invalid or expired refresh token")

    # Step 2: Get workspace info
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT workspace_id FROM users WHERE email = $1", user_id)
        if not row:
            raise HTTPException(401, "User not found")

    # Step 3: Issue new tokens
    new_access_token = jwt.encode(
        {"sub": user_id, "email": user_id, "workspace_id": str(row["workspace_id"]), "exp": int(time.time()) + 900},
        JWT_SECRET, algorithm="HS256"
    )
    new_refresh_token = await create_refresh_token(user_id)

    logger.info(f"Refresh token rotated for {user_id}")
    return {"access_token": new_access_token, "refresh_token": new_refresh_token}

@app.post("/auth/logout")
async def logout(data: RefreshRequest):
    """Revoke refresh token on logout."""
    if data.refresh_token:
        await redis_client.delete(f"refresh:{data.refresh_token}")
    return {"status": "logged out"}

@app.post("/documents")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    await rate_limit_check(user["user_id"])
    await check_workspace_quotas(user["workspace_id"])
    await log_event("DOC_UPLOAD", user["user_id"], f"filename={file.filename}")
    total = 0
    first_chunk = None
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_FILE_MB * 1024 * 1024:
            raise HTTPException(400, f"File too large, max {MAX_FILE_MB}MB")
        if first_chunk is None:
            first_chunk = chunk
            mime = magic.from_buffer(first_chunk, mime=True)
            if not is_supported_type(mime):
                raise HTTPException(400, f"Unsupported file type: {mime}")
    await file.seek(0)
    ext = file.filename.split(".")[-1].lower()
    document_id = str(uuid.uuid4())
    async with db_pool.acquire() as conn:
        await conn.execute("INSERT INTO documents (id, workspace_id, filename) VALUES ($1, $2, $3)",
                           document_id, user["workspace_id"], file.filename)
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    background_tasks.add_task(process_document_async, tmp_path, file.filename, user["workspace_id"], document_id)
    return {"document_id": document_id, "filename": file.filename, "status": "processing"}

@app.get("/documents")
async def list_documents(user: dict = Depends(get_current_user)):
    rows = await db_pool.fetch("SELECT id, filename, created_at FROM documents WHERE workspace_id = $1 ORDER BY created_at DESC", user["workspace_id"])
    return [{"id": str(r["id"]), "filename": r["filename"], "created_at": r["created_at"]} for r in rows]

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str, user: dict = Depends(get_current_user)):
    await db_pool.execute("DELETE FROM chunks WHERE document_id = $1", document_id)
    await db_pool.execute("DELETE FROM documents WHERE id = $1 AND workspace_id = $2", document_id, user["workspace_id"])
    pc.Index(PINECONE_INDEX).delete(filter={"document_id": document_id}, namespace=f"ws_{user['workspace_id']}")
    await rebuild_bm25_for_workspace(user["workspace_id"])
    await log_event("DOC_DELETE", user["user_id"], f"document_id={document_id}")
    return {"status": "deleted"}

@app.post("/query")
async def query_endpoint(request: QueryRequest, req: Request, user: dict = Depends(get_current_user)):
    await rate_limit_check(user["user_id"])
    start_time = time.time()
    retrieved = await hybrid_search(request.query, user["workspace_id"], request.document_ids)
    if not retrieved:
        async def error_gen():
            yield f"data: {json.dumps({'type': 'error', 'message': 'No relevant documents found'})}\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")
    context = ""
    citations = []
    seen = set()
    for i, chunk in enumerate(retrieved):
        doc_id = chunk["metadata"].get("document_id", "unknown")
        if doc_id not in seen:
            citations.append({"documentId": doc_id, "snippet": chunk["text"][:200]})
            seen.add(doc_id)
        context += f"[Source {i+1}] {chunk['text']}\n\n"

    async def generate():
        agen = stream_llm_sse(request.query, context, citations, str(uuid.uuid4()))
        try:
            async for sse in agen:
                if await req.is_disconnected():
                    await agen.aclose()
                    break
                yield sse
        finally:
            await agen.aclose()
        response_time = (time.time() - start_time) * 1000
        confidence = sum(c["score"] for c in retrieved) / len(retrieved) if retrieved else 0
        await db_pool.execute("INSERT INTO query_logs (id, workspace_id, query, retrieved_count, response_time_ms, confidence_score) VALUES ($1, $2, $3, $4, $5, $6)",
                              str(uuid.uuid4()), user["workspace_id"], request.query, len(retrieved), response_time, confidence)
        yield f"data: {json.dumps({'type': 'citations', 'data': citations})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'confidence': confidence})}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/feedback/{query_id}")
async def feedback(query_id: str, data: FeedbackRequest, user: dict = Depends(get_current_user)):
    await db_pool.execute("UPDATE query_logs SET feedback = $1 WHERE id = $2", data.feedback, query_id)
    return {"status": "ok"}

# Health endpoint with comprehensive checks moved below

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(REGISTRY), media_type="text/plain")

# ========== FRONTEND ==========
@app.get("/")
async def frontend():
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head><title>RAG MVP Alpha</title><style>body{font-family:monospace;max-width:800px;margin:40px auto;padding:20px}textarea{width:100%;height:100px}#response{background:#f5f5f5;padding:10px;border-radius:5px}</style></head>
<body>
<h1>RAG MVP Alpha</h1>
<div><input type="email" id="email" placeholder="Email"><br><input type="password" id="password" placeholder="Password"><br><input type="text" id="workspace" placeholder="Workspace name"><br><button onclick="register()">Register</button> <button onclick="login()">Login</button></div>
<hr>
<div id="app" style="display:none">
<h3>Upload Document</h3><input type="file" id="file"><button onclick="upload()">Upload</button>
<h3>Documents</h3><ul id="docList"></ul>
<h3>Ask</h3><textarea id="query"></textarea><button onclick="ask()">Ask</button>
<div id="response"></div><div id="citations"></div>
</div>
<script>
let token=localStorage.getItem('token');
if(token){document.getElementById('app').style.display='block';}
async function register(){const e=document.getElementById('email').value,p=document.getElementById('password').value,w=document.getElementById('workspace').value;const r=await fetch('/auth/register',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({email:e,password:p,workspace_name:w})});const d=await r.json();if(r.ok){localStorage.setItem('token',d.access_token);token=d.access_token;document.getElementById('app').style.display='block';loadDocs();}}
async function login(){const e=document.getElementById('email').value,p=document.getElementById('password').value;const r=await fetch('/auth/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({email:e,password:p})});const d=await r.json();if(r.ok){localStorage.setItem('token',d.access_token);token=d.access_token;document.getElementById('app').style.display='block';loadDocs();}}
async function upload(){const f=document.getElementById('file').files[0];if(!f)return;const fd=new FormData();fd.append('file',f);await fetch('/documents',{method:'POST',headers:{'Authorization':'Bearer '+token},body:fd});if(token) loadDocs();}
async function loadDocs(){const r=await fetch('/documents',{headers:{'Authorization':'Bearer '+token}});if(!r.ok)return;const docs=await r.json();if(!r.ok)return;document.getElementById('docList').innerHTML=docs.map(d=>`<li>${d.filename} <button onclick="del('${d.id}')">Del</button></li>`).join('');}
async function del(id){await fetch('/documents/'+id,{method:'DELETE',headers:{'Authorization':'Bearer '+token}});if(token) loadDocs();}
async function ask(){const q=document.getElementById('query').value;const docsRes=await fetch('/documents',{headers:{'Authorization':'Bearer '+token}});const docs=await docsRes.json();const docIds=docs.map(d=>d.id);const res=await fetch('/query',{method:'POST',headers:{'Content-Type':'application/json','Authorization':'Bearer '+token},body:JSON.stringify({query:q,document_ids:docIds})});const reader=res.body.getReader();const decoder=new TextDecoder();document.getElementById('response').innerHTML='';document.getElementById('citations').innerHTML='';while(true){const {done,value}=await reader.read();if(done)break;const chunk=decoder.decode(value);const lines=chunk.split('\\n');for(const line of lines){if(line.startsWith('data:')){const d=JSON.parse(line.slice(6));if(d.type==='chunk')document.getElementById('response').innerHTML+=d.text;else if(d.type==='citations')document.getElementById('citations').innerHTML='<strong>Citations:</strong><br>'+d.data.map(c=>c.snippet).join('<br>');else if(d.type==='done')document.getElementById('citations').innerHTML+=`<br><strong>Confidence: ${d.confidence}</strong>`;}}}}
if(token) loadDocs();
</script>
</body>
</html>
    """)

@app.get("/health")
async def health():
    """Comprehensive health check including all dependencies."""
    checks = {}

    # Database
    try:
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"error: {str(e)}"

    # Redis
    try:
        await redis_client.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {str(e)}"

    # Pinecone
    try:
        pc.Index(PINECONE_INDEX).describe_index_stats()
        checks["pinecone"] = "ok"
    except Exception as e:
        checks["pinecone"] = f"error: {str(e)}"

    # Ollama
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(
                f"{OLLAMA_EMBEDDING_URL}/api/embeddings",
                json={"model": OLLAMA_EMBEDDING_MODEL, "prompt": "test"},
                timeout=5.0
            )
        checks["ollama"] = "ok"
    except Exception as e:
        checks["ollama"] = f"error: {str(e)}"

    all_ok = all("ok" in v for v in checks.values())
    status_code = 200 if all_ok else 503
    return Response(
        content=json.dumps({"status": "ok" if all_ok else "degraded", "checks": checks}),
        status_code=status_code,
        media_type="application/json"
    )


# Add readiness check (optional, for Kubernetes)
@app.get("/health/ready")
async def readiness():
    """Readiness check: only true if all dependencies are ready."""
    try:
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        await redis_client.ping()
        return {"status": "ready"}
    except Exception:
        raise HTTPException(503, "Service not ready")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
