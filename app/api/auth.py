import time
import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
import bcrypt
import jwt


from app.config import get_settings
from app.middleware.request_id import get_request_id
from app.middleware.logging import logger, log_event

router = APIRouter(prefix="/auth", tags=["auth"])
settings = get_settings()


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)
    workspace_name: str = Field(..., min_length=1, max_length=100)
    company_name: str = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    refresh_token: str = None


@router.post("/register")
async def register(data: RegisterRequest):
    from app.main import db_pool, redis_client  # lazy import
    if len(data.password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters"
        )
    
    hashed = bcrypt.hashpw(data.password.encode(), bcrypt.gensalt()).decode()
    
    try:
        async with db_pool.acquire() as conn:
            async with conn.transaction():
                ws_id = await conn.fetchval(
                    "INSERT INTO workspaces (name) VALUES ($1) RETURNING id", 
                    data.workspace_name
                )
                
                await conn.execute(
                    "INSERT INTO users (email, password_hash, workspace_id) VALUES ($1, $2, $3)",
                    data.email, hashed, ws_id
                )
                
                await conn.execute(
                    """INSERT INTO subscriptions (workspace_id, plan, tokens_balance, tokens_limit_monthly) 
                       VALUES ($1, 'free', 50000, 50000)""",
                    ws_id
                )
                
                access_token = jwt.encode(
                    {
                        "sub": data.email,
                        "email": data.email,
                        "workspace_id": str(ws_id),
                        "exp": int(time.time()) + 3600
                    },
                    settings.JWT_SECRET, 
                    algorithm="HS256"
                )
                refresh_token = await create_refresh_token(data.email)
        
        await log_event("REGISTER", data.email, f"workspace={data.workspace_name}")
        
        async with db_pool.acquire() as conn:
            sub = await conn.fetchrow(
                "SELECT plan, tokens_balance, tokens_limit_monthly FROM subscriptions WHERE workspace_id = $1",
                ws_id
            )
        
        return {
            "success": True,
            "data": {
                "user_id": str(uuid.uuid4()),
                "email": data.email,
                "workspace_id": str(ws_id),
                "workspace_name": data.workspace_name,
                "access_token": access_token,
                "token_type": "bearer",
                "expires_in": 3600,
                "subscription": {
                    "plan": sub["plan"],
                    "tokens_balance": sub["tokens_balance"],
                    "tokens_limit_monthly": sub["tokens_limit_monthly"]
                }
            },
            "request_id": get_request_id()
        }
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        if "unique" in str(e).lower() or "email" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already exists"
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login")
async def login(data: LoginRequest):
    from app.main import db_pool, redis_client  # lazy import
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, password_hash, workspace_id, email FROM users WHERE email = $1", 
            data.email
        )
    
    if not row or not bcrypt.checkpw(data.password.encode(), row["password_hash"].encode()):
        await log_event("AUTH_FAIL", data.email, "Invalid credentials")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    access_token = jwt.encode(
        {
            "sub": data.email,
            "email": data.email,
            "workspace_id": str(row["workspace_id"]),
            "exp": int(time.time()) + 3600
        },
        settings.JWT_SECRET, 
        algorithm="HS256"
    )
    refresh_token = await create_refresh_token(data.email)
    
    await log_event("AUTH_SUCCESS", data.email, "Login successful")
    
    return {
        "success": True,
        "data": {
            "user_id": str(row["id"]),
            "email": data.email,
            "workspace_id": str(row["workspace_id"]),
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": 3600
        },
        "request_id": get_request_id()
    }


@router.post("/refresh")
async def refresh(data: RefreshRequest):
    from app.main import db_pool, redis_client  # lazy import
    user_id = await consume_refresh_token(data.refresh_token)
    if not user_id:
        logger.warning("Refresh token rejected: invalid or reused")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token"
        )
    
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT workspace_id FROM users WHERE email = $1", 
            user_id
        )
        if not row:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
    
    new_access_token = jwt.encode(
        {
            "sub": user_id,
            "email": user_id,
            "workspace_id": str(row["workspace_id"]),
            "exp": int(time.time()) + 3600
        },
        settings.JWT_SECRET, 
        algorithm="HS256"
    )
    new_refresh_token = await create_refresh_token(user_id)
    
    logger.info(f"Refresh token rotated for {user_id}")
    
    return {
        "success": True,
        "data": {
            "access_token": new_access_token,
            "expires_in": 3600
        },
        "request_id": get_request_id()
    }


@router.post("/logout")
async def logout(data: LogoutRequest):
    from app.main import redis_client  # lazy import
    if data.refresh_token:
        await redis_client.delete(f"refresh:{data.refresh_token}")
    
    return {
        "success": True,
        "data": None,
        "request_id": get_request_id()
    }


async def create_refresh_token(user_id: str) -> str:
    token = str(uuid.uuid4())
    payload = {
        "user_id": user_id,
        "issued_at": int(time.time()),
        "rotation_count": 0
    }
    await redis_client.setex(f"refresh:{token}", 86400 * 7, __import__('json').dumps(payload))
    return token


async def consume_refresh_token(token: str):
    import json
    try:
        data = await redis_client.get(f"refresh:{token}")
        if not data:
            return None
        
        payload = json.loads(data)
        user_id = payload["user_id"]
        rotations = payload.get("rotation_count", 0)
        
        if rotations > 100:
            logger.warning(f"Token reuse attack detected: {user_id} after {rotations} rotations")
            await redis_client.delete(f"refresh:{token}")
            return None
        
        await redis_client.delete(f"refresh:{token}")
        return user_id
    except Exception as e:
        logger.error(f"Token validation failed: {e}")
        return None
