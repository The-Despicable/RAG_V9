import uuid
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, EmailStr

from app.main import db_pool
from app.config import get_settings
from app.middleware.request_id import get_request_id

router = APIRouter(prefix="/workspace", tags=["workspace"])
settings = get_settings()


class WorkspaceUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    company_name: Optional[str] = Field(None, max_length=200)


class InviteMemberRequest(BaseModel):
    email: EmailStr
    role: str = Field(default="member", regex="^(member|admin)$")


@router.get("")
async def get_workspace(user: dict = Depends(lambda: {"workspace_id": "test", "email": "test@test.com"})):
    async with db_pool.acquire() as conn:
        ws = await conn.fetchrow(
            "SELECT id, name, created_at FROM workspaces WHERE id = $1",
            user["workspace_id"]
        )
        
        if not ws:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Workspace not found"
            )
        
        members = await conn.fetch(
            "SELECT id, email, is_admin, created_at FROM users WHERE workspace_id = $1",
            user["workspace_id"]
        )
        
        sub = await conn.fetchrow(
            """SELECT plan, tokens_balance, renewal_date 
               FROM subscriptions WHERE workspace_id = $1""",
            user["workspace_id"]
        )
    
    return {
        "success": True,
        "data": {
            "workspace_id": str(ws["id"]),
            "name": ws["name"],
            "company_name": None,
            "created_at": ws["created_at"].isoformat() + "Z" if ws["created_at"] else None,
            "members": [
                {
                    "id": str(m["id"]),
                    "email": m["email"],
                    "role": "owner" if m["is_admin"] else "member",
                    "joined_at": m["created_at"].isoformat() + "Z" if m["created_at"] else None
                }
                for m in members
            ],
            "subscription": {
                "plan": sub["plan"] if sub else "free",
                "tokens_balance": sub["tokens_balance"] if sub else 50000,
                "renewal_date": sub["renewal_date"].isoformat() + "Z" if sub and sub["renewal_date"] else None
            }
        },
        "request_id": get_request_id()
    }


@router.put("")
async def update_workspace(request: WorkspaceUpdateRequest, user: dict = Depends(lambda: {"workspace_id": "test"})):
    async with db_pool.acquire() as conn:
        if request.name:
            await conn.execute(
                "UPDATE workspaces SET name = $1, updated_at = NOW() WHERE id = $2",
                request.name, user["workspace_id"]
            )
    
    return await get_workspace(user)


@router.post("/members")
async def invite_member(request: InviteMemberRequest, user: dict = Depends(lambda: {"workspace_id": "test", "email": "owner@test.com"})):
    invitation_id = str(uuid.uuid4())
    expires_at = datetime.utcnow() + timedelta(days=7)
    
    async with db_pool.acquire() as conn:
        existing = await conn.fetchrow(
            "SELECT id FROM users WHERE email = $1",
            request.email
        )
        
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User already exists in workspace"
            )
        
        return {
            "success": True,
            "data": {
                "invitation_id": invitation_id,
                "email": request.email,
                "role": request.role,
                "status": "pending",
                "expires_at": expires_at.isoformat() + "Z"
            },
            "request_id": get_request_id()
        }


@router.delete("/members/{member_id}")
async def remove_member(member_id: str, user: dict = Depends(lambda: {"workspace_id": "test", "email": "owner@test.com"})):
    async with db_pool.acquire() as conn:
        member = await conn.fetchrow(
            "SELECT id, is_admin FROM users WHERE id = $1 AND workspace_id = $2",
            member_id, user["workspace_id"]
        )
        
        if not member:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Member not found"
            )
        
        if member["is_admin"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot remove workspace owner"
            )
        
        await conn.execute("DELETE FROM users WHERE id = $1", member_id)
    
    return {
        "success": True,
        "data": {"status": "removed"},
        "request_id": get_request_id()
    }