import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request, Query
from pydantic import BaseModel, Field

from app.main import db_pool
from app.core.security import get_current_user
from app.config import get_settings
from app.middleware.request_id import get_request_id
from app.middleware.logging import logger

router = APIRouter(prefix="/billing", tags=["billing"])
settings = get_settings()


class UpgradeRequest(BaseModel):
    plan: str = Field(..., regex="^(pro|enterprise)$")


class WebhookRequest(BaseModel):
    event: str
    payload: dict


PLAN_PRICES = {
    "pro": 10000,
    "enterprise": 50000
}


@router.get("/usage")
async def get_usage(user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        sub = await conn.fetchrow(
            """SELECT plan, tokens_balance, tokens_limit_monthly, renewal_date
               FROM subscriptions WHERE workspace_id = $1""",
            user["workspace_id"]
        )
        
        if not sub:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No subscription found"
            )
        
        tokens_used = sub["tokens_limit_monthly"] - sub["tokens_balance"]
        percentage_used = (tokens_used / sub["tokens_limit_monthly"] * 100) if sub["tokens_limit_monthly"] > 0 else 0
        
        daily_burn = await conn.fetchval(
            """SELECT COALESCE(AVG(daily_tokens), 0) FROM (
               SELECT SUM(tokens_deducted) as daily_tokens
               FROM token_ledger
               WHERE workspace_id = $1 AND created_at > NOW() - INTERVAL '30 days'
               GROUP BY DATE(created_at)
               ) sub""",
            user["workspace_id"]
        )
        
        estimated_days = int(sub["tokens_balance"] / daily_burn) if daily_burn > 0 else 999
        
        queries = await conn.fetchval(
            "SELECT COUNT(*) FROM query_logs WHERE workspace_id = $1 AND created_at > NOW() - INTERVAL '30 days'",
            user["workspace_id"]
        )
        
        renewal = sub["renewal_date"] or (datetime.utcnow() + timedelta(days=30))
    
    return {
        "success": True,
        "data": {
            "plan": sub["plan"],
            "tokens_used": tokens_used,
            "tokens_limit": sub["tokens_limit_monthly"],
            "tokens_remaining": sub["tokens_balance"],
            "percentage_used": round(percentage_used, 1),
            "reset_date": renewal.isoformat() + "Z",
            "daily_burn_rate": int(daily_burn),
            "estimated_days_remaining": estimated_days,
            "queries_this_month": queries
        },
        "request_id": get_request_id()
    }


@router.get("/usage/history")
async def get_usage_history(
    user: dict = Depends(get_current_user),
    days: int = Query(30, ge=1, le=90),
    granularity: str = Query("daily", regex="^(hourly|daily|weekly)$")
):
    async with db_pool.acquire() as conn:
        if granularity == "daily":
            rows = await conn.fetch(
                """SELECT DATE(created_at) as date,
                          SUM(tokens_deducted) as tokens_used,
                          SUM(input_tokens) as tokens_input,
                          SUM(output_tokens) as tokens_output,
                          COUNT(*) as queries
                   FROM token_ledger
                   WHERE workspace_id = $1 AND created_at > NOW() - INTERVAL '2 days'
                   GROUP BY DATE(created_at)
                   ORDER BY date""",
                user["workspace_id"]
            )
        else:
            rows = []
    
    chart_data = []
    for r in rows:
        chart_data.append({
            "date": r["date"].isoformat() if r["date"] else None,
            "tokens_used": r["tokens_used"] or 0,
            "tokens_input": r["tokens_input"] or 0,
            "tokens_output": r["tokens_output"] or 0,
            "queries": r["queries"] or 0
        })
    
    return {
        "success": True,
        "data": {"chart_data": chart_data},
        "request_id": get_request_id()
    }


@router.post("/upgrade")
async def upgrade_plan(request: UpgradeRequest, user: dict = Depends(get_current_user)):
    amount = PLAN_PRICES.get(request.plan)
    if not amount:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid plan"
        )
    
    razorpay_order_id = f"order_{uuid.uuid4().hex[:12]}"
    
    async with db_pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO razorpay_orders (workspace_id, razorpay_order_id, amount_paise, plan, status)
               VALUES ($1, $2, $3, $4, 'created')""",
            user["workspace_id"], razorpay_order_id, amount, request.plan
        )
    
    return {
        "success": True,
        "data": {
            "razorpay_order_id": razorpay_order_id,
            "amount_paise": amount,
            "plan": request.plan,
            "currency": "INR",
            "razorpay_key_id": settings.get("RAZORPAY_KEY_ID", "rzp_test_xxx"),
            "checkout_url": f"https://checkout.razorpay.com/{razorpay_order_id}"
        },
        "request_id": get_request_id()
    }


@router.post("/webhook/razorpay")
async def razorpay_webhook(request: Request):
    body = await request.body()
    signature = request.headers.get("X-Razorpay-Signature", "")
    
    logger.info(f"Razorpay webhook received: {request.app.state.get('event', 'unknown')}")
    
    try:
        payload = json.loads(body)
        event = payload.get("event")
        
        if event == "payment.authorized":
            payment = payload.get("payload", {}).get("payment", {}).get("entity", {})
            order_id = payment.get("order_id")
            payment_id = payment.get("id")
            amount = payment.get("amount")
            
            async with db_pool.acquire() as conn:
                order = await conn.fetchrow(
                    "SELECT workspace_id, plan FROM razorpay_orders WHERE razorpay_order_id = $1",
                    order_id
                )
                
                if order:
                    await conn.execute(
                        """UPDATE razorpay_orders 
                           SET razorpay_payment_id = $1, status = 'paid', verified = TRUE
                           WHERE razorpay_order_id = $2""",
                        payment_id, order_id
                    )
                    
                    tokens = 1000000 if order["plan"] == "pro" else 5000000
                    await conn.execute(
                        """UPDATE subscriptions 
                           SET plan = $1, tokens_balance = tokens_balance + $2, 
                               tokens_limit_monthly = $2, is_active = TRUE, updated_at = NOW()
                           WHERE workspace_id = $3""",
                        order["plan"], tokens, order["workspace_id"]
                    )
        
        return {
            "success": True,
            "data": {
                "event": event,
                "processed": True
            },
            "request_id": get_request_id()
        }
    except Exception as e:
        logger.error(f"Webhook processing failed: {e}")
        return {
            "success": True,
            "data": {
                "event": "unknown",
                "processed": False
            },
            "request_id": get_request_id()
        }


@router.get("/subscription")
async def get_subscription(user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        sub = await conn.fetchrow(
            """SELECT plan, tokens_balance, tokens_limit_monthly, renewal_date, 
                      is_active, payment_provider, created_at
               FROM subscriptions WHERE workspace_id = $1""",
            user["workspace_id"]
        )
    
    if not sub:
        return {
            "success": True,
            "data": {
                "plan": "free",
                "status": "active",
                "tokens_balance": 50000,
                "tokens_limit_monthly": 50000,
                "renewal_date": None,
                "is_active": True,
                "payment_method": None,
                "razorpay_subscription_id": None,
                "created_at": None
            },
            "request_id": get_request_id()
        }
    
    return {
        "success": True,
        "data": {
            "plan": sub["plan"],
            "status": "active" if sub["is_active"] else "inactive",
            "tokens_balance": sub["tokens_balance"],
            "tokens_limit_monthly": sub["tokens_limit_monthly"],
            "renewal_date": sub["renewal_date"].isoformat() + "Z" if sub["renewal_date"] else None,
            "is_active": sub["is_active"],
            "payment_method": sub["payment_provider"],
            "razorpay_subscription_id": None,
            "created_at": sub["created_at"].isoformat() + "Z" if sub["created_at"] else None
        },
        "request_id": get_request_id()
    }