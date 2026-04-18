import traceback
import os
from typing import Optional
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime
from app.middleware.request_id import get_request_id
from app.middleware.logging import logger


class ErrorCode:
    INTERNAL_ERROR = "INTERNAL_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    BAD_REQUEST = "BAD_REQUEST"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    NO_DOCUMENTS_FOUND = "NO_DOCUMENTS_FOUND"


class AppException(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 400,
        details: dict = None
    ):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}


async def app_exception_handler(request: Request, exc: AppException):
    request_id = getattr(request.state, "request_id", "unknown")
    start_time = getattr(request.state, "start_time", datetime.now().timestamp())
    latency_ms = int((datetime.now().timestamp() - start_time) * 1000)
    
    logger.error(
        f"AppException: {exc.code}",
        code=exc.code,
        message=exc.message,
        status_code=exc.status_code,
        endpoint=str(request.url),
        method=request.method,
        details=exc.details
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "data": None,
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details if exc.details else None
            },
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "latency_ms": latency_ms
        }
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.warning(
        f"Validation error: {str(exc)}",
        request_id=request_id,
        endpoint=str(request.url)
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "data": None,
            "error": {
                "code": ErrorCode.VALIDATION_ERROR,
                "message": "Invalid request data",
                "details": exc.errors()
            },
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    )


async def generic_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.error(
        f"Unhandled exception: {type(exc).__name__}",
        request_id=request_id,
        error=str(exc),
        traceback=traceback.format_exc()
    )
    
    message = "Internal server error" if os.getenv("ENVIRONMENT", "development") == "production" else str(exc)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "data": None,
            "error": {
                "code": ErrorCode.INTERNAL_ERROR,
                "message": message,
                "details": None
            },
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    )


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except HTTPException as e:
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "success": False,
                    "data": None,
                    "error": {
                        "code": f"HTTP_{e.status_code}",
                        "message": e.detail,
                        "details": None
                    },
                    "request_id": get_request_id()
                }
            )
        except Exception as e:
            request_id = get_request_id()
            logger.error(
                f"Unhandled exception: {str(e)}",
                extra={
                    "request_id": request_id,
                    "path": request.url.path,
                    "method": request.method,
                    "traceback": traceback.format_exc()
                }
            )
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "data": None,
                    "error": {
                        "code": ErrorCode.INTERNAL_ERROR,
                        "message": "An internal error occurred",
                        "details": {"request_id": request_id} if request_id else None
                    },
                    "request_id": request_id
                }
            )


def create_error_response(code: str, message: str, status_code: int = 400, details: dict = None, request_id: Optional[str] = None):
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "data": None,
            "error": {
                "code": code,
                "message": message,
                "details": details
            },
            "request_id": request_id or get_request_id()
        }
    )