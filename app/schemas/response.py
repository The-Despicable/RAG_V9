from typing import Any, Generic, Optional, TypeVar, Dict
from pydantic import BaseModel
from enum import Enum

T = TypeVar("T")


class ResponseStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"


class ApiResponse(BaseModel, Generic[T]):
    success: bool
    data: Optional[T] = None
    error: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class StandardResponse:
    @staticmethod
    def success(data: Any = None, request_id: str = None) -> Dict[str, Any]:
        return {
            "success": True,
            "data": data,
            "error": None,
            "request_id": request_id
        }
    
    @staticmethod
    def error(code: str, message: str, details: Dict = None, request_id: str = None) -> Dict[str, Any]:
        return {
            "success": False,
            "data": None,
            "error": {
                "code": code,
                "message": message,
                "details": details
            },
            "request_id": request_id
        }