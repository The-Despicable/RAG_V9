import logging
import json
from typing import Any, Dict
from datetime import datetime
from app.middleware.request_id import get_request_id


class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(handler)
    
    def _log(self, level: str, message: str, extra: Dict[str, Any] = None):
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level,
            "message": message,
            "request_id": get_request_id() or "none"
        }
        if extra:
            log_data["extra"] = extra
        
        if level == "ERROR":
            self.logger.error(json.dumps(log_data))
        elif level == "WARNING":
            self.logger.warning(json.dumps(log_data))
        elif level == "DEBUG":
            self.logger.debug(json.dumps(log_data))
        else:
            self.logger.info(json.dumps(log_data))
    
    def info(self, message: str, **kwargs):
        self._log("INFO", message, kwargs)
    
    def error(self, message: str, **kwargs):
        self._log("ERROR", message, kwargs)
    
    def warning(self, message: str, **kwargs):
        self._log("WARNING", message, kwargs)
    
    def debug(self, message: str, **kwargs):
        self._log("DEBUG", message, kwargs)


logger = StructuredLogger("rag")


def log_event(event_type: str, user_id: str, details: str = ""):
    logger.info(
        f"EVENT: {event_type}",
        event_type=event_type,
        user_id=user_id,
        details=details
    )