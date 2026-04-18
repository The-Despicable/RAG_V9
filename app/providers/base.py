from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional, Dict, Any
import logging

logger = logging.getLogger("providers")


class LLMProvider(ABC):
    name: str = ""
    
    @abstractmethod
    async def stream(self, query: str, context: str, **kwargs) -> AsyncGenerator[str, None]:
        pass
    
    @abstractmethod
    async def embed(self, text: str) -> list:
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        pass
    
    def get_usage(self) -> Dict[str, Any]:
        return {}
    
    def normalize_tokens(self, response: Dict) -> Dict[str, int]:
        return {"input_tokens": 0, "output_tokens": 0}


class ProviderError(Exception):
    pass


class ProviderTimeoutError(ProviderError):
    pass


class ProviderRateLimitError(ProviderError):
    pass


class ProviderAuthenticationError(ProviderError):
    pass