import os
import json
import httpx
from typing import AsyncGenerator, Dict, Any
from app.providers.base import LLMProvider, ProviderError, ProviderTimeoutError, ProviderRateLimitError
from app.config import get_settings

settings = get_settings()


class OllamaProvider(LLMProvider):
    name = "ollama"
    
    def __init__(self):
        self.url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", "llama3.2")
        self.api_key = os.getenv("OLLAMA_API_KEY", "")
        self.temperature = settings.OLLAMA_TEMPERATURE
        self.max_tokens = settings.OLLAMA_MAX_TOKENS
        self.timeout = settings.OLLAMA_TIMEOUT_SECONDS
        self.embedding_url = os.getenv("OLLAMA_EMBEDDING_URL", "http://localhost:11434")
        self.embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    
    async def stream(self, query: str, context: str, **kwargs) -> AsyncGenerator[str, None]:
        payload = {
            "model": self.model,
            "prompt": query,
            "system": f"Answer based on context:\n{context}",
            "stream": True,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", f"{self.url}/api/generate", json=payload, headers=headers) as resp:
                    if resp.status_code == 429:
                        raise ProviderRateLimitError("Ollama rate limit exceeded")
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]
                        except json.JSONDecodeError:
                            continue
        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(f"Ollama timeout: {e}")
        except httpx.HTTPStatusError as e:
            raise ProviderError(f"Ollama HTTP error: {e}")
    
    async def embed(self, text: str) -> list:
        try:
            async with httpx.AsyncClient(timeout=settings.OLLAMA_EMBEDDING_TIMEOUT) as client:
                resp = await client.post(
                    f"{self.embedding_url}/api/embeddings",
                    json={"model": self.embedding_model, "prompt": text}
                )
                resp.raise_for_status()
                return resp.json()["embedding"]
        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(f"Ollama embedding timeout: {e}")
        except Exception as e:
            raise ProviderError(f"Ollama embedding error: {e}")
    
    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.post(
                    f"{self.embedding_url}/api/embeddings",
                    json={"model": self.embedding_model, "prompt": "test"}
                )
                return resp.status_code == 200
        except Exception:
            return False
    
    def get_usage(self) -> Dict[str, Any]:
        return {"provider": self.name, "model": self.model}