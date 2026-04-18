import os
import json
import httpx
from typing import AsyncGenerator, Dict, Any
from app.providers.base import LLMProvider, ProviderError, ProviderTimeoutError, ProviderRateLimitError, ProviderAuthenticationError
from app.config import get_settings

settings = get_settings()


class OpenRouterProvider(LLMProvider):
    name = "openrouter"
    
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set")
        self.model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-2-70b-chat")
        self.temperature = settings.OPENROUTER_TEMPERATURE
        self.max_tokens = settings.OPENROUTER_MAX_TOKENS
        self.timeout = settings.OPENROUTER_TIMEOUT_SECONDS
        self.embedding_model = os.getenv("OPENROUTER_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    async def stream(self, query: str, context: str, **kwargs) -> AsyncGenerator[str, None]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": settings.BACKEND_URL,
            "X-Title": "RAG SaaS"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": f"Answer based on context:\n{context}"},
                {"role": "user", "content": query}
            ],
            "stream": True,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload
                ) as resp:
                    if resp.status_code == 401:
                        raise ProviderAuthenticationError("OpenRouter authentication failed")
                    if resp.status_code == 429:
                        raise ProviderRateLimitError("OpenRouter rate limit exceeded")
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        yield delta["content"]
                            except json.JSONDecodeError:
                                continue
        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(f"OpenRouter timeout: {e}")
        except httpx.HTTPStatusError as e:
            raise ProviderError(f"OpenRouter HTTP error: {e}")
    
    async def embed(self, text: str) -> list:
        raise NotImplementedError("OpenRouter embedding not implemented - use Ollama or OpenAI")
    
    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(
                    "https://openrouter.ai/api/v1/models",
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
                return resp.status_code == 200
        except Exception:
            return False
    
    def get_usage(self) -> Dict[str, Any]:
        return {"provider": self.name, "model": self.model}