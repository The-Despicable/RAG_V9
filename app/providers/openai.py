import os
from typing import AsyncGenerator, Dict, Any
from app.providers.base import LLMProvider, ProviderError, ProviderTimeoutError, ProviderRateLimitError, ProviderAuthenticationError
from app.config import get_settings

settings = get_settings()


class OpenAIProvider(LLMProvider):
    name = "openai"
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4-turbo")
        self.embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.temperature = settings.OPENAI_TEMPERATURE
        self.max_tokens = settings.OPENAI_MAX_TOKENS
        self.timeout = settings.OPENAI_TIMEOUT_SECONDS
    
    async def stream(self, query: str, context: str, **kwargs) -> AsyncGenerator[str, None]:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ProviderError("OpenAI SDK not installed")
        
        client = AsyncOpenAI(api_key=self.api_key, timeout=self.timeout)
        
        try:
            stream = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"Answer based on context:\n{context}"},
                    {"role": "user", "content": query}
                ],
                stream=True,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            if "rate_limit" in str(e).lower():
                raise ProviderRateLimitError(f"OpenAI rate limit: {e}")
            if "authentication" in str(e).lower():
                raise ProviderAuthenticationError(f"OpenAI auth error: {e}")
            if "timeout" in str(e).lower():
                raise ProviderTimeoutError(f"OpenAI timeout: {e}")
            raise ProviderError(f"OpenAI error: {e}")
    
    async def embed(self, text: str) -> list:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ProviderError("OpenAI SDK not installed")
        
        client = AsyncOpenAI(api_key=self.api_key, timeout=self.timeout)
        resp = await client.embeddings.create(input=text, model=self.embedding_model)
        return resp.data[0].embedding
    
    async def health_check(self) -> bool:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            return False
        
        try:
            client = AsyncOpenAI(api_key=self.api_key, timeout=5)
            await client.models.list()
            return True
        except Exception:
            return False
    
    def normalize_tokens(self, response: Dict) -> Dict[str, int]:
        return {
            "input_tokens": response.get("usage", {}).get("prompt_tokens", 0),
            "output_tokens": response.get("usage", {}).get("completion_tokens", 0)
        }
    
    def get_usage(self) -> Dict[str, Any]:
        return {"provider": self.name, "model": self.model}