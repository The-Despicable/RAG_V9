import logging
from typing import AsyncGenerator, Optional
from app.providers.base import LLMProvider, ProviderError, ProviderTimeoutError, ProviderRateLimitError
from app.providers.ollama import OllamaProvider
from app.providers.openrouter import OpenRouterProvider
from app.providers.openai import OpenAIProvider
from app.utils.retry import get_circuit_breaker, CircuitBreakerOpenError
from app.config import get_settings

logger = logging.getLogger("providers.router")
settings = get_settings()


class ProviderRouter:
    def __init__(self):
        self.primary = settings.PRIMARY_LLM_PROVIDER
        fallback_order = settings.FALLBACK_PROVIDERS.split(",")
        self.fallbacks = [p.strip() for p in fallback_order]
        self._providers = {}
        self._init_providers()
    
    def _init_providers(self):
        for name in [self.primary] + self.fallbacks:
            if name == "ollama":
                try:
                    self._providers["ollama"] = OllamaProvider()
                except Exception as e:
                    logger.warning(f"Failed to init Ollama: {e}")
            elif name == "openrouter":
                try:
                    self._providers["openrouter"] = OpenRouterProvider()
                except Exception as e:
                    logger.warning(f"Failed to init OpenRouter: {e}")
            elif name == "openai":
                try:
                    self._providers["openai"] = OpenAIProvider()
                except Exception as e:
                    logger.warning(f"Failed to init OpenAI: {e}")
    
    async def stream(self, query: str, context: str, **kwargs) -> AsyncGenerator[str, None]:
        tried_providers = []
        last_error = None
        
        for provider_name in [self.primary] + self.fallbacks:
            if provider_name not in self._providers:
                continue
            
            provider = self._providers[provider_name]
            circuit_breaker = get_circuit_breaker(provider_name)
            
            tried_providers.append(provider_name)
            
            try:
                circuit_breaker_state = circuit_breaker.state.value if hasattr(circuit_breaker.state, 'value') else circuit_breaker.state
                if circuit_breaker_state == "open":
                    logger.info(f"Circuit breaker open for {provider_name}, skipping")
                    continue
                
                logger.info(f"Attempting provider: {provider_name}")
                async for token in provider.stream(query, context, **kwargs):
                    yield token
                
                logger.info(f"Successfully used provider: {provider_name}")
                return
                
            except (ProviderTimeoutError, ProviderRateLimitError) as e:
                logger.warning(f"Provider {provider_name} failed: {e}")
                last_error = e
                continue
            except ProviderError as e:
                logger.error(f"Provider {provider_name} error: {e}")
                last_error = e
                continue
            except Exception as e:
                logger.error(f"Unexpected error from {provider_name}: {e}")
                last_error = e
                continue
        
        raise ProviderError(f"All providers failed. Tried: {tried_providers}. Last error: {last_error}")
    
    async def embed(self, text: str) -> list:
        for provider_name in [self.primary] + self.fallbacks:
            if provider_name not in self._providers:
                continue
            
            provider = self._providers[provider_name]
            
            try:
                return await provider.embed(text)
            except Exception as e:
                logger.warning(f"Embedding failed with {provider_name}: {e}")
                continue
        
        raise ProviderError("All embedding providers failed")
    
    async def health_check_all(self) -> dict:
        results = {}
        for name, provider in self._providers.items():
            try:
                results[name] = await provider.health_check()
            except Exception as e:
                results[name] = False
                logger.error(f"Health check failed for {name}: {e}")
        return results


router = ProviderRouter()


async def provider_stream(query: str, context: str, citations: list = None) -> AsyncGenerator[str, None]:
    context_str = context
    if citations:
        context_str = f"{context}\n\nSources: {[c.get('snippet', '') for c in citations]}"
    async for token in router.stream(query, context_str):
        yield token


async def get_embedding(text: str) -> list:
    return await router.embed(text)