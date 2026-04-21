"""Simple provider selection without over-abstraction."""
import os
import json
import httpx
import logging
from typing import AsyncGenerator

logger = logging.getLogger("providers")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "ollama").lower()

# ========== OLLAMA ==========

async def ollama_stream(query: str, context: str) -> AsyncGenerator[str, None]:
    """Stream from Ollama."""
    url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3")
    api_key = os.getenv("OLLAMA_API_KEY", "")

    payload = {
        "model": model,
        "prompt": query,
        "system": f"Answer based on context:\n{context}",
        "stream": True,
    }
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", f"{url}/api/generate", json=payload, headers=headers) as resp:
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

async def ollama_embed(text: str) -> list:
    """Embed with Ollama."""
    url = os.getenv("OLLAMA_EMBEDDING_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(f"{url}/api/embeddings", json={"model": model, "prompt": text})
        resp.raise_for_status()
        return resp.json()["embedding"]

# ========== OPENAI ==========

async def openai_stream(query: str, context: str) -> AsyncGenerator[str, None]:
    """Stream from OpenAI."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        logger.error("OpenAI SDK not installed. Install with: pip install openai")
        raise

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    model = os.getenv("OPENAI_MODEL", "gpt-4-turbo")

    client = AsyncOpenAI(api_key=api_key, timeout=30)

    stream = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": f"Answer based on context:\n{context}"},
            {"role": "user", "content": query}
        ],
        stream=True,
        temperature=0.7,
        max_tokens=2000
    )

    async for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

async def openai_embed(text: str) -> list:
    """Embed with OpenAI."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        logger.error("OpenAI SDK not installed. Install with: pip install openai")
        raise

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    client = AsyncOpenAI(api_key=api_key, timeout=30)
    resp = await client.embeddings.create(input=text, model=model)
    return resp.data[0].embedding

# ========== OPENROUTER ==========

async def openrouter_stream(query: str, context: str) -> AsyncGenerator[str, None]:
    """Stream from OpenRouter (cloud Ollama alternative)."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")
    model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-2-70b-chat")

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": f"Answer based on context:\n{context}"},
                    {"role": "user", "content": query}
                ],
                "stream": True,
            }
        ) as resp:
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

# ========== ROUTER ==========

async def provider_stream(query: str, context: str, citations: list = None) -> AsyncGenerator[str, None]:
    """Route to configured LLM provider."""
    # Build context with citations if provided
    context_str = context
    if citations:
        context_str = f"{context}\n\nSources: {[c.get('snippet', '') for c in citations]}"

    if LLM_PROVIDER == "openai":
        async for token in openai_stream(query, context_str):
            yield token
    elif LLM_PROVIDER == "openrouter":
        async for token in openrouter_stream(query, context_str):
            yield token
    else:  # ollama (default)
        async for token in ollama_stream(query, context_str):
            yield token

async def get_embedding(text: str) -> list:
    """Route to configured embedding provider."""
    if EMBEDDING_PROVIDER == "openai":
        return await openai_embed(text)
    else:  # ollama (default)
        return await ollama_embed(text)
