import os
import json
import httpx
from typing import AsyncGenerator

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
PRIMARY_PROVIDER = os.getenv("PRIMARY_PROVIDER", "ollama")

async def ollama_generate(prompt: str, system_prompt: str = "") -> AsyncGenerator[str, None]:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": system_prompt,
        "stream": True,
        "options": {"num_predict": 2000, "temperature": 0.7}
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", f"{OLLAMA_URL}/api/generate", json=payload) as resp:
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

_openai_fallback = None

def set_openai_stream_impl(func):
    global _openai_fallback
    _openai_fallback = func

async def provider_stream(query: str, context: str, citations: list) -> AsyncGenerator[str, None]:
    system_prompt = f"""You are a helpful assistant. Answer based only on the context. Cite sources as [Source N]. If unsure, say so.

Context:
{context}
"""
    if PRIMARY_PROVIDER == "ollama":
        try:
            async for token in ollama_generate(query, system_prompt):
                yield token
            return
        except Exception as e:
            print(f"Ollama failed: {e}. Falling back to OpenAI.")

    if _openai_fallback:
        async for token in _openai_fallback(query, context, citations):
            yield token
    else:
        raise RuntimeError("No OpenAI fallback available")