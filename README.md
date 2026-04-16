# RAG MVP – Alpha Release (Ollama + OpenAI Fallback)

## Quick Start

1. **Install Ollama** (if using local LLM)  
   https://ollama.com  
   Pull a model: `ollama pull llama3`

2. **Set environment variables**  
   Copy `.env.example` to `.env` and fill in your Pinecone & OpenAI keys.

3. **Run with Docker Compose**  
   ```bash
   docker compose up --build
   ```
   Open browser at http://localhost:8000

## How it works
- **Primary LLM**: Ollama (local, free) – uses llama3 model by default.
- **Fallback LLM**: OpenAI GPT-4 Turbo – used if Ollama fails or is unreachable.
- All RAG features (hybrid search, reranking, multi‑tenant, streaming) work with either provider.

## Configuration
| Env var | Description |
|---------|-------------|
| `PRIMARY_PROVIDER` | `ollama` (default) or `openai` |
| `OLLAMA_URL` | Ollama API endpoint (default `http://localhost:11434`) |
| `OLLAMA_MODEL` | Model name in Ollama (default `llama3`) |
| `OPENAI_API_KEY` | Required for fallback |

## Development without Docker
```bash
pip install -r requirements.txt
ollama pull llama3
export PRIMARY_PROVIDER=ollama
export OLLAMA_URL=http://localhost:11434
uvicorn main:app --reload
```

## API Endpoints
- `POST /auth/register` – create account
- `POST /auth/login` – get tokens
- `POST /auth/refresh` – refresh access token
- `POST /documents` – upload PDF/TXT (async processing)
- `GET /documents` – list documents
- `DELETE /documents/{id}` – delete document
- `POST /query` – ask a question (SSE stream)
- `POST /feedback/{query_id}` – submit feedback
- `GET /health` – health check
- `GET /metrics` – Prometheus metrics

## Production considerations
- Replace BackgroundTasks with Celery for durable document processing.
- Add token revocation and refresh rotation.
- Split main.py into modules for maintainability.
- Use a reverse proxy (nginx) with buffering disabled for SSE.

## License
MIT