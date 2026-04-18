import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    ENVIRONMENT: str = "development"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    APP_DEBUG: bool = False

    FRONTEND_URL: str = "http://localhost:3000"
    BACKEND_URL: str = "http://localhost:8000"
    CORS_ORIGINS: str = '["http://localhost:3000"]'

    DATABASE_URL: str
    DB_POOL_MIN_SIZE: int = 5
    DB_POOL_MAX_SIZE: int = 20
    DB_ECHO: bool = False

    REDIS_URL: str
    REDIS_CLUSTER_ENABLED: bool = False
    CACHE_TTL_SESSIONS: int = 3600
    CACHE_TTL_BM25: int = 86400
    CACHE_TTL_EMBEDDINGS: int = 604800

    PINECONE_API_KEY: str
    PINECONE_INDEX: str = "rag-prod"
    PINECONE_ENVIRONMENT: str = "us-east-1"
    PINECONE_NAMESPACE_PREFIX: str = "ws_"
    EMBEDDING_DIM: int = 768

    OLLAMA_PROVIDER_ENABLED: bool = True
    OLLAMA_URL: str = "https://ollama.com"
    OLLAMA_API_KEY: Optional[str] = None
    OLLAMA_MODEL: str = "llama3.2"
    OLLAMA_TEMPERATURE: float = 0.7
    OLLAMA_MAX_TOKENS: int = 2000
    OLLAMA_TIMEOUT_SECONDS: int = 120
    OLLAMA_EMBEDDING_URL: str = "http://host.docker.internal:11434"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"
    OLLAMA_EMBEDDING_TIMEOUT: int = 60

    OPENROUTER_PROVIDER_ENABLED: bool = True
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_MODEL: str = "meta-llama/llama-2-70b-chat"
    OPENROUTER_TEMPERATURE: float = 0.7
    OPENROUTER_MAX_TOKENS: int = 2000
    OPENROUTER_TIMEOUT_SECONDS: int = 120

    OPENAI_PROVIDER_ENABLED: bool = True
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4-turbo"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_TEMPERATURE: float = 0.7
    OPENAI_MAX_TOKENS: int = 2000
    OPENAI_TIMEOUT_SECONDS: int = 30

    PRIMARY_LLM_PROVIDER: str = "ollama"
    FALLBACK_PROVIDERS: str = "openrouter,openai"
    PROVIDER_HEALTH_CHECK_INTERVAL: int = 300
    PROVIDER_CIRCUIT_BREAKER_THRESHOLD: int = 5
    PROVIDER_CIRCUIT_BREAKER_RESET: int = 600

    JWT_SECRET: str
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = 24
    JWT_REFRESH_EXPIRATION_DAYS: int = 30

    PASSWORD_MIN_LENGTH: int = 8
    SESSION_TIMEOUT_MINUTES: int = 60
    MAX_SESSIONS_PER_USER: int = 3

    RATE_LIMIT_REQUESTS: int = 30
    RATE_LIMIT_WINDOW_SECONDS: int = 60
    RATE_LIMIT_BY_IP: bool = False

    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: str = '["GET", "POST", "PUT", "DELETE", "OPTIONS"]'
    CORS_ALLOW_HEADERS: str = '["*"]'
    CORS_EXPOSE_HEADERS: str = '["X-Total-Count", "X-Request-ID", "X-RateLimit-Remaining"]'

    RAZORPAY_KEY_ID: Optional[str] = None
    RAZORPAY_KEY_SECRET: Optional[str] = None
    RAZORPAY_WEBHOOK_SECRET: Optional[str] = None
    RAZORPAY_DEFAULT_CURRENCY: str = "INR"

    FREE_PLAN_TOKENS: int = 50000
    PRO_PLAN_TOKENS: int = 1000000
    PRO_PLAN_PRICE_INR: int = 100
    PRO_PLAN_DURATION_DAYS: int = 30
    ENTERPRISE_PLAN_TOKENS: int = 10000000
    ENTERPRISE_PLAN_PRICE_INR: int = 1000

    TOKEN_COST_INPUT: float = 0.00001
    TOKEN_COST_OUTPUT: float = 0.00002
    TOKEN_BUFFER_PERCENTAGE: int = 5

    BILLING_CYCLE_DAY: int = 15
    AUTO_RENEW_ENABLED: bool = True

    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    LOG_TO_FILE: bool = False
    LOG_FILE_PATH: str = "/var/log/rag-saas/app.log"
    LOG_FILE_MAX_SIZE_MB: int = 100
    LOG_FILE_BACKUP_COUNT: int = 10

    PROMETHEUS_ENABLED: bool = True
    PROMETHEUS_PORT: int = 9090
    PROMETHEUS_NAMESPACE: str = "rag_saas"

    SENTRY_ENABLED: bool = False
    SENTRY_DSN: Optional[str] = None

    EMAIL_PROVIDER: str = "resend"
    RESEND_API_KEY: Optional[str] = None
    SENDGRID_API_KEY: Optional[str] = None
    EMAIL_FROM: str = "noreply@yourdomain.com"
    EMAIL_FROM_NAME: str = "RAG SaaS"

    SLACK_ENABLED: bool = False
    SLACK_WEBHOOK_URL: Optional[str] = None

    POSTHOG_ENABLED: bool = False
    POSTHOG_API_KEY: Optional[str] = None
    POSTHOG_API_HOST: str = "https://app.posthog.com"

    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    MIN_CHUNK_SIZE: int = 100
    SEMANTIC_CHUNKING_ENABLED: bool = True

    ENABLE_VISION: bool = True
    ENABLE_WHISPER: bool = True
    ENABLE_OCR: bool = True
    ENABLE_YOLO: bool = False

    OCR_LANGUAGE: str = "en"
    OCR_BATCH_SIZE: int = 4
    OCR_TIMEOUT_SECONDS: int = 60

    WHISPER_MODEL_SIZE: str = "base"
    WHISPER_DEVICE: str = "cpu"
    WHISPER_TIMEOUT_SECONDS: int = 300

    MAX_IMAGES_PER_DOCUMENT: int = 10
    MIN_IMAGE_SIZE_BYTES: int = 1000
    IMAGE_QUALITY: str = "high"

    CELERY_BROKER_URL: Optional[str] = None
    CELERY_RESULT_BACKEND: Optional[str] = None
    CELERY_TASK_TIMEOUT: int = 600
    CELERY_MAX_RETRIES: int = 3
    CELERY_RETRY_BACKOFF: int = 300

    TOP_K_DENSE: int = 20
    TOP_K_SPARSE: int = 20
    TOP_K_FUSED: int = 10
    TOP_K_FINAL: int = 5

    HYBRID_WEIGHT_DENSE: float = 0.6
    HYBRID_WEIGHT_SPARSE: float = 0.4

    ENABLE_RERANKING: bool = True
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANKER_BATCH_SIZE: int = 32
    RERANKER_THRESHOLD: float = 0.3

    ENABLE_METADATA_FILTERS: bool = True
    ENABLE_DATE_FILTERS: bool = True
    ENABLE_TYPE_FILTERS: bool = True

    MAX_FILE_MB: int = 10
    MAX_FILES_PER_BATCH: int = 20
    MAX_DOCUMENTS_PER_WORKSPACE: int = 1000
    MAX_STORAGE_GB_PER_WORKSPACE: float = 100.0

    WORKERS: int = 4
    WORKER_THREADS: int = 4
    SOCKET_TIMEOUT: int = 120
    MAX_MEMORY_MB: int = 2048
    GRACEFUL_SHUTDOWN_TIMEOUT: int = 30

    FEATURE_HYBRID_SEARCH: bool = True
    FEATURE_RERANKING: bool = True
    FEATURE_CITATIONS: bool = True
    FEATURE_FOLLOW_UP_PROMPTS: bool = True
    FEATURE_DOCUMENT_PREVIEW: bool = True
    FEATURE_BULK_UPLOAD: bool = True
    FEATURE_WORKSPACE_TEAMS: bool = True
    FEATURE_API_KEYS: bool = True

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()