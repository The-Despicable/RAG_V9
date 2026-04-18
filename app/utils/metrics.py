from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
from functools import wraps
from typing import Callable
import time


class Metrics:
    def __init__(self):
        self.registry = CollectorRegistry()
        
        self.http_requests = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.http_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request latency in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        self.active_users = Gauge(
            'active_users_total',
            'Number of active users',
            registry=self.registry
        )
        
        self.document_processing = Counter(
            'documents_processing_total',
            'Total documents processed',
            ['status'],
            registry=self.registry
        )
        
        self.llm_requests = Counter(
            'llm_requests_total',
            'Total LLM requests',
            ['provider', 'model', 'status'],
            registry=self.registry
        )
        
        self.llm_duration = Histogram(
            'llm_request_duration_seconds',
            'LLM request latency in seconds',
            ['provider', 'model'],
            registry=self.registry
        )
        
        self.token_usage = Counter(
            'tokens_used_total',
            'Total tokens used',
            ['type', 'model'],
            registry=self.registry
        )
        
        self.cache_hits = Counter(
            'cache_hits_total',
            'Cache hits',
            ['cache_type'],
            registry=self.registry
        )
        
        self.cache_misses = Counter(
            'cache_misses_total',
            'Cache misses',
            ['cache_type'],
            registry=self.registry
        )
    
    def track_request(self, method: str, endpoint: str):
        self.http_requests.labels(method=method, endpoint=endpoint, status='total').inc()
    
    def track_response(self, method: str, endpoint: str, status: int):
        self.http_requests.labels(method=method, endpoint=endpoint, status=str(status)).inc()
    
    def track_latency(self, method: str, endpoint: str, duration: float):
        self.http_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    def track_llm(self, provider: str, model: str, status: str, duration: float = None):
        self.llm_requests.labels(provider=provider, model=model, status=status).inc()
        if duration:
            self.llm_duration.labels(provider=provider, model=model).observe(duration)
    
    def track_tokens(self, token_type: str, model: str, count: int):
        self.token_usage.labels(type=token_type, model=model).inc(count)
    
    def track_cache_hit(self, cache_type: str):
        self.cache_hits.labels(cache_type=cache_type).inc()
    
    def track_cache_miss(self, cache_type: str):
        self.cache_misses.labels(cache_type=cache_type).inc()
    
    def generate_metrics(self) -> bytes:
        return generate_latest(self.registry)


metrics = Metrics()


def timed(metric_name: str = None):
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                duration = time.time() - start
                if metric_name:
                    metrics.http_duration.labels(method='unknown', endpoint=metric_name).observe(duration)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.time() - start
                if metric_name:
                    metrics.http_duration.labels(method='unknown', endpoint=metric_name).observe(duration)
        
        if hasattr(func, '__code__') and func.__code__.co_flags & 0x200:
            return async_wrapper
        return sync_wrapper
    
    return decorator