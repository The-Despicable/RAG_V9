import time
import asyncio
import random
from functools import wraps
from typing import Callable, Optional, Any, TypeVar
from enum import Enum

T = TypeVar('T')

from app.middleware.logging import logger


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpenError(Exception):
    pass


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
    
    def call(self, func: Callable, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info(f"{self.name}: Entering HALF_OPEN state")
            else:
                raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    async def call_async(self, func: Callable, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info(f"{self.name}: Entering HALF_OPEN state")
            else:
                raise CircuitBreakerOpenError(f"Circuit {self.name} is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self.success_count = 0
            logger.info(f"{self.name}: Recovered, returning to CLOSED state")
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.error(f"{self.name}: Circuit breaker OPEN after {self.failure_count} failures")
    
    def _should_attempt_reset(self) -> bool:
        if not self.last_failure_time:
            return True
        return time.time() - self.last_failure_time >= self.recovery_timeout


circuit_breakers = {
    "ollama": CircuitBreaker(name="ollama", failure_threshold=5, recovery_timeout=600),
    "openrouter": CircuitBreaker(name="openrouter", failure_threshold=3, recovery_timeout=300),
    "openai": CircuitBreaker(name="openai", failure_threshold=3, recovery_timeout=300),
}


def get_circuit_breaker(provider: str) -> CircuitBreaker:
    return circuit_breakers.get(provider, CircuitBreaker(name=provider))


def retry_with_backoff(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: tuple = (Exception,)
):
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        logger.error(
                            f"Max retries exceeded for {func.__name__}",
                            attempts=max_attempts,
                            error=str(e)
                        )
                        raise
                    
                    if jitter:
                        delay_with_jitter = delay * random.uniform(0.5, 1.5)
                    else:
                        delay_with_jitter = delay
                    
                    delay_with_jitter = min(delay_with_jitter, max_delay)
                    
                    logger.warning(
                        f"Retry attempt {attempt + 1}/{max_attempts} for {func.__name__}",
                        delay=delay_with_jitter,
                        error=str(e)
                    )
                    await asyncio.sleep(delay_with_jitter)
                    delay *= exponential_base
            
            raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        logger.error(
                            f"Max retries exceeded for {func.__name__}",
                            attempts=max_attempts,
                            error=str(e)
                        )
                        raise
                    
                    if jitter:
                        delay_with_jitter = delay * random.uniform(0.5, 1.5)
                    else:
                        delay_with_jitter = delay
                    
                    delay_with_jitter = min(delay_with_jitter, max_delay)
                    
                    logger.warning(
                        f"Retry attempt {attempt + 1}/{max_attempts} for {func.__name__}",
                        delay=delay_with_jitter,
                        error=str(e)
                    )
                    time.sleep(delay_with_jitter)
                    delay *= exponential_base
            
            raise last_exception
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retriable_exceptions: tuple = (Exception,)
):
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retriable_exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts:
                        logger.error(f"Max retries exceeded for {func.__name__}")
                        raise
                    
                    delay_with_jitter = delay * random.uniform(0.5, 1.5) if jitter else delay
                    delay_with_jitter = min(delay_with_jitter, max_delay)
                    
                    logger.warning(f"Retry {attempt}/{max_attempts} for {func.__name__}")
                    time.sleep(delay_with_jitter)
                    delay *= exponential_base
            
            raise last_exception
        
        return wrapper
    return decorator