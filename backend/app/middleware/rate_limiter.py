"""
API Rate Limiting System - Enterprise Grade

Implements brute-force prevention and API throttling.
Protects against:
- Brute-force login attacks (5 attempts/minute)
- General API abuse (100 requests/minute)
- Distributed attacks (IP-based tracking)

Hardening Feature #2: API Rate Limiting

Supports both Redis (production) and in-memory backends.
"""

import logging
from typing import Optional, Dict, Tuple
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import wraps
import time

logger = logging.getLogger(__name__)


class RateLimitBackend(ABC):
    """Abstract base class for rate limiting backends"""
    
    @abstractmethod
    def is_allowed(self, key: str, limit: int, window_seconds: int) -> Tuple[bool, Dict[str, int]]:
        """Check if request is allowed and return stats"""
        pass


class InMemoryRateLimiter(RateLimitBackend):
    """
    In-memory rate limiter (fallback)
    
    Stores request counts in memory.
    Suitable for development and single-instance deployments.
    """
    
    def __init__(self):
        """Initialize in-memory rate limiter"""
        self.requests: Dict[str, list] = defaultdict(list)
        logger.info("✓ Initialized in-memory rate limiter (fallback)")
    
    def is_allowed(self, key: str, limit: int, window_seconds: int) -> Tuple[bool, Dict[str, int]]:
        """
        Check if request is allowed
        
        Args:
            key: Rate limit key (e.g., "login:192.168.1.1")
            limit: Max requests allowed
            window_seconds: Time window in seconds
        
        Returns:
            Tuple of (allowed: bool, stats: dict with current_count and reset_time)
        """
        now = time.time()
        window_start = now - window_seconds
        
        # Remove old requests outside the window
        self.requests[key] = [
            timestamp for timestamp in self.requests[key]
            if timestamp > window_start
        ]
        
        # Check if limit exceeded
        current_count = len(self.requests[key])
        allowed = current_count < limit
        
        # Add current request
        self.requests[key].append(now)
        
        # Calculate reset time
        if self.requests[key]:
            reset_time = int(self.requests[key][0] + window_seconds)
        else:
            reset_time = int(now + window_seconds)
        
        return allowed, {
            "limit": limit,
            "current": current_count,
            "remaining": max(0, limit - current_count - 1),
            "reset": reset_time
        }


class RedisRateLimiter(RateLimitBackend):
    """
    Redis-based rate limiter (production)
    
    Uses Redis INCR and EXPIRE commands for atomic, distributed rate limiting.
    Suitable for multi-instance deployments.
    """
    
    def __init__(self, redis_url: str):
        """
        Initialize Redis rate limiter
        
        Args:
            redis_url: Redis connection URL
        """
        try:
            import redis
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            self.prefix = "rate_limit:"
            logger.info("✓ Initialized Redis rate limiter (production)")
        except ImportError:
            raise ImportError("redis package required for Redis rate limiting. Install: pip install redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def is_allowed(self, key: str, limit: int, window_seconds: int) -> Tuple[bool, Dict[str, int]]:
        """
        Check if request is allowed using Redis
        
        Args:
            key: Rate limit key
            limit: Max requests allowed
            window_seconds: Time window in seconds
        
        Returns:
            Tuple of (allowed: bool, stats: dict)
        """
        try:
            full_key = f"{self.prefix}{key}"
            now = int(time.time())
            
            # Redis INCR returns the new count after incrementing
            current_count = self.redis_client.incr(full_key)
            
            # Set TTL on first request in window
            if current_count == 1:
                self.redis_client.expire(full_key, window_seconds)
            
            # Get TTL for reset time
            ttl = self.redis_client.ttl(full_key)
            reset_time = now + (ttl if ttl > 0 else window_seconds)
            
            allowed = current_count <= limit
            
            return allowed, {
                "limit": limit,
                "current": current_count,
                "remaining": max(0, limit - current_count),
                "reset": reset_time
            }
        except Exception as e:
            logger.error(f"Redis rate limit error: {e}")
            # Fail open: if Redis is down, allow requests
            return True, {"limit": limit, "current": 0, "remaining": limit, "reset": int(time.time() + window_seconds)}


class RateLimiter:
    """
    Rate Limiter Manager
    
    Provides centralized interface for rate limiting.
    Automatically selects backend (Redis or in-memory).
    """
    
    _instance: Optional['RateLimiter'] = None
    _backend: Optional[RateLimitBackend] = None
    
    def __new__(cls, redis_url: Optional[str] = None):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, redis_url: Optional[str] = None):
        """Initialize rate limiter"""
        if self._backend is not None:
            return
        
        try:
            if redis_url:
                self._backend = RedisRateLimiter(redis_url)
                logger.info("✓ Rate limiter using Redis backend")
            else:
                self._backend = InMemoryRateLimiter()
                logger.info("✓ Rate limiter using in-memory backend")
        except Exception as e:
            logger.warning(f"Failed to initialize Redis rate limiter, using in-memory: {e}")
            self._backend = InMemoryRateLimiter()
    
    def is_allowed(self, key: str, limit: int, window_seconds: int) -> Tuple[bool, Dict[str, int]]:
        """
        Check if request is allowed
        
        Args:
            key: Rate limit key (e.g., "login:192.168.1.1")
            limit: Max requests allowed
            window_seconds: Time window in seconds
        
        Returns:
            Tuple of (allowed: bool, stats: dict)
        """
        if not self._backend:
            logger.error("Rate limiter backend not initialized")
            return True, {"limit": limit, "current": 0}
        
        return self._backend.is_allowed(key, limit, window_seconds)


# Global instance (lazy initialization in main.py)
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter(redis_url: Optional[str] = None) -> RateLimiter:
    """
    Get or create rate limiter instance
    
    Args:
        redis_url: Redis URL for production
    
    Returns:
        RateLimiter instance
    """
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(redis_url)
    return _rate_limiter


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded"""
    
    def __init__(self, limit: int, window_seconds: int, reset_time: int):
        self.limit = limit
        self.window_seconds = window_seconds
        self.reset_time = reset_time
        super().__init__(f"Rate limit exceeded: {limit} requests per {window_seconds}s")
