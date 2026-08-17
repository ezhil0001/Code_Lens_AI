"""
JWT Token Blacklist & Revocation System - Enterprise Grade

Implements token revocation for logout functionality.
Supports both Redis (production) and in-memory cache (fallback).

Hardening Feature #1: JWT Revocation & Blacklisting
- Redis-based blacklist (production)
- In-memory fallback (development)
- JTI (JWT ID) tracking
- Automatic TTL-based expiration
"""

import logging
from typing import Optional, Set
from datetime import datetime, timedelta, timezone
from abc import ABC, abstractmethod
import json

logger = logging.getLogger(__name__)


def _to_aware_utc(dt: datetime) -> datetime:
    """Normalize a datetime to aware-UTC (naive values are assumed UTC)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class TokenBlacklistBackend(ABC):
    """Abstract base class for token blacklist backends"""
    
    @abstractmethod
    def add_token(self, jti: str, expiration_time: datetime) -> bool:
        """Add token JTI to blacklist"""
        pass
    
    @abstractmethod
    def is_blacklisted(self, jti: str) -> bool:
        """Check if token JTI is blacklisted"""
        pass
    
    @abstractmethod
    def clear_expired(self) -> int:
        """Clear expired tokens, returns count"""
        pass


class InMemoryTokenBlacklist(TokenBlacklistBackend):
    """
    In-memory token blacklist (fallback)
    
    Stores JTI and expiration time in memory.
    Suitable for development and single-instance deployments.
    Note: Not suitable for multi-instance deployments.
    """
    
    def __init__(self):
        """Initialize in-memory blacklist"""
        self.blacklist: dict[str, datetime] = {}
        logger.info("✓ Initialized in-memory token blacklist (fallback)")
    
    def add_token(self, jti: str, expiration_time: datetime) -> bool:
        """
        Add token JTI to blacklist
        
        Args:
            jti: JWT ID (unique identifier)
            expiration_time: Token expiration datetime
        
        Returns:
            True if added successfully
        """
        try:
            self.blacklist[jti] = _to_aware_utc(expiration_time)
            logger.debug(f"Token added to blacklist: {jti}")
            return True
        except Exception as e:
            logger.error(f"Failed to add token to blacklist: {e}")
            return False
    
    def is_blacklisted(self, jti: str) -> bool:
        """
        Check if token JTI is blacklisted
        
        Args:
            jti: JWT ID to check
        
        Returns:
            True if blacklisted, False otherwise
        """
        if jti not in self.blacklist:
            return False
        
        # Check if token has expired
        expiration_time = self.blacklist[jti]
        if datetime.now(timezone.utc) > expiration_time:
            # Remove expired token
            del self.blacklist[jti]
            return False
        
        return True
    
    def clear_expired(self) -> int:
        """
        Clear expired tokens from blacklist
        
        Returns:
            Number of tokens removed
        """
        now = datetime.now(timezone.utc)
        expired_jtis = [
            jti for jti, exp_time in self.blacklist.items()
            if exp_time < now
        ]
        
        for jti in expired_jtis:
            del self.blacklist[jti]
        
        if expired_jtis:
            logger.debug(f"Cleared {len(expired_jtis)} expired tokens")
        
        return len(expired_jtis)


class RedisTokenBlacklist(TokenBlacklistBackend):
    """
    Redis-based token blacklist (production)
    
    Stores JTI and expiration time in Redis.
    Suitable for distributed deployments and high-scale systems.
    Automatic expiration via Redis TTL.
    """
    
    def __init__(self, redis_url: str):
        """
        Initialize Redis token blacklist
        
        Args:
            redis_url: Redis connection URL (redis://localhost:6379)
        """
        try:
            import redis
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            self.prefix = "token_blacklist:"
            logger.info("✓ Initialized Redis token blacklist (production)")
        except ImportError:
            raise ImportError("redis package required for Redis blacklist. Install: pip install redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def add_token(self, jti: str, expiration_time: datetime) -> bool:
        """
        Add token JTI to Redis blacklist with TTL
        
        Args:
            jti: JWT ID (unique identifier)
            expiration_time: Token expiration datetime
        
        Returns:
            True if added successfully
        """
        try:
            key = f"{self.prefix}{jti}"
            # Calculate TTL in seconds
            ttl = int((_to_aware_utc(expiration_time) - datetime.now(timezone.utc)).total_seconds())
            
            if ttl <= 0:
                logger.debug(f"Token already expired, not adding to blacklist: {jti}")
                return False
            
            # Set in Redis with TTL
            self.redis_client.setex(key, ttl, "revoked")
            logger.debug(f"Token added to Redis blacklist: {jti} (TTL: {ttl}s)")
            return True
        except Exception as e:
            logger.error(f"Failed to add token to Redis blacklist: {e}")
            return False
    
    def is_blacklisted(self, jti: str) -> bool:
        """
        Check if token JTI is in Redis blacklist
        
        Args:
            jti: JWT ID to check
        
        Returns:
            True if blacklisted, False otherwise
        """
        try:
            key = f"{self.prefix}{jti}"
            result = self.redis_client.exists(key)
            return bool(result)
        except Exception as e:
            logger.error(f"Failed to check Redis blacklist: {e}")
            # Fail open: if Redis is down, don't block legitimate requests
            return False
    
    def clear_expired(self) -> int:
        """
        Clear expired tokens from Redis
        
        Note: Redis automatically expires keys via TTL,
        so this is mostly for cleanup of orphaned keys.
        
        Returns:
            Number of tokens removed
        """
        try:
            cursor = 0
            removed = 0
            
            while True:
                cursor, keys = self.redis_client.scan(
                    cursor,
                    match=f"{self.prefix}*",
                    count=100
                )
                
                for key in keys:
                    # Redis automatically expires keys, so this is a no-op
                    # Just count existing keys
                    pass
                
                if cursor == 0:
                    break
            
            logger.debug(f"Redis token blacklist cleanup completed")
            return removed
        except Exception as e:
            logger.error(f"Failed to clear Redis blacklist: {e}")
            return 0


class TokenBlacklistManager:
    """
    Token Blacklist Manager
    
    Provides centralized interface for token revocation.
    Automatically selects backend (Redis or in-memory).
    """
    
    _instance: Optional['TokenBlacklistManager'] = None
    _backend: Optional[TokenBlacklistBackend] = None
    
    def __new__(cls, redis_url: Optional[str] = None):
        """Singleton pattern for token blacklist manager"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize token blacklist manager
        
        Args:
            redis_url: Redis URL for production. If None, uses in-memory fallback.
        """
        if self._backend is not None:
            return  # Already initialized
        
        try:
            if redis_url:
                self._backend = RedisTokenBlacklist(redis_url)
                logger.info("✓ Token blacklist using Redis backend")
            else:
                self._backend = InMemoryTokenBlacklist()
                logger.info("✓ Token blacklist using in-memory backend")
        except Exception as e:
            # In-memory revocation is per-process: with >1 worker, or after a
            # restart, a "revoked" token becomes valid again. If Redis was
            # explicitly configured, silently downgrading is a security
            # regression, so make it impossible to miss.
            logger.error(
                "REDIS TOKEN BLACKLIST UNAVAILABLE (%s) — falling back to "
                "in-memory. Logout will NOT revoke tokens across processes or "
                "restarts. Install `redis` and verify REDIS_URL.", e,
            )
            self._backend = InMemoryTokenBlacklist()
    
    def revoke_token(self, jti: str, expiration_time: datetime) -> bool:
        """
        Revoke a token by adding its JTI to the blacklist
        
        Args:
            jti: JWT ID (unique identifier)
            expiration_time: Token expiration datetime
        
        Returns:
            True if revocation successful
        """
        if not self._backend:
            logger.error("Token blacklist backend not initialized")
            return False
        
        return self._backend.add_token(jti, expiration_time)
    
    def is_token_revoked(self, jti: str) -> bool:
        """
        Check if a token has been revoked
        
        Args:
            jti: JWT ID to check
        
        Returns:
            True if token is revoked, False otherwise
        """
        if not self._backend:
            logger.error("Token blacklist backend not initialized")
            return False
        
        return self._backend.is_blacklisted(jti)
    
    def cleanup_expired_tokens(self) -> int:
        """
        Clean up expired tokens from blacklist
        
        Returns:
            Number of tokens removed
        """
        if not self._backend:
            return 0
        
        return self._backend.clear_expired()


# Global instance (lazy initialization in main.py)
_token_blacklist_manager: Optional[TokenBlacklistManager] = None


def get_token_blacklist_manager(redis_url: Optional[str] = None) -> TokenBlacklistManager:
    """
    Get or create token blacklist manager instance
    
    Args:
        redis_url: Redis URL for production
    
    Returns:
        TokenBlacklistManager instance
    """
    global _token_blacklist_manager
    if _token_blacklist_manager is None:
        _token_blacklist_manager = TokenBlacklistManager(redis_url)
    return _token_blacklist_manager
