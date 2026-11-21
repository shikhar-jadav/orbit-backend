"""
Upstash Redis Client for Distributed Caching, Fast Session Authentication,
and Global Rate Limiting across Orbit Workers.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("orbit.redis")

_redis_instance = None
_redis_initialized = False


def get_redis():
    """
    Returns an Upstash Redis client instance if configured in the environment.
    Gracefully returns None if credentials are missing or connection fails.
    """
    global _redis_instance, _redis_initialized
    if _redis_initialized:
        return _redis_instance

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    url = os.getenv("UPSTASH_REDIS_REST_URL", "").strip()
    token = os.getenv("UPSTASH_REDIS_REST_TOKEN", "").strip()

    if not url or not token:
        _redis_initialized = True
        _redis_instance = None
        return None

    try:
        from upstash_redis import Redis

        _redis_instance = Redis(url=url, token=token)
        # Test connection
        _redis_instance.ping()
        logger.info("[Redis] Upstash Redis distributed cache connected successfully.")
    except Exception as e:
        logger.warning(f"[Redis] Could not connect to Upstash Redis ({e}); falling back to in-memory.")
        _redis_instance = None

    _redis_initialized = True
    return _redis_instance


# ── Session Caching ──────────────────────────────────────────────────────────

SESSION_CACHE_PREFIX = "orbit:session:"
DEFAULT_SESSION_TTL = 86400  # 24 hours
_in_memory_sessions: Dict[str, Tuple[Dict[str, Any], float]] = {}


def get_cached_session(hashed_token: str) -> Optional[Dict[str, Any]]:
    """Retrieve cached user session from Redis (or in-memory fallback) with sub-millisecond response time."""
    redis = get_redis()
    if not redis:
        item = _in_memory_sessions.get(hashed_token)
        if item:
            data, expires_at = item
            if time.time() < expires_at:
                return data
            _in_memory_sessions.pop(hashed_token, None)
        return None
    try:
        data = redis.get(f"{SESSION_CACHE_PREFIX}{hashed_token}")
        if data:
            if isinstance(data, str):
                return json.loads(data)
            return data
    except Exception as e:
        logger.debug(f"[Redis] Session cache lookup failed: {e}")
    return None


def set_cached_session(
    hashed_token: str,
    user_data: Dict[str, Any],
    ttl_seconds: int = DEFAULT_SESSION_TTL,
) -> None:
    """Store authenticated user session in Redis (or in-memory fallback) with expiration TTL."""
    _in_memory_sessions[hashed_token] = (user_data, time.time() + ttl_seconds)
    redis = get_redis()
    if not redis:
        return
    try:
        payload = json.dumps(user_data, ensure_ascii=False)
        redis.set(f"{SESSION_CACHE_PREFIX}{hashed_token}", payload, ex=ttl_seconds)
    except Exception as e:
        logger.debug(f"[Redis] Session cache store failed: {e}")


def invalidate_session(hashed_token: str) -> None:
    """Evict a session token from Redis and in-memory cache (e.g. on logout or password change)."""
    _in_memory_sessions.pop(hashed_token, None)
    redis = get_redis()
    if not redis:
        return
    try:
        redis.delete(f"{SESSION_CACHE_PREFIX}{hashed_token}")
    except Exception as e:
        logger.debug(f"[Redis] Session cache invalidation failed: {e}")


# ── Distributed Rate Limiter ─────────────────────────────────────────────────

class DistributedRateLimiter:
    """
    Sliding-window distributed rate limiter backed by Upstash Redis.
    Falls back gracefully to local in-memory sliding window if Redis is unavailable.
    """

    def __init__(
        self,
        max_attempts: int = 5,
        window_seconds: int = 900,
        prefix: str = "orbit:ratelimit:",
    ):
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self.prefix = prefix
        self._local_history: Dict[str, list[float]] = {}

    def check(self, key: str) -> Tuple[bool, int]:
        """
        Check if an action is permitted under the rate limit.
        Returns: (is_allowed: bool, retry_after_seconds: int)
        """
        redis = get_redis()
        now = time.time()
        redis_key = f"{self.prefix}{key}"

        if redis:
            try:
                # Use a sorted set window or counter in Redis
                # Prune items older than window
                cutoff = now - self.window_seconds
                redis.zremrangebyscore(redis_key, 0, cutoff)
                count = redis.zcard(redis_key)
                if count >= self.max_attempts:
                    oldest = redis.zrange(redis_key, 0, 0, withscores=True)
                    retry_after = (
                        int(oldest[0][1] + self.window_seconds - now) + 1
                        if oldest
                        else self.window_seconds
                    )
                    return False, max(1, retry_after)
                return True, 0
            except Exception as e:
                logger.debug(f"[Redis] Rate limiter check error: {e}")

        # Local in-memory fallback
        timestamps = self._local_history.get(key, [])
        valid = [t for t in timestamps if now - t < self.window_seconds]
        self._local_history[key] = valid
        if len(valid) >= self.max_attempts:
            oldest = valid[0]
            retry_after = int(oldest + self.window_seconds - now) + 1
            return False, max(1, retry_after)
        return True, 0

    def record_failure(self, key: str) -> None:
        """Record an attempt/failure against the rate limit window."""
        redis = get_redis()
        now = time.time()
        redis_key = f"{self.prefix}{key}"

        if redis:
            try:
                redis.zadd(redis_key, {f"{now}:{os.urandom(4).hex()}": now})
                redis.expire(redis_key, self.window_seconds + 60)
            except Exception as e:
                logger.debug(f"[Redis] Rate limiter record error: {e}")

        # Local in-memory fallback
        timestamps = self._local_history.setdefault(key, [])
        timestamps.append(now)

    def reset(self, key: str) -> None:
        """Reset the rate limit window for a specific key (e.g. on successful login)."""
        redis = get_redis()
        redis_key = f"{self.prefix}{key}"
        if redis:
            try:
                redis.delete(redis_key)
            except Exception:
                pass
        self._local_history.pop(key, None)
