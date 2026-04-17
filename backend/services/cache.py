import json
import os
from functools import lru_cache

import redis.asyncio as redis


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_ENABLED = os.getenv("REDIS_ENABLED", "false").lower() == "true"
_redis_client = redis.from_url(REDIS_URL, decode_responses=True) if REDIS_ENABLED else None


def local_cache_key(power: float, time: float, antenna_type: str, model_name: str | None) -> str:
    return f"{round(power,3)}:{round(time,3)}:{antenna_type}:{model_name or 'best'}"


@lru_cache(maxsize=1024)
def local_cached_prediction(key: str) -> str | None:
    return None


async def get_prediction_cache(key: str) -> dict | None:
    if _redis_client:
        payload = await _redis_client.get(key)
        return json.loads(payload) if payload else None
    return None


async def set_prediction_cache(key: str, value: dict, ttl_seconds: int = 3600) -> None:
    if _redis_client:
        await _redis_client.set(key, json.dumps(value), ex=ttl_seconds)
