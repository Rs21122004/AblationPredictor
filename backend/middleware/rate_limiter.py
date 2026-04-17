import os

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address


ANON_LIMIT = os.getenv("RATE_LIMIT_ANON", "60/minute")
AUTH_LIMIT = os.getenv("RATE_LIMIT_AUTH", "200/minute")

limiter = Limiter(key_func=get_remote_address)


def prediction_rate_limit(request: Request) -> str:
    auth_header = request.headers.get("authorization")
    return AUTH_LIMIT if auth_header else ANON_LIMIT
