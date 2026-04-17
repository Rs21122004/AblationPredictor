from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response


REQUEST_COUNT = Counter("api_requests_total", "Total API requests", ["method", "path", "status"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["method", "path"])


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        method = request.method
        path = request.url.path
        with REQUEST_LATENCY.labels(method=method, path=path).time():
            response = await call_next(request)
        REQUEST_COUNT.labels(method=method, path=path, status=response.status_code).inc()
        return response


def metrics_response() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
