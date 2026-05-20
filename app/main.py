"""
FastAPI application factory.

Thin as possible — just wires up middleware, routers, exception handlers,
and the lifespan context. No business logic here.
"""
from __future__ import annotations

# pyrefly: ignore [missing-import]
from redis import asyncio as aioredis
# pyrefly: ignore [missing-import]
from fastapi import FastAPI
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import employee
import app.models  # noqa: F401
from app.core.config import settings
from app.core.exception_handlers import app_error_handler, unhandled_exception_handler
from app.core.exceptions import AppError
from app.core.lifespan import lifespan
from app.core.logging import configure_logging
from app.core.middleware import RequestTracingMiddleware

configure_logging(log_level=settings.log_level, json_logs=settings.is_production)


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs" if not settings.is_production else None,
        redoc_url=None,
    )

    # Middleware (order matters outermost registered last)
    app.add_middleware(RequestTracingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handlers
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)

    # Routers
    app.include_router(employee.router, prefix="/api/v1")

    @app.on_event("startup")
    async def _init_redis() -> None:
        app.state.redis = aioredis.from_url(str(settings.redis_url))

    return app


app = create_app()