"""
FastAPI application factory

บางที่สุดเท่าที่จะทำได้ แค่ wire middleware, router,
exception handler, และ lifespan context ไม่มี business logic ที่นี่เลย
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import attendance, employee, face
from app.api.v1 import health
from app.websocket import ws_router

from app.core.config import settings
from app.core.exception_handlers import app_error_handler, unhandled_exception_handler
from app.core.exceptions import AppError
from app.core.lifespan import lifespan
from app.core.logging import configure_logging
from app.core.middleware import RequestTracingMiddleware

# ตั้งค่า logging ก่อนสร้าง app ต้องเป็นบรรทัดแรกเสมอ
configure_logging(log_level=settings.log_level, json_logs=settings.is_production)


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        lifespan=lifespan,
        # ปิด docs ใน production เพื่อ security hardening
        docs_url="/docs" if not settings.is_production else None,
        redoc_url=None,
    )

    # Middleware (ลำดับสำคัญ ตัวนอกสุด register ทีหลัง)
    app.add_middleware(RequestTracingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handlers จัดการทุก exception ที่ bubble ขึ้นมา
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)

    # API Routers
    app.include_router(health.router)  # /health (root level)
    app.include_router(employee.router, prefix="/api/v1")
    app.include_router(face.router, prefix="/api/v1")
    app.include_router(attendance.router, prefix="/api/v1")

    # WebSocket real-time attendance feed (/ws/attendance)
    app.include_router(ws_router)

    return app


app = create_app()