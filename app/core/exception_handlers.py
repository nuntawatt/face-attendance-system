"""
Global FastAPI exception handlers

นี่คือ **ที่เดียว** ที่แปลง domain exception เป็น HTTP response
Router ไม่ catch exception เลยปล่อยให้ bubble ขึ้นมาถึงที่นี่
"""
from __future__ import annotations

import structlog
from fastapi import Request
from fastapi.responses import JSONResponse

from app.core.exceptions import (
    AppError,
    ConflictError,
    NotFoundError,
    ServiceUnavailableError,
    ValidationError,
)

logger = structlog.get_logger(__name__)


def _error_response(status: int, code: str, message: str) -> JSONResponse:
    """สร้าง error response ในรูปแบบมาตรฐานของระบบ"""
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message}},
    )


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """จัดการ domain exception ทั้งหมด แปลงเป็น HTTP response ที่เหมาะสม"""
    error_class = type(exc).__name__

    if isinstance(exc, NotFoundError):
        logger.warning("not_found", error=error_class, message=exc.message)
        return _error_response(404, error_class, exc.message)

    if isinstance(exc, ConflictError):
        logger.warning("conflict", error=error_class, message=exc.message)
        return _error_response(409, error_class, exc.message)

    if isinstance(exc, ValidationError):
        logger.warning("validation_error", error=error_class, message=exc.message)
        return _error_response(422, error_class, exc.message)

    if isinstance(exc, ServiceUnavailableError):
        logger.error("service_unavailable", error=error_class, message=exc.message)
        return _error_response(503, error_class, exc.message)

    # Exception ที่ไม่คาดคิด log full stack trace
    logger.exception("unhandled_app_error", error=error_class)
    return _error_response(500, "InternalError", "เกิดข้อผิดพลาดภายในระบบ")


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """จัดการ exception ที่ไม่ใช่ AppError สิ่งที่ไม่ควรเกิดขึ้นใน production"""
    logger.exception("unhandled_exception", exc_type=type(exc).__name__)
    return _error_response(500, "InternalError", "เกิดข้อผิดพลาดภายในระบบ")