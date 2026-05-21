"""
Request tracing middleware

Request ทุกรายการที่เข้ามาจะได้รับ trace_id ที่ inject เข้าไปใน structlog contextvars
หมายความว่า log ทุกบรรทัดที่ emit ระหว่าง request นั้นในทุก layer
จะมี trace_id โดยอัตโนมัติโดยไม่ต้องส่งต่อ parameter

ใน Kibana หรือ Datadog: filter ด้วย trace_id เดียว เห็นทุก log ตั้งแต่
router -> service -> repository -> AI engine ทั้งหมดในหนึ่ง request
"""
from __future__ import annotations

import time
import uuid
from typing import Any

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = structlog.get_logger(__name__)


class RequestTracingMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        # สร้าง trace_id ใหม่สำหรับแต่ละ request
        trace_id = str(uuid.uuid4())
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            trace_id=trace_id,
            method=request.method,
            path=request.url.path,
        )

        start = time.perf_counter()
        response: Response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        logger.info(
            "request_completed",
            status_code=response.status_code,
            duration_ms=duration_ms,
        )

        # ส่ง trace_id กลับใน response header เพื่อ debug ฝั่ง client
        response.headers["X-Trace-ID"] = trace_id
        return response