"""
Health check endpoint

ทุก production system ต้องมี health check สำหรับ:
- Load balancer probe (ALB, Nginx, K8s liveness)
- Monitoring dashboard (Uptime Robot, Datadog)
- CI/CD deployment verification

Health check ตรวจสอบ dependency ที่สำคัญ (DB, Redis, AI engine)
ถ้าตัวใดตัวหนึ่ง unhealthy ก็ยังคืน 200 แต่ระบุ status ให้ชัดเจน
เพราะ partial degradation ดีกว่า full downtime
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.ai.engine import face_engine
from app.ai.recognition import embedding_index
from app.websocket.manager import attendance_manager

router = APIRouter(tags=["ระบบ"])


class HealthResponse(BaseModel):
    status: str
    database: str
    redis: str
    ai_engine: str
    embedding_index_size: int
    ws_connections: int


@router.get("/health", response_model=HealthResponse, summary="ตรวจสอบสถานะระบบ")
async def health_check(request: Request) -> HealthResponse:
    """ตรวจสอบว่า dependency หลักทั้งหมดพร้อมใช้งาน"""

    # ตรวจ Database
    database_status = "unhealthy"
    try:
        from app.database.session import async_session_factory
        from sqlalchemy import text

        async with async_session_factory() as session:
            await session.execute(text("SELECT 1"))
        database_status = "connected"
    except Exception:
        pass

    # ตรวจ Redis
    redis_status = "unhealthy"
    try:
        redis = request.app.state.redis
        await redis.ping()
        redis_status = "healthy"
    except Exception:
        pass

    # ตรวจ AI engine
    ai_status = "healthy" if face_engine.is_ready else "not_loaded"

    return HealthResponse(
        status="ok",
        database=database_status,
        redis=redis_status,
        ai_engine=ai_status,
        embedding_index_size=embedding_index.size,
        ws_connections=attendance_manager.active_count,
    )
