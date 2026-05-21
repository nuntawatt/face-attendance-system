"""
Application lifespan manager

ลำดับ startup สำคัญมาก:
1. เชื่อมต่อ Redis
2. โหลด AI model (blocking ต้องเสร็จก่อน serve request)
3. สร้าง embedding index จาก DB ใหม่
4. เริ่ม camera worker เป็น background task

ลำดับ shutdown (ย้อนกลับ):
1. ส่งสัญญาณให้ camera worker หยุด
2. รอ task drain
3. คืน resource (Redis, DB connection)

ถ้ากล้องตัวหนึ่ง fail กล้องอื่นๆ ต้องทำงานต่อ
asyncio.gather(*tasks, return_exceptions=True) ทำให้แน่ใจเรื่องนี้
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from redis import asyncio as aioredis
from fastapi import FastAPI

from app.ai.engine import face_engine
from app.ai.recognition import embedding_index
from app.attendance.engine import AttendanceEngine
from app.camera.stream_reader import CameraConfig
from app.core.config import settings
from app.database.session import async_session_factory
from app.services.embedding_cache_service import EmbeddingCacheService

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("กำลังเริ่มต้น application")

    # 1. เชื่อมต่อ Redis ต้องพร้อมก่อน camera worker ที่ใช้ dedup
    app.state.redis = aioredis.from_url(str(settings.redis_url))
    logger.info("redis_connected", url=settings.redis_host)

    # 2. โหลด AI engine (blocking จงใจ)
    face_engine.load()

    # 3. สร้าง embedding index ใหม่
    async with async_session_factory() as session:
        cache_service = EmbeddingCacheService(session)
        await cache_service.rebuild_index()

    # 4. เริ่ม camera worker
    stop_event = asyncio.Event()
    camera_tasks: list[asyncio.Task] = []

    for cam_config in settings.camera_configs:
        engine = AttendanceEngine(
            config=cam_config,
            session_factory=async_session_factory,
            redis_client=app.state.redis,
            stop_event=stop_event,
        )
        task = asyncio.create_task(
            engine.run(),
            name=f"camera-{cam_config.camera_id}",
        )
        camera_tasks.append(task)

    logger.info("application พร้อมใช้งาน", จำนวนกล้อง=len(camera_tasks))
    yield  # app ทำงานอยู่ที่นี่

    # Shutdown
    logger.info("กำลังปิด application")
    stop_event.set()  # ส่งสัญญาณให้ทุก camera worker หยุด

    if camera_tasks:
        await asyncio.gather(*camera_tasks, return_exceptions=True)

    # คืน resource ปิด Redis connection ป้องกัน connection leak
    await app.state.redis.close()
    logger.info("application ปิดแล้ว")
