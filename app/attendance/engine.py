"""
Realtime attendance engine — the core production worker.

This is NOT a FastAPI route. It's a long-running asyncio coroutine
started in the application lifespan and runs for the lifetime of the app.

Architecture:
  CameraStreamReader -> frame_queue -> AttendanceEngine
      (producer)                            (consumer)

The queue decouples producers from the consumer. If recognition is slow,
frames back-pressure and are dropped (LIFO with maxsize=1 per camera).
Dropping frames is intentional for attendance, we care about the most
recent frame, not a backlog of old ones.

Recognition deduplication:
  Redis stores "recognized::{employee_id}::{date}" with a TTL of N hours.
  If the key exists, we skip DB write the employee is already checked in.
  This prevents duplicate attendance records when the person stands in
  front of the camera for multiple seconds.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import UUID

# pyrefly: ignore [missing-import]
import structlog

from app.ai.engine import face_engine
from app.ai.recognition import embedding_index
from app.camera.face_tracker import FaceTracker
from app.camera.stream_reader import CameraConfig, stream_frames
from app.core.config import settings

logger = structlog.get_logger(__name__)

ATTENDANCE_DEDUP_TTL = 4 * 3600  # 4 hours in seconds


class AttendanceEngine:
    """
    Orchestrates: stream reading -> face detection -> recognition -> DB write.
    One instance per camera. Run all cameras concurrently via asyncio.gather.
    """

    def __init__(
        self,
        config: CameraConfig,
        session_factory,
        redis_client,
        stop_event: asyncio.Event,
    ) -> None:
        self._config = config
        self._session_factory = session_factory
        self._redis = redis_client
        self._stop = stop_event
        self._tracker = FaceTracker()

    async def run(self) -> None:
        logger.info("attendance_engine_start", camera_id=self._config.camera_id)
        async for camera_id, frame in stream_frames(self._config, self._stop):
            try:
                await self._process_frame(frame)
            except Exception:
                logger.exception("frame_processing_error", camera_id=camera_id)
        logger.info("attendance_engine_stop", camera_id=self._config.camera_id)

    async def _process_frame(self, frame) -> None:
        faces = await face_engine.analyze_frame(frame)
        if not faces:
            return

        bboxes = [f.bbox for f in faces]
        track_indices = self._tracker.update(bboxes)

        recognition_tasks = []
        for i, (face, track_idx) in enumerate(zip(faces, track_indices)):
            if self._tracker.is_recognized(track_idx):
                continue  # already recognized this visit, skip
            recognition_tasks.append(
                self._recognize_and_record(face.embedding, track_idx)
            )

        if recognition_tasks:
            await asyncio.gather(*recognition_tasks, return_exceptions=True)

    async def _recognize_and_record(self, embedding, track_idx: int) -> None:
        match = await embedding_index.find_match(embedding)
        if match is None or not match.is_confident:
            return

        employee_id = match.employee_id
        self._tracker.mark_recognized(track_idx, employee_id)

        # Deduplication check
        dedup_key = self._dedup_key(employee_id)
        if await self._redis.exists(dedup_key):
            logger.debug("attendance_dedup_hit", employee_id=str(employee_id))
            return

        # Record attendance
        async with self._session_factory() as session:
            from app.repositories.attendance import AttendanceRepository
            repo = AttendanceRepository(session)
            from app.models.attendance import AttendanceRecord
            record = AttendanceRecord(
                employee_id=employee_id,
                check_in_time=datetime.now(timezone.utc),
                camera_id=self._config.camera_id,
                confidence_score=match.similarity,
            )
            await repo.create(record)
            await session.commit()

        await self._redis.setex(dedup_key, ATTENDANCE_DEDUP_TTL, "1")
        logger.info(
            "attendance_recorded",
            employee_id=str(employee_id),
            camera_id=self._config.camera_id,
            confidence=round(match.similarity, 3),
        )

    def _dedup_key(self, employee_id: UUID) -> str:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return f"attendance:dedup:{employee_id}:{today}"