"""
Realtime attendance engine — worker หลักของ production

นี่ไม่ใช่ FastAPI route แต่เป็น long-running asyncio coroutine
ที่เริ่มใน application lifespan และรันตลอดอายุของ app

สถาปัตยกรรม:
CameraStreamReader -> frame_queue -> AttendanceEngine -> WebSocket broadcast
    (producer)                           (consumer)             (dashboard)

Queue แยก producer ออกจาก consumer ถ้า recognition ช้า
frame จะ back-pressure และ drop (LIFO maxsize=1 ต่อกล้อง)
การ drop frame เป็นเจตนา — สำหรับ attendance เราต้องการ frame ล่าสุด
ไม่ใช่คิว backlog ของ frame เก่า

การกำจัดซ้ำ (Deduplication) แยก check-in กับ check-out:
- Check-in dedup: Redis key "attendance:checkin:{employee_id}:{date}" TTL 4 ชั่วโมง
    บันทึก check-in ครั้งเดียวต่อวัน ป้องกัน duplicate record
- Check-out dedup: Redis key "attendance:checkout:{employee_id}:{date}" TTL 5 นาที
    อนุญาตให้ update check-out ทุก 5 นาที เพื่อจับเวลาออกงานล่าสุด
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import UUID

from app.core.timezone import get_local_now

import structlog

from app.ai.engine import face_engine
from app.ai.recognition import embedding_index
from app.camera.face_tracker import FaceTracker
from app.camera.stream_reader import CameraConfig, stream_frames
from app.core.config import settings
from app.websocket.manager import attendance_manager, AttendanceEvent

logger = structlog.get_logger(__name__)

CHECKIN_DEDUP_TTL = 4 * 3600   # 4 ชั่วโมง check-in ครั้งเดียวต่อวัน
CHECKOUT_DEDUP_TTL = 5 * 60    # 5 นาที update check-out ได้ทุก 5 นาที

# Quality gates สำหรับ real-time stream
MIN_DET_SCORE = 0.6            # กรอง detection ที่ไม่ชัดออก (หน้าเบลอ, ไกลเกิน)
MIN_QUALITY_SCORE = 0.3        # กรองภาพที่มืด/เบลอมากเกินไป


class AttendanceEngine:
    """
    ประสาน: อ่าน stream -> ตรวจจับใบหน้า -> จำแนก -> เขียน DB -> broadcast WS
    1 instance ต่อ 1 กล้อง รัน concurrent ทุกกล้องด้วย asyncio.gather
    """

    def __init__(self, config: CameraConfig, session_factory, redis_client, stop_event: asyncio.Event) -> None:
        self._config = config
        self._session_factory = session_factory
        self._redis = redis_client
        self._stop = stop_event
        self._tracker = FaceTracker()  # แต่ละกล้องมี tracker ของตัวเอง

    async def run(self) -> None:
        logger.info("attendance_engine_start", camera_id=self._config.camera_id)
        async for camera_id, frame in stream_frames(self._config, self._stop):
            try:
                await self._process_frame(frame)
            except Exception:
                # log แต่ไม่ crash กล้องต้องทำงานต่อเสมอ
                logger.exception("frame_processing_error", camera_id=camera_id)
        logger.info("attendance_engine_stop", camera_id=self._config.camera_id)

    async def _process_frame(self, frame) -> None:
        """ประมวลผล 1 frame: ตรวจจับ -> กรองคุณภาพ -> track -> จำแนกเฉพาะใบหน้าใหม่"""
        faces = await face_engine.analyze_frame(frame)
        if not faces:
            return

        # Quality gate: กรองใบหน้าที่ detection score ต่ำหรือภาพเบลอออก
        # ช่วยลด false positive จากภาพสะท้อน, โปสเตอร์, หรือหน้าจอมือถือ
        good_faces = [
            f for f in faces
            if f.det_score >= MIN_DET_SCORE and f.quality_score >= MIN_QUALITY_SCORE
        ]
        if not good_faces:
            return

        bboxes = [f.bbox for f in good_faces]
        track_indices = self._tracker.update(bboxes)

        recognition_tasks = []
        for i, (face, track_idx) in enumerate(zip(good_faces, track_indices)):
            if self._tracker.is_recognized(track_idx):
                continue  # จำแนกแล้วในการเข้างานนี้ ไม่ต้องจำแนกซ้ำ
            recognition_tasks.append(
                self._recognize_and_record(face, track_idx, frame)
            )

        # รันทุก recognition พร้อมกันไม่รอทีละคน
        if recognition_tasks: 
            await asyncio.gather(*recognition_tasks, return_exceptions=True)

    async def _recognize_and_record(self, face, track_idx: int, frame) -> None:
        """จำแนกใบหน้า 1 ใบ และบันทึกการเข้างานถ้าจำแนกได้"""
        match = await embedding_index.find_match(face.embedding)
        if match is None or not match.is_confident:
            return  # ไม่แน่ใจพอ ไม่บันทึก

        employee_id = match.employee_id
        self._tracker.mark_recognized(track_idx, employee_id)

        checkin_key = self._dedup_key("checkin", employee_id)
        checkout_key = self._dedup_key("checkout", employee_id)

        async with self._session_factory() as session:
            from app.repositories.attendance import AttendanceRepository
            from app.repositories.employee import EmployeeRepository
            from app.models.attendance import AttendanceRecord

            repo = AttendanceRepository(session)
            now_time = get_local_now()
            work_date = now_time.date()

            record = await repo.get_today_record(employee_id)

            # ดึงข้อมูลพนักงานสำหรับ WebSocket event (แสดงชื่อบน dashboard)
            emp_repo = EmployeeRepository(session)
            employee = await emp_repo.get_by_id(employee_id)
            emp_code = employee.employee_code if employee else None
            emp_name = employee.full_name if employee else None

            if record is None:
                # สร้าง record ใหม่ (ครั้งเดียวต่อวัน)
                if await self._redis.exists(checkin_key):
                    return  # check-in แล้ววันนี้

                # Crop และ upload
                image_url = await self._crop_and_upload_face(face, frame)

                record = AttendanceRecord(
                    employee_id=employee_id,
                    work_date=work_date,
                    check_in_time=now_time,
                    camera_id=self._config.camera_id,
                    confidence_score=match.similarity,
                    image_url=image_url,
                )
                await repo.create(record)
                await session.commit()

                await self._redis.setex(checkin_key, CHECKIN_DEDUP_TTL, "1")
                logger.info(
                    "attendance_checkin",
                    employee_id=str(employee_id),
                    camera_id=self._config.camera_id,
                    confidence=round(match.similarity, 3),
                )

                # Broadcast check-in event ไป dashboard
                await self._broadcast_event(
                    "check_in", employee_id, emp_code, emp_name,
                    match.similarity, now_time,
                )
            else:
                # update เวลาออกงาน (ทุก 5 นาที)
                if await self._redis.exists(checkout_key):
                    return  # เพิ่ง update check-out ไป ยังไม่ถึงเวลา

                # Crop และ upload
                image_url = await self._crop_and_upload_face(face, frame)

                record.check_out_time = now_time
                record.camera_id = self._config.camera_id
                record.image_url = image_url
                await session.commit()

                await self._redis.setex(checkout_key, CHECKOUT_DEDUP_TTL, "1")
                logger.info(
                    "attendance_checkout",
                    employee_id=str(employee_id),
                    camera_id=self._config.camera_id,
                    confidence=round(match.similarity, 3),
                )

                # Broadcast check-out event ไป dashboard
                await self._broadcast_event(
                    "check_out", employee_id, emp_code, emp_name,
                    match.similarity, now_time,
                )

    async def _broadcast_event(
        self,
        event_type: str,
        employee_id: UUID,
        employee_code: str | None,
        full_name: str | None,
        confidence: float,
        timestamp: datetime,
    ) -> None:
        """
        Broadcast attendance event ไป WebSocket clients ทั้งหมด

        Fire-and-forget: ถ้า broadcast ล้มเหลว ไม่ affect main pipeline
        เพราะ attendance record ถูก commit ลง DB แล้ว
        """
        try:
            event = AttendanceEvent(
                event_type=event_type,
                employee_id=str(employee_id),
                employee_code=employee_code,
                full_name=full_name,
                camera_id=self._config.camera_id,
                confidence=round(confidence, 3),
                timestamp=timestamp.isoformat(),
            )
            await attendance_manager.broadcast(event)
        except Exception:
            # WebSocket broadcast ล้มเหลวไม่ควร crash camera worker
            logger.warning("ws_broadcast_failed", employee_id=str(employee_id))

    def _dedup_key(self, action: str, employee_id: UUID) -> str:
        """สร้าง Redis key ที่ unique ต่อ action, ต่อพนักงาน, ต่อวัน"""
        today = get_local_now().strftime("%Y-%m-%d")
        return f"attendance:{action}:{employee_id}:{today}"

    async def _crop_and_upload_face(self, face, frame) -> str:
        """Helper ในการ crop ใบหน้าและ upload ขึ้น MinIO"""
        import cv2
        import uuid
        from app.services.minio_service import minio_service

        x1, y1, x2, y2 = face.bbox
        h, w = frame.shape[:2]
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))
        face_crop = frame[y1:y2, x1:x2]

        _, buffer = cv2.imencode(".jpg", face_crop)
        crop_bytes = buffer.tobytes()

        image_filename = f"{uuid.uuid4()}.jpg"
        image_url = await asyncio.to_thread(
            minio_service.upload_image, crop_bytes, image_filename
        )
        return image_url