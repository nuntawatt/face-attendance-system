"""
Face registration service

Service นี้เป็นเจ้าของ workflow การลงทะเบียนทั้งหมด:
1. Decode image bytes
2. ตรวจพบใบหน้าพอดีหนึ่งใบหน้า
3. ตรวจสอบ quality threshold
4. Extract และเก็บ embedding
5. Invalidate recognition cache

การ invalidate embedding cache ผ่าน Redis pub/sub ทำให้
RTSP processing worker (อาจอยู่ node อื่น) reload index ใน memory
หลังลงทะเบียนใหม่ได้ทันที
"""

from __future__ import annotations

import io
import uuid
from uuid import UUID

import cv2
import numpy as np
import structlog
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings

from app.services.minio_service import minio_service

from app.ai.engine import face_engine, crop_and_encode_face
from app.core.exceptions import (
    AIEngineNotReadyError,
    EmployeeNotFoundError,
    FaceNotDetectedError,
    ImageQualityError,
    MultipleFacesError,
)
from app.models.face_embedding import FaceEmbedding
from app.repositories.employee import EmployeeRepository
from app.repositories.face_embedding import FaceEmbeddingRepository
from app.schemas.face import FaceRegistrationResponse

logger = structlog.get_logger(__name__)

# คะแนนคุณภาพขั้นต่ำสำหรับลงทะเบียน
MIN_QUALITY_THRESHOLD = 0.4
# เวอร์ชันโมเดลที่ใช้สำหรับ embedding นี้ เก็บไว้เพื่อ compatibility ในอนาคต
MODEL_VERSION = f"{settings.face_model_pack}_v1"


class FaceRegistrationService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._employee_repo = EmployeeRepository(session)
        self._embedding_repo = FaceEmbeddingRepository(session)

    async def register_face(
        self, employee_id: UUID, image_bytes: bytes
    ) -> FaceRegistrationResponse:
        if not face_engine.is_ready:
            raise AIEngineNotReadyError()

        logger.info("face_registration_start", employee_id=str(employee_id))

        # ตรวจสอบว่าพนักงานมีอยู่จริง
        employee = await self._employee_repo.get_by_id(employee_id)
        if not employee:
            raise EmployeeNotFoundError(employee_id)

        # Decode ภาพ
        frame = self._decode_image(image_bytes)

        # รัน face detection
        faces = await face_engine.analyze_frame(frame)

        # ตรวจสอบว่าพบใบหน้าพอดี 1 ใบหน้า
        if len(faces) == 0:
            raise FaceNotDetectedError()
        # ถ้าพบมากกว่า 1 ใบหน้า
        if len(faces) > 1:
            raise MultipleFacesError()

        # เลือกใบหน้าที่พบ
        face = faces[0]

        # ตรวจสอบคุณภาพ
        if face.quality_score < MIN_QUALITY_THRESHOLD:
            raise ImageQualityError(face.quality_score, MIN_QUALITY_THRESHOLD)

        # Serialize embedding เป็น bytes สำหรับเก็บใน DB
        embedding_bytes = face.embedding.tobytes()

        # Crop และแปลงเป็น jpeg bytes ด้วยฟังก์ชันส่วนกลาง
        crop_bytes = crop_and_encode_face(frame, face.bbox)

        # อัพโหลดขึ้น MinIO ด้วย UUID filename แบบ async แท้
        image_filename = f"{uuid.uuid4()}.jpg"
        image_url = await minio_service.upload_image_async(crop_bytes, image_filename)

        # Upsert ลง database
        embedding_record = FaceEmbedding(
            employee_id=employee_id,
            embedding_vector=embedding_bytes,
            model_version=MODEL_VERSION,
            image_quality_score=face.quality_score,
            image_url=image_url,
        )
        await self._embedding_repo.upsert(embedding_record)
        await self._employee_repo.set_face_registered(employee_id, registered=True)
        await self._session.commit()

        # Rebuild in-memory index ทันทีหลัง commit
        # ถ้าไม่ rebuild, AI pipeline จะจำพนักงานคนนี้ไม่ได้จนกว่า server restart
        from app.services.embedding_cache_service import EmbeddingCacheService

        cache_service = EmbeddingCacheService(self._session)
        await cache_service.rebuild_index()

        logger.info(
            "face_registered", employee_id=str(employee_id), quality=face.quality_score
        )

        # ตอบกลับ Client
        return FaceRegistrationResponse(
            employee_id=employee_id,
            success=True,
            message="Face registered successfully",
            quality_score=face.quality_score,
            model_version=MODEL_VERSION,
        )

    @staticmethod
    # แปลง image bytes เป็น numpy array
    def _decode_image(image_bytes: bytes) -> np.ndarray:
        # ลองใช้ Pillow ในการ decode เพื่อรองรับ format กว้างขวางขึ้น (.png, .webp, .bmp)
        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            # แปลงเป็น BGR สำหรับ OpenCV
            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            return frame
        except Exception:
            # ถ้า Pillow decode ไม่สำเร็จ ลอง fallback ไปใช้ OpenCV ซึ่งอาจจะรองรับบางกรณีที่ Pillow ไม่รองรับ
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                raise FaceNotDetectedError()
            return frame
