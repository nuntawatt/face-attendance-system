"""
Face registration service.

This service owns the full registration workflow:
  1. Decode image bytes
  2. Detect exactly one face
  3. Check quality threshold
  4. Extract and store embedding
  5. Invalidate the recognition cache

The embedding cache invalidation via Redis pub/sub ensures that the
RTSP processing workers (potentially on different nodes) reload their
in-memory index after registration.
"""
from __future__ import annotations

import io
from uuid import UUID

import cv2
import numpy as np
import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.engine import face_engine
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

MIN_QUALITY_THRESHOLD = 0.4
MODEL_VERSION = "buffalo_s_v1"


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

        # Verify employee exists
        employee = await self._employee_repo.get_by_id(employee_id)
        if not employee:
            raise EmployeeNotFoundError(employee_id)

        # Decode image
        frame = self._decode_image(image_bytes)

        # Run face detection
        faces = await face_engine.analyze_frame(frame)

        if len(faces) == 0:
            raise FaceNotDetectedError()
        if len(faces) > 1:
            raise MultipleFacesError()

        face = faces[0]

        if face.quality_score < MIN_QUALITY_THRESHOLD:
            raise ImageQualityError(face.quality_score, MIN_QUALITY_THRESHOLD)

        # Serialize embedding
        embedding_bytes = face.embedding.tobytes()

        # Upsert into DB
        embedding_record = FaceEmbedding(
            employee_id=employee_id,
            embedding_vector=embedding_bytes,
            model_version=MODEL_VERSION,
            image_quality_score=face.quality_score,
        )
        await self._embedding_repo.upsert(embedding_record)
        await self._employee_repo.set_face_registered(employee_id, registered=True)
        await self._session.commit()

        logger.info(
            "face_registered",
            employee_id=str(employee_id),
            quality=face.quality_score,
        )

        return FaceRegistrationResponse(
            employee_id=employee_id,
            success=True,
            message="Face registered successfully",
            quality_score=face.quality_score,
            model_version=MODEL_VERSION,
        )

    @staticmethod
    def _decode_image(image_bytes: bytes) -> np.ndarray:
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise FaceNotDetectedError()
        return frame