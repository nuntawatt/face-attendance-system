# pyrefly: ignore [missing-import]
import pytest
import numpy as np
from uuid import uuid4
from unittest.mock import AsyncMock

from app.core.exceptions import (
    EmployeeNotFoundError,
    FaceNotDetectedError,
    MultipleFacesError,
    ImageQualityError,
)
from app.models.employee import Employee
from app.database.session import async_session_factory, async_engine
from app.repositories.employee import EmployeeRepository
from app.repositories.face_embedding import FaceEmbeddingRepository
from app.services.face_service import FaceRegistrationService


@pytest.mark.asyncio
async def test_face_registration_flow(mocker):
    """ทดสอบเวิร์กโฟลว์ลงทะเบียนใบหน้าพนักงานปกติแบบจำลอง AI"""
    mocker.patch("app.ai.engine.FaceEngine.is_ready", new_callable=mocker.PropertyMock, return_value=True)
    
    from app.ai.engine import DetectedFace
    fake_embedding = np.random.randn(512).astype(np.float32)
    fake_embedding = fake_embedding / np.linalg.norm(fake_embedding)
    
    mock_face = DetectedFace(
        embedding=fake_embedding,
        bbox=(10, 10, 100, 100),
        det_score=0.98,
        quality_score=0.85
    )
    
    mocker.patch(
        "app.services.face_service.face_engine.analyze_frame",
        new_callable=AsyncMock,
        return_value=[mock_face]
    )
    mocker.patch(
        "app.services.face_service.crop_and_encode_face",
        return_value=b"fake_crop_bytes"
    )
    mocker.patch(
        "app.services.face_service.minio_service.upload_image_async",
        new_callable=AsyncMock,
        return_value="http://localhost:9000/images/fake-uuid.jpg"
    )
    mocker.patch(
        "app.services.embedding_cache_service.EmbeddingCacheService.rebuild_index",
        new_callable=AsyncMock
    )
    mocker.patch.object(
        FaceRegistrationService,
        "_decode_image",
        return_value=np.zeros((100, 100, 3), dtype=np.uint8)
    )

    try:
        async with async_session_factory() as session:
            emp_repo = EmployeeRepository(session)
            emp_code = f"TEST-FACE-{uuid4()}".upper()[:20]
            emp = Employee(
                employee_code=emp_code,
                full_name="Face Test Subject",
                department="QA",
                position="Tester",
            )
            emp = await emp_repo.create(emp)
            await session.flush()
            
            face_service = FaceRegistrationService(session)
            fake_image_bytes = b"fake_png_data"
            
            response = await face_service.register_face(emp.id, fake_image_bytes)
            await session.flush()

            assert response.success is True
            assert response.employee_id == emp.id
            
            updated_emp = await emp_repo.get_by_id(emp.id)
            assert updated_emp.face_registered is True
            
            embedding_repo = FaceEmbeddingRepository(session)
            embedding_record = await embedding_repo.get_by_employee_id(emp.id)
            assert embedding_record is not None
            assert embedding_record.model_version == "edgeface_xs_v1"
            assert embedding_record.image_quality_score == 0.85
            assert embedding_record.image_url == "http://localhost:9000/images/fake-uuid.jpg"

            await session.rollback()
    finally:
        await async_engine.dispose()


@pytest.mark.asyncio
async def test_face_registration_failures(mocker):
    """ทดสอบกรณีลงทะเบียนใบหน้าแล้วเกิดข้อผิดพลาดต่างๆ (Negative & Failure Test Cases)"""
    mocker.patch("app.ai.engine.FaceEngine.is_ready", new_callable=mocker.PropertyMock, return_value=True)
    mocker.patch("app.services.face_service.crop_and_encode_face", return_value=b"fake_crop")
    mocker.patch("app.services.face_service.minio_service.upload_image_async", new_callable=AsyncMock, return_value="http://url.jpg")
    mocker.patch.object(FaceRegistrationService, "_decode_image", return_value=np.zeros((100, 100, 3), dtype=np.uint8))

    from app.ai.engine import DetectedFace

    try:
        async with async_session_factory() as session:
            emp_repo = EmployeeRepository(session)
            face_service = FaceRegistrationService(session)

            emp_code = f"ERR-TEST-{uuid4()}".upper()[:20]
            emp = Employee(
                employee_code=emp_code,
                full_name="Error Test Subject",
                department="QA",
                position="Tester",
            )
            emp = await emp_repo.create(emp)
            await session.flush()

            # --- กรณีที่ 1: ตรวจไม่พบใบหน้าในภาพเลย (FaceNotDetectedError) ---
            mocker.patch(
                "app.services.face_service.face_engine.analyze_frame",
                new_callable=AsyncMock,
                return_value=[]
            )
            with pytest.raises(FaceNotDetectedError):
                await face_service.register_face(emp.id, b"fake_bytes")

            # --- กรณีที่ 2: ตรวจพบมากกว่า 1 ใบหน้าในภาพ (MultipleFacesError) ---
            mock_face_1 = DetectedFace(embedding=np.zeros(512), bbox=(0, 0, 10, 10), det_score=0.9, quality_score=0.9)
            mock_face_2 = DetectedFace(embedding=np.zeros(512), bbox=(20, 20, 30, 30), det_score=0.9, quality_score=0.9)
            mocker.patch(
                "app.services.face_service.face_engine.analyze_frame",
                new_callable=AsyncMock,
                return_value=[mock_face_1, mock_face_2]
            )
            with pytest.raises(MultipleFacesError):
                await face_service.register_face(emp.id, b"fake_bytes")

            # --- กรณีที่ 3: คุณภาพรูปภาพต่ำกว่าเกณฑ์ขั้นต่ำ (ImageQualityError) ---
            mock_low_quality_face = DetectedFace(
                embedding=np.zeros(512), bbox=(0, 0, 10, 10), det_score=0.9, quality_score=0.15
            )
            mocker.patch(
                "app.services.face_service.face_engine.analyze_frame",
                new_callable=AsyncMock,
                return_value=[mock_low_quality_face]
            )
            with pytest.raises(ImageQualityError):
                await face_service.register_face(emp.id, b"fake_bytes")

            # --- กรณีที่ 4: พยายามลงทะเบียนใบหน้าให้พนักงานที่ไม่มีตัวตนจริง (EmployeeNotFoundError) ---
            mocker.patch(
                "app.services.face_service.face_engine.analyze_frame",
                new_callable=AsyncMock,
                return_value=[mock_face_1]
            )
            non_existent_id = uuid4()
            with pytest.raises(EmployeeNotFoundError):
                await face_service.register_face(non_existent_id, b"fake_bytes")

            await session.rollback()
    finally:
        await async_engine.dispose()
