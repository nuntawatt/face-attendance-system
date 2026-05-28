# pyrefly: ignore [missing-import]
import pytest
import numpy as np
import uuid
from uuid import uuid4, UUID
from datetime import datetime, date, timezone
from unittest.mock import AsyncMock

from app.core.timezone import get_local_now, get_local_today, BANGKOK_TZ
from app.core.exceptions import EmployeeCodeConflictError
from app.models.employee import Employee
from app.models.face_embedding import FaceEmbedding
from app.models.attendance import AttendanceRecord
from app.database.session import async_session_factory, async_engine
from app.repositories.employee import EmployeeRepository
from app.repositories.face_embedding import FaceEmbeddingRepository
from app.repositories.attendance import AttendanceRepository
from app.services.employee_service import EmployeeService
from app.services.face_service import FaceRegistrationService
from app.services.minio_service import minio_service
from app.schemas.employee import EmployeeCreate
from app.ai.recognition import EmployeeEmbeddingIndex, RecognitionMatch


# ==============================================================================
# SECTION 1: UNIT TESTS FOR UTILITIES & HELPER FUNCTIONS
# ==============================================================================

def test_timezone_utils():
    """Verify that Bangkok timezone helpers return correct, localized values."""
    now = get_local_now()
    assert now.tzinfo == BANGKOK_TZ

    today = get_local_today()
    assert today == datetime.now(BANGKOK_TZ).date()


def test_soft_delete_mixin_attributes():
    """Verify that models correctly inherit soft delete mixin and store image URL attributes."""
    emp = Employee(employee_code="EMP-TEST-99", full_name="Test Soft Delete")
    assert hasattr(emp, "deleted_at")
    assert emp.deleted_at is None

    emb = FaceEmbedding(model_version="test")
    assert hasattr(emb, "deleted_at")
    assert hasattr(emb, "image_url")
    assert emb.deleted_at is None

    rec = AttendanceRecord(confidence_score=0.99)
    assert hasattr(rec, "deleted_at")
    assert hasattr(rec, "image_url")
    assert rec.deleted_at is None


@pytest.mark.asyncio
async def test_minio_upload_mocked(mocker):
    """Verify that MinIO upload correctly maps types and constructs public URLs."""
    # Mock the client's put_object to isolate from actual connection
    mock_put = mocker.patch.object(minio_service.client, "put_object")

    image_bytes = b"fake-jpeg-data"
    filename = "test-face.jpg"

    url = minio_service.upload_image(image_bytes, filename)

    assert "test-face.jpg" in url
    assert "images" in url
    mock_put.assert_called_once()


# ==============================================================================
# SECTION 2: UNIT TESTS FOR SERVICES (PURELY MOCKED / NO DB CONNECTION NEEDED)
# ==============================================================================

@pytest.mark.asyncio
async def test_create_employee_success(mocker):
    """ทดสอบกรณีสร้างพนักงานสำเร็จ โดยจำลองการทำงานของ DB Session และ Repository"""
    # Mock the database session
    mock_session = AsyncMock()

    # Create the service
    service = EmployeeService(session=mock_session)

    # Mock the repository methods
    mocker.patch.object(
        service._repo,
        "employee_code_exists",
        new_callable=AsyncMock,
        return_value=False,
    )

    mock_employee = Employee(
        id=uuid.UUID("123e4567-e89b-12d3-a456-426614174000"),
        employee_code="EMP-001",
        full_name="John Doe",
        department="Engineering",
        position="Backend Developer",
        email="john@example.com",
        is_active=True,
        face_registered=False,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    mocker.patch.object(
        service._repo, "create", new_callable=AsyncMock, return_value=mock_employee
    )

    # Test data
    data = EmployeeCreate(
        employee_code="EMP-001",
        full_name="John Doe",
        department="Engineering",
        position="Backend Developer",
        email="john@example.com",
    )

    # Run the method
    await service.create_employee(data)

    # Verify repository calls
    service._repo.employee_code_exists.assert_called_once_with("EMP-001")
    service._repo.create.assert_called_once()
    mock_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_create_employee_conflict(mocker):
    """ทดสอบกรณีสร้างพนักงานด้วยรหัสพนักงานซ้ำซ้อน ระบบต้องโยน Exception เพื่อสกัดข้อผิดพลาด"""
    # Mock the database session
    mock_session = AsyncMock()

    # Create the service
    service = EmployeeService(session=mock_session)

    # Mock the repository methods to simulate existing employee
    mocker.patch.object(
        service._repo, "employee_code_exists", new_callable=AsyncMock, return_value=True
    )
    mocker.patch.object(service._repo, "create", new_callable=AsyncMock)

    # Test data
    data = EmployeeCreate(
        employee_code="EMP-001",
        full_name="John Doe",
        department="Engineering",
        position="Backend Developer",
    )

    # Verify exception is raised
    with pytest.raises(EmployeeCodeConflictError):
        await service.create_employee(data)

    service._repo.employee_code_exists.assert_called_once_with("EMP-001")
    service._repo.create.assert_not_called()


# ==============================================================================
# SECTION 3: IN-MEMORY AI EMBEDDING SEARCH & INDEX MATCHING TESTS
# ==============================================================================

@pytest.mark.asyncio
async def test_embedding_index_matching():
    """ระบบเปรียบเทียบในหน่วยความจำและการจับคู่ใบหน้า (In-Memory Embedding Index & Search)

    ทดสอบประสิทธิภาพการค้นหาและแมตช์ความถูกต้องของโครงสร้างดัชนีจำพวก Multi-Class Cosine Similarity
    """
    index = EmployeeEmbeddingIndex()
    
    # 1. จำลองข้อมูล Face Embeddings 2 คน
    emp_id_a = uuid4()
    emp_id_b = uuid4()
    
    # สร้าง Vector ขนาด 512 มิติแบบ Normalize แล้ว
    vec_a = np.zeros(512, dtype=np.float32)
    vec_a[0] = 1.0  # Vector แรกชี้ไปแนวแกนที่ 0
    
    vec_b = np.zeros(512, dtype=np.float32)
    vec_b[1] = 1.0  # Vector สองชี้ไปแนวแกนที่ 1 (ตั้งฉากกัน คล้ายคลึงต่ำ)
    
    # โหลดลง index
    embeddings = {
        emp_id_a: vec_a,
        emp_id_b: vec_b
    }
    await index.rebuild(embeddings)
    
    assert index.size == 2

    # 2. กรณีเปรียบเทียบกับภาพที่ใกล้เคียงกับ A (ความแม่นยำสูง)
    probe_a = np.zeros(512, dtype=np.float32)
    probe_a[0] = 0.99
    probe_a[1] = 0.05
    # Normalize probe
    probe_a = probe_a / np.linalg.norm(probe_a)
    
    match_a = await index.find_match(probe_a)
    
    assert match_a is not None
    assert match_a.employee_id == emp_id_a
    assert match_a.similarity > 0.95
    assert match_a.is_confident is True  # มั่นใจว่าใช่คน A

    # 3. กรณีเปรียบเทียบกับภาพที่ไม่เหมือนใครเลย (คนแปลกหน้า / Stranger)
    probe_stranger = np.zeros(512, dtype=np.float32)
    probe_stranger[10] = 1.0  # ชี้ไปแกนที่ 10 ซึ่งไม่มีในระบบ
    
    match_stranger = await index.find_match(probe_stranger)
    
    assert match_stranger is not None
    assert match_stranger.similarity < 0.2
    assert match_stranger.is_confident is False  # ไม่บันทึก/ปฏิเสธเป็นคนแปลกหน้า


# ==============================================================================
# SECTION 4: INTEGRATION TESTS WITH DATABASE TRANSACTIONS
# ==============================================================================

@pytest.mark.asyncio
async def test_db_soft_delete_integration():
    """Verify soft delete functionality end-to-end using a real local PostgreSQL database transaction."""
    try:
        async with async_session_factory() as session:
            repo = EmployeeRepository(session)

            # 1. Create a temporary employee
            emp_code = f"TEMP-{uuid4()}"[:20]
            emp = Employee(
                employee_code=emp_code,
                full_name="Integration Test Emp",
                department="Testing",
                position="QA",
            )
            emp = await repo.create(emp)
            await session.flush()
            emp_id = emp.id

            # 2. Verify it is searchable via repo
            found = await repo.get_by_id(emp_id)
            assert found is not None
            assert found.deleted_at is None

            # 3. Soft delete it
            await repo.delete(found)
            await session.flush()

            # 4. Verify repo.get_by_id no longer finds it
            not_found = await repo.get_by_id(emp_id)
            assert not_found is None

            # 5. Verify the code exists check no longer finds it
            exists_check = await repo.employee_code_exists(emp_code)
            assert exists_check is False

            # 6. Query raw SQL directly to check that the row still exists in DB but has deleted_at set
            from sqlalchemy import select

            raw_result = await session.execute(
                select(Employee).where(Employee.id == emp_id)
            )
            raw_emp = raw_result.scalar_one_or_none()
            assert raw_emp is not None
            assert raw_emp.deleted_at is not None

            # Rollback so we don't leave dirty data in database
            await session.rollback()
    finally:
        await async_engine.dispose()


@pytest.mark.asyncio
async def test_employee_service_deactivate_soft_delete():
    """ตรวจสอบการปิดใช้งานพนักงาน (Soft Delete) ผ่านระดับ Service Layer ว่าใช้งานได้จริงและเก็บข้อมูลเวลาที่ลบลง DB"""
    try:
        async with async_session_factory() as session:
            repo = EmployeeRepository(session)
            service = EmployeeService(session)

            # 1. สร้างข้อมูลพนักงานชั่วคราวขึ้นมาในระบบ
            emp_code = f"SERV-DEL-{uuid4()}"[:20]
            emp = Employee(
                employee_code=emp_code,
                full_name="Service Soft Delete",
                department="Engineering",
                position="Engineer",
            )
            emp = await repo.create(emp)
            await session.flush()
            emp_id = emp.id

            # 2. ทำการสั่งปิดการใช้งาน (deactivate) ผ่านทาง Service
            await service.deactivate_employee(emp_id)
            await session.flush()

            # 3. ยืนยันว่าไม่สามารถค้นหาข้อมูลผ่าน repository ปกติได้แล้ว (เพราะโดนกรอง soft delete ออก)
            not_found = await repo.get_by_id(emp_id)
            assert not_found is None

            # 4. ดึงข้อมูลดิบจาก Database มาตรวจสอบว่ามีสถานะ is_active = False และมีวันเวลา deleted_at บันทึกไว้จริง
            from sqlalchemy import select
            raw_result = await session.execute(
                select(Employee).where(Employee.id == emp_id)
            )
            raw_emp = raw_result.scalar_one_or_none()
            assert raw_emp is not None
            assert raw_emp.is_active is False
            assert raw_emp.deleted_at is not None

            # ทำการ Rollback ข้อมูลทิ้งเพื่อรักษาความสะอาดของระบบฐานข้อมูลทดสอบ
            await session.rollback()
    finally:
        await async_engine.dispose()


@pytest.mark.asyncio
async def test_employee_lifecycle_flow():
    """วงจรชีวิตของพนักงานแบบครบกระบวนการ (สร้าง -> ตรวจสอบข้อมูลซ้ำ -> ค้นหา -> ปิดใช้งาน Soft Delete)

    ทดสอบการไหลของข้อมูลพนักงานในระบบแบบ End-to-End เพื่อยืนยันว่าการทำงานระดับ Database ทำงานถูกต้อง
    """
    try:
        async with async_session_factory() as session:
            repo = EmployeeRepository(session)
            service = EmployeeService(session)

            # 1. สร้างพนักงานใหม่สำเร็จ (ตัวพิมพ์ใหญ่ทั้งหมดเนื่องจากระบบจัดเก็บเป็น Uppercase)
            emp_code = f"TEST-FLOW-{uuid4()}".upper()[:20]
            data = EmployeeCreate(
                employee_code=emp_code,
                full_name="John Flow Test",
                department="Engineering",
                position="Software Engineer",
                email=f"{uuid4()}@flow.com",
            )
            
            created_emp = await service.create_employee(data)
            await session.flush()
            assert created_emp.id is not None
            assert created_emp.employee_code == emp_code
            assert created_emp.face_registered is False

            # 2. ป้องกันรหัสพนักงานซ้ำซ้อน (Conflict Check)
            with pytest.raises(EmployeeCodeConflictError):
                await service.create_employee(data)

            # 3. ตรวจสอบการค้นหาข้อมูล
            found_emp = await repo.get_by_id(created_emp.id)
            assert found_emp is not None
            assert found_emp.full_name == "John Flow Test"

            # 4. ลบพนักงานแบบนุ่มนวล (Soft Delete) ผ่านระดับ Service
            await service.deactivate_employee(created_emp.id)
            await session.flush()

            # 5. ยืนยันว่าค้นหาผ่าน repo ปกติจะไม่เจอแล้ว
            not_found = await repo.get_by_id(created_emp.id)
            assert not_found is None

            # 6. ยืนยันข้อมูลยังอยู่ใน DB ดิบแต่ deleted_at ไม่เป็น None
            from sqlalchemy import select
            raw_result = await session.execute(
                select(Employee).where(Employee.id == created_emp.id)
            )
            raw_emp = raw_result.scalar_one_or_none()
            assert raw_emp is not None
            assert raw_emp.is_active is False
            assert raw_emp.deleted_at is not None

            # โรลแบ็กข้อมูลทดสอบออกจากฐานข้อมูล
            await session.rollback()
    finally:
        await async_engine.dispose()


@pytest.mark.asyncio
async def test_face_registration_flow(mocker):
    """การลงทะเบียนใบหน้าพนักงาน (ถอดรหัสภาพ -> รันโมเดล AI -> บันทึก Embedding -> อัปเดตแฟลก)

    ทดสอบเวิร์กโฟลว์การลงทะเบียนใบหน้าโดยจำลองส่วนการอัปโหลดและประมวลผลรูปภาพ
    """
    # 1. Mocking ส่วนประกอบภายนอก (MinIO และ AI Engine) เพื่อทำการทดสอบแยกส่วน (Isolate)
    # ใช้ PropertyMock เพื่อจำลอง Property is_ready แบบ Read-Only บน FaceEngine class
    mocker.patch("app.ai.engine.FaceEngine.is_ready", new_callable=mocker.PropertyMock, return_value=True)
    
    # สร้าง Mock Face Object ที่ตรวจจับได้จากเฟรมภาพ
    from app.ai.engine import DetectedFace
    fake_embedding = np.random.randn(512).astype(np.float32)
    # ทำการ Normalize embedding เสมอ
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
    # ปิดการรัน Rebuild Index จริงๆ ในเทสนี้
    mocker.patch(
        "app.services.embedding_cache_service.EmbeddingCacheService.rebuild_index",
        new_callable=AsyncMock
    )
    # Mock ตัวถอดรหัสรูปภาพเพื่อไม่ให้ล้มเหลวจาก fake byte array ในเทส
    mocker.patch.object(
        FaceRegistrationService,
        "_decode_image",
        return_value=np.zeros((100, 100, 3), dtype=np.uint8)
    )

    try:
        async with async_session_factory() as session:
            # สร้างพนักงานจำลองสำหรับเทสลงทะเบียนใบหน้า
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
            
            # 2. เริ่มต้นระบบลงทะเบียน
            face_service = FaceRegistrationService(session)
            fake_image_bytes = b"fake_png_data"
            
            response = await face_service.register_face(emp.id, fake_image_bytes)
            await session.flush()

            # 3. ตรวจสอบความถูกต้องหลังลงทะเบียนสำเร็จ
            assert response.success is True
            assert response.employee_id == emp.id
            
            # ตรวจสอบใน DB ว่า face_registered อัปเดตเป็น True หรือไม่
            updated_emp = await emp_repo.get_by_id(emp.id)
            assert updated_emp.face_registered is True
            
            # ตรวจสอบว่ามีข้อมูล FaceEmbedding ผูกอยู่ถูกต้องในระบบ (โดยใช้ get_by_employee_id แทน get_by_id)
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
async def test_attendance_lifecycle_flow(mocker):
    """ระบบบันทึกการเข้างานและออกงาน (Attendance Flow & De-duplication)

    ทดสอบการปั๊มบันทึกบันทึกการเข้างาน-ออกงาน เพื่อยืนยันว่าการทำงานสอดคล้องกับพนักงานและการตรวจสอบซ้ำทำงานดี
    """
    try:
        async with async_session_factory() as session:
            emp_repo = EmployeeRepository(session)
            attendance_repo = AttendanceRepository(session)

            # 1. จำลองพนักงาน
            emp_code = f"TEST-ATT-{uuid4()}".upper()[:20]
            emp = Employee(
                employee_code=emp_code,
                full_name="Worker Bee",
                department="Production",
                position="Staff",
            )
            emp = await emp_repo.create(emp)
            await session.flush()

            # 2. ตรวจสอบการลงชื่อเข้างานครั้งแรกของวัน
            now_time = get_local_now()
            work_date = now_time.date()
            
            # ยืนยันว่ายังไม่มีการบันทึก
            existing_record = await attendance_repo.get_today_record(emp.id)
            assert existing_record is None
            
            # บันทึก Check-in
            record = AttendanceRecord(
                employee_id=emp.id,
                work_date=work_date,
                check_in_time=now_time,
                camera_id="CAM-ENTRANCE-01",
                confidence_score=0.92,
                image_url="http://localhost:9000/attendance/face-01.jpg",
            )
            await attendance_repo.create(record)
            await session.flush()
            
            # ค้นหาอีกครั้งเพื่อยืนยันว่าบันทึกแล้ว
            record_db = await attendance_repo.get_today_record(emp.id)
            assert record_db is not None
            assert record_db.employee_id == emp.id
            assert record_db.check_in_time == now_time
            assert record_db.check_out_time is None  # ยังไม่ออกงาน

            # 3. อัปเดตเวลาออกงาน (Check-out)
            out_time = get_local_now()
            record_db.check_out_time = out_time
            record_db.camera_id = "CAM-EXIT-01"
            await session.flush()
            
            # ยืนยันจาก DB ว่าบันทึกการออกงานแล้ว
            record_final = await attendance_repo.get_today_record(emp.id)
            assert record_final is not None
            assert record_final.check_out_time == out_time
            assert record_final.camera_id == "CAM-EXIT-01"

            await session.rollback()
    finally:
        await async_engine.dispose()
