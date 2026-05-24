import pytest
from datetime import datetime
from uuid import uuid4

from app.core.timezone import get_local_now, get_local_today, BANGKOK_TZ
from app.models.employee import Employee
from app.models.face_embedding import FaceEmbedding
from app.models.attendance import AttendanceRecord
from app.services.minio_service import minio_service


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


@pytest.mark.asyncio
async def test_db_soft_delete_integration():
    """Verify soft delete functionality end-to-end using a real local PostgreSQL database transaction."""
    from app.database.session import async_session_factory
    from app.repositories.employee import EmployeeRepository

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
        # This bypasses the repo's automated filtering
        from sqlalchemy import select

        raw_result = await session.execute(
            select(Employee).where(Employee.id == emp_id)
        )
        raw_emp = raw_result.scalar_one_or_none()
        assert raw_emp is not None
        assert raw_emp.deleted_at is not None

        # Rollback so we don't leave dirty data in database
        await session.rollback()

    from app.database.session import async_engine
    await async_engine.dispose()


@pytest.mark.asyncio
async def test_employee_service_deactivate_soft_delete():
    """Verify that calling EmployeeService.deactivate_employee(employee_id) correctly soft deletes and deactivates the employee."""
    from app.database.session import async_session_factory
    from app.services.employee_service import EmployeeService
    from app.repositories.employee import EmployeeRepository
    from sqlalchemy import select

    async with async_session_factory() as session:
        repo = EmployeeRepository(session)
        service = EmployeeService(session)

        # 1. Create employee
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

        # 2. Deactivate using service
        await service.deactivate_employee(emp_id)
        await session.flush()

        # 3. Verify they are no longer searchable via repo.get_by_id
        not_found = await repo.get_by_id(emp_id)
        assert not_found is None

        # 4. Verify raw DB record has is_active = False and deleted_at populated
        raw_result = await session.execute(
            select(Employee).where(Employee.id == emp_id)
        )
        raw_emp = raw_result.scalar_one_or_none()
        assert raw_emp is not None
        assert raw_emp.is_active is False
        assert raw_emp.deleted_at is not None

        # Rollback transaction
        await session.rollback()

    from app.database.session import async_engine
    await async_engine.dispose()
