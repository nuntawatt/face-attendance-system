# pyrefly: ignore [missing-import]
import pytest
import uuid
from uuid import uuid4
from datetime import datetime, timezone
from unittest.mock import AsyncMock

from app.core.exceptions import EmployeeCodeConflictError
from app.models.employee import Employee
from app.database.session import async_session_factory, async_engine
from app.repositories.employee import EmployeeRepository
from app.services.employee_service import EmployeeService
from app.schemas.employee import EmployeeCreate


@pytest.mark.asyncio
async def test_create_employee_success(mocker):
    """ทดสอบการสร้างพนักงานใหม่สำเร็จระดับ Service Layer (Mocked)"""
    mock_session = AsyncMock()
    service = EmployeeService(session=mock_session)

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

    data = EmployeeCreate(
        employee_code="EMP-001",
        full_name="John Doe",
        department="Engineering",
        position="Backend Developer",
        email="john@example.com",
    )

    await service.create_employee(data)

    service._repo.employee_code_exists.assert_called_once_with("EMP-001")
    service._repo.create.assert_called_once()
    mock_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_create_employee_conflict(mocker):
    """ทดสอบว่าระบบบล็อกการลงชื่อพนักงานซ้ำซ้อนระดับ Service Layer (Mocked)"""
    mock_session = AsyncMock()
    service = EmployeeService(session=mock_session)

    mocker.patch.object(
        service._repo, "employee_code_exists", new_callable=AsyncMock, return_value=True
    )
    mocker.patch.object(service._repo, "create", new_callable=AsyncMock)

    data = EmployeeCreate(
        employee_code="EMP-001",
        full_name="John Doe",
        department="Engineering",
        position="Backend Developer",
    )

    with pytest.raises(EmployeeCodeConflictError):
        await service.create_employee(data)

    service._repo.employee_code_exists.assert_called_once_with("EMP-001")
    service._repo.create.assert_not_called()


@pytest.mark.asyncio
async def test_db_soft_delete_integration():
    """ทดสอบระบบ Soft Delete ในระดับ Database End-to-End ร่วมกับ SQL"""
    try:
        async with async_session_factory() as session:
            repo = EmployeeRepository(session)

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

            found = await repo.get_by_id(emp_id)
            assert found is not None
            assert found.deleted_at is None

            await repo.delete(found)
            await session.flush()

            not_found = await repo.get_by_id(emp_id)
            assert not_found is None

            exists_check = await repo.employee_code_exists(emp_code)
            assert exists_check is False

            from sqlalchemy import select
            raw_result = await session.execute(
                select(Employee).where(Employee.id == emp_id)
            )
            raw_emp = raw_result.scalar_one_or_none()
            assert raw_emp is not None
            assert raw_emp.deleted_at is not None

            await session.rollback()
    finally:
        await async_engine.dispose()


@pytest.mark.asyncio
async def test_employee_service_deactivate_soft_delete():
    """ทดสอบปิดใช้งานพนักงาน (Soft Delete) ผ่าน EmployeeService"""
    try:
        async with async_session_factory() as session:
            repo = EmployeeRepository(session)
            service = EmployeeService(session)

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

            await service.deactivate_employee(emp_id)
            await session.flush()

            not_found = await repo.get_by_id(emp_id)
            assert not_found is None

            from sqlalchemy import select
            raw_result = await session.execute(
                select(Employee).where(Employee.id == emp_id)
            )
            raw_emp = raw_result.scalar_one_or_none()
            assert raw_emp is not None
            assert raw_emp.is_active is False
            assert raw_emp.deleted_at is not None

            await session.rollback()
    finally:
        await async_engine.dispose()


@pytest.mark.asyncio
async def test_employee_lifecycle_flow():
    """ทดสอบวงจรข้อมูลพนักงานครบกระบวนการ (สร้าง -> ป้องกันซ้ำ -> ตรวจเช็ค -> ปิดใช้งาน)"""
    try:
        async with async_session_factory() as session:
            repo = EmployeeRepository(session)
            service = EmployeeService(session)

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

            with pytest.raises(EmployeeCodeConflictError):
                await service.create_employee(data)

            found_emp = await repo.get_by_id(created_emp.id)
            assert found_emp is not None
            assert found_emp.full_name == "John Flow Test"

            await service.deactivate_employee(created_emp.id)
            await session.flush()

            not_found = await repo.get_by_id(created_emp.id)
            assert not_found is None

            from sqlalchemy import select
            raw_result = await session.execute(
                select(Employee).where(Employee.id == created_emp.id)
            )
            raw_emp = raw_result.scalar_one_or_none()
            assert raw_emp is not None
            assert raw_emp.is_active is False
            assert raw_emp.deleted_at is not None

            await session.rollback()
    finally:
        await async_engine.dispose()
