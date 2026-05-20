# pyrefly: ignore [missing-import]
import pytest
from unittest.mock import AsyncMock

from app.schemas.employee import EmployeeCreate
from app.services.employee_service import EmployeeService
from app.core.exceptions import EmployeeCodeConflictError
from app.models.face_embedding import FaceEmbedding  # noqa: F401
from app.models.attendance import AttendanceRecord  # noqa: F401

@pytest.mark.asyncio
async def test_create_employee_success(mocker):
    # Mock the database session
    mock_session = AsyncMock()
    
    # Create the service
    service = EmployeeService(session=mock_session)
    
    # Mock the repository methods
    mocker.patch.object(service._repo, 'employee_code_exists', new_callable=AsyncMock, return_value=False)
    
    from app.models.employee import Employee
    from datetime import datetime, timezone
    import uuid
    
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
        updated_at=datetime.now(timezone.utc)
    )
    mocker.patch.object(service._repo, 'create', new_callable=AsyncMock, return_value=mock_employee)
    
    # Test data
    data = EmployeeCreate(
        employee_code="EMP-001",
        full_name="John Doe",
        department="Engineering",
        position="Backend Developer",
        email="john@example.com"
    )
    
    # Run the method
    await service.create_employee(data)

    # Verify repository calls
    service._repo.employee_code_exists.assert_called_once_with("EMP-001")
    service._repo.create.assert_called_once()
    mock_session.commit.assert_called_once()

@pytest.mark.asyncio
async def test_create_employee_conflict(mocker):
    # Mock the database session
    mock_session = AsyncMock()
    
    # Create the service
    service = EmployeeService(session=mock_session)
    
    # Mock the repository methods to simulate existing employee
    mocker.patch.object(service._repo, 'employee_code_exists', new_callable=AsyncMock, return_value=True)
    mocker.patch.object(service._repo, 'create', new_callable=AsyncMock)
    
    # Test data
    data = EmployeeCreate(
        employee_code="EMP-001",
        full_name="John Doe",
        department="Engineering",
        position="Backend Developer"
    )
    
    # Verify exception is raised
    with pytest.raises(EmployeeCodeConflictError):
        await service.create_employee(data)
        
    service._repo.employee_code_exists.assert_called_once_with("EMP-001")
    service._repo.create.assert_not_called()
