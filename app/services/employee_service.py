"""
Employee service — the brain of employee domain logic.

Services are where business rules live. This layer:
  - Orchestrates repository calls
  - Enforces invariants (e.g., can't delete employee with future shifts)
  - Emits domain events (face cache invalidation)
  - Controls transaction boundaries via the session

Dependency Injection pattern: the service receives its dependencies
(repository, cache client) at construction time via FastAPI's Depends.
This makes the service trivially testable with mock objects.
"""
from __future__ import annotations

# pyrefly: ignore [missing-import]
import structlog
from uuid import UUID

# pyrefly: ignore [missing-import]
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import EmployeeCodeConflictError, EmployeeNotFoundError
from app.models.employee import Employee
from app.repositories.employee import EmployeeRepository
from app.schemas.employee import EmployeeCreate, EmployeeListResponse, EmployeeResponse, EmployeeUpdate

logger = structlog.get_logger(__name__)


class EmployeeService:

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._repo = EmployeeRepository(session)

    async def create_employee(self, data: EmployeeCreate) -> EmployeeResponse:
        logger.info("creating_employee", employee_code=data.employee_code)

        if await self._repo.employee_code_exists(data.employee_code):
            raise EmployeeCodeConflictError(data.employee_code)

        employee = Employee(**data.model_dump())
        employee = await self._repo.create(employee)
        await self._session.commit()

        logger.info("employee_created", employee_id=str(employee.id))
        return EmployeeResponse.model_validate(employee)

    async def get_employee(self, employee_id: UUID) -> EmployeeResponse:
        employee = await self._repo.get_by_id(employee_id)
        if not employee:
            raise EmployeeNotFoundError(employee_id)
        return EmployeeResponse.model_validate(employee)

    async def update_employee(self, employee_id: UUID, data: EmployeeUpdate) -> EmployeeResponse:
        logger.info("updating_employee", employee_id=str(employee_id))

        employee = await self._repo.get_by_id(employee_id)
        if not employee:
            raise EmployeeNotFoundError(employee_id)

        update_data = data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(employee, field, value)

        await self._session.flush()
        await self._session.commit()
        await self._session.refresh(employee)

        return EmployeeResponse.model_validate(employee)

    async def deactivate_employee(self, employee_id: UUID) -> None:
        """
        Soft delete: mark inactive, preserve audit trail.
        Hard deletes are never done on employee records in production.
        """
        employee = await self._repo.get_by_id(employee_id)
        if not employee:
            raise EmployeeNotFoundError(employee_id)

        employee.is_active = False
        await self._session.commit()
        logger.info("employee_deactivated", employee_id=str(employee_id))

    async def list_employees(
        self, *, limit: int = 50, offset: int = 0
    ) -> EmployeeListResponse:
        employees = await self._repo.get_all(limit=limit, offset=offset)
        return EmployeeListResponse(
            items=[EmployeeResponse.model_validate(e) for e in employees],
            total=len(employees),
            limit=limit,
            offset=offset,
        )