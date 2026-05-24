"""
Employee service — ศูนย์กลาง business logic ของ employee domain

Service คือที่อยู่ของ business rule เท่านั้น Layer นี้:
    - ประสาน repository call
    - บังคับใช้ invariant (เช่น ไม่ลบพนักงานที่มี shift ในอนาคต)
    - ส่ง domain event (face cache invalidation)
    - ควบคุม transaction boundary ผ่าน session

รูปแบบ Dependency Injection: service รับ dependency (repository, cache)
ตอน construction ผ่าน FastAPI Depends ทำให้ test ง่ายมากด้วย mock object
"""

from __future__ import annotations

import structlog
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import EmployeeCodeConflictError, EmployeeNotFoundError
from app.models.employee import Employee
from app.repositories.employee import EmployeeRepository
from app.schemas.employee import (
    EmployeeCreate,
    EmployeeListResponse,
    EmployeeResponse,
    EmployeeUpdate,
)

logger = structlog.get_logger(__name__)


class EmployeeService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._repo = EmployeeRepository(session)

    async def create_employee(self, data: EmployeeCreate) -> EmployeeResponse:
        logger.info("creating_employee", employee_code=data.employee_code)

        # ตรวจสอบรหัสซ้ำก่อน create ป้องกัน race condition ด้วย unique constraint ของ DB
        if await self._repo.employee_code_exists(data.employee_code):
            raise EmployeeCodeConflictError(data.employee_code)

        employee = Employee(**data.model_dump())
        employee = await self._repo.create(employee)
        await self._session.commit()  # commit เฉพาะใน service ไม่ commit ใน repository

        logger.info("employee_created", employee_id=str(employee.id))
        return EmployeeResponse.model_validate(employee)

    async def get_employee(self, employee_id: UUID) -> EmployeeResponse:
        employee = await self._repo.get_by_id(employee_id)
        if not employee:
            raise EmployeeNotFoundError(employee_id)
        return EmployeeResponse.model_validate(employee)

    async def update_employee(
        self, employee_id: UUID, data: EmployeeUpdate
    ) -> EmployeeResponse:
        logger.info("updating_employee", employee_id=str(employee_id))

        employee = await self._repo.get_by_id(employee_id)
        if not employee:
            raise EmployeeNotFoundError(employee_id)

        # exclude_unset=True: อัพเดทเฉพาะ field ที่ส่งมา ไม่ทับค่าที่ไม่ได้ส่งมา
        update_data = data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(employee, field, value)

        await self._session.flush()
        await self._session.commit()
        await self._session.refresh(employee)

        return EmployeeResponse.model_validate(employee)

    async def deactivate_employee(self, employee_id: UUID) -> None:
        """
        Soft delete: ทำเครื่องหมายว่าไม่ active รักษา audit trail ไว้
        ห้ามลบ record พนักงานออกจาก DB ใน production เด็ดขาด
        """
        employee = await self._repo.get_by_id(employee_id)
        if not employee:
            raise EmployeeNotFoundError(employee_id)

        employee.is_active = False
        await self._repo.delete(employee)
        await self._session.commit()
        logger.info("employee_deactivated", employee_id=str(employee_id))

    async def list_employees(
        self, *, limit: int = 50, offset: int = 0
    ) -> EmployeeListResponse:
        employees = await self._repo.get_all(limit=limit, offset=offset)
        total = await self._repo.count()
        return EmployeeListResponse(
            items=[EmployeeResponse.model_validate(e) for e in employees],
            total=total,
            limit=limit,
            offset=offset,
        )
