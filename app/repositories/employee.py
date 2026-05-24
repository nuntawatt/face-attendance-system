"""
Repository เฉพาะสำหรับ Employee domain

Query ที่เป็นของ domain นี้อยู่ที่นี่เท่านั้น ไม่ใช่ใน service
Service layer เรียกใช้แบบระบุชื่อ method ชัดเจน
ไม่ควรสร้าง SQL condition เองใน service layer เด็ดขาด
"""
from __future__ import annotations

from typing import Sequence
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.employee import Employee
from app.repositories.base import BaseRepository


class EmployeeRepository(BaseRepository[Employee]):

    # EmployeeRepository จะสืบทอดจาก BaseRepository โดยระบุ ModelT เป็น Employee
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(Employee, session)

    async def get_by_employee_code(self, code: str) -> Employee | None:
        """ค้นหาพนักงานด้วยรหัสพนักงาน"""
        result = await self._session.execute(
            select(Employee).where(Employee.employee_code == code, Employee.deleted_at.is_(None))
        )
        return result.scalar_one_or_none()

    # get_by_department จะ return list ของ Employee ที่อยู่ใน department นี้ โดยรองรับ pagination ด้วย limit และ offset
    async def get_by_department(
        self, department: str, *, limit: int = 100, offset: int = 0
    ) -> Sequence[Employee]:
        result = await self._session.execute(
            select(Employee)
            .where(Employee.department == department, Employee.deleted_at.is_(None))
            .limit(limit)
            .offset(offset)
        )
        return result.scalars().all()

    # get_active_employees จะ return list ของ Employee ที่มี is_active = True
    async def get_active_employees(self) -> Sequence[Employee]:
        result = await self._session.execute(
            select(Employee).where(Employee.is_active == True, Employee.deleted_at.is_(None))  # noqa: E712
        )
        return result.scalars().all()

    # set_face_registered จะ update field face_registered ของ Employee ที่มี id นี้
    async def set_face_registered(self, employee_id: UUID, *, registered: bool) -> None:
        await self._session.execute(
            update(Employee)
            .where(Employee.id == employee_id)
            .values(face_registered=registered)
        )
        await self._session.flush()

    # employee_code_exists จะตรวจสอบว่ามี Employee ที่มี employee_code นี้อยู่ใน database หรือไม่
    async def employee_code_exists(self, code: str) -> bool:
        result = await self._session.execute(
            select(Employee.id).where(Employee.employee_code == code, Employee.deleted_at.is_(None))
        )
        return result.scalar_one_or_none() is not None