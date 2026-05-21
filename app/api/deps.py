"""
FastAPI dependency providers

Database session, service instance และ auth check ทั้งหมด
ถูก provide เป็น dependency ทำให้ router บางมาก ไม่มี setup, ไม่มี teardown
Session lifecycle จัดการที่นี่ผ่าน async generator

Session-per-request: HTTP request แต่ละรายการได้รับ AsyncSession ของตัวเอง
นี่คือ pattern มาตรฐาน ห้าม share session ข้าม request เด็ดขาด
"""
from __future__ import annotations

from typing import AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import async_session_factory
from app.services.employee_service import EmployeeService
from app.services.face_service import FaceRegistrationService


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Async generator คืน session ใหม่ต่อ request, rollback อัตโนมัติเมื่อ error"""
    async with async_session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def get_employee_service(session: AsyncSession = Depends(get_db_session)) -> EmployeeService:
    return EmployeeService(session)


async def get_face_service(session: AsyncSession = Depends(get_db_session)) -> FaceRegistrationService:
    return FaceRegistrationService(session)