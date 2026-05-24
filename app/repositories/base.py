"""
Generic async repository สำหรับ CRUD operations พื้นฐาน

Domain repository ทั้งหมดสืบทอดมาจากนี้ รูปแบบ Generic[T] ทำให้
มี type safety ในทุก implementation ORM model type จะไหลผ่านไปถึง
IDE completion และ static analysis ได้อัตโนมัติ
"""
from __future__ import annotations

from typing import Generic, Sequence, Type, TypeVar
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.base import Base
from app.core.timezone import get_local_now

# Generic type ของ ORM model
# bound=Base หมายถึง ModelT ต้องสืบทอดจาก Base เท่านั้น
ModelT = TypeVar("ModelT", bound=Base)


class BaseRepository(Generic[ModelT]):
    """
    Async-capable generic repository

    ทุก method จะรับ session ที่ inject มาจากภายนอก
    ทำให้ caller (services) สามารถควบคุม transaction boundary ได้
    รองรับการทำ multi-repository atomic operations ใน transaction เดียวกัน
    """
    def __init__(self, model: Type[ModelT], session: AsyncSession) -> None:
        self._model = model
        self._session = session

    # CRUD operations
    async def get_by_id(self, entity_id: UUID) -> ModelT | None:
        stmt = select(self._model).where(self._model.id == entity_id)
        if hasattr(self._model, "deleted_at"):
            stmt = stmt.where(self._model.deleted_at.is_(None))
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    # get_all รองรับ pagination ด้วย limit และ offset
    async def get_all(self, *, limit: int = 100, offset: int = 0) -> Sequence[ModelT]:
        stmt = select(self._model).limit(limit).offset(offset)
        if hasattr(self._model, "deleted_at"):
            stmt = stmt.where(self._model.deleted_at.is_(None))
        result = await self._session.execute(stmt)
        return result.scalars().all()

    # สร้าง instance ใหม่ใน database และ return instance ที่มี id และข้อมูลล่าสุดจาก database
    async def create(self, instance: ModelT) -> ModelT:
        self._session.add(instance)
        await self._session.flush()  # flush ไม่ใช่ commit ผู้เรียกเป็นคนควบคุม transaction
        await self._session.refresh(instance) # refresh เพื่อดึงข้อมูลล่าสุดจาก database (เช่น id ที่ถูก auto-generated)
        return instance

    # update จะรับ instance ที่มี id อยู่แล้ว และจะ update ข้อมูลใน database ตาม instance นั้น
    async def delete(self, instance: ModelT) -> None:
        if hasattr(instance, "deleted_at"):
            instance.deleted_at = get_local_now()
            await self._session.flush()
        else:
            await self._session.delete(instance)
            await self._session.flush()

    # ตรวจสอบว่ามี entity ที่มี id นี้อยู่ใน database หรือไม่
    async def exists(self, entity_id: UUID) -> bool:
        stmt = select(self._model.id).where(self._model.id == entity_id)
        if hasattr(self._model, "deleted_at"):
            stmt = stmt.where(self._model.deleted_at.is_(None))
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none() is not None

    # นับจำนวน row ทั้งหมด สำหรับ pagination response
    async def count(self) -> int:
        stmt = select(func.count()).select_from(self._model)
        if hasattr(self._model, "deleted_at"):
            stmt = stmt.where(self._model.deleted_at.is_(None))
        result = await self._session.execute(stmt)
        return result.scalar_one()
