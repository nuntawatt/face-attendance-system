"""
Declarative base และ mixin ร่วมสำหรับ ORM model ทั้งหมด

การใช้ mixin สำหรับ field ร่วม (id, timestamps) ทำให้ทุก model มีความสอดคล้องกัน
และลดการ copy-paste ระหว่าง model UUID primary key ถูกสร้างฝั่ง application
เพื่อความคาดเดาได้และหลีกเลี่ยง round-trip ไป database
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    """เพิ่ม created_at และ updated_at โดยใช้ server-side default อัตโนมัติ"""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class UUIDMixin:
    """UUID primary key ที่สร้างฝั่ง application ไม่รอ database generate"""

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )


class SoftDeleteMixin:
    """เพิ่ม deleted_at สำหรับ Soft Delete"""

    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
    )