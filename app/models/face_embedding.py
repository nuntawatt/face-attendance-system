"""
ORM model สำหรับ face embedding

Embedding vector เก็บเป็น BYTEA (numpy array ที่ serialize แล้ว)
ในการ deploy ขนาดใหญ่ ให้เปลี่ยนเป็น pgvector สำหรับ ANN similarity search
แต่สำหรับโรงงานที่มีพนักงาน < 5,000 คน in-memory FAISS index เร็วกว่า
DB-side ANN และหลีกเลี่ยง overhead การ operate pgvector
"""
from __future__ import annotations

import uuid

from sqlalchemy import ForeignKey, LargeBinary, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin, UUIDMixin, SoftDeleteMixin
from app.models.employee import Employee


class FaceEmbedding(UUIDMixin, TimestampMixin, SoftDeleteMixin, Base):
    __tablename__ = "face_embeddings"

    # 1 พนักงาน = 1 embedding เสมอ (unique constraint)
    employee_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("employees.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True,
    )
    # เก็บ numpy float32 array ขนาด 512 มิติ ที่ serialize แล้วเป็น bytes ใน PostgreSQL
    embedding_vector: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    # เวอร์ชันโมเดลที่ใช้สร้าง embedding นี้ สำหรับ migration เมื่อเปลี่ยนโมเดล
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    image_quality_score: Mapped[float | None] = mapped_column(nullable=True)
    image_url: Mapped[str | None] = mapped_column(String(512), nullable=True)

    employee: Mapped["Employee"] = relationship("Employee", back_populates="face_embedding")  # noqa: F821