"""
ORM model สำหรับบันทึกการเข้างาน

Attendance record เป็น write-once จาก AI pipeline
check_out_time เป็น nullable เพราะระบบบันทึกการเข้างานทันที
แต่การออกงานบันทึกเมื่อตรวจพบใบหน้าที่กล้อง exit หรือ end-of-shift cron
"""
from __future__ import annotations

import uuid
from datetime import date, datetime

from sqlalchemy import Date, DateTime, Float, ForeignKey, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin, UUIDMixin, SoftDeleteMixin
from app.models.employee import Employee


class AttendanceRecord(UUIDMixin, TimestampMixin, SoftDeleteMixin, Base):
    __tablename__ = "attendance_records"
    __table_args__ = (
        UniqueConstraint("employee_id", "work_date", name="uq_attendance_employee_date"),
    )

    employee_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("employees.id", ondelete="CASCADE"),
        nullable=False,
        index=True, # index เพราะ query บ่อยตาม employee_id
    )
    work_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    check_in_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    check_out_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    camera_id: Mapped[str] = mapped_column(String(100), nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)

    # สถานะ: present=มาตรงเวลา, late=มาสาย, early_leave=กลับก่อนเวลา
    status: Mapped[str] = mapped_column(String(20), default="present", nullable=False)
    image_url: Mapped[str | None] = mapped_column(String(512), nullable=True)

    employee: Mapped["Employee"] = relationship("Employee", back_populates="attendance_records")  # noqa: F821