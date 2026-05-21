from __future__ import annotations

from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AttendanceRecordResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    employee_id: UUID
    work_date: date
    check_in_time: datetime
    check_out_time: datetime | None
    camera_id: str
    confidence_score: float
    status: str
    created_at: datetime


class AttendanceQueryParams(BaseModel):
    start_date: date
    end_date: date
    employee_id: UUID | None = None

    @model_validator(mode="after")
    def validate_date_range(self) -> "AttendanceQueryParams":
        """ตรวจสอบช่วงวันที่ห้ามเกิน 90 วัน ป้องกัน query ที่หนักเกินไป"""
        if self.end_date < self.start_date:
            raise ValueError("end_date must be >= start_date")
        delta = (self.end_date - self.start_date).days
        if delta > 90:
            raise ValueError("Date range cannot exceed 90 days")
        return self