"""
Attendance API router

Router สำหรับดูประวัติการเข้างาน
Attendance record ถูกสร้างโดย AI pipeline (AttendanceEngine) ไม่ใช่ API
Router นี้เป็น read-only สำหรับ dashboard/frontend
"""
from __future__ import annotations

from datetime import date
from uuid import UUID

from fastapi import APIRouter, Depends, Query

from app.api.deps import get_db_session
from app.repositories.attendance import AttendanceRepository
from app.schemas.attendance import AttendanceRecordResponse

from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/attendance", tags=["การเข้างาน"])


@router.get(
    "/today/{employee_id}",
    response_model=AttendanceRecordResponse | None,
    summary="ดูการเข้างานวันนี้ของพนักงาน",
)
async def get_today_attendance(
    employee_id: UUID,
    session: AsyncSession = Depends(get_db_session),
) -> AttendanceRecordResponse | None:
    repo = AttendanceRepository(session)
    record = await repo.get_today_record(employee_id)
    if not record:
        return None
    return AttendanceRecordResponse.model_validate(record)


@router.get(
    "/history/{employee_id}",
    response_model=list[AttendanceRecordResponse],
    summary="ดูประวัติการเข้างานตามช่วงวันที่",
)
async def get_attendance_history(
    employee_id: UUID,
    start_date: date = Query(..., description="วันที่เริ่มต้น (YYYY-MM-DD)"),
    end_date: date = Query(..., description="วันที่สิ้นสุด (YYYY-MM-DD)"),
    session: AsyncSession = Depends(get_db_session),
) -> list[AttendanceRecordResponse]:
    repo = AttendanceRepository(session)
    records = await repo.get_by_date_range(employee_id, start_date, end_date)
    return [AttendanceRecordResponse.model_validate(r) for r in records]
