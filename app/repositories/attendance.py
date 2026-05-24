"""
Repository สำหรับบันทึกการเข้างาน

การออกแบบที่สำคัญ: attendance record เป็นแบบ append-only จาก AI engine
ไม่มีการลบ record ใดๆ มีแค่ soft-invalidate เท่านั้น
เพื่อรักษา audit trail ไว้ตลอดเวลา
"""
from __future__ import annotations

from datetime import date, datetime
from uuid import UUID

from app.core.timezone import get_local_today

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.attendance import AttendanceRecord
from app.repositories.base import BaseRepository


class AttendanceRepository(BaseRepository[AttendanceRecord]):

    # AttendanceRepository จะสืบทอดจาก BaseRepository โดยระบุ ModelT เป็น AttendanceRecord
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(AttendanceRecord, session)

    async def get_today_record(self, employee_id: UUID) -> AttendanceRecord | None:
        """ดึง record การเข้างานของวันนี้ สำหรับตรวจสอบก่อนบันทึกซ้ำ"""
        today = get_local_today()
        result = await self._session.execute(
            select(AttendanceRecord).where(
                and_(
                    AttendanceRecord.employee_id == employee_id,
                    AttendanceRecord.work_date == today,
                    AttendanceRecord.deleted_at.is_(None),
                )
            )
        )
        return result.scalar_one_or_none()

    async def get_by_date_range(
        self,
        employee_id: UUID,
        start_date: date,
        end_date: date,
    ) -> list[AttendanceRecord]:
        """ดึงประวัติการเข้างานตามช่วงวันที่ ระบุ เรียงจากล่าสุด"""
        result = await self._session.execute(
            select(AttendanceRecord).where(
                and_(
                    AttendanceRecord.employee_id == employee_id,
                    AttendanceRecord.work_date >= start_date,
                    AttendanceRecord.work_date <= end_date,
                    AttendanceRecord.deleted_at.is_(None),
                )
            ).order_by(AttendanceRecord.work_date.desc(), AttendanceRecord.check_in_time.desc())
        )
        return list(result.scalars().all())

    async def mark_checkout(self, record_id: UUID, checkout_time: datetime) -> AttendanceRecord | None:
        """บันทึกเวลาออกงานจะเรียกจาก exit camera หรือ end-of-shift cron"""
        record = await self.get_by_id(record_id)
        if record:
            record.check_out_time = checkout_time
            await self._session.flush() # flush เพื่อบันทึกการเปลี่ยนแปลงใน session ก่อน refresh
            await self._session.refresh(record) # refresh เพื่อดึงข้อมูลล่าสุดจาก database หลังจาก update
        return record