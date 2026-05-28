# pyrefly: ignore [missing-import]
import pytest
from uuid import uuid4

from app.core.timezone import get_local_now
from app.models.employee import Employee
from app.models.attendance import AttendanceRecord
from app.database.session import async_session_factory, async_engine
from app.repositories.employee import EmployeeRepository
from app.repositories.attendance import AttendanceRepository


@pytest.mark.asyncio
async def test_attendance_lifecycle_flow(mocker):
    """ระบบบันทึกการเข้างานและออกงาน (Attendance Flow & De-duplication)

    ทดสอบการบันทึกประวัติการเข้างาน-ออกงาน เพื่อยืนยันว่าการทำงานระดับฐานข้อมูลสอดคล้องกันอย่างถูกต้อง
    """
    try:
        async with async_session_factory() as session:
            emp_repo = EmployeeRepository(session)
            attendance_repo = AttendanceRepository(session)

            # 1. จำลองพนักงาน
            emp_code = f"TEST-ATT-{uuid4()}".upper()[:20]
            emp = Employee(
                employee_code=emp_code,
                full_name="Worker Bee",
                department="Production",
                position="Staff",
            )
            emp = await emp_repo.create(emp)
            await session.flush()

            # 2. ตรวจสอบการลงชื่อเข้างานครั้งแรกของวัน
            now_time = get_local_now()
            work_date = now_time.date()
            
            # ยืนยันว่ายังไม่มีการบันทึก
            existing_record = await attendance_repo.get_today_record(emp.id)
            assert existing_record is None
            
            # บันทึก Check-in
            record = AttendanceRecord(
                employee_id=emp.id,
                work_date=work_date,
                check_in_time=now_time,
                camera_id="CAM-ENTRANCE-01",
                confidence_score=0.92,
                image_url="http://localhost:9000/attendance/face-01.jpg",
            )
            await attendance_repo.create(record)
            await session.flush()
            
            # ค้นหาอีกครั้งเพื่อยืนยันว่าบันทึกแล้ว
            record_db = await attendance_repo.get_today_record(emp.id)
            assert record_db is not None
            assert record_db.employee_id == emp.id
            assert record_db.check_in_time == now_time
            assert record_db.check_out_time is None  # ยังไม่ออกงาน

            # 3. อัปเดตเวลาออกงาน (Check-out)
            out_time = get_local_now()
            record_db.check_out_time = out_time
            record_db.camera_id = "CAM-EXIT-01"
            await session.flush()
            
            # ยืนยันจาก DB ว่าบันทึกการออกงานแล้ว
            record_final = await attendance_repo.get_today_record(emp.id)
            assert record_final is not None
            assert record_final.check_out_time == out_time
            assert record_final.camera_id == "CAM-EXIT-01"

            await session.rollback()
    finally:
        await async_engine.dispose()
