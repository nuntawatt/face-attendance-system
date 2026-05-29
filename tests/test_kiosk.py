# pyrefly: ignore [missing-import]
import pytest
import numpy as np
import time
from uuid import uuid4
from unittest.mock import AsyncMock

from gui.run_kiosk import _recognize, _handle_force
from app.ai.recognition import RecognitionMatch, embedding_index


@pytest.mark.asyncio
async def test_kiosk_recognize_cooldown(mocker):
    """ทดสอบระบบแจ้งเตือนสเตตัส 5 วินาทีของ Kiosk (checked_in -> recognized)

    ตรวจสอบว่าเมื่อสแกนผ่านสำเร็จ แถบสถานะต้องโชว์เด่นสีเขียว (checked_in) เพียง 5 วินาที
    จากนั้นเมื่อเวลาผ่านไป ต้องสลับกลับมาแสดงผลปกติเป็นสีฟ้าหรูหรา (recognized) โดยอัตโนมัติ
    """
    emp_id = uuid4()
    emp_id_str = str(emp_id)
    emp_names = {emp_id_str: "Test Employee"}
    emp_status = {}
    events = []

    # Mock face
    mock_face = AsyncMock()
    mock_face.embedding = np.zeros(512, dtype=np.float32)

    # จำลองการแมตช์เจอในฐานข้อมูลแบบมั่นใจ (Similarity 0.95)
    mock_match = RecognitionMatch(
        employee_id=emp_id,
        similarity=0.95,
        is_confident=True
    )
    mocker.patch.object(embedding_index, "find_match", new_callable=AsyncMock, return_value=mock_match)
    
    # จำลองให้ embedding_index มีขนาดมากกว่า 0 เพื่อไม่ให้สลัดออกทันที
    mocker.patch("app.ai.recognition.EmployeeEmbeddingIndex.size", new_callable=mocker.PropertyMock, return_value=1)

    # จำลองเวลาเริ่มต้น
    start_time = 1000.0
    mocker.patch("time.monotonic", return_value=start_time)

    # --- การสแกนครั้งแรก: ต้องโชว์ checked_in (สีเขียวเด่น) ---
    name, eid, conf, status = await _recognize(mock_face, emp_names, emp_status, events)
    assert name == "Test Employee"
    assert eid == emp_id_str
    assert status == "checked_in"
    assert len(events) == 1
    assert events[0]["type"] == "check_in"

    # --- การสแกนครั้งที่สอง ทันที (เวลาผ่านไป 2 วินาที): ต้องคงสเตตัส checked_in ---
    mocker.patch("time.monotonic", return_value=start_time + 2.0)
    name, eid, conf, status = await _recognize(mock_face, emp_names, emp_status, events)
    assert status == "checked_in"

    # --- การสแกนครั้งที่สาม (เวลาผ่านไป 6 วินาที > 5 วินาที): ต้องสลับเป็น recognized อัตโนมัติ! ---
    mocker.patch("time.monotonic", return_value=start_time + 6.0)
    name, eid, conf, status = await _recognize(mock_face, emp_names, emp_status, events)
    assert status == "recognized"  # แถบแจ้งเตือนหายไปแล้ว เปลี่ยนเป็นสเตตัสสแกนติดปกติ!


@pytest.mark.asyncio
async def test_kiosk_recognize_threshold_prevention(mocker):
    """ทดสอบระบบป้องกันสแกนคนแปลกหน้าเป็นชื่อคนแรก (False Positive Prevention)

    ตรวจสอบว่าหากความคล้ายคลึงของใบหน้าต่ำกว่าเกณฑ์ความปลอดภัยสูงตัวใหม่ (0.58)
    ระบบจะไม่ทำการจับคู่มั่วหรือสแกนผ่านเด็ดขาด และยังคงแสดงผลเป็นสถานะ SCANNING เสมอ
    """
    emp_id = uuid4()
    emp_id_str = str(emp_id)
    emp_names = {emp_id_str: "Test Employee"}
    emp_status = {}
    events = []

    # Mock face
    mock_face = AsyncMock()
    mock_face.embedding = np.zeros(512, dtype=np.float32)

    # จำลองการเปรียบเทียบแล้วมีความคล้ายคลึงต่ำแค่ 0.48 (ต่ำกว่าเกณฑ์ใหม่ 0.58) -> is_confident = False
    mock_match = RecognitionMatch(
        employee_id=emp_id,
        similarity=0.48,
        is_confident=False
    )
    mocker.patch.object(embedding_index, "find_match", new_callable=AsyncMock, return_value=mock_match)
    
    # จำลองให้ embedding_index มีขนาดมากกว่า 0
    mocker.patch("app.ai.recognition.EmployeeEmbeddingIndex.size", new_callable=mocker.PropertyMock, return_value=1)

    # ทำการสแกนใบหน้า: ต้องส่งคืนผลลัพธ์เป็น scanning ไม่ดึงชื่อหรือบันทึกเวลาสแกนผ่านมั่วซั่ว
    name, eid, conf, status = await _recognize(mock_face, emp_names, emp_status, events)
    assert name is None
    assert eid is None
    assert conf == 0.48
    assert status == "scanning"
    assert len(events) == 0  # ต้องไม่มีการสร้าง event เข้างานใดๆ
