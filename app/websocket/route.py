"""
WebSocket route สำหรับ real-time attendance feed

Client เชื่อมที่ ws://host/ws/attendance แล้วจะได้รับ JSON event
ทุกครั้งที่ AI pipeline ตรวจพบพนักงาน check-in หรือ check-out

ตัวอย่าง event ที่ client จะได้รับ:
{
    "event_type": "check_in",
    "employee_id": "550e8400-e29b-41d4-a716-446655440000",
    "employee_code": "EMP-001",
    "full_name": "สมชาย ใจดี",
    "camera_id": "entrance-cam-01",
    "confidence": 0.892,
    "timestamp": "2026-05-21T08:01:23+07:00"
}

Client สามารถส่ง "ping" message กลับมาเพื่อ keep-alive
Server จะตอบ "pong" กลับไป

WebSocket connection ไม่ต้อง auth ในตอนนี้ (เพิ่มทีหลังพร้อมกับ HTTP auth)
"""
from __future__ import annotations

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.websocket.manager import attendance_manager

logger = structlog.get_logger(__name__)

router = APIRouter()


@router.websocket("/ws/attendance")
async def attendance_feed(websocket: WebSocket) -> None:
    """
    WebSocket endpoint สำหรับ real-time attendance event stream

    Flow:
    1. Client connect -> server accept + เก็บใน connection manager
    2. Server broadcast attendance events ไปทุก client
    3. Client ส่ง "ping" -> server ตอบ "pong" (keep-alive)
    4. Client disconnect -> ลบออกจาก connection manager
    """
    await attendance_manager.connect(websocket)
    try:
        # รอรับ message จาก client (keep connection alive)
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        await attendance_manager.disconnect(websocket)
    except Exception:
        logger.exception("ws_unexpected_error")
        await attendance_manager.disconnect(websocket)
