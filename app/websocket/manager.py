from __future__ import annotations

import asyncio

import structlog
from fastapi import WebSocket
from pydantic import BaseModel

logger = structlog.get_logger(__name__)


# WebSocket Event Schemas


class AttendanceEvent(BaseModel):
    """
    Event ที่ broadcast เมื่อ AI pipeline จำแนกใบหน้าและบันทึก attendance
    """

    event_type: str  # "check_in" | "check_out"
    employee_id: str
    employee_code: str | None = None
    full_name: str | None = None
    camera_id: str
    confidence: float
    timestamp: str  # ISO 8601

    def to_json(self) -> str:
        return self.model_dump_json()


class ConnectionManager:
    """
    Thread-safe WebSocket connection manager

    ใช้ set แทน list เพราะ:
    - add/remove เป็น O(1) ไม่ใช่ O(n)
    - ไม่มี duplicate connection
    - iteration ยังเร็วพอสำหรับ broadcast

    Lock ป้องกัน race condition เมื่อ client connect/disconnect
    พร้อมกับ broadcast จาก camera worker thread
    """

    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        """รับ connection ใหม่ ส่ง welcome message กลับ"""
        await websocket.accept()
        async with self._lock: self._connections.add(websocket)
        logger.info("ws_client_connected", total_connections=len(self._connections))

    async def disconnect(self, websocket: WebSocket) -> None:
        """ลบ connection ที่หลุด"""
        async with self._lock: self._connections.discard(websocket)
        logger.info("ws_client_disconnected", total_connections=len(self._connections))

    @property
    def active_count(self) -> int:
        return len(self._connections)

    async def broadcast(self, event: AttendanceEvent) -> None:
        """
        ส่ง event ไปทุก client ที่เชื่อมอยู่

        Client ที่ส่งไม่สำเร็จ (หลุดแล้ว) จะถูก disconnect อัตโนมัติ
        ใช้ gather เพื่อส่งพร้อมกัน ไม่รอทีละ client
        """
        if not self._connections:
            return

        message = event.to_json()

        async with self._lock:
            clients = list(self._connections)

        # ส่งพร้อมกันทุก client
        dead_clients: list[WebSocket] = []
        tasks = []
        for ws in clients:
            tasks.append(self._safe_send(ws, message, dead_clients))

        await asyncio.gather(*tasks)

        # ลบ client ที่ตายแล้ว
        if dead_clients:
            async with self._lock:
                for ws in dead_clients:
                    self._connections.discard(ws)

    async def _safe_send(self, ws: WebSocket, message: str, dead_list: list[WebSocket]) -> None:
        """ส่ง message ไป client ถ้าล้มเหลว เพิ่มเข้า dead_list"""
        try:
            await ws.send_text(message)
        except Exception:
            dead_list.append(ws)


# Singleton ใช้ร่วมกันทั้ง app (WebSocket route + AttendanceEngine)
attendance_manager = ConnectionManager()
