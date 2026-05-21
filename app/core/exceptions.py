"""
Domain exception hierarchy

ใช้ typed exception แทน ValueError/RuntimeError ทั่วไป เพื่อให้
global handler map exception ไปยัง HTTP status code แบบ declarative
ห้าม raise HTTPException จาก service หรือ repository layer เด็ดขาด layer เหล่านั้นไม่รู้ว่าตัวเองถูกเรียกจาก HTTP
"""
from __future__ import annotations

from uuid import UUID


class AppError(Exception):
    """รากของทุก application exception"""
    message: str = "เกิดข้อผิดพลาดที่ไม่คาดคิด"

    def __init__(self, message: str | None = None) -> None:
        self.message = message or self.__class__.message
        super().__init__(self.message)


# --- Error 404 Not Found ---

class NotFoundError(AppError):
    message = "ไม่พบข้อมูลที่ต้องการ"


class EmployeeNotFoundError(NotFoundError):
    def __init__(self, employee_id: UUID | str) -> None:
        super().__init__(f"ไม่พบพนักงาน '{employee_id}'")


# --- Error 409 Conflict ---

class ConflictError(AppError):
    message = "ข้อมูลซ้ำกัน"


class EmployeeCodeConflictError(ConflictError):
    def __init__(self, code: str) -> None:
        super().__init__(f"รหัสพนักงาน '{code}' มีอยู่ในระบบแล้ว")


# --- Error 422 Validation ---

class ValidationError(AppError):
    message = "ข้อมูลไม่ถูกต้อง"


class FaceNotDetectedError(ValidationError):
    message = "ไม่พบใบหน้าในภาพที่ส่งมา"


class MultipleFacesError(ValidationError):
    message = "พบหลายใบหน้าในภาพ กรุณาส่งภาพที่มีใบหน้าเดียว"


class ImageQualityError(ValidationError):
    def __init__(self, score: float, threshold: float) -> None:
        super().__init__(
            f"คุณภาพภาพ {score:.2f} ต่ำกว่าเกณฑ์ขั้นต่ำ {threshold:.2f}"
        )


# --- Error 503 Service Unavailable ---

class ServiceUnavailableError(AppError):
    message = "บริการไม่พร้อมใช้งานชั่วคราว"


class AIEngineNotReadyError(ServiceUnavailableError):
    message = "ระบบจำแนกใบหน้ายังไม่พร้อม"


class CameraConnectionError(ServiceUnavailableError):
    def __init__(self, camera_id: str) -> None:
        super().__init__(f"ไม่สามารถเชื่อมต่อกล้อง '{camera_id}' ได้")