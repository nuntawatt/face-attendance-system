"""
Face registration API router

Router สำหรับลงทะเบียนใบหน้าพนักงาน
รับ image file upload แล้วส่งให้ FaceRegistrationService จัดการ
"""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, UploadFile, File, status

from app.api.deps import get_face_service
from app.schemas.face import FaceRegistrationResponse
from app.services.face_service import FaceRegistrationService

router = APIRouter(prefix="/employees", tags=["ลงทะเบียนใบหน้า"])


@router.post(
    "/{employee_id}/face",
    response_model=FaceRegistrationResponse,
    status_code=status.HTTP_200_OK,
    summary="ลงทะเบียนใบหน้าพนักงาน",
)
async def register_face(
    employee_id: UUID,
    image: UploadFile = File(..., description="ภาพใบหน้าพนักงาน (JPEG/PNG)"),
    service: FaceRegistrationService = Depends(get_face_service),
) -> FaceRegistrationResponse:
    """
    อัพโหลดภาพใบหน้าเพื่อลงทะเบียน:
    - ภาพต้องมีใบหน้าเพียง 1 ใบ
    - คุณภาพภาพต้องผ่านเกณฑ์ขั้นต่ำ
    - ถ้าพนักงานมี embedding อยู่แล้ว จะถูกแทนที่ด้วยอันใหม่
    """
    image_bytes = await image.read()
    return await service.register_face(employee_id, image_bytes)
