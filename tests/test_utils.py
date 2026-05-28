# pyrefly: ignore [missing-import]
import pytest
from datetime import datetime
from app.core.timezone import get_local_now, get_local_today, BANGKOK_TZ
from app.models.employee import Employee
from app.models.face_embedding import FaceEmbedding
from app.models.attendance import AttendanceRecord
from app.services.minio_service import minio_service


def test_timezone_utils():
    """Verify that Bangkok timezone helpers return correct, localized values."""
    now = get_local_now()
    assert now.tzinfo == BANGKOK_TZ

    today = get_local_today()
    assert today == datetime.now(BANGKOK_TZ).date()


def test_soft_delete_mixin_attributes():
    """Verify that models correctly inherit soft delete mixin and store image URL attributes."""
    emp = Employee(employee_code="EMP-TEST-99", full_name="Test Soft Delete")
    assert hasattr(emp, "deleted_at")
    assert emp.deleted_at is None

    emb = FaceEmbedding(model_version="test")
    assert hasattr(emb, "deleted_at")
    assert hasattr(emb, "image_url")
    assert emb.deleted_at is None

    rec = AttendanceRecord(confidence_score=0.99)
    assert hasattr(rec, "deleted_at")
    assert hasattr(rec, "image_url")
    assert rec.deleted_at is None


@pytest.mark.asyncio
async def test_minio_upload_mocked(mocker):
    """Verify that MinIO upload correctly maps types and constructs public URLs."""
    # Mock the client's put_object to isolate from actual connection
    mock_put = mocker.patch.object(minio_service.client, "put_object")

    image_bytes = b"fake-jpeg-data"
    filename = "test-face.jpg"

    url = minio_service.upload_image(image_bytes, filename)

    assert "test-face.jpg" in url
    assert "images" in url
    mock_put.assert_called_once()
