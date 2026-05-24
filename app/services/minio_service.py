"""
MinIO client service for managing face images and attendance snapshots.
"""

from __future__ import annotations

import io
import json
import asyncio
import structlog
from minio import Minio
from minio.error import S3Error

from app.core.config import settings

logger = structlog.get_logger(__name__)


class MinioService:
    """Handles object storage uploads to MinIO for registered and detected face images."""

    def __init__(self) -> None:
        self.bucket_name = settings.minio_bucket_name
        self.client = Minio(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )

    def ensure_bucket_exists(self) -> None:
        """Checks if the images bucket exists. If not, creates it and sets a public read-only policy."""
        try:
            found = self.client.bucket_exists(self.bucket_name)
            if not found:
                logger.info("minio_bucket_not_found", bucket=self.bucket_name)
                self.client.make_bucket(self.bucket_name)
                logger.info("minio_bucket_created", bucket=self.bucket_name)

                # Configure public read policy so returned links can be loaded directly in browser
                policy = {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"AWS": ["*"]},
                            "Action": ["s3:GetObject"],
                            "Resource": [f"arn:aws:s3:::{self.bucket_name}/*"],
                        }
                    ],
                }
                self.client.set_bucket_policy(self.bucket_name, json.dumps(policy))
                logger.info("minio_bucket_policy_configured", bucket=self.bucket_name)
            else:
                logger.debug("minio_bucket_exists", bucket=self.bucket_name)
        except S3Error as e:
            logger.error("minio_bucket_initialization_failed", error=str(e))
            raise e

    def upload_image(self, image_bytes: bytes, filename: str) -> str:
        """
        Uploads an image byte array to MinIO.
        Returns the external HTTP URL.
        """
        content_type = "image/jpeg"
        if filename.lower().endswith(".png"):
            content_type = "image/png"
        elif filename.lower().endswith(".webp"):
            content_type = "image/webp"
        elif filename.lower().endswith(".bmp"):
            content_type = "image/bmp"

        data_stream = io.BytesIO(image_bytes)
        try:
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=filename,
                data=data_stream,
                length=len(image_bytes),
                content_type=content_type,
            )
            # Construct external URL
            external_url = f"{settings.minio_external_endpoint.rstrip('/')}/{self.bucket_name}/{filename}"
            logger.info("minio_image_uploaded", filename=filename, url=external_url)
            return external_url
        except S3Error as e:
            logger.error("minio_image_upload_failed", filename=filename, error=str(e))
            raise e

    async def upload_image_async(self, image_bytes: bytes, filename: str) -> str:
        """อัปโหลดรูปภาพขึ้น MinIO แบบ Non-blocking (รันใน thread pool)"""
        return await asyncio.to_thread(self.upload_image, image_bytes, filename)


# Global singleton instance
minio_service = MinioService()
