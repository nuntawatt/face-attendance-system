"""
Environment-based configuration ด้วย Pydantic Settings

ทุกค่ามาจาก environment variable (หรือ .env file)
ห้าม hardcode URL, credentials, หรือ threshold ในโค้ดเด็ดขาด

@cached_property บน camera_configs หมายความว่า parsing เกิดขึ้นครั้งเดียว
ไม่ใช่ทุกครั้งที่เข้าถึง
"""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings

from app.camera.stream_reader import CameraConfig


class Settings(BaseSettings):
    model_config = {
        "env_file": Path(__file__).resolve().parent.parent.parent / ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    # App
    app_name: str = "Face Attendance System"
    environment: Literal["development", "staging", "production"] = "production"
    log_level: str = "INFO"

    # CORS อนุญาตทุก origin ใน dev, จำกัดใน production
    cors_origins_str: str = Field("*", alias="CORS_ORIGINS")

    # Database Components
    postgres_host: str = Field(..., alias="POSTGRES_HOST")
    postgres_port: int = Field(5432, alias="POSTGRES_PORT")
    postgres_user: str = Field(..., alias="POSTGRES_USER")
    postgres_password: str = Field(..., alias="POSTGRES_PASSWORD")
    postgres_db: str = Field(..., alias="POSTGRES_DB")

    db_pool_size: int = 10
    db_max_overflow: int = 20

    @property
    def database_url(self) -> str:
        """Construct the SQLAlchemy AsyncPG connection string."""
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    # Redis Components
    redis_host: str = Field(..., alias="REDIS_HOST")
    redis_port: int = Field(6379, alias="REDIS_PORT")

    @property
    def redis_url(self) -> str:
        """Construct the Redis connection string."""
        return f"redis://{self.redis_host}:{self.redis_port}/0"

    # MinIO Components
    minio_endpoint: str = Field("localhost:9000", alias="MINIO_ENDPOINT")
    minio_access_key: str = Field("moragon", alias="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field("moragon1234", alias="MINIO_SECRET_KEY")
    minio_bucket_name: str = Field("images", alias="MINIO_BUCKET_NAME")
    minio_secure: bool = Field(False, alias="MINIO_SECURE")
    minio_external_endpoint: str = Field(
        "http://localhost:9000", alias="MINIO_EXTERNAL_ENDPOINT"
    )

    # AI — YuNet + EdgeFace
    face_det_size: int = 320
    face_det_threshold: float = 0.5
    face_recognition_threshold: float = 0.45
    min_image_quality: float = 0.4

    # Control background camera processing in API server lifespan
    enable_camera_workers: bool = Field(False, alias="ENABLE_CAMERA_WORKERS")

    # Camera JSON array of camera config dicts
    cameras_json: str = Field("[]", alias="CAMERAS_JSON")

    @cached_property
    def camera_configs(self) -> list[CameraConfig]:
        raw = json.loads(self.cameras_json)
        return [CameraConfig(**c) for c in raw]

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def cors_origins(self) -> list[str]:
        """Parse CORS origins จาก comma-separated string"""
        if self.cors_origins_str == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins_str.split(",") if o.strip()]


settings = Settings()
