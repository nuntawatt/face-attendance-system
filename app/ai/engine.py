"""
InsightFace ONNX engine wrapper

รูปแบบ Singleton: โมเดลถูกโหลดครั้งเดียวตอน startup และใช้ซ้ำตลอด
การโหลดโมเดล ONNX ขนาด 300MB ต่อ request จะทำให้ระบบล่มแน่นอน

Thread-safety: ONNX Runtime session ปลอดภัยสำหรับ inference พร้อมกันหลาย thread
เราใช้ asyncio.to_thread() รัน CPU-bound inference โดยไม่ block event loop

ปุ่มปรับ performance:
- inter_op_num_threads / intra_op_num_threads: ปรับตาม hardware จริง
- ใช้ CUDAExecutionProvider ถ้ามี GPU, fallback เป็น CPU
- face_det_size: ยิ่งเล็กยิ่งเร็ว แต่แม่นยำน้อยลงกับใบหน้าเล็ก
"""

from __future__ import annotations

import asyncio
from typing import NamedTuple

import cv2
import numpy as np
import structlog
from insightface.app import FaceAnalysis

logger = structlog.get_logger(__name__)


class DetectedFace(NamedTuple):
    embedding: np.ndarray  # shape: (512,) normalized float32
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    det_score: float  # คะแนน detection ความแน่ใจของการตรวจพบ
    quality_score: float  # คะแนนคุณภาพภาพ (0.0 - 1.0)


class FaceEngine:
    """
    Wraps InsightFace สำหรับ detection + embedding extraction

    model_pack: 'buffalo_l' (แม่นยำ) หรือ 'buffalo_s' (เร็ว)
    สำหรับโรงงาน buffalo_s มักเพียงพอและประหยัด CPU ~40%
    เมื่อเทียบกับ buffalo_l
    """

    def __init__(
        self,
        model_pack: str = "buffalo_s",
        det_size: tuple[int, int] = (320, 320),
        det_thresh: float = 0.5,
        providers: list[str] | None = None,
    ) -> None:
        self._model_pack = model_pack
        self._det_size = det_size
        self._det_thresh = det_thresh
        self._providers = providers or ["CPUExecutionProvider"]
        self._app: FaceAnalysis | None = None

    def load(self) -> None:
        """Blocking load เรียกจาก startup event ไม่ใช่จาก coroutine"""
        logger.info("กำลังโหลด face engine", model=self._model_pack)
        self._app = FaceAnalysis(
            name=self._model_pack,
            providers=self._providers,
        )
        self._app.prepare(
            ctx_id=0, det_size=self._det_size, det_thresh=self._det_thresh
        )
        logger.info("face engine พร้อมใช้งาน", model=self._model_pack)

    @property
    def is_ready(self) -> bool:
        return self._app is not None

    async def analyze_frame(self, frame: np.ndarray) -> list[DetectedFace]:
        """
        รัน detection + embedding บน frame แบบ async
        asyncio.to_thread ทำให้ CPU-bound ONNX inference ไม่ block event loop
        รักษา API responsiveness ไว้ได้แม้ขณะประมวลผลภาพ
        """
        if not self.is_ready:
            raise RuntimeError("FaceEngine ยังไม่ได้โหลด")
        return await asyncio.to_thread(self._analyze_sync, frame)

    def _analyze_sync(self, frame: np.ndarray) -> list[DetectedFace]:
        """ฟังก์ชัน blocking รันใน thread pool เสมอ ห้ามเรียกตรงจาก async code"""
        faces = self._app.get(frame)
        results: list[DetectedFace] = []
        for face in faces:
            embedding = face.normed_embedding.astype(np.float32)
            bbox = tuple(face.bbox.astype(int).tolist())
            quality = self._estimate_quality(frame, bbox)
            results.append(
                DetectedFace(
                    embedding=embedding,
                    bbox=bbox,
                    det_score=float(face.det_score),
                    quality_score=quality,
                )
            )
        return results

    @staticmethod
    def _estimate_quality(frame: np.ndarray, bbox: tuple) -> float:
        """
        Heuristic วัดคุณภาพภาพแบบเบา: Laplacian variance (ความคมชัด)
        คะแนน > 100 = ใช้ได้, > 200 = คุณภาพดีสำหรับลงทะเบียน
        ตั้งใจให้ถูก ไม่ใช้ ML model, ใช้แค่ CV math เพื่อประหยัด CPU
        """
        x1, y1, x2, y2 = bbox
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return min(float(variance) / 200.0, 1.0)  # normalize เป็น [0, 1]


def crop_and_encode_face(
    frame: np.ndarray, bbox: tuple[int, int, int, int], quality: int = 95
) -> bytes:
    """
    Crops a face from a BGR frame based on bounding box coordinates,
    guards against out-of-bound edge cases, and encodes it into JPEG bytes.
    """
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]

    # Safe boundary coordinate clipping
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))

    face_crop = frame[y1:y2, x1:x2]
    if face_crop.size == 0:
        raise ValueError("Invalid bounding box size resulted in an empty face crop.")

    _, buffer = cv2.imencode(".jpg", face_crop, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buffer.tobytes()


# Singleton ระดับ application ใช้ค่าจาก config
from app.core.config import settings

face_engine = FaceEngine(
    model_pack=settings.face_model_pack,
    det_size=(settings.face_det_size, settings.face_det_size),
    det_thresh=settings.face_det_threshold,
)
