"""
YuNet + EdgeFace Engine — เบาและเร็วกว่า InsightFace 5-10x

Pipeline: Frame → YuNet (detect+landmarks) → Align 112×112 → EdgeFace (embedding 512)

YuNet:     ~0.3 MB, built-in OpenCV (cv2.FaceDetectorYN), ~2-5 ms/frame
EdgeFace:  ~6.9 MB, ONNX Runtime, ~5-10 ms/face, 512-dim embedding
รวม:       ~7.1 MB (เทียบ InsightFace buffalo_l = 326 MB)
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import onnxruntime as ort
import structlog

from app.core.config import settings

logger = structlog.get_logger(__name__)

# โฟลเดอร์เก็บโมเดล ONNX
_MODELS_DIR = Path(__file__).resolve().parent / "models"

# ArcFace alignment template — 5 จุดอ้างอิงบนภาพ 112×112
_ARCFACE_REF = np.array([
    [38.2946, 51.6963],   # ตาซ้าย
    [73.5318, 51.5014],   # ตาขวา
    [56.0252, 71.7366],   # จมูก
    [41.5493, 92.3655],   # มุมปากซ้าย
    [70.7299, 92.2041],   # มุมปากขวา
], dtype=np.float32)


class DetectedFace(NamedTuple):
    """ใบหน้าที่ตรวจจับได้ — interface เดิมทุกอย่าง"""
    embedding: np.ndarray               # shape (512,) normalized float32
    bbox: tuple[int, int, int, int]     # x1, y1, x2, y2
    det_score: float                    # คะแนน detection
    quality_score: float                # คะแนนคุณภาพ (Laplacian)


class FaceEngine:
    """
    YuNet detector + EdgeFace recognizer

    API เหมือน InsightFace engine เดิม:
      - face_engine.load()
      - await face_engine.analyze_frame(frame) → list[DetectedFace]
    """

    def __init__(
        self,
        det_size: tuple[int, int] = (320, 320),
        det_thresh: float = 0.5,
    ) -> None:
        self._det_size = det_size
        self._det_thresh = det_thresh
        self._detector: cv2.FaceDetectorYN | None = None
        self._recognizer: ort.InferenceSession | None = None
        self._rec_input_name: str = ""

    def load(self) -> None:
        """โหลด YuNet + EdgeFace (blocking — เรียกตอน startup)"""
        det_path = str(_MODELS_DIR / "yunet.onnx")
        rec_path = str(_MODELS_DIR / "edgeface_xs.onnx")

        # --- YuNet detector ---
        logger.info("กำลังโหลด YuNet detector", path=det_path)
        self._detector = cv2.FaceDetectorYN.create(
            model=det_path,
            config="",
            input_size=self._det_size,
            score_threshold=self._det_thresh,
            nms_threshold=0.3,
            top_k=10,
        )

        # --- EdgeFace recognizer ---
        logger.info("กำลังโหลด EdgeFace recognizer", path=rec_path)
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = 2
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        self._recognizer = ort.InferenceSession(rec_path, opts, providers=providers)
        self._rec_input_name = self._recognizer.get_inputs()[0].name

        # ตรวจสอบ output shape
        out_shape = self._recognizer.get_outputs()[0].shape
        logger.info("face engine พร้อมใช้งาน",
                     det_size=self._det_size, rec_output=out_shape)

    @property
    def is_ready(self) -> bool:
        return self._detector is not None and self._recognizer is not None

    async def analyze_frame(self, frame: np.ndarray) -> list[DetectedFace]:
        """รัน detection + embedding แบบ async (ไม่ block event loop)"""
        if not self.is_ready:
            raise RuntimeError("FaceEngine ยังไม่ได้โหลด")
        return await asyncio.to_thread(self._analyze_sync, frame)

    # --- internal methods (blocking) ---
    def _analyze_sync(self, frame: np.ndarray) -> list[DetectedFace]:
        """Blocking pipeline: detect → align → embed"""
        h, w = frame.shape[:2]
        self._detector.setInputSize((w, h))
        _, detections = self._detector.detect(frame)

        if detections is None:
            return []

        results: list[DetectedFace] = []
        for det in detections:
            score = float(det[14])
            if score < self._det_thresh:
                continue

            # bbox (x, y, w, h) → (x1, y1, x2, y2)
            bx, by, bw, bh = det[0:4].astype(int)
            x1 = max(0, int(bx))
            y1 = max(0, int(by))
            x2 = min(w, int(bx + bw))
            y2 = min(h, int(by + bh))
            if x2 <= x1 or y2 <= y1:
                continue

            # 5 landmarks จาก YuNet: (right_eye, left_eye, nose, right_mouth, left_mouth)
            landmarks = det[4:14].reshape(5, 2).astype(np.float32)

            # Align face → 112×112
            aligned = self._align_face(frame, landmarks)
            if aligned is None:
                continue

            # สร้าง embedding
            embedding = self._extract_embedding(aligned)

            # วัดคุณภาพ (Laplacian variance)
            quality = self._estimate_quality(frame, (x1, y1, x2, y2))

            results.append(DetectedFace(
                embedding=embedding,
                bbox=(x1, y1, x2, y2),
                det_score=score,
                quality_score=quality,
            ))

        return results

    def _align_face(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray | None:
        """
        Align ใบหน้าด้วย similarity transform จาก 5 landmarks → ArcFace template 112×112
        """
        # YuNet landmarks (viewer's left to right) มีลำดับตรงกับ ArcFace template coordinates อยู่แล้ว
        # ลำดับคือ: 0=ตาขวาคน(ซ้ายรูป), 1=ตาซ้ายคน(ขวารูป), 2=จมูก, 3=มุมปากขวาคน(ซ้ายรูป), 4=มุมปากซ้ายคน(ขวารูป)
        src_pts = landmarks

        # คำนวณ similarity transform
        M, _ = cv2.estimateAffinePartial2D(src_pts, _ARCFACE_REF, method=cv2.RANSAC)
        if M is None:
            return None

        aligned = cv2.warpAffine(frame, M, (112, 112), borderValue=(0, 0, 0))
        return aligned

    def _extract_embedding(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        รัน EdgeFace ONNX → 512-dim normalized embedding

        Preprocessing: BGR → RGB, normalize [-1, 1], HWC → CHW, batch
        """
        img = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (img - 127.5) / 128.0         # normalize เป็น [-1, 1]
        img = np.transpose(img, (2, 0, 1))   # HWC → CHW
        img = np.expand_dims(img, axis=0)     # batch dimension

        output = self._recognizer.run(None, {self._rec_input_name: img})[0]
        embedding = output[0].astype(np.float32)

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    @staticmethod
    def _estimate_quality(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
        """Laplacian variance — วัดความคมชัด (เหมือนเดิม)"""
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return min(float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 200.0, 1.0)


def crop_and_encode_face(
    frame: np.ndarray, bbox: tuple[int, int, int, int], quality: int = 95
) -> bytes:
    """Crop ใบหน้าจาก frame แล้ว encode เป็น JPEG bytes (ไม่เปลี่ยน)"""
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("Invalid bbox: empty face crop")

    _, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes()


# Singleton — ใช้ค่าจาก config
face_engine = FaceEngine(
    det_size=(settings.face_det_size, settings.face_det_size),
    det_thresh=settings.face_det_threshold,
)
