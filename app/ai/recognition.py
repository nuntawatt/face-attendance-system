"""
Face recognition: จับคู่ embedding กับ employee index ใน memory

Employee index คือ numpy matrix ใน memory สำหรับ N พนักงาน
การจำแนกคือ matrix multiplication ครั้งเดียว: O(N * 512)
สำหรับ N=5,000 คน ใช้เวลา ~0.3ms บน CPU สมัยใหม่ — ไม่ต้องใช้ GPU เลย

Index ถูกโหลดตอน startup จาก DB และ refresh ผ่าน Redis pub/sub
เมื่อมีการลงทะเบียนใบหน้าใหม่

ทำไมไม่ใช้ FAISS?
สำหรับพนักงานน้อยกว่า 10,000 คน numpy matmul เร็วกว่า FAISS ANN
เพราะ FAISS overhead (quantization, index lookup) สูงกว่าต้นทุน brute-force
ที่ scale นี้ เปลี่ยนเป็น FAISS เมื่อ N > 50,000 คน
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import NamedTuple
from uuid import UUID

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# ค่า threshold cosine similarity ปรับตาม environment จริง
# ยิ่งสูงยิ่งเข้มงวด ลด false positive แต่อาจ miss พนักงานที่เหนื่อยหรือสวมแมสก์
RECOGNITION_THRESHOLD = 0.45


class RecognitionMatch(NamedTuple):
    employee_id: UUID
    similarity: float  # ค่า cosine similarity (0.0 - 1.0)
    is_confident: bool # True ถ้า similarity >= threshold


@dataclass
class EmployeeEmbeddingIndex:
    """
    In-memory embedding index ที่ปลอดภัยสำหรับ thread

    _matrix shape: (N, 512) stacked normalized embeddings
    _employee_ids: list UUID ในลำดับเดียวกับ row ของ matrix

    การ update เป็นแบบ full reload (copy-on-write) ไม่ใช่ in-place mutation
    เพื่อหลีกเลี่ยง race condition ระหว่างการจำแนกใบหน้า
    """

    _matrix: np.ndarray = field(default_factory=lambda: np.empty((0, 512), dtype=np.float32))
    _employee_ids: list[UUID] = field(default_factory=list)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def size(self) -> int:
        return len(self._employee_ids)

    async def rebuild(self, embeddings: dict[UUID, np.ndarray]) -> None:
        """แทนที่ index ทั้งหมดแบบ atomic เรียกหลัง registration ใหม่"""
        if not embeddings:
            async with self._lock:
                self._matrix = np.empty((0, 512), dtype=np.float32)
                self._employee_ids = []
            return

        ids = list(embeddings.keys())
        matrix = np.stack([embeddings[eid] for eid in ids]).astype(np.float32)

        async with self._lock:
            self._employee_ids = ids
            self._matrix = matrix

        logger.info("embedding_index_rebuilt", size=len(ids))

    async def find_match(self, probe: np.ndarray) -> RecognitionMatch | None:
        """
        ค้นหาด้วย cosine similarity คืน match ที่ดีที่สุดที่เหนือ threshold
        probe ต้องเป็น normalized (L2) float32 vector shape (512,)
        """
        async with self._lock:
            if self._matrix.shape[0] == 0:
                return None
            matrix = self._matrix
            ids = self._employee_ids

        # Cosine similarity = dot product ของ normalized vector
        similarities = matrix @ probe  # shape: (N,)
        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])

        return RecognitionMatch(
            employee_id=ids[best_idx],
            similarity=best_sim,
            is_confident=best_sim >= RECOGNITION_THRESHOLD,
        )


# Singleton ระดับ application
embedding_index = EmployeeEmbeddingIndex()