from __future__ import annotations

# pyrefly: ignore [missing-import]
import structlog
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.recognition import embedding_index
from app.repositories.face_embedding import FaceEmbeddingRepository

logger = structlog.get_logger(__name__)


class EmbeddingCacheService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._repo = FaceEmbeddingRepository(session)

    async def rebuild_index(self) -> None:
        logger.info("rebuilding_embedding_index")
        embeddings = await self._repo.get_all_active_embeddings()

        cache = {}
        for emp in embeddings:
            vector = np.frombuffer(emp.embedding_vector, dtype=np.float32)
            cache[emp.employee_id] = vector

        await embedding_index.rebuild(cache)
        logger.info("embedding_index_rebuilt_successfully", count=len(cache))
