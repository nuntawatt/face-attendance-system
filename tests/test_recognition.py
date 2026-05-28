# pyrefly: ignore [missing-import]
import pytest
import numpy as np
from uuid import uuid4
from app.ai.recognition import EmployeeEmbeddingIndex, RecognitionMatch


@pytest.mark.asyncio
async def test_embedding_index_matching():
    """ระบบเปรียบเทียบในหน่วยความจำและการจับคู่ใบหน้า (In-Memory Embedding Index & Search)

    ทดสอบประสิทธิภาพการค้นหาและแมตช์ความถูกต้องของโครงสร้างดัชนีจำพวก Multi-Class Cosine Similarity
    """
    index = EmployeeEmbeddingIndex()
    
    # 1. จำลองข้อมูล Face Embeddings 2 คน
    emp_id_a = uuid4()
    emp_id_b = uuid4()
    
    # สร้าง Vector ขนาด 512 มิติแบบ Normalize แล้ว
    vec_a = np.zeros(512, dtype=np.float32)
    vec_a[0] = 1.0  # Vector แรกชี้ไปแนวแกนที่ 0
    
    vec_b = np.zeros(512, dtype=np.float32)
    vec_b[1] = 1.0  # Vector สองชี้ไปแนวแกนที่ 1 (ตั้งฉากกัน คล้ายคลึงต่ำ)
    
    # โหลดลง index
    embeddings = {
        emp_id_a: vec_a,
        emp_id_b: vec_b
    }
    await index.rebuild(embeddings)
    
    assert index.size == 2

    # 2. กรณีเปรียบเทียบกับภาพที่ใกล้เคียงกับ A (ความแม่นยำสูง)
    probe_a = np.zeros(512, dtype=np.float32)
    probe_a[0] = 0.99
    probe_a[1] = 0.05
    # Normalize probe
    probe_a = probe_a / np.linalg.norm(probe_a)
    
    match_a = await index.find_match(probe_a)
    
    assert match_a is not None
    assert match_a.employee_id == emp_id_a
    assert match_a.similarity > 0.95
    assert match_a.is_confident is True  # มั่นใจว่าใช่คน A

    # 3. กรณีเปรียบเทียบกับภาพที่ไม่เหมือนใครเลย (คนแปลกหน้า / Stranger)
    probe_stranger = np.zeros(512, dtype=np.float32)
    probe_stranger[10] = 1.0  # ชี้ไปแกนที่ 10 ซึ่งไม่มีในระบบ
    
    match_stranger = await index.find_match(probe_stranger)
    
    assert match_stranger is not None
    assert match_stranger.similarity < 0.2
    assert match_stranger.is_confident is False  # ไม่บันทึก/ปฏิเสธเป็นคนแปลกหน้า
