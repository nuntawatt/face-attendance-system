FROM python:3.11-slim

# ป้องกัน Python เขียนไฟล์ .pyc และตั้งค่า Log ให้แสดงผลแบบเรียลไทม์ (unbuffered)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ติดตั้งเครื่องมือคอมไพล์และไลบรารีระบบที่ OpenCV และ InsightFace C++ จำเป็นต้องใช้
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ติดตั้ง Python Packages โดยอิงจาก Docker layer caching เพื่อความรวดเร็ว
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# ความปลอดภัยระดับ Production: กำหนดสิทธิ์ให้รันด้วยผู้ใช้ทั่วไป (non-root) ป้องกันการเจาะระบบโฮสต์
RUN useradd -U -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
