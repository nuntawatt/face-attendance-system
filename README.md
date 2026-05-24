# Face Attendance System 🎭

A highly performant, AI-powered Face Recognition and Attendance tracking system built with modern Python technologies. This backend application leverages **FastAPI** for lightning-fast API responses, **InsightFace** for state-of-the-art facial recognition, **MinIO** for secure face image storage, and background camera stream processing.

---

## 🌟 Features

- **AI Face Recognition**: Powered by InsightFace (`buffalo_l` / `buffalo_s` model) for highly accurate face detection, landmark extraction, and recognition.
- **MinIO Image Storage**: Automatically crops detected faces, uploads them under random UUID names to a public `images` bucket in MinIO, and persists direct image links in the database.
- **Robust Timezone Alignment**: Configured with `Asia/Bangkok` timezone rules to eliminate server-dependent date mismatches (e.g. AWS/Render UTC issues).
- **Soft Delete Support**: Protects auditing trails across all database entities using a `deleted_at` timestamp.
- **Asynchronous Architecture**: Built entirely on `asyncio`, utilizing FastAPI, AsyncPG, and async Redis for non-blocking I/O operations.
- **Background Camera Workers**: Continuously processes RTSP streams in background tasks without blocking the main API thread.
- **In-Memory Embedding Cache**: Automatically rebuilds face embedding indexes in memory on startup for real-time comparison.
- **Clean Architecture**: Domain-driven directory structure with clear separation of routers, services, repositories, and models.

---

## 🛠️ Tech Stack

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/)
- **Database**: PostgreSQL (with [SQLAlchemy 2.0](https://www.sqlalchemy.org/) & AsyncPG)
- **Object Storage**: [MinIO](https://min.io/) (with standard `minio` SDK)
- **Caching & Message Broker**: Redis (`redis.asyncio`)
- **AI/CV Models**: [InsightFace](https://github.com/deepinsight/insightface), ONNXRuntime, OpenCV, Pillow
- **Validation**: Pydantic v2
- **Testing**: Pytest with `pytest-asyncio` & `pytest-mock`

---

## 📚 Data Dictionary (พจนานุกรมข้อมูล)

ระบบ Face Attendance System ใช้ PostgreSQL เป็นฐานข้อมูลหลัก มีตารางข้อมูล (Tables) ทั้งหมด 3 ตารางหลัก ดังนี้:

### 1. Table: `employees`
**คำอธิบาย:** เก็บข้อมูลประวัติส่วนตัวและการทำงานของพนักงาน รวมถึงสถานะว่าลงทะเบียนใบหน้าแล้วหรือยัง

| Column Name | Data Type (SQL) | Constraints | Default | Description (คำอธิบาย) |
| :--- | :--- | :--- | :--- | :--- |
| `id` | `UUID` | **PK**, Not Null | `uuid4()` | รหัสประจำตัวอ้างอิงระดับฐานข้อมูล |
| `employee_code` | `VARCHAR(50)` | **Unique**, Indexed, Not Null | - | รหัสพนักงาน (เช่น EMP-001) ใช้ล็อกอิน/อ้างอิง |
| `full_name` | `VARCHAR(200)` | Not Null | - | ชื่อ-นามสกุล ของพนักงาน |
| `department` | `VARCHAR(100)` | Indexed, Not Null | - | แผนกหรือฝ่ายที่สังกัด |
| `position` | `VARCHAR(100)` | Not Null | - | ตำแหน่งงาน |
| `email` | `VARCHAR(255)` | **Unique**, Nullable | `NULL` | อีเมลสำหรับการติดต่อ |
| `phone` | `VARCHAR(20)` | Nullable | `NULL` | เบอร์โทรศัพท์ |
| `notes` | `TEXT` | Nullable | `NULL` | หมายเหตุหรือข้อมูลเพิ่มเติม |
| `is_active` | `BOOLEAN` | Not Null | `TRUE` | สถานะการทำงาน (`TRUE`=ทำงาน, `FALSE`=ลาออก/พักงาน) |
| `face_registered` | `BOOLEAN` | Not Null | `FALSE` | สถานะการลงทะเบียนใบหน้า (`TRUE`=ลงทะเบียนแล้ว) |
| `created_at` | `TIMESTAMPTZ` | Not Null | `now()` | วัน-เวลาที่สร้าง record นี้ |
| `updated_at` | `TIMESTAMPTZ` | Not Null | `now()` | วัน-เวลาที่แก้ไข record นี้ล่าสุด |
| `deleted_at` | `TIMESTAMPTZ` | Nullable | `NULL` | วัน-เวลาที่พนักงานถูกลบออก (สำหรับ **Soft Delete**) |

---

### 2. Table: `face_embeddings`
**คำอธิบาย:** เก็บข้อมูลลักษณะทางชีวมิติของใบหน้า (Biometrics Vector) แยกตารางเพื่อลดภาระการโหลดข้อมูลพนักงานทั่วไป

| Column Name | Data Type (SQL) | Constraints | Default | Description (คำอธิบาย) |
| :--- | :--- | :--- | :--- | :--- |
| `id` | `UUID` | **PK**, Not Null | `uuid4()` | รหัสประจำตัวอ้างอิงระดับฐานข้อมูล |
| `employee_id` | `UUID` | **FK** (`employees.id`), **Unique**, Indexed, Not Null | - | อ้างอิงรหัสพนักงาน (1 พนักงาน มีได้ 1 ใบหน้าเท่านั้น) |
| `embedding_vector` | `BYTEA` | Not Null | - | ข้อมูล Vector 512 มิติ ที่ได้จาก AI (Serialize เป็น Binary) |
| `model_version` | `VARCHAR(50)` | Not Null | - | เวอร์ชันของ AI Model ที่สร้าง Vector (เช่น `buffalo_l_v1`) |
| `image_quality_score` | `FLOAT` | Nullable | `NULL` | คะแนนความคมชัดของใบหน้าตอนลงทะเบียน (0.0 - 1.0) |
| `image_url` | `VARCHAR(512)` | Nullable | `NULL` | ลิงก์รูปภาพใบหน้าพนักงานที่เซฟไว้ใน MinIO |
| `created_at` | `TIMESTAMPTZ` | Not Null | `now()` | วัน-เวลาที่สร้าง record นี้ |
| `updated_at` | `TIMESTAMPTZ` | Not Null | `now()` | วัน-เวลาที่แก้ไข record นี้ล่าสุด |
| `deleted_at` | `TIMESTAMPTZ` | Nullable | `NULL` | วัน-เวลาที่ข้อมูลใบหน้าถูกลบออก (สำหรับ **Soft Delete**) |

> ⚠️ **Constraint:** คอลัมน์ `employee_id` ตั้งค่าไว้เป็น **Cascade Delete** หากพนักงานถูกลบออกจากตาราง `employees` ข้อมูลใบหน้าจะหายไปอัตโนมัติ

---

### 3. Table: `attendance_records`
**คำอธิบาย:** เก็บประวัติการเข้า-ออกงานของพนักงานแบบรายวัน

| Column Name | Data Type (SQL) | Constraints | Default | Description (คำอธิบาย) |
| :--- | :--- | :--- | :--- | :--- |
| `id` | `UUID` | **PK**, Not Null | `uuid4()` | รหัสประจำตัวอ้างอิงระดับฐานข้อมูล |
| `employee_id` | `UUID` | **FK** (`employees.id`), Indexed, Not Null | - | อ้างอิงรหัสพนักงานที่ทำการสแกน |
| `work_date` | `DATE` | Indexed, Not Null | - | วันที่ทำงาน (อิงตามเขตเวลา `Asia/Bangkok`) |
| `check_in_time` | `TIMESTAMPTZ` | Not Null | - | เวลาเช็คอินครั้งแรกของวัน |
| `check_out_time` | `TIMESTAMPTZ` | Nullable | `NULL` | เวลาเช็คเอาท์ (อัปเดตเมื่อเกิน 10 นาทีหรือจบวัน) |
| `camera_id` | `VARCHAR(100)` | Not Null | - | รหัสของกล้องวงจรปิดหรือ Kiosk ที่ใช้สแกน |
| `confidence_score` | `FLOAT` | Not Null | - | ค่าความมั่นใจของ AI ว่าเป็นบุคคลนี้จริง (%) |
| `status` | `VARCHAR(20)` | Not Null | `'present'` | สถานะ: `present` (มา), `late` (สาย), `early_leave` (กลับก่อน) |
| `image_url` | `VARCHAR(512)` | Nullable | `NULL` | ลิงก์รูปภาพขณะสแกนเช็คอิน/เช็คเอาท์ในวันนั้น เซฟใน MinIO |
| `created_at` | `TIMESTAMPTZ` | Not Null | `now()` | วัน-เวลาที่สร้าง record นี้ |
| `updated_at` | `TIMESTAMPTZ` | Not Null | `now()` | วัน-เวลาที่แก้ไข record นี้ล่าสุด |
| `deleted_at` | `TIMESTAMPTZ` | Nullable | `NULL` | วัน-เวลาที่บันทึกนี้ถูกลบออก (สำหรับ **Soft Delete**) |

> ⚠️ **Constraint (Multi-column Unique):** ตารางนี้มีการบังคับ `UniqueConstraint("employee_id", "work_date")` เพื่อป้องกันพนักงานเช็คอินซ้ำซ้อนในวันเดียวกัน

---

## ✨ Getting Started

### Prerequisites
- Python 3.11+
- PostgreSQL 17
- Redis 8
- Docker

### 1. Docker Service Commands (MinIO, Postgres, Redis)

The orchestration spins up MinIO, PostgreSQL, and Redis automatically.
- **MinIO Web Portal**: [http://localhost:9001](http://localhost:9001)
- **MinIO API Port**: `9000`
- **Root Admin User**: `moragon`
- **Root Admin Password**: `moragon1234`

```bash
# Start all background services (MinIO, Postgres, Redis)
docker compose up -d minio postgres redis

# Build and run complete application stack
docker compose up -d --build

# View real-time logs
docker compose logs -f

# Stop containers
docker compose stop
```

### 2. Local Backend Server Setup

To run the FastAPI server locally on your host machine:

1. **Activate your virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Apply Database Schema Migration**:
   ```bash
   alembic upgrade head
   ```

4. **Start the API server**:
   ```bash
   uvicorn app.main:app --reload
   ```

---

## 🧪 Testing

The project utilizes `pytest` alongside standard DB transaction rollbacks to test functionality safely:

```bash
# Run all tests (including timezone, soft delete DB transaction, and S3 upload tests)
PYTHONPATH=. ./.venv/bin/pytest -v
```

---

## 📂 Project Structure

```
.
├── alembic/           # Database migration revisions and env config
├── app/
│   ├── ai/            # InsightFace ONNX engine and recognition logic
│   ├── api/           # FastAPI routers (v1 endpoints)
│   ├── attendance/    # Background camera workers and attendance engine
│   ├── core/          # Lifespan, Exceptions, Logging, Config, Timezone
│   ├── database/      # SQLAlchemy AsyncSession factories and Mixins
│   ├── models/        # Database ORM entity models
│   ├── repositories/  # Database access layer and query builders
│   ├── schemas/       # Pydantic data schemas
│   └── services/      # Business logic, caching, MinIO uploading
├── tests/             # Pytest unit and integration test suites
├── Dockerfile         # API Docker build instructions
├── docker-compose.yml # Service container orchestration
├── requirements.txt   # Declared Python dependencies
└── run_kiosk.py       # Standalone OpenCV terminal Kiosk app
```