# Face Attendance System 🎭

A highly performant, AI-powered Face Recognition and Attendance tracking system built with modern Python technologies. This backend application leverages **FastAPI** for lightning-fast API responses, **InsightFace** for state-of-the-art facial recognition, and background camera stream processing.

## Features

- **AI Face Recognition**: Powered by InsightFace (buffalo_s model) for highly accurate face detection, landmark extraction, and recognition.
- **Asynchronous Architecture**: Built entirely on `asyncio`, utilizing FastAPI, AsyncPG, and async Redis for non-blocking I/O operations.
- **Background Camera Workers**: Continuously processes camera streams in background tasks without blocking the main API thread.
- **In-Memory Embedding Cache**: Automatically rebuilds face embedding indexes in memory on startup for real-time comparison.
- **Clean Architecture**: Domain-driven directory structure with clear separation of routers, services, repositories, and models.

---

## 🛠️ Tech Stack

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/)
- **Database**: PostgreSQL (with [SQLAlchemy 2.0](https://www.sqlalchemy.org/) & AsyncPG)
- **Caching & Message Broker**: Redis (`redis.asyncio`)
- **AI/CV Models**: [InsightFace](https://github.com/deepinsight/insightface), ONNXRuntime, OpenCV
- **Validation**: Pydantic v2
- **Testing**: Pytest with `pytest-asyncio`

---

## ✨ Getting Started

### Prerequisites

- Python 3.11+
- PostgreSQL 17
- Redis 8
- Docker & Docker Compose (Optional, for containerized environments)

### 1. Docker Service Commands

```bash
# Run
docker compose up -d

# Build the image and run
docker compose up -d --build

# Stop and cleanup volumes
docker compose down -v

# Stop containers
docker compose stop

# Restart containers
docker compose restart

# View logs
docker compose logs -f
```

### 2. Local Development Setup

If you prefer to run the FastAPI server directly on your machine:

1. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   uvicorn app.main:app --reload
   ```

---

## Testing

The project uses `pytest` for unit and integration testing. 

To run the test suite:

```bash
# Ensure you are in the virtual environment
pytest tests/ -v
```

---

## Project Structure

```
.
├── app/
│   ├── ai/            # InsightFace engine and recognition logic
│   ├── api/           # FastAPI routers (v1)
│   ├── attendance/    # Background camera workers and attendance logic
│   ├── core/          # Lifespan, Exceptions, Logging, Config
│   ├── database/      # SQLAlchemy AsyncSession factory
│   ├── models/        # Database ORM Models
│   ├── repositories/  # Database access layer
│   ├── schemas/       # Pydantic validation schemas
│   └── services/      # Business logic and use-cases
├── tests/             # Pytest test cases
├── Dockerfile         # Docker build instructions
├── docker-compose.yml # Container orchestration
└── requirements.txt   # Python dependencies
```
