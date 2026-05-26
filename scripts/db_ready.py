#!/usr/bin/env python3
"""
Infrastructure helper to check database availability before application startup.
"""

import asyncio
import os
import sys
from sqlalchemy.ext.asyncio import create_async_engine

async def wait_for_postgres(timeout_seconds: int = 30) -> None:
    """Polls PostgreSQL until a successful connection is established or timeout is reached."""
    print("Connecting to database...", flush=True)
    
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "nuntawat838")
    host = os.getenv("POSTGRES_HOST", "postgres")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "face_attendance")
    
    # Constructing asyncpg connection url directly to bypass any dependency boot circular imports
    database_url = f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db}"
    engine = create_async_engine(database_url)
    
    for attempt in range(1, timeout_seconds + 1):
        try:
            async with engine.connect():
                print("Database is ready to accept connections.", flush=True)
                sys.exit(0)
        except Exception:
            print(f"Database unavailable. Retrying ({attempt}/{timeout_seconds})...", flush=True)
            await asyncio.sleep(1)
            
    print(f"Error: Database connection timed out after {timeout_seconds} seconds.", file=sys.stderr, flush=True)
    sys.exit(1)

if __name__ == "__main__":
    asyncio.run(wait_for_postgres())
