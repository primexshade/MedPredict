"""
src/db/session.py — Async SQLAlchemy session factory.

Uses asyncpg driver for maximum Postgres throughput.
Provides get_db() FastAPI dependency for connection-per-request pattern.
"""
from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.config import get_settings

settings = get_settings()

# Convert postgresql:// → postgresql+asyncpg:// for async driver
_async_url = str(settings.database_url).replace(
    "postgresql://", "postgresql+asyncpg://", 1
)

engine = create_async_engine(
    _async_url,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,       # Detect stale connections before use
    echo=not settings.is_production,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: yields one async session per request."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
