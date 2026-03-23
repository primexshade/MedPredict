"""
src/db/session.py — Async SQLAlchemy session factory.

Supports both PostgreSQL (asyncpg) and SQLite (aiosqlite) drivers.
Provides get_db() FastAPI dependency for connection-per-request pattern.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Determine the async URL based on database type
_db_url = str(settings.database_url)

if _db_url.startswith("postgresql://"):
    # Convert postgresql:// → postgresql+asyncpg:// for async driver
    _async_url = _db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
elif _db_url.startswith("sqlite://"):
    # Convert sqlite:// → sqlite+aiosqlite:// for async driver
    _async_url = _db_url.replace("sqlite://", "sqlite+aiosqlite://", 1)
else:
    # Already has async driver specified
    _async_url = _db_url

# SQLite-specific settings
_is_sqlite = "sqlite" in _async_url
_engine_kwargs = {
    "echo": not settings.is_production,
}

if not _is_sqlite:
    # PostgreSQL-specific pool settings
    _engine_kwargs.update({
        "pool_size": 10,
        "max_overflow": 20,
        "pool_pre_ping": True,
    })
else:
    # SQLite requires connect_args for async
    _engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_async_engine(_async_url, **_engine_kwargs)

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
        except Exception as exc:
            logger.error("Database transaction failed, rolling back: %s", exc)
            await session.rollback()
            raise
