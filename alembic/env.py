"""
alembic/env.py — Async-compatible Alembic environment.

Key design decisions:
- Uses SQLAlchemy async engine (asyncpg) so it matches the production app engine.
- Falls back to synchronous mode for `alembic check` / --sql offline dumps.
- Reads DATABASE_URL from the environment or falls back to a local SQLite path
  so developers can run migrations without a running Postgres instance.
- Autogenerate compares against all ORM models imported from src.db.models.
"""
from __future__ import annotations

import asyncio
import os
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# ── Import ORM Base so autogenerate discovers all models ─────────────────────
# Side-effect: importing Base causes all mapped classes to register their
# metadata, which is exactly what target_metadata needs.
from src.db.models import Base  # noqa: E402

# ── Alembic Config object ─────────────────────────────────────────────────────
config = context.config

# Inject the DATABASE_URL from the environment (overrides alembic.ini).
# In production: postgresql+asyncpg://user:pass@host/db
# For local dev/CI: falls back to SQLite (sync driver used for offline mode).
_db_url = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://medpredict:medpredict@localhost:5432/medpredict",
)
config.set_main_option("sqlalchemy.url", _db_url)

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ── Target metadata ───────────────────────────────────────────────────────────
target_metadata = Base.metadata


# ── Offline mode (generates SQL script, no DB connection needed) ──────────────
def run_migrations_offline() -> None:
    """Generate SQL without connecting to the DB (alembic upgrade --sql)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # Render UUID as VARCHAR for portability in offline SQL scripts
        render_as_batch=False,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


# ── Online mode (connects to DB and applies migrations directly) ──────────────
def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,          # detect column type changes
        compare_server_default=True,# detect server_default changes
        render_as_batch=False,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an async engine and run migrations inside a sync wrapper."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,    # never pool during migrations
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    """Entry point for online migration (used by alembic upgrade head)."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
