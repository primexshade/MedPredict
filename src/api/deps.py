"""
src/api/deps.py — FastAPI shared dependencies.

Centralizes:
- JWT authentication + RBAC enforcement
- Redis connection
- Database session
"""
from __future__ import annotations

from typing import Annotated, Any

import redis.asyncio as aioredis
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.auth.security import decode_token, has_permission
from src.config import get_settings
from src.db.session import get_db

settings = get_settings()
bearer_scheme = HTTPBearer(auto_error=True)

# ── Redis Pool (singleton) ───────────────────────────────────────────────────
_redis_pool: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis:
    """Return the shared async Redis connection pool."""
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = aioredis.from_url(
            str(settings.redis_url),
            encoding="utf-8",
            decode_responses=True,
        )
    return _redis_pool


# ── JWT Auth Dependency ──────────────────────────────────────────────────────

async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Security(bearer_scheme)],
) -> dict:
    """
    Validate Bearer token and return token payload.
    Raises 401 if token is missing/invalid, 403 if token is a refresh token.
    """
    token = credentials.credentials
    try:
        payload = decode_token(token)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    if payload.type != "access":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Refresh tokens cannot be used for API access.",
        )

    return {"user_id": payload.sub, "role": payload.role, "jti": payload.jti}


def require_permission(permission: str):
    """
    Dependency factory for RBAC enforcement.

    Usage:
        @router.delete("/patient/{id}")
        async def delete_patient(
            _=Depends(require_permission("delete")),
            ...
        ): ...
    """
    async def _check(
        current_user: Annotated[dict, Depends(get_current_user)],
    ) -> dict:
        role = current_user["role"]
        if not has_permission(role, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role}' does not have '{permission}' permission.",
            )
        return current_user

    return _check
