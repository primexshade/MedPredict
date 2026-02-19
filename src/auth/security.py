"""
src/auth/security.py — JWT authentication and password hashing.

Implements access + refresh token pair with Redis-backed blacklisting
for secure logout without server-side session state.

Note: Uses bcrypt directly (passlib 1.7 is incompatible with bcrypt 5.x).
"""
from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
from jose import JWTError, jwt
from pydantic import BaseModel

from src.config import get_settings

settings = get_settings()


class TokenPayload(BaseModel):
    sub: str        # user_id
    role: str
    type: str       # "access" | "refresh"
    jti: str        # JWT ID — unique per token for blacklisting
    iat: datetime
    exp: datetime


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


def hash_password(plain: str) -> str:
    """Bcrypt hash with cost factor 12 (recommended for 2025+ hardware)."""
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(plain.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


def create_token(
    user_id: str,
    role: str,
    token_type: str,
    expires_delta: timedelta,
) -> str:
    now = datetime.now(tz=timezone.utc)
    payload = {
        "sub": user_id,
        "role": role,
        "type": token_type,
        "jti": secrets.token_hex(16),  # Unique token ID for blacklisting
        "iat": now,
        "exp": now + expires_delta,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def create_token_pair(user_id: str, role: str) -> TokenPair:
    access = create_token(
        user_id, role, "access",
        timedelta(minutes=settings.access_token_expire_minutes),
    )
    refresh = create_token(
        user_id, role, "refresh",
        timedelta(days=settings.refresh_token_expire_days),
    )
    return TokenPair(
        access_token=access,
        refresh_token=refresh,
        expires_in=settings.access_token_expire_minutes * 60,
    )


def decode_token(token: str) -> TokenPayload:
    """
    Decode and validate a JWT token.

    Raises:
        JWTError: If token is invalid, expired, or malformed.
    """
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
        )
        return TokenPayload(**payload)
    except JWTError as exc:
        raise ValueError(f"Invalid token: {exc}") from exc


# ─── RBAC ─────────────────────────────────────────────────────────────────────

ROLE_PERMISSIONS: dict[str, set[str]] = {
    "superadmin": {"read", "write", "delete", "admin", "deploy"},
    "admin":      {"read", "write", "delete", "admin"},
    "clinician":  {"read", "write"},
    "patient":    {"read"},
    "researcher": {"read"},
}


def has_permission(role: str, permission: str) -> bool:
    return permission in ROLE_PERMISSIONS.get(role, set())
