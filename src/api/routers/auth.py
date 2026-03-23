"""src/api/routers/auth.py — Authentication endpoints (login, refresh, logout)."""
from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.security import (
    create_token_pair,
    decode_token,
    hash_password,
    verify_password,
)
from src.config import get_settings
from src.db.models import User
from src.db.session import get_db
from src.api.deps import get_redis

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()
bearer_scheme = HTTPBearer(auto_error=False)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class RefreshRequest(BaseModel):
    refresh_token: str


async def _get_user_by_email(db: AsyncSession, email: str) -> User | None:
    """Fetch user by email from database."""
    result = await db.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()


async def _create_default_admin_if_missing(db: AsyncSession) -> None:
    """Create a default admin user if no users exist (first-run bootstrap)."""
    result = await db.execute(select(User).limit(1))
    if result.scalar_one_or_none() is None:
        admin = User(
            email="admin@medpredict.local",
            hashed_password=hash_password("changeme123"),
            role="admin",
            full_name="System Administrator",
            is_active=True,
        )
        db.add(admin)
        await db.commit()
        logger.info("Created default admin user: admin@medpredict.local")


@router.post("/login", response_model=LoginResponse)
async def login(
    payload: LoginRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> LoginResponse:
    """
    Authenticate user and return access + refresh token pair.
    
    On first run, creates a default admin account if no users exist.
    """
    # Bootstrap: create default admin on first run
    await _create_default_admin_if_missing(db)
    
    # Query user from database
    user = await _get_user_by_email(db, payload.email)
    
    if user is None:
        logger.warning("Login attempt for non-existent user: %s", payload.email)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    
    if not user.is_active:
        logger.warning("Login attempt for inactive user: %s", payload.email)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account is disabled",
        )
    
    if not verify_password(payload.password, user.hashed_password):
        logger.warning("Invalid password for user: %s", payload.email)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    
    # Create token pair
    pair = create_token_pair(user_id=user.id, role=user.role)
    
    logger.info("User logged in successfully: %s", payload.email)
    return LoginResponse(
        access_token=pair.access_token,
        refresh_token=pair.refresh_token,
        expires_in=pair.expires_in,
    )


@router.post("/refresh", response_model=LoginResponse)
async def refresh_token(
    payload: RefreshRequest,
    redis: Annotated[Any, Depends(get_redis)],
) -> LoginResponse:
    """
    Exchange a valid refresh token for a new access + refresh token pair.
    """
    try:
        token_payload = decode_token(payload.refresh_token)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        ) from exc
    
    if token_payload.type != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is not a refresh token",
        )
    
    # Check if token is blacklisted
    blacklist_key = f"token_blacklist:{token_payload.jti}"
    if await redis.get(blacklist_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
        )
    
    # Create new token pair
    pair = create_token_pair(user_id=token_payload.sub, role=token_payload.role)
    
    return LoginResponse(
        access_token=pair.access_token,
        refresh_token=pair.refresh_token,
        expires_in=pair.expires_in,
    )


@router.post("/logout")
async def logout(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(bearer_scheme)],
    redis: Annotated[Any, Depends(get_redis)],
) -> dict:
    """
    Invalidate the current access token by adding its JTI to Redis blacklist.
    """
    if credentials is None:
        return {"message": "Already logged out"}
    
    try:
        token_payload = decode_token(credentials.credentials)
        
        # Add token JTI to blacklist with TTL matching token expiry
        blacklist_key = f"token_blacklist:{token_payload.jti}"
        ttl = max(1, int((token_payload.exp - token_payload.iat).total_seconds()))
        await redis.setex(blacklist_key, ttl, "revoked")
        
        logger.info("Token revoked for user: %s", token_payload.sub)
    except (ValueError, Exception) as exc:
        logger.warning("Logout with invalid token: %s", exc)
    
    return {"message": "Logged out successfully"}
