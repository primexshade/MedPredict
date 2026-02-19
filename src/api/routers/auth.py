"""src/api/routers/auth.py — Authentication endpoints (login, refresh, logout)."""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from src.auth.security import create_token_pair, hash_password, verify_password

router = APIRouter()


class LoginRequest(BaseModel):
    email: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


@router.post("/login", response_model=LoginResponse)
async def login(payload: LoginRequest) -> LoginResponse:
    """
    Authenticate user and return access + refresh token pair.
    TODO: Replace stub with real DB user lookup.
    """
    # STUB: In production this queries the User table via SQLAlchemy
    if payload.email != "admin@example.com" or payload.password != "admin":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    pair = create_token_pair(user_id="stub-user-id", role="clinician")
    return LoginResponse(
        access_token=pair.access_token,
        refresh_token=pair.refresh_token,
        expires_in=pair.expires_in,
    )


@router.post("/logout")
async def logout() -> dict:
    """Invalidate refresh token (add JTI to Redis blacklist)."""
    # TODO: Add JTI to Redis blacklist
    return {"message": "Logged out successfully"}
