"""
tests/unit/test_security.py — Unit tests for JWT authentication and RBAC.
"""
from __future__ import annotations

import time
from datetime import timedelta

import pytest
from jose import JWTError

from src.auth.security import (
    create_token,
    create_token_pair,
    decode_token,
    has_permission,
    hash_password,
    verify_password,
)


class TestPasswordHashing:
    def test_hash_is_not_plaintext(self):
        hashed = hash_password("mysecretpassword")
        assert hashed != "mysecretpassword"

    def test_verify_correct_password(self):
        hashed = hash_password("correct-horse-battery-staple")
        assert verify_password("correct-horse-battery-staple", hashed) is True

    def test_verify_wrong_password_fails(self):
        hashed = hash_password("correctpassword")
        assert verify_password("wrongpassword", hashed) is False

    def test_same_password_produces_different_hashes(self):
        """bcrypt Salt ensures unique hashes per call."""
        h1 = hash_password("password")
        h2 = hash_password("password")
        assert h1 != h2


class TestJwtTokens:
    def test_access_token_decodes_correctly(self):
        token = create_token("user123", "clinician", "access", timedelta(minutes=30))
        payload = decode_token(token)
        assert payload.sub == "user123"
        assert payload.role == "clinician"
        assert payload.type == "access"

    def test_refresh_token_type_is_refresh(self):
        pair = create_token_pair("user456", "patient")
        refresh_payload = decode_token(pair.refresh_token)
        assert refresh_payload.type == "refresh"

    def test_expired_token_raises_error(self):
        """Tokens with -1 second expiry should immediately be invalid."""
        token = create_token("user789", "admin", "access", timedelta(seconds=-1))
        with pytest.raises(ValueError, match="Invalid token"):
            decode_token(token)

    def test_tampered_token_raises_error(self):
        token = create_token("user123", "clinician", "access", timedelta(minutes=5))
        bad_token = token[:-5] + "XXXXX"  # Corrupt the signature
        with pytest.raises(ValueError, match="Invalid token"):
            decode_token(bad_token)

    def test_token_pair_both_fields_present(self):
        pair = create_token_pair("userABC", "researcher")
        assert pair.access_token
        assert pair.refresh_token
        assert pair.token_type == "bearer"
        assert pair.expires_in > 0

    def test_jti_is_unique_per_token(self):
        """Each token must have a unique JWT ID for blacklisting."""
        t1 = create_token("u1", "clinician", "access", timedelta(minutes=5))
        t2 = create_token("u1", "clinician", "access", timedelta(minutes=5))
        p1 = decode_token(t1)
        p2 = decode_token(t2)
        assert p1.jti != p2.jti


class TestRBAC:
    def test_clinician_can_read_and_write(self):
        assert has_permission("clinician", "read") is True
        assert has_permission("clinician", "write") is True

    def test_clinician_cannot_delete(self):
        assert has_permission("clinician", "delete") is False

    def test_patient_can_only_read(self):
        assert has_permission("patient", "read") is True
        assert has_permission("patient", "write") is False

    def test_superadmin_has_all_permissions(self):
        for perm in ["read", "write", "delete", "admin", "deploy"]:
            assert has_permission("superadmin", perm) is True

    def test_unknown_role_has_no_permissions(self):
        assert has_permission("hacker", "read") is False
        assert has_permission("hacker", "delete") is False
