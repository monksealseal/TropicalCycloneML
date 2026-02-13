"""Authentication and authorization for the Climate API.

Supports two auth methods:
1. API Key authentication (header: X-API-Key: clm_...)
2. Bearer token (header: Authorization: Bearer <token>)

Uses HMAC-SHA256 tokens (no external JWT dependency needed).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import base64
import time
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session

from climate_agent.db.models import ApiKey, User

TOKEN_SECRET = os.environ.get("JWT_SECRET", "change-me-in-production-please")
TOKEN_EXPIRATION_HOURS = 24


def _hmac_sign(payload_b64: bytes) -> str:
    return hmac.new(TOKEN_SECRET.encode(), payload_b64, hashlib.sha256).hexdigest()


def hash_password(password: str) -> str:
    """Hash a plaintext password with a salt using SHA-256.

    For production, swap this out for bcrypt/argon2 by installing passlib.
    """
    salt = os.urandom(16).hex()
    h = hashlib.sha256(f"{salt}${password}".encode()).hexdigest()
    return f"{salt}${h}"


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plaintext password against its hash."""
    parts = hashed.split("$", 1)
    if len(parts) != 2:
        return False
    salt, expected = parts
    h = hashlib.sha256(f"{salt}${plain}".encode()).hexdigest()
    return hmac.compare_digest(h, expected)


def create_access_token(user_id: str, email: str, plan: str) -> str:
    """Create an HMAC-signed bearer token."""
    payload = {
        "sub": user_id,
        "email": email,
        "plan": plan,
        "exp": int(time.time()) + TOKEN_EXPIRATION_HOURS * 3600,
    }
    payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode())
    sig = _hmac_sign(payload_b64)
    return f"{payload_b64.decode()}.{sig}"


def decode_access_token(token: str) -> dict[str, Any] | None:
    """Decode and verify a bearer token. Returns payload or None."""
    parts = token.rsplit(".", 1)
    if len(parts) != 2:
        return None
    payload_b64, sig = parts[0].encode(), parts[1]

    expected_sig = _hmac_sign(payload_b64)
    if not hmac.compare_digest(sig, expected_sig):
        return None

    try:
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
    except Exception:
        return None

    if payload.get("exp", 0) < time.time():
        return None
    return payload


def authenticate_api_key(db: Session, raw_key: str) -> User | None:
    """Authenticate a request using an API key."""
    key_hash = ApiKey.hash_key(raw_key)
    api_key = db.query(ApiKey).filter(ApiKey.key_hash == key_hash, ApiKey.is_active.is_(True)).first()

    if api_key is None:
        return None

    if api_key.expires_at and api_key.expires_at < datetime.utcnow():
        return None

    api_key.last_used_at = datetime.utcnow()
    db.commit()

    user = db.query(User).filter(User.id == api_key.user_id, User.is_active.is_(True)).first()
    return user


def authenticate_jwt(db: Session, token: str) -> User | None:
    """Authenticate a request using a bearer token."""
    payload = decode_access_token(token)
    if payload is None:
        return None
    user = db.query(User).filter(User.id == payload["sub"], User.is_active.is_(True)).first()
    return user


def register_user(
    db: Session,
    email: str,
    password: str,
    name: str | None = None,
    company: str | None = None,
) -> tuple[User, str]:
    """Register a new user and generate their first API key.

    Returns (user, raw_api_key).
    """
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise ValueError(f"User with email {email} already exists")

    user = User(
        email=email,
        password_hash=hash_password(password),
        name=name,
        company=company,
    )
    db.add(user)
    db.flush()

    raw_key, key_hash = ApiKey.generate()
    api_key = ApiKey(
        user_id=user.id,
        key_hash=key_hash,
        key_prefix=raw_key[:12],
        name="Default",
    )
    db.add(api_key)
    db.commit()

    return user, raw_key
