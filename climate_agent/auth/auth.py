"""Authentication and authorization for the Climate API.

Supports two auth methods:
1. API Key authentication (header: X-API-Key: clm_...)
2. JWT Bearer token (header: Authorization: Bearer <token>)
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timedelta
from typing import Any

import jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from climate_agent.db.models import ApiKey, User

JWT_SECRET = os.environ.get("JWT_SECRET", "change-me-in-production-please")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hash a plaintext password."""
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plaintext password against its hash."""
    return pwd_context.verify(plain, hashed)


def create_access_token(user_id: str, email: str, plan: str) -> str:
    """Create a JWT access token for a user."""
    payload = {
        "sub": user_id,
        "email": email,
        "plan": plan,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode and validate a JWT access token."""
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])


def authenticate_api_key(db: Session, raw_key: str) -> User | None:
    """Authenticate a request using an API key.

    Returns the associated User if the key is valid and active, else None.
    """
    key_hash = ApiKey.hash_key(raw_key)
    api_key = db.query(ApiKey).filter(ApiKey.key_hash == key_hash, ApiKey.is_active.is_(True)).first()

    if api_key is None:
        return None

    # Check expiration
    if api_key.expires_at and api_key.expires_at < datetime.utcnow():
        return None

    # Update last used timestamp
    api_key.last_used_at = datetime.utcnow()
    db.commit()

    user = db.query(User).filter(User.id == api_key.user_id, User.is_active.is_(True)).first()
    return user


def authenticate_jwt(db: Session, token: str) -> User | None:
    """Authenticate a request using a JWT bearer token."""
    try:
        payload = decode_access_token(token)
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
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
    # Check for existing user
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

    # Generate first API key
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
