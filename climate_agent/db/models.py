"""Database models for users, API keys, subscriptions, and usage tracking."""

from __future__ import annotations

import hashlib
import secrets
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker


class Base(DeclarativeBase):
    pass


class PlanTier(str, Enum):
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


PLAN_LIMITS = {
    PlanTier.FREE: {
        "name": "Explorer",
        "price_monthly_cents": 0,
        "price_yearly_cents": 0,
        "requests_per_month": 50,
        "requests_per_minute": 5,
        "agent_turns_per_request": 5,
        "tools": ["fetch_climate_indicators", "fetch_active_cyclones", "analyze_climate_trend"],
        "visualizations": False,
        "forecasting": False,
        "priority_support": False,
        "description": "Get started with basic climate data",
    },
    PlanTier.STARTER: {
        "name": "Starter",
        "price_monthly_cents": 4900,
        "price_yearly_cents": 47000,
        "requests_per_month": 1_000,
        "requests_per_minute": 20,
        "agent_turns_per_request": 15,
        "tools": "all_except_enterprise",
        "visualizations": True,
        "forecasting": True,
        "priority_support": False,
        "description": "For researchers and small teams",
    },
    PlanTier.PROFESSIONAL: {
        "name": "Professional",
        "price_monthly_cents": 19900,
        "price_yearly_cents": 190_000,
        "requests_per_month": 10_000,
        "requests_per_minute": 60,
        "agent_turns_per_request": 30,
        "tools": "all",
        "visualizations": True,
        "forecasting": True,
        "priority_support": True,
        "description": "For organizations and power users",
    },
    PlanTier.ENTERPRISE: {
        "name": "Enterprise",
        "price_monthly_cents": 0,  # Custom pricing
        "price_yearly_cents": 0,
        "requests_per_month": 999_999,
        "requests_per_minute": 300,
        "agent_turns_per_request": 50,
        "tools": "all",
        "visualizations": True,
        "forecasting": True,
        "priority_support": True,
        "description": "Custom solutions for large organizations",
    },
}


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255))
    password_hash = Column(String(255), nullable=False)
    company = Column(String(255))
    plan = Column(String(20), default=PlanTier.FREE.value, nullable=False)
    stripe_customer_id = Column(String(255), unique=True)
    stripe_subscription_id = Column(String(255))
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    api_keys = relationship("ApiKey", back_populates="user", cascade="all, delete-orphan")
    usage_records = relationship("UsageRecord", back_populates="user", cascade="all, delete-orphan")

    @property
    def plan_limits(self) -> dict[str, Any]:
        return PLAN_LIMITS.get(PlanTier(self.plan), PLAN_LIMITS[PlanTier.FREE])


class ApiKey(Base):
    __tablename__ = "api_keys"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    key_hash = Column(String(64), unique=True, nullable=False, index=True)
    key_prefix = Column(String(12), nullable=False)  # "clm_" + first 8 chars
    name = Column(String(100), default="Default")
    is_active = Column(Boolean, default=True)
    last_used_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)

    user = relationship("User", back_populates="api_keys")

    @staticmethod
    def generate() -> tuple[str, str]:
        """Generate a new API key pair: (raw_key, key_hash)."""
        raw = "clm_" + secrets.token_urlsafe(32)
        hashed = hashlib.sha256(raw.encode()).hexdigest()
        return raw, hashed

    @staticmethod
    def hash_key(raw_key: str) -> str:
        return hashlib.sha256(raw_key.encode()).hexdigest()


class UsageRecord(Base):
    __tablename__ = "usage_records"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    endpoint = Column(String(100), nullable=False)
    tool_calls = Column(Integer, default=0)
    agent_turns = Column(Integer, default=0)
    tokens_used = Column(Integer, default=0)
    latency_ms = Column(Integer, default=0)
    status_code = Column(Integer, default=200)
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    user = relationship("User", back_populates="usage_records")


class WaitlistEntry(Base):
    __tablename__ = "waitlist"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, nullable=False)
    company = Column(String(255))
    use_case = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


def create_database(database_url: str = "sqlite:///climate_agent.db") -> sessionmaker:
    """Create database tables and return a session factory."""
    engine = create_engine(database_url, echo=False)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def get_monthly_usage(session: Session, user_id: str) -> int:
    """Get the number of API requests this month for a user."""
    start_of_month = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    count = (
        session.query(func.count(UsageRecord.id))
        .filter(UsageRecord.user_id == user_id, UsageRecord.created_at >= start_of_month)
        .scalar()
    )
    return count or 0


def get_recent_usage_per_minute(session: Session, user_id: str) -> int:
    """Get the number of API requests in the last minute for a user."""
    one_minute_ago = datetime.utcnow() - timedelta(minutes=1)
    count = (
        session.query(func.count(UsageRecord.id))
        .filter(UsageRecord.user_id == user_id, UsageRecord.created_at >= one_minute_ago)
        .scalar()
    )
    return count or 0
