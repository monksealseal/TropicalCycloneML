"""
CycloneAPI — Commercial tropical cyclone intelligence service.

Subscription tiers, API key management, usage tracking, and rate limiting.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Tier(str, Enum):
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


@dataclass
class TierConfig:
    name: str
    price_monthly_cents: int          # in cents (e.g. 4900 = $49)
    price_yearly_cents: int
    requests_per_month: int
    requests_per_minute: int
    data_sources: list[str]
    features: list[str]
    model_access: list[str]
    support: str
    concurrent_monitors: int
    history_days: int                 # how far back historical queries go
    webhook_support: bool
    sla_uptime: float | None          # e.g. 99.9


TIER_CONFIGS: dict[Tier, TierConfig] = {
    Tier.FREE: TierConfig(
        name="Free",
        price_monthly_cents=0,
        price_yearly_cents=0,
        requests_per_month=100,
        requests_per_minute=5,
        data_sources=["nhc_active_storms", "nhc_rss"],
        features=["active_storms", "advisories"],
        model_access=[],
        support="community",
        concurrent_monitors=0,
        history_days=0,
        webhook_support=False,
        sla_uptime=None,
    ),
    Tier.STARTER: TierConfig(
        name="Starter",
        price_monthly_cents=4900,
        price_yearly_cents=47000,
        requests_per_month=5_000,
        requests_per_minute=30,
        data_sources=["nhc_active_storms", "nhc_rss", "nhc_arcgis", "atcf_best_track"],
        features=[
            "active_storms", "advisories", "forecast_track", "forecast_cone",
            "wind_radii", "atcf_best_track", "hf_inference",
        ],
        model_access=["weatherflow"],
        support="email",
        concurrent_monitors=1,
        history_days=90,
        webhook_support=False,
        sla_uptime=None,
    ),
    Tier.PROFESSIONAL: TierConfig(
        name="Professional",
        price_monthly_cents=19900,
        price_yearly_cents=191000,
        requests_per_month=50_000,
        requests_per_minute=120,
        data_sources=[
            "nhc_active_storms", "nhc_rss", "nhc_arcgis", "atcf_best_track",
            "ibtracs", "hurdat2",
        ],
        features=[
            "active_storms", "advisories", "forecast_track", "forecast_cone",
            "wind_radii", "atcf_best_track", "hf_inference", "ibtracs",
            "hurdat2", "full_analysis", "monitor", "bulk_export",
        ],
        model_access=["weatherflow", "autotrain-hurricane3"],
        support="priority_email",
        concurrent_monitors=5,
        history_days=365 * 3,
        webhook_support=True,
        sla_uptime=99.5,
    ),
    Tier.ENTERPRISE: TierConfig(
        name="Enterprise",
        price_monthly_cents=0,  # custom pricing
        price_yearly_cents=0,
        requests_per_month=1_000_000,
        requests_per_minute=600,
        data_sources=[
            "nhc_active_storms", "nhc_rss", "nhc_arcgis", "atcf_best_track",
            "ibtracs", "hurdat2", "jtwc", "custom",
        ],
        features=[
            "active_storms", "advisories", "forecast_track", "forecast_cone",
            "wind_radii", "atcf_best_track", "hf_inference", "ibtracs",
            "hurdat2", "full_analysis", "monitor", "bulk_export",
            "custom_models", "dedicated_endpoint", "white_label",
        ],
        model_access=["weatherflow", "autotrain-hurricane3", "custom"],
        support="dedicated_slack",
        concurrent_monitors=50,
        history_days=365 * 50,
        webhook_support=True,
        sla_uptime=99.99,
    ),
}


# ---------------------------------------------------------------------------
# API key management (in-memory for demo; swap with a DB in production)
# ---------------------------------------------------------------------------

@dataclass
class APIKey:
    key: str
    key_hash: str
    owner_email: str
    tier: Tier
    created_at: float = field(default_factory=time.time)
    stripe_customer_id: str | None = None
    stripe_subscription_id: str | None = None
    is_active: bool = True


@dataclass
class UsageRecord:
    key_hash: str
    month: str              # "2026-02"
    request_count: int = 0
    last_request_at: float = 0.0
    minute_bucket: str = ""
    minute_count: int = 0


class KeyStore:
    """In-memory API key + usage store. Replace with PostgreSQL/Redis in prod."""

    def __init__(self) -> None:
        self._keys: dict[str, APIKey] = {}          # hash -> APIKey
        self._usage: dict[str, UsageRecord] = {}    # hash:month -> UsageRecord

    @staticmethod
    def _hash(key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()

    def create_key(self, email: str, tier: Tier = Tier.FREE) -> APIKey:
        raw = f"cyc_{secrets.token_urlsafe(32)}"
        h = self._hash(raw)
        rec = APIKey(key=raw, key_hash=h, owner_email=email, tier=tier)
        self._keys[h] = rec
        return rec

    def validate(self, raw_key: str) -> APIKey | None:
        h = self._hash(raw_key)
        rec = self._keys.get(h)
        if rec and rec.is_active:
            return rec
        return None

    def get_tier_config(self, raw_key: str) -> TierConfig | None:
        rec = self.validate(raw_key)
        if rec is None:
            return None
        return TIER_CONFIGS[rec.tier]

    def record_request(self, raw_key: str) -> tuple[bool, str]:
        """
        Record a request and check rate limits.

        Returns (allowed, reason).
        """
        rec = self.validate(raw_key)
        if rec is None:
            return False, "Invalid or inactive API key"

        tier_cfg = TIER_CONFIGS[rec.tier]
        now = time.time()
        month = time.strftime("%Y-%m", time.gmtime(now))
        minute = time.strftime("%Y-%m-%dT%H:%M", time.gmtime(now))

        usage_key = f"{rec.key_hash}:{month}"
        usage = self._usage.get(usage_key)
        if usage is None:
            usage = UsageRecord(key_hash=rec.key_hash, month=month)
            self._usage[usage_key] = usage

        # Monthly limit
        if usage.request_count >= tier_cfg.requests_per_month:
            return False, f"Monthly limit reached ({tier_cfg.requests_per_month} requests)"

        # Per-minute limit
        if usage.minute_bucket == minute:
            if usage.minute_count >= tier_cfg.requests_per_minute:
                return False, f"Rate limit: {tier_cfg.requests_per_minute} req/min"
            usage.minute_count += 1
        else:
            usage.minute_bucket = minute
            usage.minute_count = 1

        usage.request_count += 1
        usage.last_request_at = now
        return True, "ok"

    def get_usage(self, raw_key: str) -> dict[str, Any]:
        rec = self.validate(raw_key)
        if rec is None:
            return {}
        month = time.strftime("%Y-%m", time.gmtime())
        usage_key = f"{rec.key_hash}:{month}"
        usage = self._usage.get(usage_key, UsageRecord(key_hash=rec.key_hash, month=month))
        tier_cfg = TIER_CONFIGS[rec.tier]
        return {
            "tier": rec.tier.value,
            "month": month,
            "requests_used": usage.request_count,
            "requests_limit": tier_cfg.requests_per_month,
            "requests_remaining": max(0, tier_cfg.requests_per_month - usage.request_count),
            "rate_limit_per_minute": tier_cfg.requests_per_minute,
        }


# Singleton store
_store = KeyStore()


def get_store() -> KeyStore:
    return _store
