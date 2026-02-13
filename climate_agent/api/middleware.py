"""Rate limiting and usage tracking middleware."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Callable

from fastapi import HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from climate_agent.db.models import (
    PlanTier,
    PLAN_LIMITS,
    UsageRecord,
    get_monthly_usage,
    get_recent_usage_per_minute,
)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting and usage tracking.

    Checks per-minute and per-month limits based on the user's plan tier.
    Records all API requests for usage analytics and billing.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for non-API routes
        if not request.url.path.startswith("/api/v1/"):
            return await call_next(request)

        # Skip for auth endpoints
        if request.url.path in ("/api/v1/auth/register", "/api/v1/auth/login"):
            return await call_next(request)

        user = getattr(request.state, "user", None)
        db = getattr(request.state, "db", None)

        if not user or not db:
            return await call_next(request)

        plan = PlanTier(user.plan)
        limits = PLAN_LIMITS[plan]

        # Check per-minute rate limit
        rpm = get_recent_usage_per_minute(db, user.id)
        if rpm >= limits["requests_per_minute"]:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "rate_limit_exceeded",
                    "message": f"Rate limit: {limits['requests_per_minute']} requests/minute "
                    f"for {limits['name']} plan",
                    "retry_after_seconds": 60,
                    "upgrade_url": "/pricing",
                },
            )

        # Check monthly limit
        monthly = get_monthly_usage(db, user.id)
        if monthly >= limits["requests_per_month"]:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "monthly_limit_exceeded",
                    "message": f"Monthly limit: {limits['requests_per_month']} requests/month "
                    f"for {limits['name']} plan",
                    "current_usage": monthly,
                    "upgrade_url": "/pricing",
                },
            )

        # Track request
        start_time = time.time()
        response = await call_next(request)
        latency_ms = int((time.time() - start_time) * 1000)

        # Record usage
        record = UsageRecord(
            user_id=user.id,
            endpoint=request.url.path,
            latency_ms=latency_ms,
            status_code=response.status_code,
        )
        db.add(record)
        db.commit()

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(limits["requests_per_minute"])
        response.headers["X-RateLimit-Remaining"] = str(max(0, limits["requests_per_minute"] - rpm - 1))
        response.headers["X-RateLimit-Monthly-Limit"] = str(limits["requests_per_month"])
        response.headers["X-RateLimit-Monthly-Remaining"] = str(max(0, limits["requests_per_month"] - monthly - 1))

        return response
