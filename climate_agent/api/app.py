"""FastAPI application for the Climate Analysis API.

This is the main web service that exposes the climate agent as a commercial API.

    uvicorn climate_agent.api.app:app --reload
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from climate_agent.api.middleware import RateLimitMiddleware
from climate_agent.auth.auth import (
    authenticate_api_key,
    authenticate_jwt,
    create_access_token,
    hash_password,
    register_user,
    verify_password,
)
from climate_agent.billing.stripe_billing import (
    create_billing_portal_session,
    create_checkout_session,
    get_plan_pricing,
    handle_webhook,
)
from climate_agent.db import (
    ApiKey,
    PlanTier,
    PLAN_LIMITS,
    User,
    UsageRecord,
    WaitlistEntry,
    create_database,
    get_monthly_usage,
)
from climate_agent.tools.registry import TOOL_DEFINITIONS, dispatch_tool

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

DESCRIPTION = """
# Climate Insight API

Autonomous AI-powered climate analysis and tropical cyclone forecasting.

## What you can do

- **Tropical cyclone tracking**: Fetch active storms, historical data, track forecasts
- **Climate indicators**: CO2, temperature anomaly, sea level, Arctic ice
- **Forecasting**: Cyclone intensity & track, multi-hazard risk assessment
- **Analysis**: Carbon budgets, climate trends, potential intensity
- **Visualization**: Storm track maps, SST maps, time series plots

## Authentication

Include your API key in the `X-API-Key` header:

```
X-API-Key: clm_your_api_key_here
```

Or use a JWT Bearer token:

```
Authorization: Bearer <token>
```
"""

app = FastAPI(
    title="Climate Insight API",
    description=DESCRIPTION,
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    contact={
        "name": "Climate Insight",
        "email": "api@climateinsight.ai",
    },
    license_info={"name": "Commercial", "url": "https://climateinsight.ai/terms"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///climate_agent.db")
SessionFactory = create_database(DATABASE_URL)

# Templates and static files
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "..", "web", "templates")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "web", "static")

templates = Jinja2Templates(directory=TEMPLATE_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

def get_db():
    """Dependency: get a database session."""
    db = SessionFactory()
    try:
        yield db
    finally:
        db.close()


async def get_current_user(
    x_api_key: str | None = Header(None),
    authorization: str | None = Header(None),
    db: Session = Depends(get_db),
) -> User:
    """Dependency: authenticate and return the current user."""
    user = None

    if x_api_key:
        user = authenticate_api_key(db, x_api_key)
    elif authorization and authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ")
        user = authenticate_jwt(db, token)

    if user is None:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "unauthorized",
                "message": "Invalid or missing API key. Get yours at /api/v1/auth/register",
            },
        )
    return user


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    email: str = Field(..., description="Email address")
    password: str = Field(..., min_length=8, description="Password (min 8 characters)")
    name: str | None = Field(None, description="Full name")
    company: str | None = Field(None, description="Company or organization")

class LoginRequest(BaseModel):
    email: str
    password: str

class AgentQueryRequest(BaseModel):
    query: str = Field(..., description="Natural language climate question or task")
    max_turns: int = Field(10, ge=1, le=50, description="Max agent reasoning turns")

class ToolCallRequest(BaseModel):
    tool: str = Field(..., description="Tool name")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Tool parameters")

class CheckoutRequest(BaseModel):
    plan: str = Field(..., description="Plan tier: starter, professional")
    billing_period: str = Field("monthly", description="monthly or yearly")

class WaitlistRequest(BaseModel):
    email: str
    company: str | None = None
    use_case: str | None = None


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------

@app.post("/api/v1/auth/register", tags=["Auth"])
def api_register(body: RegisterRequest, db: Session = Depends(get_db)):
    """Register a new account and receive an API key."""
    try:
        user, raw_key = register_user(db, body.email, body.password, body.name, body.company)
    except ValueError as e:
        raise HTTPException(status_code=409, detail={"error": "conflict", "message": str(e)})

    token = create_access_token(user.id, user.email, user.plan)

    return {
        "user": {"id": user.id, "email": user.email, "plan": user.plan},
        "api_key": raw_key,
        "access_token": token,
        "message": "Welcome! Save your API key - it won't be shown again.",
    }


@app.post("/api/v1/auth/login", tags=["Auth"])
def api_login(body: LoginRequest, db: Session = Depends(get_db)):
    """Log in and receive a JWT access token."""
    user = db.query(User).filter(User.email == body.email, User.is_active.is_(True)).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail={"error": "Invalid credentials"})

    token = create_access_token(user.id, user.email, user.plan)
    return {"access_token": token, "token_type": "bearer", "plan": user.plan}


# ---------------------------------------------------------------------------
# Agent endpoint - the core product
# ---------------------------------------------------------------------------

@app.post("/api/v1/agent/query", tags=["Agent"])
async def agent_query(
    body: AgentQueryRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Send a natural language query to the autonomous climate agent.

    The agent will autonomously select and call tools to answer your question.
    This is the primary endpoint for interacting with Climate Insight.

    **Examples:**
    - "What are the current global climate indicators?"
    - "Analyze Hurricane Ian's track and intensity"
    - "Assess climate risk for Miami under SSP5-8.5 by 2100"
    - "What is the remaining carbon budget for 1.5C?"
    - "Forecast intensity for a Category 2 storm at 20N 60W"
    """
    limits = PLAN_LIMITS[PlanTier(user.plan)]
    max_turns = min(body.max_turns, limits["agent_turns_per_request"])

    # Check monthly usage
    monthly = get_monthly_usage(db, user.id)
    if monthly >= limits["requests_per_month"]:
        raise HTTPException(status_code=429, detail={
            "error": "monthly_limit_exceeded",
            "message": f"You've used {monthly}/{limits['requests_per_month']} requests this month.",
            "upgrade_url": "/pricing",
        })

    start_time = time.time()

    try:
        from climate_agent.agent import ClimateAgent

        agent = ClimateAgent(max_turns=max_turns, verbose=False)
        response = agent.run(body.query)
        stats = agent.get_session_stats()

        latency_ms = int((time.time() - start_time) * 1000)

        # Record usage
        record = UsageRecord(
            user_id=user.id,
            endpoint="/api/v1/agent/query",
            tool_calls=stats["total_tool_calls"],
            agent_turns=stats["total_messages"],
            latency_ms=latency_ms,
            status_code=200,
        )
        db.add(record)
        db.commit()

        return {
            "response": response,
            "metadata": {
                "tool_calls": stats["total_tool_calls"],
                "agent_turns": stats["total_messages"],
                "latency_ms": latency_ms,
                "model": stats["model"],
            },
        }
    except EnvironmentError:
        raise HTTPException(status_code=503, detail={
            "error": "service_unavailable",
            "message": "AI backend not configured. Contact support.",
        })
    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        record = UsageRecord(
            user_id=user.id,
            endpoint="/api/v1/agent/query",
            latency_ms=latency_ms,
            status_code=500,
            error_message=str(e),
        )
        db.add(record)
        db.commit()
        raise HTTPException(status_code=500, detail={
            "error": "internal_error",
            "message": "An error occurred processing your request.",
        })


# ---------------------------------------------------------------------------
# Direct tool endpoints
# ---------------------------------------------------------------------------

@app.post("/api/v1/tools/call", tags=["Tools"])
def tool_call(
    body: ToolCallRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Call a specific climate tool directly.

    For users who want fine-grained control over which tools to invoke,
    bypassing the autonomous agent.
    """
    # Check tool access based on plan
    limits = PLAN_LIMITS[PlanTier(user.plan)]
    allowed_tools = limits.get("tools", [])

    if isinstance(allowed_tools, list) and body.tool not in allowed_tools:
        raise HTTPException(status_code=403, detail={
            "error": "tool_not_available",
            "message": f"Tool '{body.tool}' is not available on the {limits['name']} plan.",
            "upgrade_url": "/pricing",
        })

    start_time = time.time()
    result_json = dispatch_tool(body.tool, body.parameters)
    latency_ms = int((time.time() - start_time) * 1000)

    result = json.loads(result_json)

    record = UsageRecord(
        user_id=user.id,
        endpoint=f"/api/v1/tools/{body.tool}",
        tool_calls=1,
        latency_ms=latency_ms,
        status_code=200 if "error" not in result else 400,
    )
    db.add(record)
    db.commit()

    if "error" in result:
        raise HTTPException(status_code=400, detail=result)

    return {"result": result, "tool": body.tool, "latency_ms": latency_ms}


@app.get("/api/v1/tools", tags=["Tools"])
def list_tools(user: User = Depends(get_current_user)):
    """List all available climate tools and their schemas."""
    limits = PLAN_LIMITS[PlanTier(user.plan)]
    allowed = limits.get("tools", [])

    tools = []
    for tool in TOOL_DEFINITIONS:
        available = True
        if isinstance(allowed, list):
            available = tool["name"] in allowed
        tools.append({
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["input_schema"],
            "available_on_your_plan": available,
        })
    return {"tools": tools, "total": len(tools), "plan": limits["name"]}


# ---------------------------------------------------------------------------
# Billing endpoints
# ---------------------------------------------------------------------------

@app.get("/api/v1/pricing", tags=["Billing"])
def get_pricing():
    """Get pricing information for all plans."""
    return {"plans": get_plan_pricing()}


@app.post("/api/v1/billing/checkout", tags=["Billing"])
def create_checkout(
    body: CheckoutRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a Stripe checkout session to upgrade your plan."""
    try:
        plan = PlanTier(body.plan)
    except ValueError:
        raise HTTPException(status_code=400, detail={"error": f"Invalid plan: {body.plan}"})

    try:
        result = create_checkout_session(db, user, plan, body.billing_period)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail={"error": str(e)})


@app.post("/api/v1/billing/portal", tags=["Billing"])
def billing_portal(user: User = Depends(get_current_user)):
    """Get a Stripe billing portal URL to manage your subscription."""
    try:
        return create_billing_portal_session(user)
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": str(e)})


@app.post("/api/v1/billing/webhook", tags=["Billing"], include_in_schema=False)
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    """Process Stripe webhook events."""
    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    try:
        result = handle_webhook(payload, sig, db)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": str(e)})


# ---------------------------------------------------------------------------
# Account & usage endpoints
# ---------------------------------------------------------------------------

@app.get("/api/v1/account", tags=["Account"])
def get_account(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get your account details and current usage."""
    limits = PLAN_LIMITS[PlanTier(user.plan)]
    monthly_usage = get_monthly_usage(db, user.id)

    return {
        "user": {
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "company": user.company,
            "plan": user.plan,
            "created_at": user.created_at.isoformat() if user.created_at else None,
        },
        "plan_details": {
            "name": limits["name"],
            "requests_per_month": limits["requests_per_month"],
            "requests_per_minute": limits["requests_per_minute"],
            "agent_turns_per_request": limits["agent_turns_per_request"],
        },
        "usage_this_month": {
            "requests": monthly_usage,
            "limit": limits["requests_per_month"],
            "pct_used": round(monthly_usage / limits["requests_per_month"] * 100, 1),
        },
    }


@app.post("/api/v1/account/api-keys", tags=["Account"])
def create_api_key(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Generate a new API key."""
    raw_key, key_hash = ApiKey.generate()
    api_key = ApiKey(
        user_id=user.id,
        key_hash=key_hash,
        key_prefix=raw_key[:12],
    )
    db.add(api_key)
    db.commit()
    return {
        "api_key": raw_key,
        "prefix": raw_key[:12],
        "message": "Save this key - it won't be shown again.",
    }


# ---------------------------------------------------------------------------
# Waitlist (for enterprise / pre-launch)
# ---------------------------------------------------------------------------

@app.post("/api/v1/waitlist", tags=["Waitlist"])
def join_waitlist(body: WaitlistRequest, db: Session = Depends(get_db)):
    """Join the waitlist for early access or enterprise plan."""
    existing = db.query(WaitlistEntry).filter(WaitlistEntry.email == body.email).first()
    if existing:
        return {"message": "You're already on the waitlist!", "position": "confirmed"}

    entry = WaitlistEntry(email=body.email, company=body.company, use_case=body.use_case)
    db.add(entry)
    db.commit()

    count = db.query(WaitlistEntry).count()
    return {"message": "You've been added to the waitlist!", "position": count}


# ---------------------------------------------------------------------------
# Landing page
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def landing_page(request: Request):
    """Serve the marketing landing page."""
    plans = get_plan_pricing()
    return templates.TemplateResponse("landing.html", {"request": request, "plans": plans})


@app.get("/pricing", response_class=HTMLResponse, include_in_schema=False)
async def pricing_page(request: Request):
    """Serve the pricing page."""
    plans = get_plan_pricing()
    return templates.TemplateResponse("landing.html", {"request": request, "plans": plans, "scroll_to": "pricing"})


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Climate Insight API", "version": "1.0.0"}
