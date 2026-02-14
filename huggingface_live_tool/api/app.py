"""
CycloneAPI — FastAPI web service.

Production tropical cyclone intelligence API with:
- Real-time government data (NHC, ATCF, IBTrACS, HURDAT2)
- HuggingFace model inference (monkseal555)
- API key authentication and tier-based rate limiting
- Usage tracking and billing integration
"""

from __future__ import annotations

import os
import time
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles

from ..billing.subscriptions import TIER_CONFIGS, Tier, get_store
from .. import government_data as gov
from .. import huggingface_client as hf
from .. import pipeline

DESCRIPTION = """
# CycloneAPI

Real-time tropical cyclone intelligence powered by government data and ML models.

## Data Sources
- **NHC** — Active storms, forecast tracks, cones, wind radii, advisories
- **ATCF** — Best-track and forecast aid data
- **IBTrACS** — Global best-track archive (updated 3x/week)
- **HURDAT2** — Historical Atlantic & East Pacific database (1851–present)

## ML Models
- **WeatherFlow** — Tropical cyclone track and intensity forecasting
- **AutoTrain Hurricane** — Transformer-based hurricane prediction

## Authentication
All endpoints require an API key via the `X-API-Key` header.
Get a free key at [cycloneapi.com](/).
"""

app = FastAPI(
    title="CycloneAPI",
    version="1.0.0",
    description=DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static assets for landing page
_static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
store = get_store()


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

def require_key(api_key: str | None = Security(api_key_header)) -> str:
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    rec = store.validate(api_key)
    if rec is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    allowed, reason = store.record_request(api_key)
    if not allowed:
        raise HTTPException(status_code=429, detail=reason)
    return api_key


def require_feature(feature: str):
    """Return a dependency that checks the caller's tier includes *feature*."""
    def _check(api_key: str = Depends(require_key)) -> str:
        cfg = store.get_tier_config(api_key)
        if cfg is None or feature not in cfg.features:
            raise HTTPException(
                status_code=403,
                detail=f"Feature '{feature}' requires a higher subscription tier",
            )
        return api_key
    return _check


# ---------------------------------------------------------------------------
# Landing page
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def landing_page():
    tpl_path = os.path.join(os.path.dirname(__file__), "..", "templates", "landing.html")
    if os.path.exists(tpl_path):
        with open(tpl_path) as f:
            return f.read()
    return "<h1>CycloneAPI</h1><p>Visit <a href='/docs'>/docs</a> for API documentation.</p>"


# ---------------------------------------------------------------------------
# Account & billing
# ---------------------------------------------------------------------------

@app.post("/v1/auth/register", tags=["Account"])
async def register(email: str, tier: Tier = Tier.FREE):
    """Register for an API key. Returns the key (shown only once)."""
    rec = store.create_key(email, tier)
    return {
        "api_key": rec.key,
        "tier": rec.tier.value,
        "email": rec.owner_email,
        "message": "Store this key securely — it will not be shown again.",
    }


@app.get("/v1/account/usage", tags=["Account"])
async def usage(api_key: str = Depends(require_key)):
    """View your current billing period usage."""
    return store.get_usage(api_key)


@app.get("/v1/account/tier", tags=["Account"])
async def tier_info(api_key: str = Depends(require_key)):
    """View your subscription tier details."""
    rec = store.validate(api_key)
    cfg = TIER_CONFIGS[rec.tier]
    return {
        "tier": rec.tier.value,
        "name": cfg.name,
        "price_monthly": f"${cfg.price_monthly_cents / 100:.2f}" if cfg.price_monthly_cents else "Custom",
        "price_yearly": f"${cfg.price_yearly_cents / 100:.2f}" if cfg.price_yearly_cents else "Custom",
        "requests_per_month": cfg.requests_per_month,
        "rate_limit_per_minute": cfg.requests_per_minute,
        "data_sources": cfg.data_sources,
        "features": cfg.features,
        "model_access": cfg.model_access,
        "support": cfg.support,
        "concurrent_monitors": cfg.concurrent_monitors,
        "history_days": cfg.history_days,
        "webhook_support": cfg.webhook_support,
        "sla_uptime": cfg.sla_uptime,
    }


@app.get("/v1/pricing", tags=["Account"])
async def pricing():
    """View all available subscription tiers and pricing."""
    tiers = []
    for tier, cfg in TIER_CONFIGS.items():
        tiers.append({
            "tier": tier.value,
            "name": cfg.name,
            "price_monthly": f"${cfg.price_monthly_cents / 100:.2f}" if cfg.price_monthly_cents else "Custom",
            "price_yearly": f"${cfg.price_yearly_cents / 100:.2f}" if cfg.price_yearly_cents else "Custom",
            "requests_per_month": cfg.requests_per_month,
            "rate_limit_per_minute": cfg.requests_per_minute,
            "features": cfg.features,
            "model_access": cfg.model_access,
            "support": cfg.support,
            "sla_uptime": cfg.sla_uptime,
        })
    return {"tiers": tiers}


# ---------------------------------------------------------------------------
# Real-time data endpoints
# ---------------------------------------------------------------------------

@app.get("/v1/storms", tags=["Real-Time Data"])
async def active_storms(api_key: str = Depends(require_feature("active_storms"))):
    """List all currently active tropical cyclones (from NHC)."""
    return {"storms": gov.list_active_storms()}


@app.get("/v1/storms/raw", tags=["Real-Time Data"])
async def active_storms_raw(api_key: str = Depends(require_feature("active_storms"))):
    """Raw NHC CurrentStorms.json response."""
    return gov.fetch_current_storms()


@app.get("/v1/snapshot", tags=["Real-Time Data"])
async def snapshot(basin: str = "AL", api_key: str = Depends(require_feature("active_storms"))):
    """Combined real-time snapshot: active storms, forecast, advisories."""
    return gov.get_realtime_snapshot(basin)


@app.get("/v1/forecast/track", tags=["Real-Time Data"])
async def forecast_track(basin: str = "AL", api_key: str = Depends(require_feature("forecast_track"))):
    """NHC forecast track as GeoJSON."""
    return gov.fetch_forecast_track(basin)


@app.get("/v1/forecast/cone", tags=["Real-Time Data"])
async def forecast_cone(basin: str = "AL", api_key: str = Depends(require_feature("forecast_cone"))):
    """NHC forecast cone as GeoJSON."""
    return gov.fetch_forecast_cone(basin)


@app.get("/v1/forecast/wind-radii", tags=["Real-Time Data"])
async def wind_radii(
    basin: str = "AL",
    threshold_kt: int = 34,
    api_key: str = Depends(require_feature("wind_radii")),
):
    """Wind radii polygons as GeoJSON (34, 50, or 64 kt)."""
    return gov.fetch_wind_radii(basin, threshold_kt)


@app.get("/v1/advisories", tags=["Real-Time Data"])
async def advisories(basin: str = "AL", api_key: str = Depends(require_feature("advisories"))):
    """NHC RSS advisories for a basin."""
    return {"advisories": gov.fetch_nhc_rss(basin)}


# ---------------------------------------------------------------------------
# ATCF best-track data
# ---------------------------------------------------------------------------

@app.get("/v1/atcf/files", tags=["ATCF"])
async def atcf_files(api_key: str = Depends(require_feature("atcf_best_track"))):
    """List available ATCF best-track files."""
    return {"files": gov.list_atcf_best_track_files()}


@app.get("/v1/atcf/track/{storm_id}", tags=["ATCF"])
async def atcf_track(storm_id: str, api_key: str = Depends(require_feature("atcf_best_track"))):
    """Fetch ATCF best-track records for a specific storm."""
    return {"storm_id": storm_id, "track": gov.fetch_atcf_best_track(storm_id)}


# ---------------------------------------------------------------------------
# IBTrACS
# ---------------------------------------------------------------------------

@app.get("/v1/ibtracs", tags=["IBTrACS"])
async def ibtracs(
    dataset: str = "last3years",
    storm: str | None = None,
    limit: int = 100,
    api_key: str = Depends(require_feature("ibtracs")),
):
    """Fetch IBTrACS global best-track data."""
    rows = gov.fetch_ibtracs(dataset=dataset, max_rows=limit if not storm else None)
    if storm:
        rows = [r for r in rows if storm.upper() in (r.get("NAME") or "").upper()][:limit]
    return {"dataset": dataset, "filter": storm, "count": len(rows), "records": rows}


# ---------------------------------------------------------------------------
# HURDAT2
# ---------------------------------------------------------------------------

@app.get("/v1/hurdat2", tags=["HURDAT2"])
async def hurdat2(
    basin: str = "AL",
    storm: str | None = None,
    limit: int = 10,
    api_key: str = Depends(require_feature("hurdat2")),
):
    """Fetch HURDAT2 historical hurricane database."""
    storms = gov.fetch_hurdat2(basin)
    if storm:
        storms = [s for s in storms if storm.upper() in s.get("name", "").upper()]
    return {"basin": basin, "filter": storm, "count": len(storms), "storms": storms[-limit:]}


# ---------------------------------------------------------------------------
# HuggingFace integration
# ---------------------------------------------------------------------------

@app.get("/v1/models", tags=["ML Models"])
async def list_models(api_key: str = Depends(require_key)):
    """List available HuggingFace models and spaces."""
    return hf.discover_all()


@app.get("/v1/models/{model_id}/info", tags=["ML Models"])
async def model_info(model_id: str, api_key: str = Depends(require_key)):
    """Get metadata for a specific model."""
    from ..config import HF_MODELS
    full_id = HF_MODELS.get(model_id, model_id)
    return hf.get_model_info(full_id)


@app.post("/v1/inference/live", tags=["ML Inference"])
async def inference_live(
    model: str = "weatherflow",
    basin: str = "AL",
    api_key: str = Depends(require_feature("hf_inference")),
):
    """Fetch current active storms and run ML inference on each."""
    return pipeline.run_live_inference(model_id=model, basin=basin)


@app.post("/v1/inference/forecast", tags=["ML Inference"])
async def inference_forecast(
    model: str = "weatherflow",
    basin: str = "AL",
    api_key: str = Depends(require_feature("hf_inference")),
):
    """Run ML inference on the current NHC forecast track."""
    return pipeline.run_forecast_inference(model_id=model, basin=basin)


@app.post("/v1/inference/atcf/{storm_id}", tags=["ML Inference"])
async def inference_atcf(
    storm_id: str,
    model: str = "weatherflow",
    api_key: str = Depends(require_feature("hf_inference")),
):
    """Run ML inference on ATCF best-track data for a specific storm."""
    return pipeline.run_atcf_inference(storm_id=storm_id, model_id=model)


@app.post("/v1/inference/ibtracs", tags=["ML Inference"])
async def inference_ibtracs(
    model: str = "weatherflow",
    dataset: str = "last3years",
    storm: str | None = None,
    limit: int = 500,
    api_key: str = Depends(require_feature("hf_inference")),
):
    """Run ML inference on IBTrACS data."""
    return pipeline.run_ibtracs_inference(
        model_id=model, dataset=dataset, storm_name=storm, max_rows=limit,
    )


@app.post("/v1/analysis/full", tags=["ML Inference"])
async def full_analysis(
    model: str = "weatherflow",
    basin: str = "AL",
    api_key: str = Depends(require_feature("full_analysis")),
):
    """Comprehensive analysis: all data sources + ML predictions."""
    return pipeline.full_analysis(model_id=model, basin=basin)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"], include_in_schema=False)
async def health():
    return {"status": "ok", "service": "CycloneAPI", "version": "1.0.0"}
