"""
Pipeline that connects real-time government data sources to HuggingFace
models / spaces from monkseal555.

The pipeline supports three main workflows:

1. **Live inference** – fetch current storm data → format → send to HF model
2. **Historical replay** – load HURDAT2 / IBTrACS records → send to HF model
3. **Monitor mode** – poll for new data at an interval, run inference on updates
"""

from __future__ import annotations

import io
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable

from . import config
from . import government_data as gov
from . import huggingface_client as hf

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data formatters – transform raw government data into model-ready payloads
# ---------------------------------------------------------------------------

def _format_storm_for_inference(storm: dict) -> dict:
    """Convert an active-storm dict to a model inference payload."""
    return {
        "inputs": {
            "latitude": storm.get("lat"),
            "longitude": storm.get("lon"),
            "intensity_kt": storm.get("intensity_kt"),
            "pressure_mb": storm.get("pressure_mb"),
            "movement_dir": storm.get("movement_dir"),
            "movement_speed_mph": storm.get("movement_speed_mph"),
            "classification": storm.get("classification"),
            "name": storm.get("name"),
            "basin": storm.get("basin"),
        }
    }


def _format_atcf_track_for_inference(records: list[dict]) -> dict:
    """Convert ATCF best-track records into a sequence payload."""
    track_points = []
    for rec in records:
        track_points.append({
            "datetime": rec.get("datetime"),
            "lat": rec.get("lat"),
            "lon": rec.get("lon"),
            "vmax_kt": rec.get("vmax_kt"),
            "mslp_mb": rec.get("mslp_mb"),
        })
    return {"inputs": {"track": track_points}}


def _format_geojson_for_inference(geojson: dict) -> dict:
    """Wrap GeoJSON forecast data as model input."""
    features = geojson.get("features", [])
    points = []
    for feat in features:
        props = feat.get("properties", {})
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [None, None])
        points.append({
            "lon": coords[0] if coords else None,
            "lat": coords[1] if len(coords) > 1 else None,
            "storm_name": props.get("STORMNAME"),
            "storm_type": props.get("STORMTYPE"),
            "max_wind_kt": props.get("MAXWIND"),
            "gust_kt": props.get("GUST"),
            "forecast_hour": props.get("TAU"),
            "valid_time": props.get("VALIDTIME"),
        })
    return {"inputs": {"forecast_points": points}}


def _format_ibtracs_for_inference(rows: list[dict]) -> dict:
    """Convert IBTrACS CSV rows to model input."""
    track = []
    for row in rows:
        track.append({
            "sid": row.get("SID"),
            "name": row.get("NAME"),
            "iso_time": row.get("ISO_TIME"),
            "lat": _safe_float(row.get("LAT")),
            "lon": _safe_float(row.get("LON")),
            "wmo_wind_kt": _safe_float(row.get("WMO_WIND")),
            "wmo_pres_mb": _safe_float(row.get("WMO_PRES")),
            "nature": row.get("NATURE"),
            "basin": row.get("BASIN"),
        })
    return {"inputs": {"track": track}}


def _safe_float(val: Any) -> float | None:
    if val is None or val == "":
        return None
    try:
        f = float(val)
        return None if f == -999 else f
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Core pipeline functions
# ---------------------------------------------------------------------------

def run_live_inference(
    model_id: str = "weatherflow",
    basin: str = "AL",
    token: str | None = None,
) -> dict:
    """
    Fetch current active storms and run inference on each through a
    HuggingFace model.

    Parameters
    ----------
    model_id : str
        Model short alias or full HF model ID.
    basin : str
        Basin filter (``"AL"``, ``"EP"``, etc.). ``None`` for all basins.
    token : str, optional
        HuggingFace API token.

    Returns
    -------
    dict
        ``{"storms": [...], "predictions": [...], "timestamp": ...}``
    """
    storms = gov.list_active_storms()
    if basin:
        storms = [s for s in storms if (s.get("basin") or "").upper().startswith(basin.upper())]

    predictions = []
    for storm in storms:
        payload = _format_storm_for_inference(storm)
        try:
            result = hf.infer(model_id, payload, token=token)
            predictions.append({
                "storm": storm.get("name") or storm.get("id"),
                "input": payload,
                "output": result,
            })
        except Exception as exc:
            log.error("Inference failed for %s: %s", storm.get("name"), exc)
            predictions.append({
                "storm": storm.get("name") or storm.get("id"),
                "input": payload,
                "error": str(exc),
            })

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model_id,
        "basin": basin,
        "storms": storms,
        "predictions": predictions,
    }


def run_forecast_inference(
    model_id: str = "weatherflow",
    basin: str = "AL",
    token: str | None = None,
) -> dict:
    """
    Fetch NHC ArcGIS forecast track and run it through a HuggingFace model.
    """
    geojson = gov.fetch_forecast_track(basin)
    payload = _format_geojson_for_inference(geojson)

    try:
        result = hf.infer(model_id, payload, token=token)
    except Exception as exc:
        result = {"error": str(exc)}

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model_id,
        "basin": basin,
        "source": "nhc_arcgis_forecast_track",
        "input_features": len(geojson.get("features", [])),
        "prediction": result,
    }


def run_atcf_inference(
    storm_id: str,
    model_id: str = "weatherflow",
    token: str | None = None,
) -> dict:
    """
    Fetch ATCF best-track data for a specific storm and run inference.

    Parameters
    ----------
    storm_id : str
        ATCF storm identifier, e.g. ``"al052024"``.
    """
    records = gov.fetch_atcf_best_track(storm_id)
    payload = _format_atcf_track_for_inference(records)

    try:
        result = hf.infer(model_id, payload, token=token)
    except Exception as exc:
        result = {"error": str(exc)}

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model_id,
        "storm_id": storm_id,
        "source": "nhc_atcf_best_track",
        "track_points": len(records),
        "prediction": result,
    }


def run_ibtracs_inference(
    model_id: str = "weatherflow",
    dataset: str = "last3years",
    storm_name: str | None = None,
    max_rows: int = 500,
    token: str | None = None,
) -> dict:
    """
    Fetch IBTrACS data and run inference. Optionally filter by storm name.
    """
    rows = gov.fetch_ibtracs(dataset=dataset, max_rows=max_rows if not storm_name else None)
    if storm_name:
        rows = [r for r in rows if storm_name.upper() in (r.get("NAME") or "").upper()]
        if max_rows:
            rows = rows[:max_rows]

    payload = _format_ibtracs_for_inference(rows)

    try:
        result = hf.infer(model_id, payload, token=token)
    except Exception as exc:
        result = {"error": str(exc)}

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model_id,
        "source": f"ibtracs_{dataset}",
        "filter": storm_name,
        "track_points": len(rows),
        "prediction": result,
    }


def run_space_with_live_data(
    space_id: str = "testing",
    basin: str = "AL",
    token: str | None = None,
) -> dict:
    """
    Fetch live data and send it to a Gradio Space via its API.
    """
    snapshot = gov.get_realtime_snapshot(basin)
    payload_json = json.dumps(snapshot, default=str)

    try:
        result = hf.call_space_api(space_id, data=[payload_json], token=token)
    except Exception as exc:
        log.warning("Direct API call failed, trying gradio_client: %s", exc)
        try:
            result = hf.call_space_gradio_client(space_id, payload_json, token=token)
        except Exception as exc2:
            result = {"error": str(exc2)}

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "space": space_id,
        "basin": basin,
        "active_storms": len(snapshot.get("active_storms", [])),
        "result": result,
    }


# ---------------------------------------------------------------------------
# Monitor mode – poll and run inference periodically
# ---------------------------------------------------------------------------

def monitor(
    model_id: str = "weatherflow",
    basin: str = "AL",
    interval_seconds: int = 300,
    callback: Callable[[dict], None] | None = None,
    max_iterations: int | None = None,
    token: str | None = None,
) -> None:
    """
    Continuously poll government data and run inference at *interval_seconds*.

    Parameters
    ----------
    callback : callable, optional
        Called with each result dict. If None, results are printed as JSON.
    max_iterations : int, optional
        Stop after this many iterations (None = run forever).
    """
    iteration = 0
    log.info(
        "Starting monitor: model=%s basin=%s interval=%ds",
        model_id, basin, interval_seconds,
    )
    while True:
        iteration += 1
        log.info("Monitor iteration %d", iteration)
        try:
            result = run_live_inference(model_id=model_id, basin=basin, token=token)
            if callback:
                callback(result)
            else:
                print(json.dumps(result, indent=2, default=str))
        except Exception as exc:
            log.error("Monitor iteration %d failed: %s", iteration, exc)

        if max_iterations and iteration >= max_iterations:
            log.info("Reached max iterations (%d), stopping.", max_iterations)
            break

        time.sleep(interval_seconds)


# ---------------------------------------------------------------------------
# Full snapshot with predictions
# ---------------------------------------------------------------------------

def full_analysis(
    model_id: str = "weatherflow",
    basin: str = "AL",
    token: str | None = None,
) -> dict:
    """
    Run a comprehensive analysis: fetch all available real-time data,
    run inference, and return a combined report.
    """
    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model_id,
        "basin": basin,
    }

    # 1. Real-time snapshot
    report["realtime_snapshot"] = gov.get_realtime_snapshot(basin)

    # 2. Live inference on active storms
    try:
        report["live_inference"] = run_live_inference(model_id, basin, token=token)
    except Exception as exc:
        report["live_inference"] = {"error": str(exc)}

    # 3. Forecast track inference
    try:
        report["forecast_inference"] = run_forecast_inference(model_id, basin, token=token)
    except Exception as exc:
        report["forecast_inference"] = {"error": str(exc)}

    # 4. Available ATCF files
    try:
        report["available_atcf_files"] = gov.list_atcf_best_track_files()[:20]
    except Exception as exc:
        report["available_atcf_files"] = {"error": str(exc)}

    # 5. Recent IBTrACS storms
    try:
        ibtracs = gov.fetch_ibtracs("last3years", max_rows=10)
        report["recent_ibtracs_sample"] = ibtracs
    except Exception as exc:
        report["recent_ibtracs_sample"] = {"error": str(exc)}

    # 6. HuggingFace resources
    try:
        report["hf_resources"] = hf.discover_all()
    except Exception as exc:
        report["hf_resources"] = {"error": str(exc)}

    return report
