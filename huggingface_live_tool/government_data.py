"""
Fetchers for real-time tropical cyclone data from US government sources.

Supported sources
-----------------
* NHC CurrentStorms.json          – active storm metadata
* NHC ATCF B-deck / A-deck       – best-track and forecast aid data
* NHC ArcGIS REST MapServer       – GeoJSON forecast layers
* NHC RSS feeds                   – advisory text
* IBTrACS CSV                     – global best-track archive (updated ~3×/week)
* HURDAT2                         – historical Atlantic / E-Pac database
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any

import requests

from . import config

log = logging.getLogger(__name__)

_SESSION: requests.Session | None = None


def _get_session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        _SESSION.headers.update({"User-Agent": config.USER_AGENT})
    return _SESSION


def _get(url: str, timeout: int = config.DEFAULT_TIMEOUT, **kwargs: Any) -> requests.Response:
    resp = _get_session().get(url, timeout=timeout, **kwargs)
    resp.raise_for_status()
    return resp


# ---- NHC CurrentStorms.json ------------------------------------------------

def fetch_current_storms() -> dict:
    """Return the NHC CurrentStorms JSON (active cyclones across all basins)."""
    resp = _get(config.NHC_CURRENT_STORMS_URL)
    return resp.json()


def list_active_storms() -> list[dict]:
    """Return a simplified list of active storms with key fields."""
    data = fetch_current_storms()
    storms = []
    for entry in data.get("activeStorms", []):
        storms.append({
            "id": entry.get("id"),
            "basin": entry.get("basin"),
            "name": entry.get("name"),
            "classification": entry.get("classification"),
            "lat": entry.get("lat"),
            "lon": entry.get("lon"),
            "intensity_kt": entry.get("intensity"),
            "pressure_mb": entry.get("pressure"),
            "movement_dir": entry.get("movementDir"),
            "movement_speed_mph": entry.get("movementSpeed"),
            "last_update": entry.get("lastUpdate"),
            "advisory_url": entry.get("url"),
        })
    return storms


# ---- NHC ATCF (B-deck best track) -----------------------------------------

def _parse_atcf_line(line: str) -> dict | None:
    """Parse a single ATCF ab-deck format line into a dict."""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 8:
        return None
    try:
        lat_str = parts[6]
        lon_str = parts[7]
        lat = float(lat_str[:-1]) / 10.0 * (1 if lat_str[-1] == "N" else -1)
        lon = float(lon_str[:-1]) / 10.0 * (-1 if lon_str[-1] == "W" else 1)
    except (ValueError, IndexError):
        lat, lon = None, None
    try:
        vmax = int(parts[8]) if len(parts) > 8 and parts[8] else None
    except ValueError:
        vmax = None
    try:
        mslp = int(parts[9]) if len(parts) > 9 and parts[9] else None
    except ValueError:
        mslp = None
    return {
        "basin": parts[0],
        "cyclone_num": parts[1],
        "datetime": parts[2],
        "tech": parts[4] if len(parts) > 4 else "",
        "tau": int(parts[5]) if len(parts) > 5 and parts[5].strip().isdigit() else 0,
        "lat": lat,
        "lon": lon,
        "vmax_kt": vmax,
        "mslp_mb": mslp,
    }


def fetch_atcf_best_track(storm_id: str) -> list[dict]:
    """
    Download and parse the ATCF B-deck file for *storm_id*.

    Parameters
    ----------
    storm_id : str
        E.g. ``"bal052024"`` for Atlantic storm 5 of 2024, or the filename
        without extension (``"bal052024"``).
    """
    storm_id = storm_id.lower()
    if not storm_id.startswith("b"):
        storm_id = f"b{storm_id}"
    url = f"{config.NHC_ATCF_BEST_TRACK_URL}{storm_id}.dat"
    log.info("Fetching ATCF best track: %s", url)
    resp = _get(url)
    records = []
    for line in resp.text.splitlines():
        rec = _parse_atcf_line(line)
        if rec:
            records.append(rec)
    return records


def list_atcf_best_track_files() -> list[str]:
    """Scrape the ATCF B-deck directory listing for available files."""
    resp = _get(config.NHC_ATCF_BEST_TRACK_URL)
    return re.findall(r'href="(b\w+\.dat)"', resp.text, re.IGNORECASE)


# ---- NHC ArcGIS REST (GeoJSON) --------------------------------------------

def fetch_arcgis_layer(
    base_url: str = config.ARCGIS_ATL_ACTIVE,
    layer: int = 0,
    where: str = "1=1",
    out_fields: str = "*",
    result_format: str = "geojson",
) -> dict:
    """
    Query an NHC ArcGIS MapServer layer and return GeoJSON.

    Parameters
    ----------
    base_url : str
        One of ``config.ARCGIS_ATL_ACTIVE``, ``config.ARCGIS_EPAC_ACTIVE``,
        or ``config.ARCGIS_TROPICAL_WEATHER``.
    layer : int
        Layer index (see ``config.ARCGIS_LAYERS``).
    where : str
        SQL-like where clause (default selects all features).
    out_fields : str
        Comma-separated field names or ``"*"`` for all.
    result_format : str
        ``"geojson"`` or ``"json"``.
    """
    url = f"{base_url}/{layer}/query"
    params = {
        "where": where,
        "outFields": out_fields,
        "f": result_format,
        "returnGeometry": "true",
    }
    log.info("ArcGIS query: %s  layer=%d", base_url, layer)
    resp = _get(url, params=params)
    return resp.json()


def fetch_forecast_track(basin: str = "AL") -> dict:
    """Shortcut: get the forecast track GeoJSON for a basin (AL or EP)."""
    base = config.ARCGIS_ATL_ACTIVE if basin.upper() == "AL" else config.ARCGIS_EPAC_ACTIVE
    return fetch_arcgis_layer(base, layer=config.ARCGIS_LAYERS["forecast_track"])


def fetch_forecast_cone(basin: str = "AL") -> dict:
    """Shortcut: get the forecast cone GeoJSON for a basin."""
    base = config.ARCGIS_ATL_ACTIVE if basin.upper() == "AL" else config.ARCGIS_EPAC_ACTIVE
    return fetch_arcgis_layer(base, layer=config.ARCGIS_LAYERS["forecast_cone"])


def fetch_wind_radii(basin: str = "AL", threshold_kt: int = 34) -> dict:
    """Shortcut: get wind radii GeoJSON (34, 50, or 64 kt)."""
    layer_key = f"wind_radii_{threshold_kt}kt"
    if layer_key not in config.ARCGIS_LAYERS:
        raise ValueError(f"Unsupported wind threshold: {threshold_kt} kt (use 34, 50, or 64)")
    base = config.ARCGIS_ATL_ACTIVE if basin.upper() == "AL" else config.ARCGIS_EPAC_ACTIVE
    return fetch_arcgis_layer(base, layer=config.ARCGIS_LAYERS[layer_key])


# ---- NHC RSS advisories ---------------------------------------------------

def _parse_rss(url: str) -> list[dict]:
    resp = _get(url)
    root = ET.fromstring(resp.content)
    items = []
    for item in root.iter("item"):
        items.append({
            "title": (item.findtext("title") or "").strip(),
            "link": (item.findtext("link") or "").strip(),
            "description": (item.findtext("description") or "").strip(),
            "pubDate": (item.findtext("pubDate") or "").strip(),
        })
    return items


def fetch_nhc_rss(basin: str = "AL") -> list[dict]:
    """
    Fetch NHC RSS advisory feed.

    Parameters
    ----------
    basin : str
        ``"AL"`` (Atlantic), ``"EP"`` (East Pacific), or ``"CP"`` (Central Pacific).
    """
    urls = {"AL": config.NHC_RSS_ATL_URL, "EP": config.NHC_RSS_EPAC_URL, "CP": config.NHC_RSS_CPAC_URL}
    url = urls.get(basin.upper())
    if url is None:
        raise ValueError(f"Unknown basin '{basin}'. Use AL, EP, or CP.")
    return _parse_rss(url)


# ---- IBTrACS CSV -----------------------------------------------------------

def fetch_ibtracs(
    dataset: str = "last3years",
    max_rows: int | None = None,
) -> list[dict]:
    """
    Download IBTrACS CSV and return rows as dicts.

    Parameters
    ----------
    dataset : str
        ``"last3years"``, ``"active"``, or ``"all"``.
    max_rows : int, optional
        Limit returned rows (useful for ``"all"`` which is very large).
    """
    urls = {
        "last3years": config.IBTRACS_LAST3_URL,
        "active": config.IBTRACS_ACTIVE_URL,
        "all": config.IBTRACS_ALL_URL,
    }
    url = urls.get(dataset)
    if url is None:
        raise ValueError(f"Unknown IBTrACS dataset '{dataset}'.")
    log.info("Fetching IBTrACS: %s", url)
    resp = _get(url)
    # IBTrACS CSV has a header row then a units row – skip the units row
    lines = resp.text.splitlines()
    if len(lines) < 3:
        return []
    reader = csv.DictReader(io.StringIO("\n".join([lines[0]] + lines[2:])))
    rows = []
    for i, row in enumerate(reader):
        rows.append(dict(row))
        if max_rows and i + 1 >= max_rows:
            break
    return rows


# ---- HURDAT2 ---------------------------------------------------------------

def _parse_hurdat2(text: str) -> list[dict]:
    """Parse HURDAT2 text into a list of storm dicts."""
    storms = []
    current: dict | None = None
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4 and parts[0].startswith(("AL", "EP", "CP")):
            if current:
                storms.append(current)
            current = {
                "storm_id": parts[0],
                "name": parts[1],
                "num_entries": int(parts[2]),
                "track": [],
            }
        elif current is not None and len(parts) >= 7:
            try:
                lat_str = parts[4].strip()
                lon_str = parts[5].strip()
                lat = float(lat_str[:-1]) * (1 if lat_str[-1] == "N" else -1)
                lon = float(lon_str[:-1]) * (-1 if lon_str[-1] == "W" else 1)
            except (ValueError, IndexError):
                lat, lon = None, None
            try:
                vmax = int(parts[6]) if parts[6].strip() not in ("", "-999") else None
            except ValueError:
                vmax = None
            try:
                mslp = int(parts[7]) if len(parts) > 7 and parts[7].strip() not in ("", "-999") else None
            except ValueError:
                mslp = None
            current["track"].append({
                "date": parts[0],
                "time": parts[1],
                "record_id": parts[2],
                "status": parts[3],
                "lat": lat,
                "lon": lon,
                "vmax_kt": vmax,
                "mslp_mb": mslp,
            })
    if current:
        storms.append(current)
    return storms


def fetch_hurdat2(basin: str = "AL") -> list[dict]:
    """Download and parse the full HURDAT2 database (Atlantic or E-Pac)."""
    url = config.HURDAT2_ATL_URL if basin.upper() == "AL" else config.HURDAT2_EPAC_URL
    log.info("Fetching HURDAT2: %s", url)
    resp = _get(url)
    return _parse_hurdat2(resp.text)


# ---- Convenience: aggregate snapshot ----------------------------------------

def get_realtime_snapshot(basin: str = "AL") -> dict:
    """
    Build a combined real-time snapshot for the given basin.

    Returns a dict with keys:

    * ``active_storms`` – from NHC CurrentStorms.json
    * ``forecast_track`` – GeoJSON from ArcGIS
    * ``forecast_cone`` – GeoJSON from ArcGIS
    * ``advisories`` – from NHC RSS
    * ``timestamp`` – UTC ISO-8601
    """
    snapshot: dict[str, Any] = {"timestamp": datetime.now(timezone.utc).isoformat()}
    errors: list[str] = []

    try:
        snapshot["active_storms"] = list_active_storms()
    except Exception as exc:
        log.warning("Could not fetch active storms: %s", exc)
        errors.append(f"active_storms: {exc}")
        snapshot["active_storms"] = []

    try:
        snapshot["forecast_track"] = fetch_forecast_track(basin)
    except Exception as exc:
        log.warning("Could not fetch forecast track: %s", exc)
        errors.append(f"forecast_track: {exc}")
        snapshot["forecast_track"] = {}

    try:
        snapshot["forecast_cone"] = fetch_forecast_cone(basin)
    except Exception as exc:
        log.warning("Could not fetch forecast cone: %s", exc)
        errors.append(f"forecast_cone: {exc}")
        snapshot["forecast_cone"] = {}

    try:
        snapshot["advisories"] = fetch_nhc_rss(basin)
    except Exception as exc:
        log.warning("Could not fetch advisories: %s", exc)
        errors.append(f"advisories: {exc}")
        snapshot["advisories"] = []

    if errors:
        snapshot["errors"] = errors

    return snapshot
