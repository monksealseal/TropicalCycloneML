"""
CycloneAPI Python SDK

Install:
    pip install cycloneapi

Usage:
    from cycloneapi import CycloneAPI

    client = CycloneAPI("cyc_your_api_key")
    storms = client.active_storms()
    prediction = client.predict(model="weatherflow", basin="AL")
"""

from __future__ import annotations

from typing import Any

import requests


class CycloneAPIError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


class CycloneAPI:
    """Client for the CycloneAPI tropical cyclone intelligence service."""

    DEFAULT_BASE_URL = "https://api.cycloneapi.com"

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        timeout: int = 30,
    ):
        self.api_key = api_key
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
            "User-Agent": "CycloneAPI-Python/1.0",
        })

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self.base_url}{path}"
        kwargs.setdefault("timeout", self.timeout)
        resp = self._session.request(method, url, **kwargs)
        if not resp.ok:
            detail = resp.text
            try:
                detail = resp.json().get("detail", resp.text)
            except (ValueError, AttributeError):
                pass
            raise CycloneAPIError(resp.status_code, detail)
        return resp.json()

    def _get(self, path: str, **params: Any) -> Any:
        return self._request("GET", path, params=params)

    def _post(self, path: str, **params: Any) -> Any:
        return self._request("POST", path, params=params)

    # ----- Account -----

    def usage(self) -> dict:
        """Get current billing period usage."""
        return self._get("/v1/account/usage")

    def tier(self) -> dict:
        """Get subscription tier details."""
        return self._get("/v1/account/tier")

    # ----- Real-Time Data -----

    def active_storms(self) -> list[dict]:
        """List all currently active tropical cyclones."""
        return self._get("/v1/storms").get("storms", [])

    def active_storms_raw(self) -> dict:
        """Raw NHC CurrentStorms.json response."""
        return self._get("/v1/storms/raw")

    def snapshot(self, basin: str = "AL") -> dict:
        """Combined real-time data snapshot."""
        return self._get("/v1/snapshot", basin=basin)

    def forecast_track(self, basin: str = "AL") -> dict:
        """NHC forecast track as GeoJSON."""
        return self._get("/v1/forecast/track", basin=basin)

    def forecast_cone(self, basin: str = "AL") -> dict:
        """NHC forecast cone as GeoJSON."""
        return self._get("/v1/forecast/cone", basin=basin)

    def wind_radii(self, basin: str = "AL", threshold_kt: int = 34) -> dict:
        """Wind radii polygons as GeoJSON."""
        return self._get("/v1/forecast/wind-radii", basin=basin, threshold_kt=threshold_kt)

    def advisories(self, basin: str = "AL") -> list[dict]:
        """NHC RSS advisories."""
        return self._get("/v1/advisories", basin=basin).get("advisories", [])

    # ----- ATCF -----

    def atcf_files(self) -> list[str]:
        """List available ATCF best-track files."""
        return self._get("/v1/atcf/files").get("files", [])

    def atcf_track(self, storm_id: str) -> list[dict]:
        """Fetch ATCF best-track records for a storm."""
        return self._get(f"/v1/atcf/track/{storm_id}").get("track", [])

    # ----- IBTrACS -----

    def ibtracs(
        self,
        dataset: str = "last3years",
        storm: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Fetch IBTrACS data."""
        params: dict[str, Any] = {"dataset": dataset, "limit": limit}
        if storm:
            params["storm"] = storm
        return self._get("/v1/ibtracs", **params).get("records", [])

    # ----- HURDAT2 -----

    def hurdat2(
        self,
        basin: str = "AL",
        storm: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Fetch HURDAT2 historical data."""
        params: dict[str, Any] = {"basin": basin, "limit": limit}
        if storm:
            params["storm"] = storm
        return self._get("/v1/hurdat2", **params).get("storms", [])

    # ----- ML Models -----

    def list_models(self) -> dict:
        """List available HuggingFace models and spaces."""
        return self._get("/v1/models")

    def model_info(self, model_id: str) -> dict:
        """Get metadata for a model."""
        return self._get(f"/v1/models/{model_id}/info")

    # ----- Inference -----

    def predict(self, model: str = "weatherflow", basin: str = "AL") -> dict:
        """Run ML inference on current active storms."""
        return self._post("/v1/inference/live", model=model, basin=basin)

    def predict_forecast(self, model: str = "weatherflow", basin: str = "AL") -> dict:
        """Run ML inference on NHC forecast track."""
        return self._post("/v1/inference/forecast", model=model, basin=basin)

    def predict_atcf(self, storm_id: str, model: str = "weatherflow") -> dict:
        """Run ML inference on ATCF best-track data."""
        return self._post(f"/v1/inference/atcf/{storm_id}", model=model)

    def predict_ibtracs(
        self,
        model: str = "weatherflow",
        dataset: str = "last3years",
        storm: str | None = None,
        limit: int = 500,
    ) -> dict:
        """Run ML inference on IBTrACS data."""
        params: dict[str, Any] = {"model": model, "dataset": dataset, "limit": limit}
        if storm:
            params["storm"] = storm
        return self._post("/v1/inference/ibtracs", **params)

    def full_analysis(self, model: str = "weatherflow", basin: str = "AL") -> dict:
        """Comprehensive analysis: all data sources + ML predictions."""
        return self._post("/v1/analysis/full", model=model, basin=basin)
