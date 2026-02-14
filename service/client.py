"""
TerraScore Python Client SDK.

Provides a simple interface for interacting with the TerraScore API
from Python code, notebooks, and CI/CD pipelines.

Usage
-----
    from service.client import TerraScoreClient

    client = TerraScoreClient(api_key="ts_fre_...")

    # Single assessment
    result = client.assess(
        name="My Weather Model",
        target_timescale="weather",
        reanalysis_quality=0.9,
        signal_to_noise_ratio=5.0,
    )
    print(result["score"], result["recommendation"])

    # Quick go/no-go check
    if client.should_use_ml(name="My Problem", **params):
        print("Proceed with ML approach")

    # Batch comparison
    results = client.compare([problem1, problem2, problem3])
"""

from __future__ import annotations

import json
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError


class TerraScoreError(Exception):
    """Error from the TerraScore API."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"HTTP {status_code}: {message}")


class TerraScoreClient:
    """
    Python client for the TerraScore API.

    Parameters
    ----------
    api_key : str
        Your TerraScore API key.
    base_url : str
        API base URL. Defaults to localhost for development.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "http://localhost:5000",
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def _request(self, method: str, path: str, data: Optional[dict] = None) -> dict:
        """Make an authenticated API request."""
        url = f"{self.base_url}{path}"
        body = json.dumps(data).encode() if data else None
        req = Request(url, data=body, method=method)
        req.add_header("X-API-Key", self.api_key)
        req.add_header("Content-Type", "application/json")

        try:
            with urlopen(req) as response:
                return json.loads(response.read().decode())
        except HTTPError as e:
            body = e.read().decode()
            try:
                error_data = json.loads(body)
                msg = error_data.get("error", body)
            except json.JSONDecodeError:
                msg = body
            raise TerraScoreError(e.code, msg) from None

    def assess(self, **problem_spec) -> dict:
        """
        Assess a single problem's AI suitability.

        Pass any ProblemSpecification fields as keyword arguments.
        The 'name' field is required.

        Returns the full assessment result dict.
        """
        if "name" not in problem_spec:
            raise ValueError("'name' is required")
        resp = self._request("POST", "/api/v1/assess", problem_spec)
        return resp["assessment"]

    def quick_score(self, **problem_spec) -> dict:
        """
        Get a quick score and go/no-go recommendation.

        Returns a simplified dict with score, recommendation, and proceed flag.
        """
        if "name" not in problem_spec:
            raise ValueError("'name' is required")
        return self._request("POST", "/api/v1/quick-score", problem_spec)

    def should_use_ml(self, **problem_spec) -> bool:
        """
        Simple boolean: should you use ML for this problem?

        Returns True if the recommendation is 'proceed' or 'proceed_with_caution'.
        """
        result = self.quick_score(**problem_spec)
        return result["proceed"]

    def compare(self, problems: list, weight_profile: str = "default") -> dict:
        """
        Batch compare multiple problems.

        Parameters
        ----------
        problems : list of dict
            Each dict contains ProblemSpecification fields.
        weight_profile : str
            Weight profile to use for all assessments.

        Returns the full batch response including comparison stats.
        """
        return self._request("POST", "/api/v1/assess/batch", {
            "problems": problems,
            "weight_profile": weight_profile,
        })

    def assess_example(self, name: str, weight_profile: str = "default") -> dict:
        """Run assessment on a pre-configured example problem."""
        resp = self._request(
            "GET",
            f"/api/v1/examples/{name}/assess?weight_profile={weight_profile}",
        )
        return resp["assessment"]

    def list_examples(self) -> list:
        """List available pre-configured examples."""
        resp = self._request("GET", "/api/v1/examples")
        return resp["examples"]

    def list_dimensions(self) -> list:
        """List all scoring dimensions."""
        resp = self._request("GET", "/api/v1/dimensions")
        return resp["dimensions"]

    def list_weight_profiles(self) -> dict:
        """List available weight profiles and their weights."""
        resp = self._request("GET", "/api/v1/weight-profiles")
        return resp["profiles"]

    def generate_key(self, owner: str = "anonymous", tier: str = "free") -> dict:
        """Generate a new API key (demo only)."""
        return self._request("POST", "/api/v1/keys/generate", {
            "owner": owner,
            "tier": tier,
        })
