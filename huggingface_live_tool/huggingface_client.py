"""
Client for interacting with HuggingFace models and spaces from monkseal555.

Supports:
* HuggingFace Inference API (serverless or dedicated endpoints)
* Gradio-based Spaces via the ``gradio_client`` library
* HuggingFace Hub API for listing models / spaces / datasets
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import requests

from . import config

log = logging.getLogger(__name__)


def _hf_token() -> str | None:
    """Resolve the HuggingFace API token from the environment."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


def _headers(token: str | None = None) -> dict[str, str]:
    tok = token or _hf_token()
    headers = {"Content-Type": "application/json"}
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    return headers


# ---------------------------------------------------------------------------
# Hub API – list models / spaces / datasets for a user
# ---------------------------------------------------------------------------

def list_user_models(user: str = config.HF_USER) -> list[dict]:
    """List all public models owned by *user* on HuggingFace Hub."""
    url = f"{config.HF_API_URL}/models"
    resp = requests.get(url, params={"author": user}, timeout=config.DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def list_user_spaces(user: str = config.HF_USER) -> list[dict]:
    """List all public spaces owned by *user* on HuggingFace Hub."""
    url = f"{config.HF_API_URL}/spaces"
    resp = requests.get(url, params={"author": user}, timeout=config.DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def list_user_datasets(user: str = config.HF_USER) -> list[dict]:
    """List all public datasets owned by *user* on HuggingFace Hub."""
    url = f"{config.HF_API_URL}/datasets"
    resp = requests.get(url, params={"author": user}, timeout=config.DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_model_info(model_id: str) -> dict:
    """Return full metadata for a model on HuggingFace Hub."""
    url = f"{config.HF_API_URL}/models/{model_id}"
    resp = requests.get(url, timeout=config.DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_space_info(space_id: str) -> dict:
    """Return full metadata for a space on HuggingFace Hub."""
    url = f"{config.HF_API_URL}/spaces/{space_id}"
    resp = requests.get(url, timeout=config.DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Inference API – serverless model inference
# ---------------------------------------------------------------------------

def infer(
    model_id: str,
    payload: dict | list | str,
    token: str | None = None,
    wait_for_model: bool = True,
    timeout: int = 120,
) -> Any:
    """
    Run inference on a HuggingFace model via the Inference API.

    Parameters
    ----------
    model_id : str
        Full model ID, e.g. ``"monkseal555/autotrain-hurricane3-1415853436"``.
        You can also pass a short alias from ``config.HF_MODELS``.
    payload : dict | list | str
        Input data matching the model's expected schema. Typically
        ``{"inputs": ...}`` for text/tabular or raw image bytes for vision.
    token : str, optional
        HuggingFace API token. Falls back to ``$HF_TOKEN``.
    wait_for_model : bool
        If True, sets ``x-wait-for-model`` header so cold models are loaded.
    timeout : int
        Request timeout in seconds (default 120 for model loading).

    Returns
    -------
    Any
        Model output (JSON-decoded).
    """
    # Resolve short alias
    if model_id in config.HF_MODELS:
        model_id = config.HF_MODELS[model_id]

    url = config.HF_INFERENCE_API_URL.format(model_id=model_id)
    hdrs = _headers(token)
    if wait_for_model:
        hdrs["x-wait-for-model"] = "1"

    if isinstance(payload, (dict, list)):
        data = json.dumps(payload).encode()
    elif isinstance(payload, str):
        data = payload.encode()
    else:
        data = payload

    log.info("Inference request → %s", url)
    resp = requests.post(url, headers=hdrs, data=data, timeout=timeout)
    resp.raise_for_status()
    try:
        return resp.json()
    except ValueError:
        return resp.content


def infer_with_image(
    model_id: str,
    image_bytes: bytes,
    token: str | None = None,
    timeout: int = 120,
) -> Any:
    """
    Send raw image bytes to a vision / image-classification model.

    Parameters
    ----------
    model_id : str
        Full or short-alias model ID.
    image_bytes : bytes
        Raw PNG / JPEG bytes.
    """
    if model_id in config.HF_MODELS:
        model_id = config.HF_MODELS[model_id]

    url = config.HF_INFERENCE_API_URL.format(model_id=model_id)
    hdrs = _headers(token)
    hdrs["Content-Type"] = "application/octet-stream"
    hdrs["x-wait-for-model"] = "1"

    log.info("Image inference → %s (%d bytes)", url, len(image_bytes))
    resp = requests.post(url, headers=hdrs, data=image_bytes, timeout=timeout)
    resp.raise_for_status()
    try:
        return resp.json()
    except ValueError:
        return resp.content


# ---------------------------------------------------------------------------
# Gradio Spaces – interact with running Gradio apps
# ---------------------------------------------------------------------------

def call_space_api(
    space_id: str,
    fn_index: int = 0,
    data: list | None = None,
    token: str | None = None,
    timeout: int = 120,
) -> Any:
    """
    Call a Gradio Space's ``/api/predict`` endpoint directly.

    Parameters
    ----------
    space_id : str
        Full space ID (e.g. ``"monkseal555/testing"``) or short alias.
    fn_index : int
        The Gradio function index to call (default 0 = first function).
    data : list
        Positional arguments matching the Gradio interface inputs.
    """
    if space_id in config.HF_SPACES:
        space_id = config.HF_SPACES[space_id]

    # Gradio 4+ uses /api/predict or /call/<fn_index> pattern
    # Try the /api/predict endpoint first
    owner, name = space_id.split("/", 1)
    base = f"https://{owner}-{name}.hf.space"
    url = f"{base}/api/predict"

    payload = {"fn_index": fn_index, "data": data or []}
    hdrs = _headers(token)
    hdrs["Content-Type"] = "application/json"

    log.info("Space API call → %s  fn_index=%d", url, fn_index)
    resp = requests.post(url, headers=hdrs, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def call_space_gradio_client(
    space_id: str,
    *args: Any,
    api_name: str = "/predict",
    token: str | None = None,
) -> Any:
    """
    Call a Gradio Space using the ``gradio_client`` library (richer support).

    Requires ``pip install gradio_client``.

    Parameters
    ----------
    space_id : str
        Full or short-alias space ID.
    *args
        Positional arguments matching the Gradio interface inputs.
    api_name : str
        The API endpoint name (default ``"/predict"``).
    """
    try:
        from gradio_client import Client
    except ImportError:
        raise ImportError(
            "gradio_client is required for this method. "
            "Install with: pip install gradio_client"
        )

    if space_id in config.HF_SPACES:
        space_id = config.HF_SPACES[space_id]

    tok = token or _hf_token()
    client = Client(space_id, hf_token=tok)
    return client.predict(*args, api_name=api_name)


# ---------------------------------------------------------------------------
# Convenience: discover all resources
# ---------------------------------------------------------------------------

def discover_all(user: str = config.HF_USER) -> dict:
    """
    Discover all models, spaces, and datasets for a HuggingFace user.

    Returns
    -------
    dict
        Keys: ``models``, ``spaces``, ``datasets``, each a list of dicts.
    """
    result: dict[str, Any] = {}
    for kind, fetcher in [
        ("models", list_user_models),
        ("spaces", list_user_spaces),
        ("datasets", list_user_datasets),
    ]:
        try:
            result[kind] = fetcher(user)
        except Exception as exc:
            log.warning("Failed to list %s for %s: %s", kind, user, exc)
            result[kind] = []
    return result
