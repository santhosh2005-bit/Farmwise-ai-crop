"""
api.py — HTTP client helpers for the Streamlit frontend.

Wraps all backend API calls so that the UI code never builds URLs or
handles ``requests`` directly.
"""

from __future__ import annotations

from typing import Any

import requests

# ─── Backend base URL ─────────────────────────────────────────
# Override with FARMWISE_API_URL env var for production / ngrok deployments.
import os as _os
_BASE_URL: str = _os.getenv("FARMWISE_API_URL", "http://localhost:8000")


def health_check() -> bool:
    """Return *True* if the backend is reachable.

    Returns
    -------
    bool
        ``True`` when the ``/health`` endpoint responds with status 200.
    """
    try:
        resp = requests.get(f"{_BASE_URL}/health", timeout=5)
        return resp.status_code == 200
    except requests.ConnectionError:
        return False


def get_dataset_info() -> dict[str, Any]:
    """Fetch dataset metadata from the backend.

    Returns
    -------
    dict[str, Any]
        Keys: ``columns``, ``row_count``, ``sample_rows``.

    Raises
    ------
    requests.HTTPError
        If the backend returns a non-2xx status.
    """
    resp = requests.get(f"{_BASE_URL}/dataset/info", timeout=10)
    resp.raise_for_status()
    return resp.json()


def send_message(
    session_id: str,
    message: str,
    custom_api_key: str | None = None,
    model_choice: str | None = None,
) -> dict[str, Any]:
    """Send a chat message to the backend and return the AI response.

    Parameters
    ----------
    session_id : str
        The current session identifier.
    message : str
        The user's natural-language question.
    custom_api_key : str | None
        User provided custom API key.
    model_choice : str | None
        User chosen Llama model name.

    Returns
    -------
    dict[str, Any]
        Keys: ``answer``, ``chart_json``, ``data``, ``suggestions``, ``insights``.

    Raises
    ------
    requests.HTTPError
        If the backend returns a non-2xx status.
    """
    payload = {
        "session_id": session_id,
        "message": message,
        "custom_api_key": custom_api_key,
        "model_choice": model_choice,
    }
    resp = requests.post(f"{_BASE_URL}/chat", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def reset_session(session_id: str) -> bool:
    """Reset the conversation memory for a session.

    Parameters
    ----------
    session_id : str
        The session to reset.

    Returns
    -------
    bool
        ``True`` if the reset was successful.
    """
    try:
        payload = {"session_id": session_id}
        st = requests.post(f"{_BASE_URL}/chat/reset", json=payload, timeout=10)
        return st.status_code == 200
    except requests.ConnectionError:
        return False


def get_forecast(country: str, item: str, forecast_years: int = 5) -> list[dict[str, Any]]:
    """Fetch yield forecast data from the backend.

    Parameters
    ----------
    country : str
        Country name.
    item : str
        Crop name.
    forecast_years : int
        Number of years to forecast.

    Returns
    -------
    list[dict]
        Yield data points.
    """
    params = {"country": country, "item": item, "forecast_years": forecast_years}
    resp = requests.get(f"{_BASE_URL}/analytics/forecast", params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def get_regression(country: str, item: str) -> dict[str, Any]:
    """Fetch factor regression data from the backend.

    Parameters
    ----------
    country : str
        Country name.
    item : str
        Crop name.

    Returns
    -------
    dict
        Factor analysis results.
    """
    params = {"country": country, "item": item}
    resp = requests.get(f"{_BASE_URL}/analytics/regression", params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def get_raw_data(country: str, item: str) -> list[dict[str, Any]]:
    """Fetch raw yield and climate data records for a specific country and crop.

    Parameters
    ----------
    country : str
        Country name.
    item : str
        Crop name.

    Returns
    -------
    list[dict]
        Raw records.
    """
    params = {"country": country, "item": item}
    resp = requests.get(f"{_BASE_URL}/analytics/raw-data", params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()
