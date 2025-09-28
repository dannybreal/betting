from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
import json
from typing import Any, Mapping

import requests

API_BASE_URL = "https://v3.football.api-sports.io"
API_KEY_ENV = "API_FOOTBALL_KEY"


ENV_FILE = Path(__file__).resolve().parents[4] / ".env"


def _load_key_from_env_file() -> str | None:
    if not ENV_FILE.exists():
        return None
    for raw in ENV_FILE.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(f"{API_KEY_ENV}="):
            value = line.split("=", 1)[1].strip().strip('"').strip("'")
            return value or None
    return None


class ApiFootballError(RuntimeError):
    """Raised when API-Football returns an error or unexpected payload."""


@dataclass
class ApiFootballClient:
    api_key: str | None = None
    base_url: str = API_BASE_URL
    timeout: float = 30.0

    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key = os.getenv(API_KEY_ENV)
        if not self.api_key:
            self.api_key = _load_key_from_env_file()
        if not self.api_key:
            raise ApiFootballError(
                f"Missing API key. Set {API_KEY_ENV} environment variable or pass api_key explicitly."
            )
        self._session = requests.Session()
        self._headers = {
            "x-apisports-key": self.api_key,
        }

    def get(self, path: str, *, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        response = self._session.get(url, headers=self._headers, params=params, timeout=self.timeout)
        if response.status_code != 200:
            raise ApiFootballError(
                f"API request failed ({response.status_code}): {response.text[:200]}"
            )
        try:
            payload = response.json()
        except (UnicodeDecodeError, json.JSONDecodeError):
            text = response.content.decode('utf-8', errors='ignore')
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                text = response.content.decode('latin-1', errors='ignore')
                payload = json.loads(text)
        if payload.get("errors"):
            raise ApiFootballError(f"API returned errors: {payload['errors']}")
        return payload


