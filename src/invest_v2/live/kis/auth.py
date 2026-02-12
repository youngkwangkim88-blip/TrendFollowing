from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

from .settings import KISSettings


@dataclass
class Token:
    access_token: str
    token_type: str
    expires_at_unix: float

    def is_valid(self, leeway_sec: int = 60) -> bool:
        return bool(self.access_token) and time.time() < (self.expires_at_unix - leeway_sec)


class KISTokenProvider:
    """Fetches and caches KIS OAuth2 access tokens.

    KIS access tokens are short-lived (typically 24h); caching avoids re-issuing
    tokens repeatedly.
    """

    def __init__(self, settings: KISSettings, cache_path: Optional[str] = None, timeout_sec: int = 10):
        self.settings = settings
        self.timeout_sec = timeout_sec
        self.cache_path = Path(cache_path) if cache_path else Path(".cache/kis_token.json")

    def load_cached(self) -> Optional[Token]:
        try:
            if not self.cache_path.exists():
                return None
            obj = json.loads(self.cache_path.read_text(encoding="utf-8"))
            tok = Token(
                access_token=str(obj.get("access_token", "")),
                token_type=str(obj.get("token_type", "Bearer")),
                expires_at_unix=float(obj.get("expires_at_unix", 0.0)),
            )
            return tok if tok.is_valid() else None
        except Exception:
            return None

    def _save_cache(self, tok: Token) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(
                {
                    "access_token": tok.access_token,
                    "token_type": tok.token_type,
                    "expires_at_unix": tok.expires_at_unix,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def fetch_new(self) -> Token:
        url = f"{self.settings.base_url}/oauth2/tokenP"
        payload = {
            "grant_type": "client_credentials",
            "appkey": self.settings.app_key,
            "appsecret": self.settings.app_secret,
        }
        resp = requests.post(url, json=payload, timeout=self.timeout_sec)
        if resp.status_code != 200:
            raise RuntimeError(f"KIS token issuance failed: {resp.status_code} {resp.text}")
        data = resp.json()
        access_token = str(data.get("access_token", ""))
        token_type = str(data.get("token_type", "Bearer"))
        expires_in = float(data.get("expires_in", 0.0) or 0.0)
        if not access_token:
            raise RuntimeError(f"KIS token issuance returned empty access_token: {data}")
        expires_at = time.time() + (expires_in if expires_in > 0 else 23 * 3600)
        tok = Token(access_token=access_token, token_type=token_type, expires_at_unix=expires_at)
        self._save_cache(tok)
        return tok

    def get(self) -> Token:
        cached = self.load_cached()
        if cached:
            return cached
        return self.fetch_new()
