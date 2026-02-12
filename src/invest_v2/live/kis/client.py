from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

from .auth import KISTokenProvider
from .settings import KISSettings


@dataclass
class KISResponse:
    status_code: int
    headers: Dict[str, str]
    json: Any
    text: str


class KISClient:
    """Thin REST client for KIS OpenAPI.

    - Adds OAuth2 bearer token
    - Adds appkey/appsecret headers
    - Optionally adds hashkey for POST bodies
    """

    def __init__(self, settings: KISSettings, token_provider: Optional[KISTokenProvider] = None, timeout_sec: int = 10):
        self.settings = settings
        self.timeout_sec = timeout_sec
        self.token_provider = token_provider or KISTokenProvider(settings)

    def _auth_headers(self) -> Dict[str, str]:
        tok = self.token_provider.get()
        return {
            "authorization": f"{tok.token_type} {tok.access_token}",
            "appkey": self.settings.app_key,
            "appsecret": self.settings.app_secret,
        }

    def hashkey(self, body: Dict[str, Any]) -> str:
        """Generate hashkey for POST request body.

        KIS requires a hashkey header for some POST endpoints.
        """
        url = f"{self.settings.base_url}/uapi/hashkey"
        headers = self._auth_headers() | {"content-type": "application/json"}
        resp = requests.post(url, headers=headers, data=json.dumps(body, ensure_ascii=False), timeout=self.timeout_sec)
        if resp.status_code != 200:
            raise RuntimeError(f"KIS hashkey failed: {resp.status_code} {resp.text}")
        data = resp.json()
        hk = str(data.get("HASH", "")) or str(data.get("hash", ""))
        if not hk:
            raise RuntimeError(f"KIS hashkey returned empty HASH: {data}")
        return hk

    def request(
        self,
        method: str,
        path: str,
        *,
        tr_id: str,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
        use_hashkey: bool = False,
    ) -> KISResponse:
        url = f"{self.settings.base_url}{path}"
        headers = self._auth_headers() | {"tr_id": tr_id}
        if method.upper() in {"POST", "PUT", "PATCH"}:
            headers["content-type"] = "application/json"
            if body is None:
                body = {}
            if use_hashkey:
                headers["hashkey"] = self.hashkey(body)
            resp = requests.request(method.upper(), url, headers=headers, params=params, json=body, timeout=self.timeout_sec)
        else:
            resp = requests.request(method.upper(), url, headers=headers, params=params, timeout=self.timeout_sec)

        try:
            js = resp.json()
        except Exception:
            js = None

        return KISResponse(
            status_code=int(resp.status_code),
            headers={k.lower(): v for k, v in resp.headers.items()},
            json=js,
            text=resp.text,
        )