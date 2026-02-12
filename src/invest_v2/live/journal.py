from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class JournalRecord:
    idempotency_key: str
    created_at_unix: float
    request: Dict[str, Any]
    response: Optional[Dict[str, Any]]


class OrderJournal:
    """SQLite order journal for idempotency and auditability.

    Design goal: prevent duplicate order placement when a runner is retried.
    """

    def __init__(self, path: str = "./.cache/order_journal.sqlite3"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.path) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS order_journal (
                    idempotency_key TEXT PRIMARY KEY,
                    created_at_unix REAL NOT NULL,
                    request_json TEXT NOT NULL,
                    response_json TEXT
                )
                """
            )
            con.commit()

    def get(self, idempotency_key: str) -> Optional[JournalRecord]:
        with sqlite3.connect(self.path) as con:
            cur = con.execute(
                "SELECT idempotency_key, created_at_unix, request_json, response_json FROM order_journal WHERE idempotency_key=?",
                (idempotency_key,),
            )
            row = cur.fetchone()
            if not row:
                return None
            req = json.loads(row[2])
            resp = json.loads(row[3]) if row[3] else None
            return JournalRecord(idempotency_key=row[0], created_at_unix=float(row[1]), request=req, response=resp)

    def put_request(self, idempotency_key: str, request: Dict[str, Any]) -> None:
        with sqlite3.connect(self.path) as con:
            con.execute(
                "INSERT OR IGNORE INTO order_journal(idempotency_key, created_at_unix, request_json, response_json) VALUES(?,?,?,NULL)",
                (idempotency_key, time.time(), json.dumps(request, ensure_ascii=False)),
            )
            con.commit()

    def put_response(self, idempotency_key: str, response: Dict[str, Any]) -> None:
        with sqlite3.connect(self.path) as con:
            con.execute(
                "UPDATE order_journal SET response_json=? WHERE idempotency_key=?",
                (json.dumps(response, ensure_ascii=False), idempotency_key),
            )
            con.commit()
