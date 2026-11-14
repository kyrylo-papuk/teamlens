"""Append-only audit log. Each entry carries the hash of the one before it,
so an edited or removed entry breaks the chain and ``verify`` reports where.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

GENESIS = "0" * 64


def _digest(entry: dict) -> str:
    body = {k: v for k, v in entry.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class AuditLog:
    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def entries(self) -> list[dict]:
        if not self._path.exists():
            return []
        return [json.loads(line) for line in self._path.read_text(encoding="utf-8").splitlines() if line]

    def record(self, kind: str, actor: str, thread: str, **data) -> dict:
        previous = self.entries()
        entry = {
            "seq": len(previous) + 1,
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            "kind": kind, "actor": actor, "thread": thread, "data": data,
            "prev": previous[-1]["hash"] if previous else GENESIS,
        }
        entry["hash"] = _digest(entry)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def verify(self) -> int | None:
        """``None`` if the chain is intact, otherwise the ``seq`` of the first broken entry."""
        prev = GENESIS
        for n, entry in enumerate(self.entries(), start=1):
            if entry["seq"] != n or entry["prev"] != prev or entry["hash"] != _digest(entry):
                return n
            prev = entry["hash"]
        return None
