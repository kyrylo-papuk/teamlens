"""One copy of the demo company for every instance of the app.

The public demo keeps the fictional company in memory. A serverless host runs several
instances, each with its own memory, so a change made through one instance was invisible
through another: an engineer closed an issue, the page asked a different instance and still
showed it open.

With a Redis store configured, an instance reads a version number before each request and
loads the saved state if it is behind. After a request that may change something it saves
the state and raises the version. Requests that change state take a lock, so two instances
never save over each other. If nothing was changed for ``idle_seconds`` the demo starts over.

This keeps a *demo* consistent. It is not how production stores data: the whole state is one
document, saved as a whole. docs/DATA.md says what a deployment needs instead.

The store is Upstash Redis over its REST API, which needs no connection pool and no driver:
each command is one HTTP request, which is what a serverless function wants.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import time
import uuid
import zlib
from collections.abc import Callable
from pathlib import Path

import httpx
from langgraph.checkpoint.memory import InMemorySaver

from teamlens.jira import FakeJira
from teamlens.seed import Issue

HEAD, STATE, LOCK = "teamlens:head", "teamlens:state", "teamlens:lock"
FILES = ("audit.jsonl", "access.json", "team_settings.json")  # everything the app writes besides paused conversations
LOCK_MS = 60_000  # longer than a request that waits for the model
StoreDown = (httpx.HTTPError, ValueError)  # a demo that cannot reach its store keeps working from its own memory


class Shared:
    def __init__(self, url: str, token: str, transport: httpx.AsyncBaseTransport | None = None,
                 idle_seconds: int = 7200, clock: Callable[[], float] = time.time) -> None:
        self._url, self._auth = url.rstrip("/"), {"Authorization": f"Bearer {token}"}
        self._transport, self._idle, self._clock = transport, idle_seconds, clock
        self._http: httpx.AsyncClient | None = None
        self.version = 0  # of the state this instance holds
        self._digest = ""  # of the state as last loaded or saved, so an unchanged state is not saved again

    async def _run(self, *command: str | int):
        if self._http is None:
            self._http = httpx.AsyncClient(transport=self._transport, timeout=10)
        answer = (await self._http.post(self._url, headers=self._auth, json=list(command))).json()
        if "error" in answer:
            raise ValueError(f"The shared store refused {command[0]}: {answer['error']}")
        return answer["result"]

    async def head(self) -> tuple[int, bool]:
        """Version of the saved state, and whether it was left alone long enough for the demo to start over."""
        raw = await self._run("GET", HEAD)
        if not raw:
            return 0, False
        version, at = (int(x) for x in raw.split(":"))
        return version, self._clock() - at > self._idle

    async def load(self) -> dict | None:
        raw = await self._run("GET", STATE)
        if not raw:
            return None
        text = zlib.decompress(base64.b64decode(raw)).decode()
        self._digest = hashlib.sha256(text.encode()).hexdigest()
        return json.loads(text)

    async def save(self, state: dict) -> None:
        """Call with the lock held and after catching up, so ``version`` is the one in the store."""
        text = json.dumps(state, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest == self._digest:
            return
        await self._run("SET", STATE, base64.b64encode(zlib.compress(text.encode())).decode())
        await self._run("SET", HEAD, f"{self.version + 1}:{int(self._clock())}")
        self.version, self._digest = self.version + 1, digest

    def forget(self) -> None:
        self._digest = ""

    async def lock(self) -> str | None:
        """Wait for the right to change the state. ``None`` means: go on without it (store down, or a long wait);
        a rare lost change is better than a demo that hangs."""
        mine = uuid.uuid4().hex
        try:
            for _ in range(150):
                if await self._run("SET", LOCK, mine, "NX", "PX", LOCK_MS) == "OK":
                    return mine
                await asyncio.sleep(0.2)
        except StoreDown:
            pass
        return None

    async def unlock(self, mine: str | None) -> None:
        try:
            if mine and await self._run("GET", LOCK) == mine:
                await self._run("DEL", LOCK)
        except StoreDown:
            pass

    async def count(self, key: str) -> int:
        """Add one to a counter every instance shares, kept for two days."""
        n = int(await self._run("INCR", key))
        if n == 1:
            await self._run("EXPIRE", key, 172_800)
        return n


class Threads(InMemorySaver):
    """Paused conversations in memory, in a form that can be saved and loaded as a whole.

    The checkpointer keeps three dictionaries whose keys are tuples and whose values hold bytes. They are
    written out as plain JSON: no pickle, because this is read back from a store outside the process.
    """

    def dump(self) -> dict:
        storage = {thread: {ns: dict(found) for ns, found in by_ns.items()} for thread, by_ns in self.storage.items()}
        return {"storage": _plain(storage), "writes": _plain(dict(self.writes)), "blobs": _plain(dict(self.blobs))}

    def load(self, saved: dict | None) -> None:
        self.storage.clear()
        self.writes.clear()
        self.blobs.clear()
        if not saved:
            return
        for thread, by_ns in _original(saved["storage"]).items():
            for ns, found in by_ns.items():
                self.storage[thread][ns].update(found)
        self.writes.update(_original(saved["writes"]))
        self.blobs.update(_original(saved["blobs"]))


def _plain(x):
    if isinstance(x, bytes):
        return {"b": base64.b64encode(x).decode()}
    if isinstance(x, tuple):
        return {"t": [_plain(i) for i in x]}
    if isinstance(x, list):
        return {"l": [_plain(i) for i in x]}
    if isinstance(x, dict):
        return {"d": [[_plain(k), _plain(v)] for k, v in x.items()]}
    return x


def _original(x):
    if not isinstance(x, dict):
        return x
    (kind, value), = x.items()
    if kind == "b":
        return base64.b64decode(value)
    if kind == "t":
        return tuple(_original(i) for i in value)
    if kind == "l":
        return [_original(i) for i in value]
    return {_original(k): _original(v) for k, v in value}


def snapshot(jira: FakeJira, data: Path, threads: Threads | None) -> dict:
    return {"issues": [i.model_dump(mode="json") for i in jira.all()], "comments": jira.comments,
            "files": {name: (data / name).read_text(encoding="utf-8") for name in FILES if (data / name).exists()},
            "threads": threads.dump() if threads is not None else None}


def restore(state: dict | None, jira: FakeJira, seed: list[Issue], data: Path, threads: Threads | None) -> None:
    """Put this instance into ``state``; without one, into the state the demo starts in."""
    state = state or {"issues": None, "comments": {}, "files": {}, "threads": None}
    jira.reset(seed if state["issues"] is None else [Issue(**i) for i in state["issues"]], state["comments"])
    data.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        if name in state["files"]:
            (data / name).write_text(state["files"][name], encoding="utf-8")
        else:
            (data / name).unlink(missing_ok=True)
    if threads is not None:
        threads.load(state["threads"])
