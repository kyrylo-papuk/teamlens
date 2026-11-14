"""Tracing to Langfuse, under the same data rules as everything else.

A trace is a copy of what the agent saw and did, kept in another system. Two rules follow:
  - Restricted data cannot be in a trace, because it never enters this process (docs/DATA.md).
  - Confidential data must not identify a person there. Every name is replaced by the employee id
    before anything leaves the process: "Anton Lindqvist is away" is stored as "E002 is away".
    The id is enough to debug with, and only someone with HR access can turn it back into a person.

The replacement is not left to the call sites. It is registered as the client's ``mask`` function,
which the SDK applies to every input, output and metadata value, so a new span added later is
covered without anyone remembering to mask it.

That function only sees spans this code creates. Libraries underneath (the Anthropic SDK, the MCP
client) emit OpenTelemetry spans of their own, which the Langfuse client would export as well and
which the mask never touches. So the client is told to export this code's spans only. Today those
library spans carry no content; the filter is there so that this stays true whatever they add.

Without Langfuse keys nothing is traced and every call here is a no-op.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Any

from teamlens.hrm import Person

MAX_CHARS = 4000  # a tool result can be long; a trace needs enough to debug, not a full copy


class Masker:
    """Replaces the names of known people with their employee ids, anywhere in a value."""

    def __init__(self) -> None:
        self._ids: dict[str, str] = {}
        self._pattern: re.Pattern | None = None

    def learn(self, people: list[Person]) -> None:
        before = len(self._ids)
        first_names = [p.name.split()[0] for p in people]
        for p in people:
            first, last = p.name.split()[0], p.name.split()[-1]
            self._ids.setdefault(p.name, p.id)
            self._ids.setdefault(last, p.id)
            # A first name shared by several people cannot be turned into one id, and must not stay readable either.
            self._ids.setdefault(first, p.id if first_names.count(first) == 1 else "[person]")
        if len(self._ids) != before:  # longest first, so a full name wins over the surname inside it
            names = sorted(self._ids, key=len, reverse=True)
            self._pattern = re.compile(r"\b(" + "|".join(map(re.escape, names)) + r")\b")

    def __call__(self, *, data: Any, **_: Any) -> Any:
        if isinstance(data, str):
            text = self._pattern.sub(lambda m: self._ids[m.group(1)], data) if self._pattern else data
            return text if len(text) <= MAX_CHARS else text[:MAX_CHARS] + f"… [{len(text) - MAX_CHARS} more characters]"
        if isinstance(data, dict):
            return {key: self(data=value) for key, value in data.items()}
        if isinstance(data, (list, tuple)):
            return [self(data=value) for value in data]
        return data


def client(public_key: str, secret_key: str, host: str, masker: Masker, **options: Any):
    """A Langfuse client that masks everything it writes and exports nothing written by anyone else."""
    from langfuse import Langfuse, is_langfuse_span

    return Langfuse(public_key=public_key, secret_key=secret_key, host=host or None, mask=masker,
                    should_export_span=is_langfuse_span, **options)


class _Nothing:
    def update(self, **_: Any) -> None:
        return None


class Tracing:
    def __init__(self, client=None, masker: Masker | None = None) -> None:
        self._client, self.masker = client, masker or Masker()

    @classmethod
    def from_settings(cls, public_key: str, secret_key: str, host: str) -> Tracing:
        if not (public_key and secret_key):
            return cls()
        masker = Masker()
        return cls(client(public_key, secret_key, host, masker), masker)

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @contextmanager
    def request(self, name: str, thread: str, actor_id: str, input: Any = None):
        """One trace per request. Requests of one conversation share a session; the user is the employee id."""
        if not self._client:
            yield _Nothing()
            return
        from langfuse import propagate_attributes

        with propagate_attributes(session_id=thread, user_id=actor_id, trace_name=name):
            with self._client.start_as_current_observation(name=name, as_type="agent", input=input) as span:
                yield span

    @contextmanager
    def observe(self, name: str, kind: str = "span", **attributes: Any):
        """A step inside the current request. ``kind``: span | generation | tool | guardrail."""
        if not self._client:
            yield _Nothing()
            return
        with self._client.start_as_current_observation(name=name, as_type=kind, **attributes) as span:
            yield span

    def flush(self) -> None:
        if self._client:
            self._client.flush()
