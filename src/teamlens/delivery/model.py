"""The language model behind one small interface: Claude for real runs, a script for tests."""

from __future__ import annotations

from typing import Protocol

import anthropic
from pydantic import BaseModel

# USD per million tokens for prompts up to 100K tokens (claude-haiku-5-5).
PRICE = {"input": 0.10, "output": 0.50, "cache_read": 0.01, "cache_write": 0.125}


class ToolCall(BaseModel):
    id: str
    name: str
    input: dict


class Usage(BaseModel):
    calls: int = 0
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0

    def plus(self, other: "Usage") -> "Usage":
        return Usage(**{k: getattr(self, k) + getattr(other, k) for k in Usage.model_fields})

    @property
    def usd(self) -> float:
        return sum(getattr(self, k) * PRICE[k] for k in PRICE) / 1_000_000


class Reply(BaseModel):
    content: list[dict]  # assistant content blocks, sent back unchanged on the next turn
    text: str = ""
    tool_calls: list[ToolCall] = []
    usage: Usage = Usage()


class Model(Protocol):
    async def respond(self, system: str, messages: list[dict], tools: list[dict]) -> Reply: ...


class ClaudeModel:
    def __init__(self, model: str = "claude-haiku-5-5", effort: str = "medium",
                 client: anthropic.AsyncAnthropic | None = None, api_key: str | None = None) -> None:
        self._model, self._effort = model, effort
        self._client = client or anthropic.AsyncAnthropic(api_key=api_key or None)

    async def respond(self, system: str, messages: list[dict], tools: list[dict]) -> Reply:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=16000,
            system=system,
            tools=tools,
            messages=messages,
            output_config={"effort": self._effort},
            cache_control={"type": "ephemeral"},  # caches the stable prefix: tools, system, earlier turns
        )
        u = response.usage
        usage = Usage(calls=1, input=u.input_tokens, output=u.output_tokens,
                      cache_read=u.cache_read_input_tokens or 0, cache_write=u.cache_creation_input_tokens or 0)
        content = [b.model_dump(mode="json", exclude_none=True) for b in response.content]
        if response.stop_reason == "refusal":
            return Reply(content=[{"type": "text", "text": "The model declined this request."}],
                         text="The model declined this request.", usage=usage)
        return Reply(
            content=content,
            text="".join(b.text for b in response.content if b.type == "text").strip(),
            tool_calls=[ToolCall(id=b.id, name=b.name, input=b.input) for b in response.content if b.type == "tool_use"],
            usage=usage,
        )


class ScriptedModel:
    """Replays prepared replies in order and records what it was shown."""

    def __init__(self, replies: list[Reply]) -> None:
        self._replies = list(replies)
        self.seen: list[list[dict]] = []
        self.tools: list[dict] = []

    async def respond(self, system: str, messages: list[dict], tools: list[dict]) -> Reply:
        self.seen.append(list(messages))
        self.tools = tools
        return self._replies.pop(0)


def say(text: str) -> Reply:
    return Reply(content=[{"type": "text", "text": text}], text=text)


def call(name: str, _id: str = "call_1", **arguments) -> Reply:
    block = {"type": "tool_use", "id": _id, "name": name, "input": arguments}
    return Reply(content=[block], tool_calls=[ToolCall(id=_id, name=name, input=arguments)])
