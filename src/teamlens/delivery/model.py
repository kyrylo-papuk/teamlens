"""The language model behind one small interface: Claude for real runs, a script for tests."""

from __future__ import annotations

from typing import Protocol

import anthropic
from pydantic import BaseModel

from teamlens.tracing import Tracing

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
    async def text(self, system: str, user: str) -> tuple[str, Usage]: ...
    async def structured(self, system: str, user: str, schema: type[BaseModel]) -> tuple[BaseModel, Usage]: ...


class ClaudeModel:
    def __init__(self, model: str = "claude-haiku-5-5", effort: str = "medium",
                 client: anthropic.AsyncAnthropic | None = None, api_key: str | None = None,
                 tracing: Tracing | None = None) -> None:
        self._model, self._effort, self._tracing = model, effort, tracing or Tracing()
        self._client = client or anthropic.AsyncAnthropic(api_key=api_key or None)

    def _traced(self, span, usage: Usage, output) -> None:
        span.update(output=output, usage_details={"input": usage.input, "output": usage.output,
                                                  "cache_read_input_tokens": usage.cache_read,
                                                  "cache_creation_input_tokens": usage.cache_write},
                    cost_details={"total": usage.usd})

    async def respond(self, system: str, messages: list[dict], tools: list[dict]) -> Reply:
        # The trace gets the newest message only: earlier turns are already in earlier spans of the same trace.
        with self._tracing.observe("model", "generation", model=self._model, input=messages[-1]["content"],
                                   model_parameters={"effort": self._effort}) as span:
            reply = await self._respond(system, messages, tools)
            self._traced(span, reply.usage, {"text": reply.text, "tool_calls": [c.model_dump() for c in reply.tool_calls]})
            return reply

    async def _respond(self, system: str, messages: list[dict], tools: list[dict]) -> Reply:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=16000,
            system=system,
            messages=messages,
            **({"tools": tools} if tools else {}),
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

    async def text(self, system: str, user: str) -> tuple[str, Usage]:
        """One answer in plain text, no tools."""
        reply = await self.respond(system, [{"role": "user", "content": user}], [])
        return reply.text, reply.usage

    async def structured(self, system: str, user: str, schema: type[BaseModel]) -> tuple[BaseModel, Usage]:
        """One answer constrained to ``schema``."""
        with self._tracing.observe("model", "generation", model=self._model, input=user,
                                   model_parameters={"effort": self._effort, "schema": schema.__name__}) as span:
            response = await self._client.messages.parse(
                model=self._model, max_tokens=16000, system=system, messages=[{"role": "user", "content": user}],
                output_config={"effort": self._effort}, output_format=schema)
            u = response.usage
            usage = Usage(calls=1, input=u.input_tokens, output=u.output_tokens,
                          cache_read=u.cache_read_input_tokens or 0, cache_write=u.cache_creation_input_tokens or 0)
            self._traced(span, usage, response.parsed_output.model_dump())
            return response.parsed_output, usage


class ScriptedModel:
    """Replays prepared replies in order and records what it was shown."""

    def __init__(self, replies: list) -> None:
        self._replies = list(replies)
        self.seen: list[list[dict]] = []
        self.tools: list[dict] = []

    async def respond(self, system: str, messages: list[dict], tools: list[dict]) -> Reply:
        self.seen.append(list(messages))
        self.tools = tools
        return self._replies.pop(0)

    async def text(self, system: str, user: str) -> tuple[str, Usage]:
        reply = await self.respond(system, [{"role": "user", "content": user}], [])
        return reply.text, reply.usage

    async def structured(self, system: str, user: str, schema: type[BaseModel]) -> tuple[BaseModel, Usage]:
        """The next prepared reply must be an instance of ``schema``."""
        self.seen.append([{"role": "user", "content": user}])
        answer = self._replies.pop(0)
        assert isinstance(answer, schema), f"scripted reply is {type(answer).__name__}, expected {schema.__name__}"
        return answer, Usage(calls=1)


def say(text: str) -> Reply:
    return Reply(content=[{"type": "text", "text": text}], text=text)


def call(name: str, _id: str = "call_1", **arguments) -> Reply:
    block = {"type": "tool_use", "id": _id, "name": name, "input": arguments}
    return Reply(content=[block], tool_calls=[ToolCall(id=_id, name=name, input=arguments)])
