"""The People agent's own model, built with Microsoft Agent Framework.

It reaches the HRM only through two lookups of the HRM MCP server. The server's third tool,
which returns absence reasons, is not wrapped here, so the model cannot call it.
"""

from __future__ import annotations

import json
from datetime import date

from agent_framework_anthropic import AnthropicClient
from mcp import Client

INSTRUCTIONS = """You are the People agent of an engineering company. Other agents and managers ask you \
staffing questions: who works in which team and role, how many hours a week, and when people are away.

Today is {today}. Use the tools for every fact; do not answer from memory.
You know when someone is away, never why: you have no access to reasons for absence, pay or contact details, \
and you do not guess them. If asked, say that this information is not shared.
Refer to people by name with their employee id in brackets. Be brief: answer first, details only if needed."""


def build_advisor(hrm: Client, api_key: str, today: date, model: str = "claude-haiku-5-5"):
    """``hrm`` is a connected MCP client of the HRM server. Returns ``async (question) -> answer``."""

    async def lookup(tool: str, arguments: dict) -> str:
        result = await hrm.call_tool(tool, arguments)
        payload = result.structured_content or {}
        return json.dumps(payload.get("result", payload), ensure_ascii=False)

    async def list_people(team: str | None = None) -> str:
        """Employees with team, role and weekly hours. Pass a team key such as PAY, or nothing for everyone."""
        return await lookup("list_people", {"team": team})

    async def get_availability(employee_id: str, start: str | None = None, end: str | None = None) -> str:
        """When one employee is away between two ISO dates (default: the current sprint). Dates only, no reason."""
        return await lookup("get_availability", {"employee_id": employee_id, "start": start, "end": end})

    agent = AnthropicClient(api_key=api_key, model=model).as_agent(
        name="People agent", instructions=INSTRUCTIONS.format(today=today), tools=[list_people, get_availability])

    async def ask(question: str) -> str:
        return (await agent.run(question)).text

    return ask
