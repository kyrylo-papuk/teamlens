"""What the agent can see and do.

The model gets read tools and one way to ask for changes: ``propose_plan``.
Jira's write tools are never offered to the model; only this module calls them,
for steps that passed the policy and were approved by a person.
"""

from __future__ import annotations

import json
from datetime import date

import httpx
from mcp import Client

from teamlens.delivery.policy import Plan, Step, World
from teamlens.hrm import Person
from teamlens.jira import STATUSES
from teamlens.overview import build
from teamlens.people import People, PeopleAgentError
from teamlens.seed import Issue

MAX_ROWS = 40  # issues shown per search; more than this and the model is told to narrow the filters

OVERVIEW_TOOL = {
    "name": "team_overview",
    "description": "Your team right now: each person's open issues, sprint points against capacity, "
                   "availability, and detected problems. Call this first for any question about people or load.",
    "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
}

PEOPLE_TOOL = {
    "name": "ask_people_agent",
    "description": "Ask the People agent, which owns HR data, a staffing question in plain language: roles, weekly "
                   "hours, who is away and when, also beyond the current sprint. It never shares reasons for "
                   "absence, pay or contact details, so do not ask for them.",
    "input_schema": {"type": "object", "properties": {"question": {"type": "string"}},
                     "required": ["question"], "additionalProperties": False},
}

PLAN_TOOL = {
    "name": "propose_plan",
    "description": "The only way to change Jira. Submit every change as one plan. Each step is checked "
                   "against company policy and then shown to the user for approval; nothing runs before that. "
                   "If steps are refused you get the reasons and can submit a corrected plan.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "One sentence: what this plan achieves."},
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "tool": {"type": "string", "enum": ["assign_issue", "transition_issue", "create_issue"]},
                        "issue_key": {"type": ["string", "null"], "description": "Existing issue; null for create_issue."},
                        "assignee_id": {"type": ["string", "null"], "description": "Employee id for assign_issue; null to unassign."},
                        "status": {"type": ["string", "null"], "description": f"For transition_issue: {', '.join(STATUSES)}."},
                        "summary": {"type": ["string", "null"], "description": "Title for create_issue."},
                        "reason": {"type": "string", "description": "Why this step, in one sentence. Shown to the approver."},
                    },
                    "required": ["tool", "issue_key", "assignee_id", "status", "summary", "reason"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["summary", "steps"],
        "additionalProperties": False,
    },
}


class ToolFailure(Exception):
    pass


class Toolbox:
    def __init__(self, jira: Client, people: People, today: date) -> None:
        self._jira, self._people, self._today = jira, people, today
        self._read_tools: list[dict] | None = None

    async def definitions(self) -> list[dict]:
        """Tool list for the model. Stable order, so the prompt prefix stays cacheable."""
        if self._read_tools is None:
            listed = (await self._jira.list_tools()).tools
            self._read_tools = [
                {"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
                for t in sorted(listed, key=lambda t: t.name)
                if t.annotations is not None and t.annotations.read_only_hint
            ]
        delegate = [PEOPLE_TOOL] if hasattr(self._people, "ask") else []  # only when People is another agent
        return [OVERVIEW_TOOL, *self._read_tools, *delegate, PLAN_TOOL]

    async def _call_jira(self, name: str, arguments: dict):
        result = await self._jira.call_tool(name, arguments)
        if result.is_error:
            raise ToolFailure(" ".join(b.text for b in result.content if getattr(b, "text", None)))
        payload = result.structured_content or {}
        return payload["result"] if set(payload) == {"result"} else payload

    async def world(self, actor_id: str) -> World:
        try:
            return await self._world(actor_id)
        except (PeopleAgentError, httpx.HTTPError) as exc:
            raise ToolFailure(f"The People agent is unavailable: {exc}") from exc

    async def _world(self, actor_id: str) -> World:
        actor = await self._people.person(actor_id)
        if actor is None:
            raise ToolFailure(f"Unknown employee {actor_id}.")
        team = await self._people.team(actor.team) if actor.team else []
        availability = await self._people.availability([p.id for p in team])
        issues = [Issue(**i) for i in await self._call_jira("search_issues", {"project": actor.team})] if actor.team else []
        return World(actor=actor, overview=build(team, availability, issues, self._today),
                     issues={i.key: i for i in issues})

    async def run(self, name: str, arguments: dict, actor_id: str) -> str:
        """Run a read tool for the model and return compact text. Raises ToolFailure with a readable reason."""
        if name == "team_overview":
            overview = (await self.world(actor_id)).overview
            people = [{"id": x.employee_id, "name": x.name, "role": x.role, "open_issues": x.open_issues,
                       "sprint_points": x.sprint_points, "capacity_points": x.capacity_points, "load": x.load,
                       "status": "left the company" if not x.availability.active
                       else "available" if x.availability.available_today
                       else f"away until {x.availability.unavailable_until}"} for x in overview.loads]
            findings = [{"kind": f.kind, "employee_id": f.employee_id, "issues": f.issue_keys, "detail": f.detail}
                        for f in overview.findings]
            return json.dumps({"today": str(overview.today), "people": people, "findings": findings}, ensure_ascii=False)
        if name == PEOPLE_TOOL["name"] and hasattr(self._people, "ask"):
            try:
                return await self._people.ask(arguments["question"])
            except (PeopleAgentError, httpx.HTTPError) as exc:
                raise ToolFailure(f"The People agent is unavailable: {exc}") from exc
        if name not in {t["name"] for t in await self.definitions()} or name == PLAN_TOOL["name"]:
            raise ToolFailure(f"Tool '{name}' is not available. To change Jira, use propose_plan.")
        data = await self._call_jira(name, arguments)
        if isinstance(data, list):
            rows = [_brief(i) for i in data[:MAX_ROWS]]
            more = len(data) - len(rows)
            return json.dumps({"issues": rows, **({"not_shown": more, "hint": "Narrow the filters."} if more else {})},
                              ensure_ascii=False)
        return json.dumps(_brief(data) if "key" in data else data, ensure_ascii=False)

    async def execute(self, step: Step, actor: Person) -> str:
        """Apply one approved step in Jira and leave a comment saying who asked and why."""
        note = f"[TeamLens] Changed on behalf of {actor.name}. Reason: {step.reason}"
        if step.tool == "assign_issue":
            await self._call_jira("assign_issue", {"key": step.issue_key, "assignee_id": step.assignee_id})
            await self._call_jira("add_comment", {"key": step.issue_key, "text": note})
            return f"{step.issue_key} assigned to {step.assignee_id or 'nobody'}"
        if step.tool == "transition_issue":
            await self._call_jira("transition_issue", {"key": step.issue_key, "status": step.status})
            await self._call_jira("add_comment", {"key": step.issue_key, "text": note})
            return f"{step.issue_key} moved to {step.status}"
        created = await self._call_jira("create_issue", {"project": actor.team, "summary": step.summary})
        await self._call_jira("add_comment", {"key": created["key"], "text": note})
        return f"{created['key']} created: {step.summary}"


def _brief(issue: dict) -> dict:
    return {"key": issue["key"], "summary": issue["summary"], "type": issue["type"], "status": issue["status"],
            "assignee_id": issue["assignee_id"], "points": issue["story_points"], "in_sprint": issue["in_sprint"],
            "updated": issue["updated"]}


def parse_plan(arguments: dict) -> Plan:
    return Plan.model_validate(arguments)
