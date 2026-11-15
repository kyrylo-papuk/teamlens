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

from teamlens.access import Access
from teamlens.delivery.policy import Plan, Step, World
from teamlens.forecast import Fit, Forecast, default_fit, forecast
from teamlens.hrm import Person
from teamlens.jira import STATUSES
from teamlens.overview import build
from teamlens.people import People, PeopleAgentError
from teamlens.seed import Issue
from teamlens.similar import Finder, Match
from teamlens.tracing import Tracing

MAX_ROWS = 40  # issues shown per search; more than this and the model is told to narrow the filters

OVERVIEW_TOOL = {
    "name": "team_overview",
    "description": "Your team right now: each person's open issues, sprint points against capacity, "
                   "availability, and detected problems. Call this first for any question about people or load.",
    "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
}

FORECAST_TOOL = {
    "name": "sprint_forecast",
    "description": "How many of the points committed for this sprint the team is expected to finish, with a 90% "
                   "interval, from a regression fitted on past sprints. Use it for any question about whether the "
                   "sprint will be finished. Never estimate this yourself.",
    "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
}

SIMILAR_TOOL = {
    "name": "find_similar_issues",
    "description": "Open issues of the team that may already cover a given problem, even when worded differently. "
                   "Call it before proposing a new issue, and tell the user about any match instead of creating a copy.",
    "input_schema": {"type": "object", "properties": {"text": {"type": "string", "description": "Title of the new issue."}},
                     "required": ["text"], "additionalProperties": False},
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


def _change_tool(name: str, description: str, properties: dict) -> dict:
    properties = {**properties, "reason": {"type": "string", "description": "Why, in one sentence. Shown to the approver."}}
    return {"name": name, "description": description + " Checked against company policy, then approved by the user.",
            "strict": True, "input_schema": {"type": "object", "properties": properties, "required": list(properties),
                                             "additionalProperties": False}}


# React mode only: the same three changes a plan can contain, as tools the model calls one at a time.
CHANGE_TOOLS = {t["name"]: t for t in (
    _change_tool("assign_issue", "Give an issue to a person.", {
        "issue_key": {"type": "string"},
        "assignee_id": {"type": ["string", "null"], "description": "Employee id; null to unassign."}}),
    _change_tool("transition_issue", "Move an issue to another status.", {
        "issue_key": {"type": "string"}, "status": {"type": "string", "description": ", ".join(STATUSES)}}),
    _change_tool("create_issue", "Create an unassigned issue in the backlog.", {"summary": {"type": "string"}}),
)}


def to_step(tool: str, arguments: dict) -> Step:
    return Step(tool=tool, **arguments)


class ToolFailure(Exception):
    pass


class Toolbox:
    def __init__(self, jira: Client, people: People, today: date, access: Access | None = None,
                 finder: Finder | None = None, tracing: Tracing | None = None, teams: list[str] | None = None) -> None:
        self._jira, self._people, self._today = jira, people, today
        self.teams = teams or []  # every team there is, for people who see all of them
        self.tracing = tracing or Tracing()
        self.finder = finder or Finder()  # words only unless an embedder is passed in
        self.access = access or Access()  # without a store: the role each HR record implies
        self._fit: Fit | None = None
        self._read_tools: list[dict] | None = None

    def similar(self, text: str, world: World) -> list[Match]:
        """Open issues of the team that look like ``text``."""
        return self.finder.search(text, [(i.key, i.summary) for i in world.issues.values() if i.status != "Done"])

    def forecast(self, world: World) -> Forecast:
        self._fit = self._fit or default_fit()
        return forecast(world.overview, list(world.issues.values()), self._fit)

    async def definitions(self, direct_changes: bool = False) -> list[dict]:
        """Tool list for the model. Stable order, so the prompt prefix stays cacheable.
        ``direct_changes`` swaps the single plan tool for one tool per kind of change (react mode)."""
        if self._read_tools is None:
            listed = (await self._jira.list_tools()).tools
            self._read_tools = [
                {"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
                for t in sorted(listed, key=lambda t: t.name)
                if t.annotations is not None and t.annotations.read_only_hint
            ]
        delegate = [PEOPLE_TOOL] if hasattr(self._people, "ask") else []  # only when People is another agent
        changes = list(CHANGE_TOOLS.values()) if direct_changes else [PLAN_TOOL]
        return [OVERVIEW_TOOL, FORECAST_TOOL, SIMILAR_TOOL, *self._read_tools, *delegate, *changes]

    async def _call_jira(self, name: str, arguments: dict):
        result = await self._jira.call_tool(name, arguments)
        if result.is_error:
            raise ToolFailure(" ".join(b.text for b in result.content if getattr(b, "text", None)))
        payload = result.structured_content or {}
        return payload["result"] if set(payload) == {"result"} else payload

    async def world(self, actor_id: str, team: str | None = None) -> World:
        """The state of a team as ``actor_id`` sees it: their own team, or another one they have the right to see."""
        try:
            return await self._world(actor_id, team)
        except (PeopleAgentError, httpx.HTTPError) as exc:
            raise ToolFailure(f"The People agent is unavailable: {exc}") from exc

    async def _world(self, actor_id: str, team: str | None) -> World:
        actor = await self._people.person(actor_id)
        if actor is None:
            raise ToolFailure(f"Unknown employee {actor_id}.")
        team = team or actor.team
        rights = self.access.rights(actor, team)
        if team != actor.team and "overview.view" not in rights:
            raise PermissionError(f"{actor.name} does not have the right to see the work of team {team}.")
        members = await self._people.team(team) if team else []
        self.tracing.masker.learn([actor, *members])  # so their names never reach a trace
        availability = await self._people.availability([p.id for p in members])
        issues = [Issue(**i) for i in await self._call_jira("search_issues", {"project": team})] if team else []
        return World(actor=actor, team=team, rights=rights, overview=build(members, availability, issues, self._today),
                     issues={i.key: i for i in issues})

    async def _views(self, actor_id: str) -> list[World]:
        """The teams a question is about: the person's own, or every team for someone who belongs to none
        and has the right to see them all."""
        own = await self.world(actor_id)
        if own.team is None and "company.view" in own.rights:
            return [await self.world(actor_id, team) for team in self.teams]
        return [own]

    async def _readable(self, actor_id: str) -> set[str]:
        """Teams whose issues this person may read: their own, and any other where a role lets them look."""
        actor = await self._people.person(actor_id)
        if actor is None:
            raise ToolFailure(f"Unknown employee {actor_id}.")
        return {t for t in {actor.team, *self.teams} if t and "overview.view" in self.access.rights(actor, t)}

    async def run(self, name: str, arguments: dict, actor_id: str) -> str:
        """Run a read tool for the model and return compact text. Raises ToolFailure with a readable reason."""
        with self.tracing.observe(name, "tool", input=arguments) as span:
            try:
                result = await self._run(name, arguments, actor_id)
            except ToolFailure as exc:
                span.update(output=str(exc), level="WARNING")
                raise
            span.update(output=result)
            return result

    async def _run(self, name: str, arguments: dict, actor_id: str) -> str:
        if name == "team_overview":
            views = await self._views(actor_id)
            if len(views) == 1:
                return json.dumps(_seen(views[0].overview), ensure_ascii=False)
            return json.dumps({"teams": [{"team": w.team, **_seen(w.overview)} for w in views]}, ensure_ascii=False)
        if name == SIMILAR_TOOL["name"]:
            found = self.similar(arguments["text"], await self.world(actor_id))
            return json.dumps({"possible_duplicates": [m.model_dump() for m in found]}, ensure_ascii=False)
        if name == FORECAST_TOOL["name"]:
            views = await self._views(actor_id)
            if len(views) == 1:
                return self.forecast(views[0]).model_dump_json()
            return json.dumps({"teams": [{"team": w.team, **self.forecast(w).model_dump()} for w in views]}, ensure_ascii=False)
        if name == PEOPLE_TOOL["name"] and hasattr(self._people, "ask"):
            try:
                return await self._people.ask(arguments["question"])
            except (PeopleAgentError, httpx.HTTPError) as exc:
                raise ToolFailure(f"The People agent is unavailable: {exc}") from exc
        if name not in {t["name"] for t in await self.definitions()} or name == PLAN_TOOL["name"]:
            raise ToolFailure(f"Tool '{name}' is not available. To change Jira, use propose_plan.")
        # Jira would answer for any project. The model reads only what the person it works for may see.
        readable = await self._readable(actor_id)
        other = f"That belongs to another team; this person may read the issues of: {', '.join(sorted(readable)) or 'no team'}."
        if arguments.get("project") and arguments["project"] not in readable:
            raise ToolFailure(other)
        data = await self._call_jira(name, arguments)
        if isinstance(data, list):
            data = [i for i in data if i["project"] in readable]
            rows = [_brief(i) for i in data[:MAX_ROWS]]
            more = len(data) - len(rows)
            return json.dumps({"issues": rows, **({"not_shown": more, "hint": "Narrow the filters."} if more else {})},
                              ensure_ascii=False)
        if "key" in data and data["project"] not in readable:
            raise ToolFailure(other)
        return json.dumps(_brief(data) if "key" in data else data, ensure_ascii=False)

    async def create_in_sprint(self, team: str, summary: str, story_points: int) -> Issue:
        return Issue(**await self._call_jira("create_issue", {"project": team, "summary": summary,
                                                              "story_points": story_points, "in_sprint": True}))

    async def execute(self, step: Step, actor: Person) -> str:
        """Apply one approved step in Jira and leave a comment saying who asked and why."""
        with self.tracing.observe(f"jira:{step.tool}", "tool", input=step.model_dump()) as span:
            try:
                done = await self._execute(step, actor)
            except ToolFailure as exc:
                span.update(output=str(exc), level="ERROR")
                raise
            span.update(output=done)
            return done

    async def _execute(self, step: Step, actor: Person) -> str:
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


def _seen(overview) -> dict:
    """A team as the model sees it: people with load and whether they are in, and the problems found."""
    people = [{"id": x.employee_id, "name": x.name, "role": x.role, "open_issues": x.open_issues,
               "sprint_points": x.sprint_points, "capacity_points": x.capacity_points, "load": x.load,
               "status": "left the company" if not x.availability.active
               else "available" if x.availability.available_today
               else f"away until {x.availability.unavailable_until}"} for x in overview.loads]
    findings = [{"kind": f.kind, "employee_id": f.employee_id, "issues": f.issue_keys, "detail": f.detail}
                for f in overview.findings]
    return {"today": str(overview.today), "people": people, "findings": findings}


def _brief(issue: dict) -> dict:
    return {"key": issue["key"], "summary": issue["summary"], "type": issue["type"], "status": issue["status"],
            "assignee_id": issue["assignee_id"], "points": issue["story_points"], "in_sprint": issue["in_sprint"],
            "updated": issue["updated"]}


def parse_plan(arguments: dict) -> Plan:
    return Plan.model_validate(arguments)
