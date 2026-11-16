"""Jira through mcp-atlassian, the open-source MCP server for Atlassian, behind the same interface as the fake.

That server offers a lot: 63 Jira tools in version 0.24, and by default eleven fields per issue with a person's
name, e-mail and avatar inside. TeamLens narrows it in three places:

- the server is started with ENABLED_TOOLS, so only the tools in ``TOOLS`` exist at all;
- JIRA_PROJECTS_FILTER keeps every search inside one project;
- every read names the fields it wants, and from the answer only what an ``Issue`` holds is kept.

Nothing here depends on the site language: statuses are told apart by their category, issue types by their
untranslated name, custom fields by their type.
"""

from __future__ import annotations

import json
import os
import shlex
from contextlib import asynccontextmanager
from datetime import date

from mcp import StdioServerParameters
from mcp.client import Client

from teamlens.config import Settings
from teamlens.jira import STATUSES, JiraError
from teamlens.seed import Issue

TOOLS = ("jira_search", "jira_get_issue", "jira_search_fields", "jira_get_project_issue_types", "jira_get_transitions",
         "jira_assign_issue", "jira_transition_issue", "jira_add_comment", "jira_create_issue",
         "jira_get_agile_boards", "jira_get_sprints_from_board", "jira_add_issues_to_sprint")
FIELDS = ("summary", "status", "issuetype", "assignee", "created", "updated")  # plus two custom fields, found at start
SPRINT_FIELD = "com.pyxis.greenhopper.jira:gh-sprint"
POINTS_FIELD = "com.pyxis.greenhopper.jira:jsw-story-points"
PAGE = 50  # the most the server returns per search


def server_env(settings: Settings, tools: tuple[str, ...] = TOOLS) -> dict[str, str]:
    """What the server process is started with: where Jira is, and how little of it to offer."""
    return {"JIRA_URL": settings.jira_base_url, "JIRA_USERNAME": settings.jira_email,
            "JIRA_API_TOKEN": settings.jira_api_token, "JIRA_PROJECTS_FILTER": settings.jira_project_key,
            "TOOLSETS": "all", "ENABLED_TOOLS": ",".join(tools), "PYTHONIOENCODING": "utf-8"}


@asynccontextmanager
async def connect(settings: Settings, extra_tools: tuple[str, ...] = ()):
    """An MCP client connected to mcp-atlassian. ``TEAMLENS_JIRA_MCP`` is either the URL of a running server
    (it then has its own environment, see docker-compose.jira.yml) or the command that starts one over stdio."""
    target = settings.teamlens_jira_mcp
    if target.startswith(("http://", "https://")):
        server: str | StdioServerParameters = target
    else:
        command, *args = shlex.split(target, posix=os.name != "nt")
        server = StdioServerParameters(command=command, args=args, env=server_env(settings, TOOLS + extra_tools))
    async with Client(server, mode="legacy") as client:
        yield client


class AtlassianJira:
    def __init__(self, client: Client, settings: Settings, users: dict[str, str] | None = None) -> None:
        """``users`` maps HRM employee ids to Jira account ids."""
        self._client = client
        self._project = settings.jira_project_key
        self._team = settings.jira_team_key
        self._account = dict(users or {})
        self._employee = {v: k for k, v in self._account.items()}
        self._status_id: dict[str, str] = {}
        self._status_name: dict[str, str] = {}
        self._ready = False

    async def _call(self, tool: str, **arguments):
        try:
            result = await self._client.call_tool(tool, arguments)
        except Exception as exc:  # the connection, not Jira's answer
            raise JiraError(f"Jira is unreachable: {exc}") from exc
        text = "".join(getattr(part, "text", "") for part in result.content)
        if result.is_error:
            raise JiraError(text)
        try:
            return json.loads(text)
        except ValueError:
            return text

    async def _field(self, keyword: str, kind: str) -> str:
        found = await self._call("jira_search_fields", keyword=keyword, limit=20)
        return next(f["id"] for f in found if f.get("schema", {}).get("custom") == kind)

    async def _setup(self) -> None:
        if self._ready:
            return
        self._sprint_field = await self._field("sprint", SPRINT_FIELD)
        self._points_field = await self._field("story point", POINTS_FIELD)
        types = await self._call("jira_get_project_issue_types", project_key=self._project)
        local = {t.get("untranslated_name") or t["name"]: t["name"] for t in types if not t.get("subtask")}
        self._type_local = {name: local.get(name, next(iter(local.values()))) for name in ("Story", "Bug", "Task")}
        self._type_name = {shown: name for name, shown in self._type_local.items()}
        self._fields = ",".join((*FIELDS, self._sprint_field, self._points_field))
        self._ready = True

    def _learn(self, transitions: list[dict]) -> None:
        """Workflow order from one issue's transitions: one "new" status, the in-between ones, one "done"."""
        if self._status_id or not transitions:
            return
        to = list({t["to"]["id"]: t["to"] for t in transitions}.values())
        middle = [s for s in to if s["statusCategory"]["key"] == "indeterminate"]
        ours = {"To Do": next(s for s in to if s["statusCategory"]["key"] == "new"),
                "In Progress": middle[0], "In Review": middle[-1],
                "Done": next(s for s in to if s["statusCategory"]["key"] == "done")}
        self._status_id = {name: s["id"] for name, s in ours.items()}
        self._status_name = {s["name"]: name for name, s in reversed(ours.items())}

    async def _statuses(self, key: str | None = None) -> bool:
        """Statuses are learnt from an issue. False while the project has none."""
        if not self._status_id:
            if key is None:
                page = await self._call("jira_search", jql=f'project = "{self._project}"', fields="summary", limit=1)
                key = page["issues"][0]["key"] if page["issues"] else None
            if key:
                self._learn(await self._call("jira_get_transitions", issue_key=key))
        return bool(self._status_id)

    def _issue(self, raw: dict) -> Issue:
        """The one place an answer of the server becomes ours. Whatever else came with it stops here."""
        account = (raw.get("assignee") or {}).get("account_id")
        sprints = (raw.get(self._sprint_field) or {}).get("value") or []
        return Issue(
            key=raw["key"], project=self._team, summary=raw["summary"],
            type=self._type_name.get(raw["issue_type"]["name"], "Task"),
            status=self._status_name.get(raw["status"]["name"], "In Progress"),
            assignee_id=self._employee.get(account, account),
            story_points=int((raw.get(self._points_field) or {}).get("value") or 0),
            in_sprint=any(s.get("state") == "active" for s in sprints),
            created=date.fromisoformat(raw["created"][:10]), updated=date.fromisoformat(raw["updated"][:10]),
        )

    def _account_id(self, assignee_id: str | None) -> str | None:
        return None if assignee_id is None else self._account.get(assignee_id, assignee_id)

    async def search(self, project=None, assignee_id=None, status=None, sprint_only=False) -> list[Issue]:
        await self._setup()
        if status and status not in STATUSES:
            raise JiraError(f"Unknown status '{status}'. Use one of: {', '.join(STATUSES)}.")
        if project not in (None, self._team) or not await self._statuses():
            return []
        jql = [f'project = "{self._project}"']
        if assignee_id:
            jql.append(f'assignee = "{self._account_id(assignee_id)}"')
        if status:
            jql.append(f"status = {self._status_id[status]}")
        if sprint_only:
            jql.append("sprint in openSprints()")
        issues, token = [], None
        while True:
            more = {"page_token": token} if token else {}
            page = await self._call("jira_search", jql=" AND ".join(jql) + " ORDER BY key", fields=self._fields,
                                    limit=PAGE, **more)
            issues += [self._issue(i) for i in page["issues"]]
            token = page.get("next_page_token")
            if not token or not page["issues"]:
                return issues

    async def get(self, key: str) -> Issue:
        await self._setup()
        try:
            raw = await self._call("jira_get_issue", issue_key=key, fields=self._fields, comment_limit=0,
                                   update_history=False)
        except JiraError as exc:
            raise JiraError(f"Issue {key} does not exist.") from exc
        await self._statuses(key)
        return self._issue(raw)

    async def assign(self, key: str, assignee_id: str | None) -> Issue:
        await self._call("jira_assign_issue", issue_key=key, assignee=self._account_id(assignee_id))
        return await self.get(key)

    async def transition(self, key: str, status: str) -> Issue:
        if status not in STATUSES:
            raise JiraError(f"Unknown status '{status}'. Use one of: {', '.join(STATUSES)}.")
        options = await self._call("jira_get_transitions", issue_key=key)
        self._learn(options)
        match = [t["id"] for t in options if t["to"]["id"] == self._status_id.get(status)]
        if not match:
            raise JiraError(f"{key} cannot move to '{status}' from its current status.")
        await self._call("jira_transition_issue", issue_key=key, transition_id=match[0])
        return await self.get(key)

    async def comment(self, key: str, text: str) -> None:
        await self._call("jira_add_comment", issue_key=key, body=text)

    async def _active_sprint(self) -> str:
        boards = await self._call("jira_get_agile_boards", project_key=self._project)
        sprints = await self._call("jira_get_sprints_from_board", board_id=str(boards[0]["id"]), state="active")
        if not sprints:
            raise JiraError("There is no active sprint.")
        return str(sprints[0]["id"])

    async def create(self, project, summary, type="Task", story_points=1, assignee_id=None, in_sprint=False) -> Issue:
        await self._setup()
        if project != self._team:
            raise JiraError(f"Project {project} does not exist.")
        who = {"assignee": self._account_id(assignee_id)} if assignee_id else {}
        made = await self._call("jira_create_issue", project_key=self._project, summary=summary,
                                issue_type=self._type_local.get(type, self._type_local["Task"]),
                                additional_fields=json.dumps({self._points_field: story_points}), **who)
        key = made["issue"]["key"]
        if in_sprint:
            await self._call("jira_add_issues_to_sprint", sprint_id=await self._active_sprint(), issue_keys=key)
        return await self.get(key)

    async def delete(self, key: str) -> None:
        """Used by the live test to clean up. The server offers this tool only when asked to: see ``connect``."""
        await self._call("jira_delete_issue", issue_key=key)
