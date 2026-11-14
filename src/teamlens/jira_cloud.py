"""Jira Cloud client behind the same interface as the fake.

Nothing here depends on the site language: issue types, statuses and custom fields are
discovered once by their stable identifiers, not by display names.
"""

from __future__ import annotations

import asyncio
from datetime import date

import httpx

from teamlens.config import Settings
from teamlens.jira import STATUSES, JiraError
from teamlens.seed import Issue

SPRINT_FIELD = "com.pyxis.greenhopper.jira:gh-sprint"
POINTS_FIELDS = ("com.pyxis.greenhopper.jira:jsw-story-points", "com.atlassian.jira.plugin.system.customfieldtypes:float")
RETRIES = 3


class JiraCloud:
    def __init__(self, settings: Settings, users: dict[str, str] | None = None,
                 client: httpx.AsyncClient | None = None) -> None:
        """``users`` maps HRM employee ids to Jira account ids."""
        self._project = settings.jira_project_key
        self._team = settings.jira_team_key
        self._account = dict(users or {})
        self._employee = {v: k for k, v in self._account.items()}
        self._http = client or httpx.AsyncClient(
            base_url=settings.jira_base_url.rstrip("/"),
            auth=(settings.jira_email, settings.jira_api_token), timeout=30,
        )
        self._ready = False

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _call(self, method: str, path: str, **kwargs) -> dict | list | None:
        for attempt in range(RETRIES):
            try:
                r = await self._http.request(method, path, **kwargs)
            except httpx.TransportError as exc:
                if attempt == RETRIES - 1:
                    raise JiraError(f"Jira is unreachable: {exc}") from exc
                await asyncio.sleep(2 ** attempt)
                continue
            if r.status_code in (429, 502, 503, 504) and attempt < RETRIES - 1:
                await asyncio.sleep(float(r.headers.get("Retry-After", 2 ** attempt)))
                continue
            if r.is_success:
                return r.json() if r.content else None
            try:
                body = r.json()
                reasons = body.get("errorMessages", []) + list(body.get("errors", {}).values())
            except ValueError:
                reasons = []
            raise JiraError("; ".join(map(str, reasons)) or f"Jira answered {r.status_code}.")
        raise JiraError("Jira kept refusing the request.")

    async def _setup(self) -> None:
        if self._ready:
            return
        fields, project, statuses = await asyncio.gather(
            self._call("GET", "/rest/api/3/field"),
            self._call("GET", f"/rest/api/3/project/{self._project}"),
            self._call("GET", f"/rest/api/3/project/{self._project}/statuses"),
        )
        custom = {f["schema"]["custom"]: f["id"] for f in fields if f.get("schema", {}).get("custom")}
        self._sprint_field = custom[SPRINT_FIELD]
        self._points_field = next(f["id"] for f in fields if f.get("schema", {}).get("custom") == POINTS_FIELDS[0])

        # This endpoint, unlike the project one, also returns the name in English.
        types = await self._call("GET", "/rest/api/3/issuetype/project", params={"projectId": project["id"]})
        standard = [t for t in types if t.get("hierarchyLevel", 0) == 0]
        by_name = {t.get("untranslatedName") or t["name"]: t["id"] for t in standard}
        self._type_id = {name: by_name.get(name, standard[0]["id"]) for name in ("Story", "Bug", "Task")}
        self._type_name = {tid: name for name, tid in self._type_id.items()}

        # Workflow order: one "new" status, the in-between ones, one "done" status.
        ordered = list({s["id"]: s for t in statuses for s in t["statuses"]}.values())
        middle = [s["id"] for s in ordered if s["statusCategory"]["key"] == "indeterminate"]
        self._status_id = {
            "To Do": next(s["id"] for s in ordered if s["statusCategory"]["key"] == "new"),
            "In Progress": middle[0],
            "In Review": middle[-1],
            "Done": next(s["id"] for s in ordered if s["statusCategory"]["key"] == "done"),
        }
        self._status_name = {sid: name for name, sid in reversed(self._status_id.items())}
        self._fields = ["summary", "issuetype", "status", "assignee", "created", "updated",
                        self._sprint_field, self._points_field]
        self._ready = True

    def _issue(self, raw: dict) -> Issue:
        f = raw["fields"]
        account = (f.get("assignee") or {}).get("accountId")
        return Issue(
            key=raw["key"], project=self._team, summary=f["summary"],
            type=self._type_name.get(f["issuetype"]["id"], "Task"),
            status=self._status_name.get(f["status"]["id"], "In Progress"),
            assignee_id=self._employee.get(account, account),
            story_points=int(f.get(self._points_field) or 0),
            in_sprint=any(s.get("state") == "active" for s in f.get(self._sprint_field) or []),
            created=date.fromisoformat(f["created"][:10]), updated=date.fromisoformat(f["updated"][:10]),
        )

    def _account_id(self, assignee_id: str | None) -> str | None:
        return None if assignee_id is None else self._account.get(assignee_id, assignee_id)

    async def search(self, project=None, assignee_id=None, status=None, sprint_only=False) -> list[Issue]:
        await self._setup()
        if project not in (None, self._team):
            return []
        jql = [f'project = "{self._project}"']
        if assignee_id:
            jql.append(f'assignee = "{self._account_id(assignee_id)}"')
        if status:
            if status not in STATUSES:
                raise JiraError(f"Unknown status '{status}'. Use one of: {', '.join(STATUSES)}.")
            jql.append(f"status = {self._status_id[status]}")
        if sprint_only:
            jql.append("sprint in openSprints()")
        issues, token = [], None
        while True:
            body = {"jql": " AND ".join(jql) + " ORDER BY key", "fields": self._fields, "maxResults": 100}
            if token:
                body["nextPageToken"] = token
            page = await self._call("POST", "/rest/api/3/search/jql", json=body)
            issues += [self._issue(i) for i in page.get("issues", [])]
            token = page.get("nextPageToken")
            if not token:
                return issues

    async def get(self, key: str) -> Issue:
        await self._setup()
        try:
            raw = await self._call("GET", f"/rest/api/3/issue/{key}", params={"fields": ",".join(self._fields)})
        except JiraError as exc:
            raise JiraError(f"Issue {key} does not exist.") from exc
        return self._issue(raw)

    async def assign(self, key: str, assignee_id: str | None) -> Issue:
        await self._setup()
        await self._call("PUT", f"/rest/api/3/issue/{key}/assignee", json={"accountId": self._account_id(assignee_id)})
        return await self.get(key)

    async def transition(self, key: str, status: str) -> Issue:
        await self._setup()
        if status not in STATUSES:
            raise JiraError(f"Unknown status '{status}'. Use one of: {', '.join(STATUSES)}.")
        options = await self._call("GET", f"/rest/api/3/issue/{key}/transitions")
        match = [t["id"] for t in options["transitions"] if t["to"]["id"] == self._status_id[status]]
        if not match:
            raise JiraError(f"{key} cannot move to '{status}' from its current status.")
        await self._call("POST", f"/rest/api/3/issue/{key}/transitions", json={"transition": {"id": match[0]}})
        return await self.get(key)

    async def comment(self, key: str, text: str) -> None:
        doc = {"type": "doc", "version": 1,
               "content": [{"type": "paragraph", "content": [{"type": "text", "text": text}]}]}
        await self._call("POST", f"/rest/api/3/issue/{key}/comment", json={"body": doc})

    async def _active_sprint(self) -> int:
        boards = await self._call("GET", "/rest/agile/1.0/board", params={"projectKeyOrId": self._project})
        board = boards["values"][0]["id"]
        sprints = await self._call("GET", f"/rest/agile/1.0/board/{board}/sprint", params={"state": "active"})
        if not sprints["values"]:
            raise JiraError("There is no active sprint.")
        return sprints["values"][0]["id"]

    async def create(self, project, summary, type="Task", story_points=1, assignee_id=None, in_sprint=False) -> Issue:
        await self._setup()
        if project != self._team:
            raise JiraError(f"Project {project} does not exist.")
        fields = {"project": {"key": self._project}, "summary": summary,
                  "issuetype": {"id": self._type_id.get(type, self._type_id["Task"])},
                  self._points_field: story_points}
        if assignee_id:
            fields["assignee"] = {"accountId": self._account_id(assignee_id)}
        created = await self._call("POST", "/rest/api/3/issue", json={"fields": fields})
        if in_sprint:
            await self._call("POST", f"/rest/agile/1.0/sprint/{await self._active_sprint()}/issue",
                             json={"issues": [created["key"]]})
        return await self.get(created["key"])

    async def delete(self, key: str) -> None:
        """Used by the demo seeder and tests to clean up; not exposed as an agent tool."""
        await self._call("DELETE", f"/rest/api/3/issue/{key}")
