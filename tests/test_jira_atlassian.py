"""Jira through the mcp-atlassian server.

The tests talk to a stand-in server over the real MCP protocol. It has the tool names and arguments of
mcp-atlassian 0.24 and answers in the shapes recorded from a real Jira Cloud site set to Russian: the site
language, people's names, e-mails and avatars are all in there, as they are in real answers.

The last test runs against a real site through the real server. Opt-in: TEAMLENS_TEST_JIRA_MCP=1, a filled
.env, and TEAMLENS_JIRA_MCP set to the command that starts mcp-atlassian.
"""

import inspect
import json
import os
from pathlib import Path

import pytest
from mcp.client import Client
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from teamlens.config import Settings
from teamlens.jira import JiraError
from teamlens.jira_atlassian import FIELDS, TOOLS, AtlassianJira, connect, server_env

SPRINT, POINTS = "customfield_10020", "customfield_10016"
STATUS = {"10001": ("К выполнению", "new"), "3": ("В работе", "indeterminate"),
          "10002": ("В процессе проверки", "indeterminate"), "10000": ("Готово", "done")}
TRANSITION = {"11": "10001", "21": "3", "31": "10002", "41": "10000"}
TYPES = {"Задача": "Task", "История": "Story", "Баг": "Bug"}
DANA, ME = "712020:dana", "712020:me"
USERS = {"E001": ME, "E002": DANA}
CUSTOM = {SPRINT: ("Спринт", "com.pyxis.greenhopper.jira:gh-sprint"),
          POINTS: ("Story point estimate", "com.pyxis.greenhopper.jira:jsw-story-points")}


class Site:
    """A Jira site as mcp-atlassian shows it, and a record of every call made to it."""

    def __init__(self, issues: int = 3) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.issues: dict[str, dict] = {}
        self.comments: dict[str, list[str]] = {}
        self.page = 50
        for n in range(issues):
            self.add(f"Issue {n + 1}", "Баг", points=2, assignee=DANA if n else None, sprint=n < 2)

    def add(self, summary, type, points=1, assignee=None, sprint=False) -> str:
        key = f"SCRUM-{len(self.issues) + 1}"
        self.issues[key] = {"summary": summary, "type": type, "status": "10001", "assignee": assignee,
                            "points": points, "sprint": sprint}
        return key

    def known(self, key: str) -> dict:
        if key not in self.issues:
            raise ToolError(f"Issue {key} not found. Verify the issue key and project access.")
        return self.issues[key]

    def shown(self, key: str, fields: str) -> dict:
        i = self.issues[key]
        person = {"account_id": i["assignee"], "display_name": "Dana Whitfield", "name": "Dana Whitfield",
                  "email": "dana@example.com", "avatar_url": "https://secure.gravatar.com/avatar/be745a"}
        name = STATUS[i["status"]][0]
        sprint = [{"id": 1, "name": "SCRUM Sprint 1", "state": "active", "boardId": 1}] if i["sprint"] else None
        every = {"summary": i["summary"], "status": {"name": name, "category": name, "color": "yellow"},
                 "issue_type": {"name": i["type"]}, "assignee": person if i["assignee"] else None,
                 "reporter": person | {"account_id": ME}, "priority": {"name": "Medium"}, "description": "Private notes.",
                 "created": "2026-11-14 18:46:07 GMT Standard Time", "updated": "2026-11-15 09:01:38 GMT Standard Time",
                 POINTS: {"value": float(i["points"])}, SPRINT: {"value": sprint}}
        asked = [{"issuetype": "issue_type"}.get(f, f) for f in fields.split(",")]
        head = {"id": key[6:], "key": key, "browse_url": f"https://site.atlassian.net/browse/{key}"}
        return head | {f: every[f] for f in asked if f in every}

    def server(self) -> MCPServer:
        server, site = MCPServer("Atlassian MCP", log_level="WARNING"), self

        def tool(fn):
            """Registers ``fn`` under its own name; the tool records the call and answers with JSON text."""
            async def recorded(**arguments):
                site.calls.append((fn.__name__, arguments))
                return json.dumps(fn(**arguments), ensure_ascii=False)
            recorded.__name__ = fn.__name__
            recorded.__signature__ = inspect.signature(fn).replace(return_annotation=str)
            recorded.__annotations__ = {**fn.__annotations__, "return": str}
            server.tool(name=fn.__name__)(recorded)
            return fn

        @tool
        def jira_search_fields(keyword: str = "", limit: int = 10):
            other = [{"id": "timespent", "name": "Затраченное время", "schema": {"type": "number", "system": "timespent"}}]
            return other + [{"id": k, "name": n, "custom": True, "schema": {"custom": c}} for k, (n, c) in CUSTOM.items()]

        @tool
        def jira_get_project_issue_types(project_key: str):
            other = [{"id": "10001", "name": "Эпик", "subtask": False, "untranslated_name": "Epic"},
                     {"id": "10002", "name": "Subtask", "subtask": True, "untranslated_name": "Subtask"}]
            return other + [{"id": str(n), "name": shown, "subtask": False, "untranslated_name": name}
                            for n, (shown, name) in enumerate(TYPES.items())]

        @tool
        def jira_search(jql: str, fields: str = "summary,description,reporter,assignee,status", limit: int = 10,
                        page_token: str | None = None):
            keys = list(site.issues)
            if "assignee = " in jql:
                keys = [k for k in keys if f'assignee = "{site.issues[k]["assignee"]}"' in jql]
            if "status = " in jql:
                keys = [k for k in keys if f"status = {site.issues[k]['status']} " in jql]
            if "openSprints()" in jql:
                keys = [k for k in keys if site.issues[k]["sprint"]]
            start, size = int(page_token or 0), min(limit, site.page)
            more = {"next_page_token": str(start + size)} if start + size < len(keys) else {}
            return {"total": -1, "start_at": 0, "max_results": limit, **more,
                    "issues": [site.shown(k, fields) for k in keys[start:start + size]]}

        @tool
        def jira_get_issue(issue_key: str, fields: str = "summary,description,reporter,assignee,status",
                           comment_limit: int = 10, update_history: bool = True):
            site.known(issue_key)
            return site.shown(issue_key, fields)

        @tool
        def jira_get_transitions(issue_key: str):
            site.known(issue_key)
            return [{"id": t, "name": STATUS[s][0], "to_status": STATUS[s][0],
                     "to": {"id": s, "name": STATUS[s][0], "statusCategory": {"key": STATUS[s][1]}}}
                    for t, s in TRANSITION.items()]

        @tool
        def jira_assign_issue(issue_key: str, assignee: str | None = None):
            site.known(issue_key)["assignee"] = assignee
            return {"message": "Issue assigned"}

        @tool
        def jira_transition_issue(issue_key: str, transition_id: str):
            site.known(issue_key)["status"] = TRANSITION[transition_id]
            return {"message": "Issue transitioned"}

        @tool
        def jira_add_comment(issue_key: str, body: str):
            site.known(issue_key)
            site.comments.setdefault(issue_key, []).append(body)
            return {"id": "1", "body": body}

        @tool
        def jira_create_issue(project_key: str, summary: str, issue_type: str, assignee: str | None = None,
                              additional_fields: dict | str | None = None):
            if issue_type not in TYPES:
                raise ToolError(f"Issue type '{issue_type}' does not exist in this project.")
            extra = json.loads(additional_fields) if isinstance(additional_fields, str) else additional_fields
            key = site.add(summary, issue_type, extra[POINTS], assignee)
            return {"message": "Issue created successfully", "issue": {"id": key[6:], "key": key, "summary": summary}}

        @tool
        def jira_get_agile_boards(project_key: str):
            return [{"id": "1", "name": "SCRUM board", "type": "simple"}]

        @tool
        def jira_get_sprints_from_board(board_id: str, state: str | None = None):
            return [{"id": "1", "name": "SCRUM Sprint 1", "state": "active"}]

        @tool
        def jira_add_issues_to_sprint(sprint_id: str, issue_keys: str):
            for key in issue_keys.split(","):
                site.known(key)["sprint"] = True
            return {"message": f"Issues added to sprint {sprint_id}"}

        return server


def settings(**kw) -> Settings:
    return Settings(_env_file=None, jira_project_key="SCRUM", jira_team_key="PAY", **kw)


async def test_every_read_names_its_fields_and_only_an_issue_comes_out():
    site = Site()
    async with Client(site.server()) as client:
        jira = AtlassianJira(client, settings(), USERS)
        found = await jira.search()
        one = await jira.get("SCRUM-2")

    wanted = ",".join((*FIELDS, SPRINT, POINTS))
    reads = [(tool, a["fields"]) for tool, a in site.calls if tool in ("jira_search", "jira_get_issue")]
    # The first search only finds one issue key, to learn the site's statuses from.
    assert reads == [("jira_search", "summary"), ("jira_search", wanted), ("jira_get_issue", wanted)]

    assert [i.key for i in found] == ["SCRUM-1", "SCRUM-2", "SCRUM-3"] and found[1] == one
    assert one.model_dump(mode="json") == {
        "key": "SCRUM-2", "project": "PAY", "summary": "Issue 2", "type": "Bug", "status": "To Do", "assignee_id": "E002",
        "story_points": 2, "in_sprint": True, "created": "2026-11-14", "updated": "2026-11-15"}
    assert found[0].assignee_id is None and not found[2].in_sprint
    carried = json.dumps([i.model_dump(mode="json") for i in found])  # what the server adds on its own stops at the adapter
    assert not any(x in carried for x in ("Dana", "example.com", "gravatar", "browse", "Medium"))


async def test_the_site_language_does_not_matter_and_filters_become_one_query():
    site = Site()
    async with Client(site.server()) as client:
        jira = AtlassianJira(client, settings(), USERS)
        assert (await jira.transition("SCRUM-3", "In Review")).status == "In Review"  # "В процессе проверки" on the site
        assert site.issues["SCRUM-3"]["status"] == "10002"
        assert [i.key for i in await jira.search(assignee_id="E002", status="In Review")] == ["SCRUM-3"]
        assert [i.key for i in await jira.search(sprint_only=True)] == ["SCRUM-1", "SCRUM-2"]
        assert await jira.search(project="KYC") == []  # one project is connected; others have nothing here
        with pytest.raises(JiraError, match="Unknown status 'Shipped'"):
            await jira.search(status="Shipped")
    jql = [a["jql"] for tool, a in site.calls if tool == "jira_search"]
    assert 'project = "SCRUM" AND assignee = "712020:dana" AND status = 10002 ORDER BY key' in jql


async def test_changes_go_through_the_servers_tools():
    site = Site()
    async with Client(site.server()) as client:
        jira = AtlassianJira(client, settings(), USERS)
        made = await jira.create("PAY", "Retry failed payouts", type="Story", story_points=5, assignee_id="E001", in_sprint=True)
        assert (made.key, made.type, made.story_points, made.assignee_id, made.in_sprint) == ("SCRUM-4", "Story", 5, "E001", True)
        assert site.issues["SCRUM-4"]["type"] == "История"  # created under the name the site uses

        assert (await jira.assign(made.key, "E002")).assignee_id == "E002"
        assert (await jira.assign(made.key, None)).assignee_id is None
        for status in ("In Progress", "In Review", "Done", "To Do"):
            assert (await jira.transition(made.key, status)).status == status
        await jira.comment(made.key, "Reassigned while Dana is away.")
        assert site.comments == {"SCRUM-4": ["Reassigned while Dana is away."]}

        with pytest.raises(JiraError, match="Issue SCRUM-99 does not exist"):
            await jira.get("SCRUM-99")
        with pytest.raises(JiraError, match="Unknown status"):
            await jira.transition(made.key, "Shipped")
        with pytest.raises(JiraError, match="Project KYC does not exist"):
            await jira.create("KYC", "Elsewhere")
    assert {tool for tool, _ in site.calls} <= set(TOOLS)  # nothing outside the list the server is started with


async def test_a_long_project_is_read_page_by_page_and_an_empty_one_is_just_empty():
    site = Site(issues=7)
    site.page = 3
    async with Client(site.server()) as client:
        assert len(await AtlassianJira(client, settings(), USERS).search()) == 7
    assert [a.get("page_token") for tool, a in site.calls if tool == "jira_search"] == [None, None, "3", "6"]

    empty = Site(issues=0)
    async with Client(empty.server()) as client:
        jira = AtlassianJira(client, settings(), USERS)
        assert await jira.search() == [] and await jira.search(status="Done") == []
        assert (await jira.create("PAY", "First issue")).status == "To Do"  # statuses are learnt from the first issue


def test_the_server_is_started_with_the_least_it_needs():
    env = server_env(settings(jira_base_url="https://site.atlassian.net", jira_email="me@example.com", jira_api_token="t"))
    assert env["ENABLED_TOOLS"].split(",") == list(TOOLS) and len(TOOLS) == 12
    assert env["JIRA_PROJECTS_FILTER"] == "SCRUM" and "jira_delete_issue" not in env["ENABLED_TOOLS"]


def test_the_compose_file_starts_the_server_with_the_same_tools():
    compose = (Path(__file__).parent.parent / "docker-compose.jira.yml").read_text(encoding="utf-8")
    listed = next(line.split(":", 1)[1].strip() for line in compose.splitlines() if line.strip().startswith("ENABLED_TOOLS:"))
    assert listed.split(",") == list(TOOLS)
    web = compose[compose.index("  web:"):]
    assert "TEAMLENS_JIRA_MCP" in web and "JIRA_API_TOKEN" not in web  # the web app is not given the Jira token


@pytest.mark.skipif(os.environ.get("TEAMLENS_TEST_JIRA_MCP") != "1", reason="needs mcp-atlassian and a real Jira site")
async def test_issue_lifecycle_on_real_jira_through_mcp_atlassian():
    s = Settings()
    extra = ("jira_delete_issue", "jira_get_user_profile")
    async with connect(s, extra_tools=extra) as client:
        assert sorted(t.name for t in (await client.list_tools()).tools) == sorted((*TOOLS, *extra))
        jira = AtlassianJira(client, s)
        me = (await jira._call("jira_get_user_profile", user_identifier="me"))["user"]["account_id"]
        issue = await jira.create(s.jira_team_key, "TeamLens live test through mcp-atlassian", type="Bug", story_points=3)
        try:
            assert issue.status == "To Do" and issue.type == "Bug" and issue.story_points == 3
            assert issue.assignee_id is None and issue.project == s.jira_team_key and not issue.in_sprint

            assert (await jira.assign(issue.key, me)).assignee_id == me
            for status in ("In Progress", "In Review", "Done"):
                assert (await jira.transition(issue.key, status)).status == status
            await jira.comment(issue.key, "Checked by the live test.")
            assert issue.key in [i.key for i in await jira.search(assignee_id=me, status="Done")]
            assert (await jira.assign(issue.key, None)).assignee_id is None
            everything = await jira.search()
            assert issue.key in [i.key for i in everything] and any(i.in_sprint for i in everything)
        finally:
            await jira.delete(issue.key)
        with pytest.raises(JiraError, match="does not exist"):
            await jira.get(issue.key)
