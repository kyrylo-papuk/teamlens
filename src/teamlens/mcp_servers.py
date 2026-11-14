"""MCP servers: thin adapters that expose Jira and HRM as tools.

They add no rules of their own. Who may call a write tool, and with what arguments,
is decided by the policy layer in front of the agent, not here.

Run one over stdio:  python -m teamlens.mcp_servers jira|hrm
"""

from __future__ import annotations

import functools
import sys
from datetime import date

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from teamlens.hrm import Availability, Hrm, Person
from teamlens.jira import FakeJira, Jira, JiraError
from teamlens.seed import Absence, Issue, generate

READ = ToolAnnotations(readOnlyHint=True)
WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False)


def _expected(fn):
    """Turn a Jira refusal into a tool error the model can read and react to."""
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        try:
            return await fn(*args, **kwargs)
        except JiraError as exc:
            raise ToolError(str(exc)) from exc
    return wrapper


def jira_server(jira: Jira) -> MCPServer:
    server = MCPServer("jira", instructions="Read and change Jira issues.")

    @server.tool(annotations=READ)
    @_expected
    async def search_issues(project: str | None = None, assignee_id: str | None = None,
                            status: str | None = None, sprint_only: bool = False) -> list[Issue]:
        """Find issues. All filters are optional and combine with AND."""
        return await jira.search(project, assignee_id, status, sprint_only)

    @server.tool(annotations=READ)
    @_expected
    async def get_issue(key: str) -> Issue:
        """One issue by key, for example PAY-12."""
        return await jira.get(key)

    @server.tool(annotations=WRITE)
    @_expected
    async def assign_issue(key: str, assignee_id: str | None) -> Issue:
        """Set the assignee. Pass null to unassign."""
        return await jira.assign(key, assignee_id)

    @server.tool(annotations=WRITE)
    @_expected
    async def transition_issue(key: str, status: str) -> Issue:
        """Move an issue to: To Do, In Progress, In Review or Done."""
        return await jira.transition(key, status)

    @server.tool(annotations=WRITE)
    @_expected
    async def add_comment(key: str, text: str) -> str:
        """Add a comment to an issue."""
        await jira.comment(key, text)
        return "ok"

    @server.tool(annotations=WRITE)
    @_expected
    async def create_issue(project: str, summary: str, type: str = "Task", story_points: int = 1,
                           assignee_id: str | None = None) -> Issue:
        """Create an issue in the backlog of a project."""
        return await jira.create(project, summary, type, story_points, assignee_id)

    return server


def hrm_server(hrm: Hrm, today: date, period_start: date, period_end: date) -> MCPServer:
    """Only the People agent connects to this server: one of its tools returns restricted data."""
    server = MCPServer("hrm", instructions="Employee directory, availability and absence records.")

    @server.tool(annotations=READ)
    async def list_people(team: str | None = None) -> list[Person]:
        """Employee directory, optionally one team. No restricted fields."""
        return hrm.people(team)

    @server.tool(annotations=READ)
    async def get_availability(employee_id: str) -> Availability:
        """Whether a person can work in the current sprint and until when they are away. No reason."""
        return hrm.availability(employee_id, period_start, period_end, today)

    @server.tool(annotations=READ, meta={"data_class": "restricted"})
    async def get_absence_records(employee_id: str) -> list[Absence]:
        """RESTRICTED. Absence records with the reason. Never pass the reason on."""
        return hrm.absence_records(employee_id)

    return server


if __name__ == "__main__":
    c = generate()
    if sys.argv[1:] == ["hrm"]:
        hrm_server(Hrm(c.employees, c.absences), c.today, c.sprint_start, c.sprint_end).run()
    else:
        jira_server(FakeJira(c.issues, c.today)).run()
