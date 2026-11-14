import json

from mcp import Client

from teamlens.hrm import Hrm
from teamlens.jira import FakeJira
from teamlens.mcp_servers import hrm_server, jira_server
from teamlens.seed import generate


def data(result):
    """Tool output as plain Python, whatever shape the SDK wraps it in."""
    assert not result.is_error, result.content
    payload = result.structured_content
    return payload["result"] if set(payload) == {"result"} else payload


async def test_jira_tools_read_and_write():
    c = generate()
    key = c.planted["assigned_while_absent"][0]
    async with Client(jira_server(FakeJira(c.issues, c.today))) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        assert tools["search_issues"].annotations.read_only_hint is True
        assert tools["assign_issue"].annotations.read_only_hint is False

        issue = data(await client.call_tool("get_issue", {"key": key}))
        mine = data(await client.call_tool("search_issues", {"assignee_id": issue["assignee_id"]}))
        assert key in [i["key"] for i in mine]

        moved = data(await client.call_tool("assign_issue", {"key": key, "assignee_id": "E003"}))
        assert moved["assignee_id"] == "E003"
        done = data(await client.call_tool("transition_issue", {"key": key, "status": "Done"}))
        assert done["status"] == "Done"
        new = data(await client.call_tool("create_issue", {"project": "PAY", "summary": "Follow up"}))
        assert new["key"].startswith("PAY-") and new["status"] == "To Do"


async def test_jira_errors_come_back_as_tool_errors():
    c = generate()
    async with Client(jira_server(FakeJira(c.issues, c.today))) as client:
        missing = await client.call_tool("get_issue", {"key": "PAY-9999"})
        assert missing.is_error and "does not exist" in json.dumps([b.model_dump() for b in missing.content])
        bad = await client.call_tool("transition_issue", {"key": "PAY-1", "status": "Shipped"})
        assert bad.is_error


async def test_hrm_tools_and_restricted_marking():
    c = generate()
    away = c.absences[0].employee_id
    server = hrm_server(Hrm(c.employees, c.absences), c.today, c.sprint_start, c.sprint_end)
    async with Client(server) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        assert tools["get_absence_records"].meta == {"data_class": "restricted"}

        team = data(await client.call_tool("list_people", {"team": "PAY"}))
        assert len(team) == 8 and "salary_eur" not in team[0]

        av = await client.call_tool("get_availability", {"employee_id": away})
        assert data(av)["available_today"] is False
        assert "sick" not in json.dumps(av.structured_content)

        records = data(await client.call_tool("get_absence_records", {"employee_id": away}))
        assert records[0]["reason"] == "sick"
