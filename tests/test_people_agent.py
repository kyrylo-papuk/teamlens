import httpx
import pytest

from teamlens.hrm import Hrm
from teamlens.people import A2APeople, LocalPeople, PeopleAgentError
from teamlens.people_agent.service import READ, Caller, build_app
from teamlens.seed import generate

C = generate()
URL = "http://people.test/"
TOKENS = {"delivery-token": Caller(name="delivery-agent", scopes={READ}),
          "weak-token": Caller(name="some-agent", scopes=set())}
AWAY = C.absences[0].employee_id


def http(token: str | None = "delivery-token", advisor=None) -> httpx.AsyncClient:
    app = build_app(Hrm(C.employees, C.absences), C.today, TOKENS, URL, advisor)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=URL, headers=headers)


def people(client: httpx.AsyncClient) -> A2APeople:
    return A2APeople(client, C.today, C.sprint_start, C.sprint_end)


async def test_card_is_public_and_says_how_to_authenticate():
    async with http(token=None) as client:
        card = (await client.get("/.well-known/agent-card.json")).json()
    assert [s["id"] for s in card["skills"]] == ["directory", "availability", "staffing_question"]
    assert card["securitySchemes"]["bearer"]["httpAuthSecurityScheme"]["scheme"] == "bearer"
    assert "pay" in card["description"]


async def test_requests_need_a_token_and_the_right_scope():
    async with http(token=None) as client:
        assert (await client.post("/", json={})).status_code == 401
    async with http(token="wrong") as client:
        assert (await client.post("/", json={})).status_code == 401
    async with http(token="weak-token") as client:
        with pytest.raises(PeopleAgentError, match="lacks the 'people:read' scope"):
            await people(client).team("PAY")


async def test_answers_match_the_hrm_and_carry_no_restricted_data():
    local = LocalPeople(Hrm(C.employees, C.absences), C.today, C.sprint_start, C.sprint_end)
    async with http() as client:
        remote = people(client)
        team = await remote.team("PAY")
        ids = [p.id for p in team]
        assert team == await local.team("PAY")
        assert await remote.person(AWAY) == await local.person(AWAY)
        assert await remote.person("E999") is None
        assert await remote.availability(ids) == await local.availability(ids)

        raw = await remote._lookup(skill="availability", employee_ids=[AWAY], start=str(C.sprint_start), end=str(C.sprint_end))
        text = str(raw) + str(await remote._lookup(skill="team", team="PAY"))
    assert raw["availability"][AWAY]["available_today"] is False and raw["availability"][AWAY]["away"]
    assert "sick" not in text and "reason" not in text and "salary" not in text


async def test_there_is_no_skill_for_restricted_data():
    async with http() as client:
        for request in ({"skill": "absence_records", "employee_id": AWAY}, {"skill": "salary", "employee_id": AWAY}, {}):
            with pytest.raises(PeopleAgentError, match="Unknown or malformed"):
                await people(client)._lookup(**request)


async def test_text_questions_go_to_the_advisor():
    async def advisor(question: str) -> str:
        return f"asked: {question}"
    async with http(advisor=advisor) as client:
        assert await people(client).ask("Who works part-time in PAY?") == "asked: Who works part-time in PAY?"
    async with http() as client:
        assert "not enabled" in await people(client).ask("Anything")


# --- the Delivery agent with People behind A2A ----------------------------------------------

async def delivery(client, tmp_path, *replies):
    from mcp import Client

    from teamlens.delivery.agent import DeliveryAgent
    from teamlens.delivery.audit import AuditLog
    from teamlens.delivery.model import ScriptedModel
    from teamlens.delivery.tools import Toolbox
    from teamlens.jira import FakeJira
    from teamlens.mcp_servers import jira_server

    jira = FakeJira(C.issues, C.today)
    mcp = Client(jira_server(jira))
    await mcp.__aenter__()
    model = ScriptedModel(list(replies))
    agent = DeliveryAgent(model, Toolbox(mcp, people(client), C.today), AuditLog(tmp_path / "audit.jsonl"), C.today)
    return agent, model, jira, mcp


async def test_delivery_agent_gets_people_data_over_a2a(tmp_path):
    from teamlens.delivery.model import call

    pay = [e for e in C.employees if e.team == "PAY"]
    manager, free, issue = pay[0].id, pay[2].id, C.planted["assigned_while_absent"][0]
    step = {"tool": "assign_issue", "issue_key": issue, "assignee_id": free, "status": None, "summary": None, "reason": "cover"}

    async def advisor(question: str) -> str:
        return "Dana Okafor [E004] works 20 hours a week."

    async with http(advisor=advisor) as client:
        agent, model, jira, mcp = await delivery(
            client, tmp_path, call("ask_people_agent", question="Who works part-time in PAY?"),
            call("propose_plan", _id="p1", summary="cover", steps=[step]))
        try:
            waiting = await agent.ask(manager, "Cover for the absent engineer", "t1")
            assert "ask_people_agent" in {t["name"] for t in model.tools}
            assert model.seen[1][-1]["content"][0]["content"] == "Dana Okafor [E004] works 20 hours a week."
            assert waiting.outcome == "needs_approval"
            assert (await agent.decide("t1", True, manager)).outcome == "executed"
            assert (await jira.get(issue)).assignee_id == free
        finally:
            await mcp.__aexit__(None, None, None)


async def test_delivery_agent_fails_safely_when_the_people_agent_refuses(tmp_path):
    from teamlens.delivery.model import say

    async with http(token="weak-token") as client:
        agent, _, _, mcp = await delivery(client, tmp_path, say("unused"))
        try:
            result = await agent.ask(C.employees[0].id, "Who is at risk?", "t1")
            assert result.outcome == "failed" and "People agent is unavailable" in result.answer
        finally:
            await mcp.__aexit__(None, None, None)
