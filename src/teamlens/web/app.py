"""API and one page over the same code the terminal uses.

  python -m teamlens.web                 real Jira (needs .env)      http://localhost:8000
  TEAMLENS_FAKE=1 python -m teamlens.web the fictional company, in memory
  TEAMLENS_A2A=1  python -m teamlens.web people data from the People agent over A2A

There is no login: the page has a "working as" switch, because this is a demo of how
different people see and may do different things. A real deployment puts SSO in front.
"""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from mcp import Client
from pydantic import BaseModel

from teamlens import live
from teamlens.config import Settings
from teamlens.delivery.agent import DeliveryAgent
from teamlens.delivery.audit import AuditLog
from teamlens.delivery.model import ClaudeModel, Model
from teamlens.delivery.tools import Toolbox, ToolFailure
from teamlens.hrm import Hrm
from teamlens.jira import FakeJira
from teamlens.jira_cloud import JiraCloud
from teamlens.mcp_servers import jira_server
from teamlens.people import A2APeople, LocalPeople
from teamlens.seed import generate

PAGE = Path(__file__).with_name("page.html")


class Ask(BaseModel):
    actor: str
    text: str


class Decide(BaseModel):
    actor: str
    thread: str
    approved: bool


def create_app(model: Model | None = None, fake: bool | None = None, audit_path: Path | None = None) -> FastAPI:
    s = Settings()
    fake = os.environ.get("TEAMLENS_FAKE") == "1" if fake is None else fake

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if fake:
            c, users = generate(), None
            jira = FakeJira(c.issues, c.today)
        else:
            c, users = live.company(), live.load_users()
            jira = JiraCloud(s, users)
        only, http = (set(users) if users else None), None
        if os.environ.get("TEAMLENS_A2A") == "1":
            http = httpx.AsyncClient(base_url=s.people_agent_url, timeout=120,
                                     headers={"Authorization": f"Bearer {s.people_agent_token}"})
            people = A2APeople(http, c.today, c.sprint_start, c.sprint_end, only=only)
        else:
            people = LocalPeople(Hrm(c.employees, c.absences), c.today, c.sprint_start, c.sprint_end, only=only)
        async with Client(jira_server(jira)) as client:
            app.state.team = s.jira_team_key
            app.state.people = people
            app.state.box = Toolbox(client, people, c.today)
            app.state.audit = AuditLog(audit_path or Path("data/audit.jsonl"))
            app.state.agent = DeliveryAgent(model or ClaudeModel(api_key=s.anthropic_api_key), app.state.box,
                                            app.state.audit, c.today)
            app.state.source = "in-memory demo data" if fake else s.jira_base_url
            yield
        if http:
            await http.aclose()
        if not fake:
            await jira.aclose()

    app = FastAPI(title="TeamLens", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def page() -> str:
        return PAGE.read_text(encoding="utf-8")

    @app.get("/api/people")
    async def people() -> dict:
        """Who can be picked in the "working as" switch."""
        team = await app.state.people.team(app.state.team)
        return {"team": app.state.team, "source": app.state.source,
                "people": [{"id": p.id, "name": p.name, "role": p.role} for p in team]}

    @app.get("/api/overview")
    async def overview(actor: str) -> dict:
        try:
            world = await app.state.box.world(actor)
        except ToolFailure as exc:
            raise HTTPException(503, str(exc)) from exc
        names = {x.employee_id: x.name for x in world.overview.loads}
        issues = {k: {"summary": i.summary, "status": i.status, "points": i.story_points, "in_sprint": i.in_sprint}
                  for k, i in world.issues.items()}
        backlog = [k for k, i in world.issues.items() if i.assignee_id is None and i.status != "Done"]
        return {"today": str(world.overview.today), "actor": world.actor.model_dump(),
                "loads": [x.model_dump(mode="json") for x in world.overview.loads],
                "findings": [{**f.model_dump(), "name": names.get(f.employee_id)} for f in world.overview.findings],
                "issues": issues, "backlog": backlog}

    def reply(thread: str, result) -> dict:
        u = result.usage
        return {"thread": thread, "outcome": result.outcome, "answer": result.answer, "plan": result.plan,
                "usage": {"calls": u.calls, "tokens": u.input + u.cache_read + u.cache_write + u.output,
                          "usd": round(u.usd, 5)}}

    @app.post("/api/ask")
    async def ask(body: Ask) -> dict:
        thread = uuid.uuid4().hex[:8]
        return reply(thread, await app.state.agent.ask(body.actor, body.text, thread))

    @app.post("/api/decide")
    async def decide(body: Decide) -> dict:
        return reply(body.thread, await app.state.agent.decide(body.thread, body.approved, body.actor))

    @app.get("/api/audit")
    async def audit(limit: int = 30) -> dict:
        log = app.state.audit
        entries = log.entries()
        return {"intact": log.verify() is None, "total": len(entries), "entries": entries[-limit:][::-1]}

    return app
