"""API and one page over the same code the terminal uses.

  python -m teamlens.web                 real Jira (needs .env)      http://localhost:8000
  TEAMLENS_FAKE=1 python -m teamlens.web the fictional company, in memory
  TEAMLENS_A2A=1  python -m teamlens.web people data from the People agent over A2A

There is no login: the page has a "working as" switch, because this is a demo of how
different people see and may do different things. A real deployment puts SSO in front.
"""

from __future__ import annotations

import asyncio
import logging
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
from teamlens.access import RIGHTS, Access
from teamlens.config import Settings
from teamlens.delivery.agent import DeliveryAgent
from teamlens.delivery import digest as digests
from teamlens.delivery.audit import AuditLog
from teamlens.delivery.autoassign import AutoAssigner, Mode
from teamlens.delivery.model import ClaudeModel, Model
from teamlens.delivery.policy import Step, check
from teamlens.delivery.tools import Toolbox, ToolFailure
from teamlens.hrm import Hrm
from teamlens.jira import FakeJira
from teamlens.jira_cloud import JiraCloud
from teamlens.mcp_servers import jira_server
from teamlens.people import A2APeople, LocalPeople
from teamlens.seed import generate
from teamlens.similar import Finder, load_embedder
from teamlens.tracing import Tracing
from teamlens.triage import Triage, load as load_triage

PAGE = Path(__file__).with_name("page.html")


class Ask(BaseModel):
    actor: str
    text: str


class Decide(BaseModel):
    actor: str
    thread: str
    approved: bool


class SetMode(BaseModel):
    actor: str
    mode: Mode


class OnIssue(BaseModel):
    actor: str
    issue_key: str


class NewIssue(BaseModel):
    actor: str
    summary: str
    points: int = 1


class OnActor(BaseModel):
    actor: str


class Similar(BaseModel):
    actor: str
    text: str


class RoleGrant(BaseModel):
    actor: str
    employee_id: str
    role: str


class RoleDefine(BaseModel):
    actor: str
    name: str
    rights: list[str] = []


def create_app(model: Model | None = None, fake: bool | None = None, audit_path: Path | None = None,
               triage: Triage | None = None) -> FastAPI:
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
            app.state.audit = AuditLog(audit_path or Path("data/audit.jsonl"))
            access_path = audit_path.with_name("access.json") if audit_path else Path("data/access.json")
            app.state.access = Access(access_path, audit=app.state.audit)
            # A scripted model means a test: skip loading the embedding model and search by words.
            finder = Finder(load_embedder() if model is None else None)
            tracing = Tracing() if model is not None else Tracing.from_settings(
                s.langfuse_public_key, s.langfuse_secret_key, s.langfuse_host)
            app.state.tracing = tracing
            app.state.box = Toolbox(client, people, c.today, app.state.access, finder, tracing)
            app.state.model = model or ClaudeModel(api_key=s.anthropic_api_key, tracing=tracing)
            app.state.agent = DeliveryAgent(app.state.model, app.state.box, app.state.audit, c.today)
            app.state.source = "in-memory demo data" if fake else s.jira_base_url
            settings_path = audit_path.with_name("team_settings.json") if audit_path else Path("data/team_settings.json")
            # Tests pass their own classifier or none; a real run uses the trained model if there is one.
            sorter = triage or (None if model is not None else load_triage(s.anthropic_api_key))
            app.state.triage = sorter.name if sorter else None
            app.state.auto = AutoAssigner(app.state.box, app.state.audit, settings_path, sorter)
            app.state.manager = next(p.id for p in await people.team(s.jira_team_key) if p.role == "Engineering Manager")

            async def watch(seconds: float) -> None:
                """Auto mode also covers issues created directly in Jira: look for unowned sprint work on a timer."""
                while True:
                    await asyncio.sleep(seconds)
                    try:
                        await app.state.auto.sweep(app.state.team, app.state.manager)
                    except Exception:  # the watcher must survive a bad pass
                        logging.getLogger("teamlens").exception("auto-assignment pass failed")

            seconds = float(os.environ.get("TEAMLENS_AUTOASSIGN_SECONDS", "0" if fake else "30"))
            watcher = asyncio.create_task(watch(seconds)) if seconds > 0 else None
            yield
            if watcher:
                watcher.cancel()
            tracing.flush()
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
        return {"team": app.state.team, "source": app.state.source, "tracing": app.state.tracing.enabled,
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
        backlog = [k for k, i in world.issues.items() if i.assignee_id is None and i.status != "Done" and not i.in_sprint]
        return {"today": str(world.overview.today), "actor": world.actor.model_dump(), "rights": sorted(world.rights),
                "loads": [x.model_dump(mode="json") for x in world.overview.loads],
                "findings": [{**f.model_dump(), "name": names.get(f.employee_id)} for f in world.overview.findings],
                "issues": issues, "backlog": backlog, "forecast": app.state.box.forecast(world).model_dump()}

    @app.post("/api/digest")
    async def digest(body: OnActor) -> dict:
        """The morning digest for this person's team. A scheduler would call the same thing once a day."""
        world = await guarded(app.state.box.world(body.actor))
        if "overview.view" not in world.rights:
            raise HTTPException(403, f"{world.actor.name} cannot see this team's work.")
        unowned = [{"issue": s.issue_key, "summary": s.summary, "points": s.points, "suggested_owner": s.assignee_name}
                   for s in await app.state.auto.suggestions(body.actor)]
        with app.state.tracing.request("digest", f"digest-{world.overview.today}", body.actor) as trace:
            d = await digests.write(app.state.model, world, app.state.box.forecast(world), unowned)
            trace.update(output={"text": d.text, "revisions": d.revisions, "problems_found": len(d.problems_found)})
        app.state.audit.record("digest", body.actor, "-", revisions=d.revisions, problems_found=len(d.problems_found),
                               problems_left=len(d.problems_left), usage=d.usage.model_dump())
        u = d.usage
        return {"text": d.text, "revisions": d.revisions, "problems_found": [p.model_dump() for p in d.problems_found],
                "problems_left": [p.model_dump() for p in d.problems_left],
                "usage": {"calls": u.calls, "tokens": u.input + u.cache_read + u.cache_write + u.output, "usd": round(u.usd, 5)}}

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

    # --- automatic assignment: plain code, no model ------------------------------------------

    async def guarded(job):
        try:
            return await job
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except ToolFailure as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.get("/api/autoassign")
    async def autoassign(actor: str) -> dict:
        auto, world = app.state.auto, await guarded(app.state.box.world(actor))
        setting = auto.setting(app.state.team)
        names = {x.employee_id: x.name for x in world.overview.loads}
        recent = [{"issue_key": e["data"]["issue"], "assignee_name": names.get(e["data"]["assignee"]),
                   "by_system": e["actor"] == "autoassign", "reason": e["data"]["reason"]}
                  for e in reversed(app.state.audit.entries())
                  if e["kind"] == "auto_assign" and e["data"].get("assignee")
                  and getattr(world.issues.get(e["data"]["issue"]), "assignee_id", None) == e["data"]["assignee"]][:5]
        return {"mode": setting["mode"], "set_by": names.get(setting["set_by"]),
                "triage": app.state.triage, "can_configure": "autoassign.configure" in world.rights, "can_create": "issues.create" in world.rights,
                "suggestions": [s.model_dump() for s in await auto.suggestions(actor)], "recent": recent}

    @app.post("/api/autoassign/mode")
    async def set_mode(body: SetMode) -> dict:
        setting = await guarded(app.state.auto.set_mode(body.actor, body.mode))
        applied = await app.state.auto.sweep(app.state.team, app.state.manager)
        return {"mode": setting["mode"], "applied": [a.model_dump() for a in applied]}

    @app.post("/api/autoassign/assign")
    async def assign(body: OnIssue) -> dict:
        return (await guarded(app.state.auto.assign(body.actor, body.issue_key))).model_dump()

    @app.post("/api/autoassign/undo")
    async def undo(body: OnIssue) -> dict:
        return (await guarded(app.state.auto.undo(body.actor, body.issue_key))).model_dump()

    @app.post("/api/similar")
    async def similar(body: Similar) -> dict:
        world = await guarded(app.state.box.world(body.actor))
        return {"mode": app.state.box.finder.mode, "matches": [m.model_dump() for m in app.state.box.similar(body.text, world)]}

    @app.post("/api/issues")
    async def new_issue(body: NewIssue) -> dict:
        """A task lands on the team: created in the sprint with no owner. In auto mode it gets one at once."""
        world = await guarded(app.state.box.world(body.actor))
        verdict = check([Step(tool="create_issue", summary=body.summary, reason="created from the page")], world)[0]
        if not verdict.allowed:
            raise HTTPException(403, verdict.reason)
        issue = await guarded(app.state.box.create_in_sprint(world.actor.team, body.summary.strip(),
                                                             max(0, min(body.points, 13))))
        app.state.audit.record("issue_created", body.actor, "-", issue=issue.key, summary=issue.summary)
        applied = await app.state.auto.sweep(app.state.team, app.state.manager)
        return {"issue_key": issue.key, "applied": [a.model_dump() for a in applied]}

    # --- roles and rights: buttons and plain code, never the model ---------------------------

    async def person(employee_id: str):
        found = await app.state.people.person(employee_id)
        if found is None:
            raise HTTPException(404, f"Unknown employee {employee_id}.")
        return found

    async def change_roles(job) -> dict:
        async def run():
            job()
        await guarded(run())
        return {"ok": True}

    @app.get("/api/roles")
    async def roles(actor: str) -> dict:
        access, me = app.state.access, await person(actor)
        team = me.team or app.state.team
        mine = access.rights(me, team)
        members = await app.state.people.team(team)
        return {
            "team": team, "catalogue": RIGHTS, "mine": sorted(mine),
            "roles": [{**r.model_dump(), "can_give": "roles.grant" in mine and set(r.rights) <= mine}
                      for r in access.roles(team) if r.name != "admin"],
            "members": [{"id": p.id, "name": p.name, "base": access.base_role(p), "granted": access.granted(p, team),
                         "rights": sorted(access.rights(p, team))} for p in members if p.active],
        }

    @app.post("/api/roles/grant")
    async def grant_role(body: RoleGrant) -> dict:
        me, target = await person(body.actor), await person(body.employee_id)
        return await change_roles(lambda: app.state.access.grant(me, target, body.role))

    @app.post("/api/roles/revoke")
    async def revoke_role(body: RoleGrant) -> dict:
        me, target = await person(body.actor), await person(body.employee_id)
        return await change_roles(lambda: app.state.access.revoke(me, target, body.role))

    @app.post("/api/roles/define")
    async def define_role(body: RoleDefine) -> dict:
        me = await person(body.actor)
        return await change_roles(lambda: app.state.access.define(me, body.name.strip(), body.rights))

    @app.post("/api/roles/delete")
    async def delete_role(body: RoleDefine) -> dict:
        me = await person(body.actor)
        return await change_roles(lambda: app.state.access.delete(me, body.name))

    @app.get("/api/audit")
    async def audit(limit: int = 30) -> dict:
        log = app.state.audit
        entries = log.entries()
        return {"intact": log.verify() is None, "total": len(entries), "entries": entries[-limit:][::-1]}

    return app
