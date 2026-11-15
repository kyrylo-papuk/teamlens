"""API and one page over the same code the terminal uses.

  python -m teamlens.web                 real Jira (needs .env)      http://localhost:8000
  TEAMLENS_FAKE=1 python -m teamlens.web the fictional company, in memory
  TEAMLENS_A2A=1  python -m teamlens.web people data from the People agent over A2A

There is no login: the page starts by asking who you are, because this is a demo of how
people in different positions see and may do different things. A real deployment puts SSO
in front, and the position comes from the HR record either way.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import anthropic
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from mcp import Client
from pydantic import BaseModel

from teamlens import live
from teamlens.access import COMPANY_WIDE, POSITIONS, RIGHTS, Access
from teamlens.config import Settings
from teamlens.delivery import digest as digests
from teamlens.delivery import direct
from teamlens.delivery.agent import DeliveryAgent
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
from teamlens.seed import LEADERSHIP, TEAMS, generate
from teamlens.similar import Finder, load_embedder
from teamlens.tracing import Tracing
from teamlens.triage import Triage
from teamlens.triage import load as load_triage
from teamlens.web.shared import Shared, StoreDown, Threads, restore, snapshot

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


class Assign(BaseModel):
    actor: str
    issue_key: str
    assignee_id: str | None = None  # nobody: the owner gives the issue up


class Move(BaseModel):
    actor: str
    issue_key: str
    status: str


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


SPENT = ("This demo has used its model allowance for today. Everything that does not need the model still works; "
         "the allowance resets at midnight UTC.")


class DailyBudget:
    """Counts the requests that reach the model and refuses once the day's allowance is used."""

    def __init__(self, allowance: int) -> None:
        self.allowance, self._day, self._used = allowance, "", 0

    def spend(self) -> None:
        today = datetime.now(UTC).date().isoformat()
        if today != self._day:
            self._day, self._used = today, 0
        if self.allowance and self._used >= self.allowance:
            raise HTTPException(429, SPENT)
        self._used += 1


def create_app(model: Model | None = None, fake: bool | None = None, audit_path: Path | None = None,
               triage: Triage | None = None, state_path: Path | None = None, daily_model_requests: int | None = None,
               shared: Shared | None = None) -> FastAPI:
    """Passing a scripted ``model`` marks a test: no embedding model, no tracing, and conversation state in
    memory unless ``state_path`` is given. ``shared`` makes this instance one of several that keep one copy
    of the demo company; outside tests it comes from the environment."""
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
        async with AsyncExitStack() as stack:
            client = await stack.enter_async_context(Client(jira_server(jira)))
            app.state.team = s.jira_team_key
            # The demo company has every team in Jira; a real site is connected one project at a time.
            app.state.teams = list(TEAMS) if fake else [s.jira_team_key]
            app.state.demo = (jira, c.issues) if fake else None  # what a reset of the demo goes back to
            app.state.people = people
            # Everything the app writes goes to one folder. Hosts with a read-only filesystem point it at /tmp.
            data = audit_path.parent if audit_path else Path(os.environ.get("TEAMLENS_DATA_DIR", "data"))
            app.state.data = data
            app.state.audit = AuditLog(audit_path or data / "audit.jsonl")
            app.state.access = Access(data / "access.json", audit=app.state.audit)
            # A scripted model means a test: skip loading the embedding model and search by words.
            finder = Finder(load_embedder() if model is None else None)
            tracing = Tracing() if model is not None else Tracing.from_settings(
                s.langfuse_public_key, s.langfuse_secret_key, s.langfuse_host)
            app.state.tracing = tracing
            app.state.box = Toolbox(client, people, c.today, app.state.access, finder, tracing, app.state.teams)
            app.state.model = model or ClaudeModel(api_key=s.anthropic_api_key, tracing=tracing)
            # Several instances of the public demo keep one copy of the company through Redis: see web/shared.py.
            url, token = s.kv_rest_api_url or s.upstash_redis_rest_url, s.kv_rest_api_token or s.upstash_redis_rest_token
            app.state.shared = shared or (Shared(url, token) if fake and model is None and url and token else None)
            app.state.threads = Threads() if app.state.shared else None  # paused conversations travel with the rest
            # A real run keeps paused conversations in SQLite, so a plan waiting for approval survives a restart.
            saved, checkpointer = state_path or (None if model is not None else data / "state.sqlite"), app.state.threads
            if saved and checkpointer is None:
                from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

                saved.parent.mkdir(parents=True, exist_ok=True)
                checkpointer = await stack.enter_async_context(AsyncSqliteSaver.from_conn_string(str(saved)))
            app.state.agent = DeliveryAgent(app.state.model, app.state.box, app.state.audit, c.today, checkpointer)
            app.state.source = "in-memory demo data" if fake else "Jira Cloud"
            settings_path = data / "team_settings.json"
            # Tests pass their own classifier or none; a real run uses the trained model if there is one.
            sorter = triage or (None if model is not None else load_triage(s.anthropic_api_key))
            app.state.triage = sorter.name if sorter else None
            app.state.auto = AutoAssigner(app.state.box, app.state.audit, settings_path, sorter)
            app.state.managers = {key: next(p.id for p in await people.team(key) if POSITIONS.get(p.role) == "manager")
                                  for key in app.state.teams}

            async def watch(seconds: float) -> None:
                """Auto mode also covers issues created directly in Jira: look for unowned sprint work on a timer."""
                while True:
                    await asyncio.sleep(seconds)
                    try:
                        for key in app.state.teams:
                            await sweep(key)
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
    budget = DailyBudget(s.teamlens_daily_model_requests if daily_model_requests is None else daily_model_requests)
    log = logging.getLogger("teamlens")

    def put(state: dict | None) -> None:
        """Put this instance into a saved state of the demo, or without one into the state it starts in."""
        jira, seed = app.state.demo
        restore(state, jira, seed, app.state.data, app.state.threads)
        app.state.access = Access(app.state.data / "access.json", audit=app.state.audit)  # roles are held in memory
        app.state.box.access = app.state.access

    @app.middleware("http")
    async def one_copy(request, call_next):
        """Before a request, catch up with what other instances did; after one that may change something, save."""
        shared = getattr(app.state, "shared", None)
        if shared is None or not request.url.path.startswith("/api/"):
            return await call_next(request)
        changing = request.method == "POST"
        mine = await shared.lock() if changing else None
        try:
            try:
                version, idle = await shared.head()
                if idle or version != shared.version:
                    put(None if idle else await shared.load())
                    shared.version = version
                    if idle:
                        shared.forget()
            except StoreDown:
                log.warning("shared store unreachable; answering from this instance's memory")
            response = await call_next(request)
            if changing:
                try:
                    await shared.save(snapshot(app.state.demo[0], app.state.data, app.state.threads))
                except StoreDown:
                    log.warning("shared store unreachable; this change stays in this instance's memory")
            return response
        finally:
            await shared.unlock(mine)

    async def spend() -> None:
        """One more request that reaches the model. Instances that share a store also share the day's count."""
        shared = getattr(app.state, "shared", None)
        if shared is not None and budget.allowance:
            try:
                used = await shared.count(f"teamlens:model:{datetime.now(UTC).date()}")
            except StoreDown:
                used = None
            if used is not None:
                if used > budget.allowance:
                    raise HTTPException(429, SPENT)
                return
        budget.spend()

    @app.get("/", response_class=HTMLResponse)
    async def page() -> str:
        return PAGE.read_text(encoding="utf-8")

    async def sweep(team: str) -> list:
        return await app.state.auto.sweep(team, app.state.managers[team])

    def lacking(who, right: str) -> HTTPException:
        return HTTPException(403, f"{who.name} does not have the right to do this: {RIGHTS[right].lower()}.")

    @app.get("/api/people")
    async def people() -> dict:
        """Everyone who can be picked on the "who are you" screen: leadership and the members of each team,
        each with the level their position brings."""
        access, listed = app.state.access, list(await app.state.people.department(LEADERSHIP))
        for key in app.state.teams:
            listed += await app.state.people.team(key)
        return {"team": app.state.team, "teams": {key: TEAMS.get(key, key) for key in app.state.teams},
                "source": app.state.source, "tracing": app.state.tracing.enabled, "notice": s.teamlens_notice, "demo": fake,
                "shared": app.state.shared is not None,
                "people": [{"id": p.id, "name": p.name, "role": p.role, "team": p.team, "level": access.base_role(p)}
                           for p in listed if access.base_role(p)]}

    @app.get("/api/me")
    async def me(actor: str) -> dict:
        """What the page needs to decide which screens this person gets."""
        p, access = await person(actor), app.state.access
        return {"id": p.id, "name": p.name, "position": p.role, "team": p.team, "level": access.base_role(p),
                "granted": access.granted(p, p.team) if p.team else [], "rights": sorted(access.rights(p))}

    @app.get("/api/company")
    async def company(actor: str) -> dict:
        """One line per team, for people who may see all of them."""
        viewer, access = await person(actor), app.state.access
        if "company.view" not in access.rights(viewer):
            raise lacking(viewer, "company.view")
        teams = []
        for key in app.state.teams:
            world = await guarded(app.state.box.world(actor, key))
            f, here = app.state.box.forecast(world), [x for x in world.overview.loads if x.availability.active]
            kinds: dict[str, int] = {}
            for finding in world.overview.findings:
                kinds[finding.kind] = kinds.get(finding.kind, 0) + 1
            teams.append({
                "key": key, "name": TEAMS.get(key, key), **managers(world),
                "people": len(here), "away": sum(not x.availability.available_today for x in here),
                # Capacity is the forecast's number (managers carry no sprint work), so the two columns agree.
                "points": sum(x.sprint_points for x in here), "capacity": f.capacity,
                "forecast": {"committed": f.committed, "done": f.done, "expected": f.expected, "at_risk": f.at_risk},
                "findings": kinds, "unowned": len(app.state.auto.unowned(world)),
            })
        return {"teams": teams}

    def managers(world) -> dict:
        """Who runs the team: the manager named in the HR record, and anyone leadership made an acting manager."""
        acting = set(app.state.access.holders("manager", world.team or ""))
        people = [{"id": x.employee_id, "name": x.name, "role": x.role} for x in world.overview.loads if x.availability.active]
        return {"manager": next((p for p in people if POSITIONS.get(p["role"]) == "manager"), None),
                "acting": [p for p in people if p["id"] in acting]}

    @app.get("/api/overview")
    async def overview(actor: str, team: str | None = None) -> dict:
        world = await guarded(app.state.box.world(actor, team))
        names = {x.employee_id: x.name for x in world.overview.loads}
        issues = {k: {"summary": i.summary, "status": i.status, "points": i.story_points, "in_sprint": i.in_sprint,
                      "assignee": i.assignee_id}
                  for k, i in world.issues.items()}
        backlog = [k for k, i in world.issues.items() if i.assignee_id is None and i.status != "Done" and not i.in_sprint]
        return {"today": str(world.overview.today), "actor": world.actor.model_dump(), "rights": sorted(world.rights),
                "team": world.team, "team_name": TEAMS.get(world.team, world.team), **managers(world),
                "loads": [x.model_dump(mode="json") for x in world.overview.loads],
                "findings": [{**f.model_dump(), "name": names.get(f.employee_id)} for f in world.overview.findings],
                "issues": issues, "backlog": backlog, "forecast": app.state.box.forecast(world).model_dump()}

    @app.post("/api/digest")
    async def digest(body: OnActor) -> dict:
        """The morning digest for this person's team. A scheduler would call the same thing once a day."""
        world = await guarded(app.state.box.world(body.actor))
        if "overview.view" not in world.rights:
            raise HTTPException(403, f"{world.actor.name} cannot see this team's work.")
        await spend()
        unowned = [{"issue": s.issue_key, "summary": s.summary, "points": s.points, "suggested_owner": s.assignee_name}
                   for s in await app.state.auto.suggestions(body.actor)]
        with app.state.tracing.request("digest", f"digest-{world.overview.today}", body.actor) as trace:
            d = await guarded(digests.write(app.state.model, world, app.state.box.forecast(world), unowned))
            trace.update(output={"text": d.text, "revisions": d.revisions, "problems_found": len(d.problems_found)})
        app.state.audit.record("digest", body.actor, "-", revisions=d.revisions, problems_found=len(d.problems_found),
                               problems_left=len(d.problems_left), usage=d.usage.model_dump())
        u = d.usage
        return {"text": d.text, "revisions": d.revisions, "problems_found": [p.model_dump() for p in d.problems_found],
                "problems_left": [p.model_dump() for p in d.problems_left],
                "usage": {"calls": u.calls, "tokens": u.input + u.cache_read + u.cache_write + u.output, "usd": round(u.usd, 5)}}

    def reply(thread: str, result) -> dict:
        if app.state.threads is not None and result.outcome != "needs_approval":
            app.state.threads.delete_thread(thread)  # only conversations waiting for a person travel between instances
        u = result.usage
        return {"thread": thread, "outcome": result.outcome, "answer": result.answer, "plan": result.plan,
                "usage": {"calls": u.calls, "tokens": u.input + u.cache_read + u.cache_write + u.output,
                          "usd": round(u.usd, 5)}}

    @app.post("/api/ask")
    async def ask(body: Ask) -> dict:
        await spend()
        thread = uuid.uuid4().hex[:8]
        return reply(thread, await guarded(app.state.agent.ask(body.actor, body.text[:2000], thread)))

    @app.post("/api/decide")
    async def decide(body: Decide) -> dict:
        return reply(body.thread, await guarded(app.state.agent.decide(body.thread, body.approved, body.actor)))

    # --- automatic assignment: plain code, no model ------------------------------------------

    async def guarded(job):
        try:
            return await job
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except anthropic.APIError as exc:
            raise HTTPException(503, "The language model is not available. Check ANTHROPIC_API_KEY. "
                                     "Everything that does not need a model still works.") from exc
        except ToolFailure as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.get("/api/autoassign")
    async def autoassign(actor: str, team: str | None = None) -> dict:
        auto, world = app.state.auto, await guarded(app.state.box.world(actor, team))
        setting = auto.setting(world.team or "")
        names = {x.employee_id: x.name for x in world.overview.loads}
        recent = [{"issue_key": e["data"]["issue"], "assignee_name": names.get(e["data"]["assignee"]),
                   "by_system": e["actor"] == "autoassign", "reason": e["data"]["reason"]}
                  for e in reversed(app.state.audit.entries())
                  if e["kind"] == "auto_assign" and e["data"].get("assignee")
                  and getattr(world.issues.get(e["data"]["issue"]), "assignee_id", None) == e["data"]["assignee"]][:5]
        return {"mode": setting["mode"], "set_by": names.get(setting["set_by"]),
                "triage": app.state.triage, "can_configure": "autoassign.configure" in world.rights,
                "can_create": "issues.create" in world.rights,
                "suggestions": [s.model_dump() for s in await auto.suggestions(actor, team)], "recent": recent}

    @app.post("/api/autoassign/mode")
    async def set_mode(body: SetMode) -> dict:
        setting = await guarded(app.state.auto.set_mode(body.actor, body.mode))
        applied = await sweep((await person(body.actor)).team)
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
        await spend()  # a new ticket is triaged by the model
        issue = await guarded(app.state.box.create_in_sprint(world.actor.team, body.summary.strip(),
                                                             max(0, min(body.points, 13))))
        app.state.audit.record("issue_created", body.actor, "-", issue=issue.key, summary=issue.summary)
        applied = await sweep(world.actor.team)
        return {"issue_key": issue.key, "applied": [a.model_dump() for a in applied]}

    # --- changes made with a button: one step, the same rules and log as a plan, no model -----

    async def change(actor: str, step: Step) -> dict:
        done = await guarded(direct.apply(app.state.box, app.state.audit, actor, step))
        if not done.ok:  # 403: you may not do this; 409: nobody may, as things stand
            raise HTTPException(403 if done.rule in ("right", "actor_team") else 409, done.detail)
        return done.model_dump()

    @app.post("/api/issues/assign")
    async def assign_issue(body: Assign) -> dict:
        """Take an issue, hand it to someone, or give it up. Which of these a person may do is the policy's call."""
        return await change(body.actor, Step(tool="assign_issue", issue_key=body.issue_key, assignee_id=body.assignee_id,
                                             reason="Changed directly on the TeamLens page."))

    @app.post("/api/issues/move")
    async def move_issue(body: Move) -> dict:
        return await change(body.actor, Step(tool="transition_issue", issue_key=body.issue_key, status=body.status,
                                             reason="Changed directly on the TeamLens page."))

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
        # Who holds which role is shown only to people who manage roles. Everyone else gets their own rights.
        manages = bool(mine & {"roles.grant", "roles.define"})
        members = await app.state.people.team(team) if manages else []
        return {
            "team": team, "catalogue": RIGHTS, "mine": sorted(mine), "can_manage": manages,
            "roles": [{**r.model_dump(), "can_give": "roles.grant" in mine and set(r.rights) <= mine}
                      for r in access.roles(team) if r.name not in COMPANY_WIDE] if manages else [],
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

    @app.post("/api/demo/reset")
    async def reset_demo() -> dict:
        """Put the fictional company back to its starting state. A shared demo needs this: one visitor's approved
        plan would otherwise be what the next visitor finds. The route does not exist when the data is real."""
        if app.state.demo is None:
            raise HTTPException(404, "This deployment does not run on demo data.")
        put(None)
        return {"ok": True}

    @app.get("/api/audit")
    async def audit(actor: str, team: str | None = None, limit: int = 30) -> dict:
        """One team's part of the log, or all of it for people who see every team. The chain is checked over the whole log."""
        viewer = await person(actor)
        team = team or viewer.team
        if "audit.view" not in app.state.access.rights(viewer, team):
            raise lacking(viewer, "audit.view")
        log = app.state.audit
        entries = log.entries()
        if team:
            members = {p.id for p in await app.state.people.team(team)}
            entries = [e for e in entries if e["actor"] in members or e["data"].get("on_behalf_of") in members
                       or e["data"].get("team") == team]
        return {"intact": log.verify() is None, "total": len(entries), "entries": entries[-limit:][::-1]}

    return app
