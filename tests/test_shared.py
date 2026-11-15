"""Several instances of the app, one copy of the demo company.

Each test starts two apps with separate memory and separate folders, as a serverless host does, and gives
them the same store. What one instance changes, the other must show.
"""
import json

import httpx
from fastapi.testclient import TestClient

from teamlens.delivery.model import ScriptedModel, call
from teamlens.seed import generate
from teamlens.web.app import create_app
from teamlens.web.shared import Shared

C = generate()
PAY = [e for e in C.employees if e.team == "PAY"]
MANAGER, AWAY, FREE = PAY[0].id, PAY[1].id, PAY[2].id
ISSUE = C.planted["assigned_while_absent"][0]


class FakeRedis:
    """The few commands the app uses, answered the way the Upstash REST API answers them."""

    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.down = False

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("store is down", request=request)
        assert request.headers["authorization"] == "Bearer secret"
        command, key, *rest = json.loads(request.content)
        if command == "GET":
            result = self.data.get(key)
        elif command == "SET":
            taken = "NX" in rest and key in self.data
            result = None if taken else "OK"
            if not taken:
                self.data[key] = rest[0]
        elif command == "DEL":
            result = int(self.data.pop(key, None) is not None)
        elif command == "INCR":
            self.data[key] = str(int(self.data.get(key, 0)) + 1)
            result = int(self.data[key])
        else:
            assert command == "EXPIRE", command
            result = 1
        return httpx.Response(200, json={"result": result})


def instance(tmp_path, name, redis, *replies, clock=None, **kw) -> TestClient:
    shared = Shared("https://store.test", "secret", transport=httpx.MockTransport(redis), **({"clock": clock} if clock else {}))
    return TestClient(create_app(ScriptedModel(list(replies)), fake=True, audit_path=tmp_path / name / "audit.jsonl",
                                 shared=shared, **kw))


def status(http, key):
    return http.get("/api/overview", params={"actor": MANAGER}).json()["issues"][key]["status"]


def test_a_change_made_through_one_instance_shows_in_the_other(tmp_path):
    redis = FakeRedis()
    with instance(tmp_path, "a", redis) as a, instance(tmp_path, "b", redis) as b:
        mine = a.get("/api/overview", params={"actor": FREE}).json()
        key = next(x for x in mine["loads"] if x["employee_id"] == FREE)["open_issues"][0]
        assert b.get("/api/people").json()["shared"] is True
        assert status(b, key) == mine["issues"][key]["status"] != "Done"

        assert a.post("/api/issues/move", json={"actor": FREE, "issue_key": key, "status": "Done"}).json()["ok"]
        assert status(b, key) == "Done"  # the case that failed on the public demo: closed on one, open on the other
        again = b.post("/api/issues/move", json={"actor": FREE, "issue_key": key, "status": "In Review"})
        assert again.status_code == 409 and "Done and cannot be changed" in again.json()["detail"]

        log = a.get("/api/audit", params={"actor": MANAGER}).json()  # one log, written through both
        assert log["intact"] and [e["data"]["allowed"] for e in log["entries"]][::-1] == [True, False]
        assert b.get("/api/audit", params={"actor": MANAGER}).json() == log


def test_a_plan_asked_on_one_instance_is_approved_on_another(tmp_path):
    step = {"tool": "assign_issue", "issue_key": ISSUE, "assignee_id": FREE, "status": None, "summary": None, "reason": "cover"}
    redis = FakeRedis()
    asked = call("propose_plan", _id="p1", summary="Cover", steps=[step])
    with instance(tmp_path, "a", redis, asked) as a, instance(tmp_path, "b", redis) as b:
        waiting = a.post("/api/ask", json={"actor": MANAGER, "text": "Cover for Anton"}).json()
        assert waiting["outcome"] == "needs_approval"
        done = b.post("/api/decide", json={"actor": MANAGER, "thread": waiting["thread"], "approved": True}).json()
        assert done["outcome"] == "executed"  # the paused conversation travelled with the rest of the state
        for http in (a, b):
            o = http.get("/api/overview", params={"actor": MANAGER}).json()
            assert ISSUE in next(x for x in o["loads"] if x["employee_id"] == FREE)["open_issues"]
        kinds = [e["kind"] for e in a.get("/api/audit", params={"actor": MANAGER}).json()["entries"]][::-1]
        assert kinds == ["request", "plan_proposed", "approval", "step", "outcome"]
        gone = b.post("/api/decide", json={"actor": MANAGER, "thread": waiting["thread"], "approved": True})
        assert gone.status_code == 400 and "No plan is waiting" in gone.json()["detail"]  # a finished plan cannot run twice


def test_roles_settings_and_reset_travel_too(tmp_path):
    redis = FakeRedis()
    with instance(tmp_path, "a", redis) as a, instance(tmp_path, "b", redis) as b:
        a.post("/api/roles/define", json={"actor": MANAGER, "name": "tech-lead", "rights": ["issues.assign"]})
        a.post("/api/roles/grant", json={"actor": MANAGER, "employee_id": FREE, "role": "tech-lead"})
        a.post("/api/autoassign/mode", json={"actor": MANAGER, "mode": "auto"})
        assert "issues.assign" in b.get("/api/me", params={"actor": FREE}).json()["rights"]
        assert b.get("/api/autoassign", params={"actor": MANAGER}).json()["mode"] == "auto"

        cto = next(e.id for e in C.employees if e.role == "Chief Technology Officer")
        hire = next(e.id for e in C.employees if e.department == "Engineering" and not e.team)
        assert a.post("/api/teams", json={"actor": cto, "key": "RSK", "name": "Risk"}).json()["ok"]
        assert b.post("/api/staff/add", json={"actor": cto, "employee_id": hire, "team": "RSK"}).json()["ok"]  # b knows the team
        assert [x["employee_id"] for x in a.get("/api/overview", params={"actor": cto, "team": "RSK"}).json()["loads"]] == [hire]

        assert b.post("/api/demo/reset", json={}).json() == {"ok": True}
        assert "RSK" not in a.get("/api/people").json()["teams"]
        assert "issues.assign" not in a.get("/api/me", params={"actor": FREE}).json()["rights"]
        assert a.get("/api/autoassign", params={"actor": MANAGER}).json()["mode"] == "suggest"
        assert a.get("/api/audit", params={"actor": MANAGER}).json()["total"] == 0


def test_the_demo_starts_over_after_nobody_touched_it_for_a_while(tmp_path):
    redis, now = FakeRedis(), [1_000_000.0]
    with instance(tmp_path, "a", redis, clock=lambda: now[0]) as a, instance(tmp_path, "b", redis, clock=lambda: now[0]) as b:
        key = next(x for x in a.get("/api/overview", params={"actor": FREE}).json()["loads"] if x["employee_id"] == FREE)["open_issues"][0]
        before = status(a, key)
        a.post("/api/issues/move", json={"actor": FREE, "issue_key": key, "status": "Done"})
        now[0] += 3600
        assert status(b, key) == "Done"  # an hour later it is still there
        now[0] += 7200
        assert status(b, key) == before and status(a, key) == before  # left alone for long enough: a clean company
        assert a.post("/api/issues/move", json={"actor": FREE, "issue_key": key, "status": "Done"}).json()["ok"]
        assert status(b, key) == "Done"  # and it is in use again


def test_the_model_allowance_is_counted_across_instances(tmp_path):
    from teamlens.delivery.model import say

    redis = FakeRedis()
    with instance(tmp_path, "a", redis, say("one"), daily_model_requests=2) as a, \
            instance(tmp_path, "b", redis, say("two"), daily_model_requests=2) as b:
        ask = {"actor": MANAGER, "text": "Anyone overloaded?"}
        assert a.post("/api/ask", json=ask).json()["answer"] == "one"
        assert b.post("/api/ask", json=ask).json()["answer"] == "two"
        assert a.post("/api/ask", json=ask).status_code == 429  # the third request of the day, whichever instance gets it


def test_without_the_store_an_instance_keeps_working_from_its_own_memory(tmp_path):
    redis = FakeRedis()
    with instance(tmp_path, "a", redis) as a:
        redis.down = True
        key = next(x for x in a.get("/api/overview", params={"actor": FREE}).json()["loads"] if x["employee_id"] == FREE)["open_issues"][0]
        assert a.post("/api/issues/move", json={"actor": FREE, "issue_key": key, "status": "Done"}).json()["ok"]
        assert status(a, key) == "Done"
