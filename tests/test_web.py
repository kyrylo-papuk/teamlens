from fastapi.testclient import TestClient

from teamlens.delivery.model import ScriptedModel, call, say
from teamlens.seed import generate
from teamlens.web.app import create_app

C = generate()
PAY = [e for e in C.employees if e.team == "PAY"]
MANAGER, AWAY, FREE = PAY[0].id, PAY[1].id, PAY[2].id
ISSUE = C.planted["assigned_while_absent"][0]


def client(tmp_path, *replies) -> TestClient:
    return TestClient(create_app(ScriptedModel(list(replies)), fake=True, audit_path=tmp_path / "audit.jsonl"))


def test_page_and_overview(tmp_path):
    with client(tmp_path) as http:
        assert "TeamLens" in http.get("/").text
        people = http.get("/api/people").json()
        assert people["team"] == "PAY" and len(people["people"]) == 8
        o = http.get("/api/overview", params={"actor": MANAGER}).json()
        assert [f["kind"] for f in o["findings"]] == ["assigned_while_absent"] and len(o["backlog"]) == 6
        away = next(x for x in o["loads"] if x["employee_id"] == AWAY)
        assert away["availability"]["available_today"] is False
        assert "salary" not in http.get("/api/overview", params={"actor": MANAGER}).text
        assert "sick" not in http.get("/api/overview", params={"actor": MANAGER}).text


def test_ask_approve_and_audit(tmp_path):
    step = {"tool": "assign_issue", "issue_key": ISSUE, "assignee_id": FREE, "status": None, "summary": None, "reason": "cover"}
    with client(tmp_path, call("propose_plan", _id="p1", summary="Cover", steps=[step])) as http:
        waiting = http.post("/api/ask", json={"actor": MANAGER, "text": "Cover for Anton"}).json()
        assert waiting["outcome"] == "needs_approval" and waiting["plan"]["steps"][0]["issue_key"] == ISSUE
        done = http.post("/api/decide", json={"actor": MANAGER, "thread": waiting["thread"], "approved": True}).json()
        assert done["outcome"] == "executed"
        o = http.get("/api/overview", params={"actor": MANAGER}).json()
        assert ISSUE in next(x for x in o["loads"] if x["employee_id"] == FREE)["open_issues"]
        audit = http.get("/api/audit").json()
        assert audit["intact"] and [e["kind"] for e in audit["entries"]][::-1] == ["request", "plan_proposed", "approval", "step", "outcome"]


def test_question(tmp_path):
    with client(tmp_path, say("Nobody is overloaded.")) as http:
        r = http.post("/api/ask", json={"actor": MANAGER, "text": "Anyone overloaded?"}).json()
        assert r["outcome"] == "answered" and r["answer"] == "Nobody is overloaded."


def test_auto_assignment_through_the_api(tmp_path):
    with client(tmp_path) as http:
        made = http.post("/api/issues", json={"actor": MANAGER, "summary": "Payout bug", "points": 2}).json()
        assert made["applied"] == []  # suggest mode: created, nobody assigned yet
        state = http.get("/api/autoassign", params={"actor": MANAGER}).json()
        assert state["mode"] == "suggest" and state["can_configure"]
        assert state["suggestions"][0]["issue_key"] == made["issue_key"] and state["suggestions"][0]["assignee_id"]
        assert not http.get("/api/autoassign", params={"actor": FREE}).json()["can_configure"]
        assert http.post("/api/autoassign/mode", json={"actor": FREE, "mode": "auto"}).status_code == 403

        switched = http.post("/api/autoassign/mode", json={"actor": MANAGER, "mode": "auto"}).json()
        assert switched["mode"] == "auto" and switched["applied"][0]["ok"]  # the waiting issue got an owner at once
        second = http.post("/api/issues", json={"actor": MANAGER, "summary": "Another one", "points": 1}).json()
        assert second["applied"][0]["ok"] and second["applied"][0]["issue_key"] == second["issue_key"]

        state = http.get("/api/autoassign", params={"actor": MANAGER}).json()
        assert state["suggestions"] == [] and {r["issue_key"] for r in state["recent"]} == {made["issue_key"], second["issue_key"]}
        assert http.post("/api/autoassign/undo", json={"actor": MANAGER, "issue_key": second["issue_key"]}).json()["ok"]
        assert http.get("/api/audit").json()["intact"]


def test_roles_through_the_api(tmp_path):
    with client(tmp_path) as http:
        seen = http.get("/api/roles", params={"actor": MANAGER}).json()
        assert [r["name"] for r in seen["roles"]] == ["manager", "engineer"] and "roles.define" in seen["mine"]
        assert http.get("/api/roles", params={"actor": FREE}).json()["mine"] == sorted(["overview.view", "issues.take", "issues.move_own", "issues.create"])

        lead = {"actor": MANAGER, "name": "tech-lead", "rights": ["issues.assign", "autoassign.configure"]}
        assert http.post("/api/roles/define", json=lead).json() == {"ok": True}
        assert http.post("/api/roles/define", json=lead).status_code == 400  # the name is taken
        assert http.post("/api/roles/define", json={**lead, "actor": FREE, "name": "mine"}).status_code == 403
        assert http.post("/api/roles/grant", json={"actor": FREE, "employee_id": FREE, "role": "manager"}).status_code == 403

        assert http.post("/api/roles/grant", json={"actor": MANAGER, "employee_id": FREE, "role": "tech-lead"}).json() == {"ok": True}
        member = next(m for m in http.get("/api/roles", params={"actor": MANAGER}).json()["members"] if m["id"] == FREE)
        assert member["granted"] == ["tech-lead"] and "issues.assign" in member["rights"]
        assert http.get("/api/autoassign", params={"actor": FREE}).json()["can_configure"]  # the role took effect

        assert http.post("/api/roles/revoke", json={"actor": MANAGER, "employee_id": FREE, "role": "tech-lead"}).json() == {"ok": True}
        assert not http.get("/api/autoassign", params={"actor": FREE}).json()["can_configure"]
        assert http.post("/api/roles/delete", json={"actor": MANAGER, "name": "tech-lead"}).json() == {"ok": True}
        kinds = [e["kind"] for e in http.get("/api/audit").json()["entries"]][::-1]
        assert kinds == ["role_defined", "role_granted", "role_revoked", "role_deleted"]
