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
