from fastapi.testclient import TestClient

from teamlens.delivery.model import ScriptedModel
from teamlens.seed import NEW_HIRES, generate
from teamlens.web.app import create_app

C = generate()
PAY = [e for e in C.employees if e.team == "PAY"]
MANAGER, AWAY, FREE = PAY[0].id, PAY[1].id, PAY[2].id
KYC_MANAGER = next(e.id for e in C.employees if e.team == "KYC" and e.role == "Engineering Manager")
CTO = next(e.id for e in C.employees if e.role == "Chief Technology Officer")
HIRES = [e.id for e in C.employees if e.department == "Engineering" and not e.team]
SALES = next(e.id for e in C.employees if e.department == "Sales")


def client(tmp_path) -> TestClient:
    return TestClient(create_app(ScriptedModel([]), fake=True, audit_path=tmp_path / "audit.jsonl"))


def team(http, actor=MANAGER, **params):
    return http.get("/api/overview", params={"actor": actor, **params}).json()


def test_the_company_has_engineers_waiting_for_a_team():
    assert len(HIRES) == len(NEW_HIRES) == 4 and len(C.employees) == 150


def test_a_manager_adds_a_new_person_to_the_team_and_removes_one(tmp_path):
    new = HIRES[0]
    add = lambda actor, who, to: http.post("/api/staff/add", json={"actor": actor, "employee_id": who, "team": to})  # noqa: E731
    remove = lambda actor, who: http.post("/api/staff/remove", json={"actor": actor, "employee_id": who})  # noqa: E731
    with client(tmp_path) as http:
        assert [p["id"] for p in http.get("/api/staff/pool", params={"actor": MANAGER}).json()["people"]] == HIRES
        assert http.get("/api/staff/pool", params={"actor": FREE}).status_code == 403  # an engineer does not place people
        assert add(FREE, new, "PAY").status_code == 403
        assert add(MANAGER, new, "KYC").status_code == 403  # not in someone else's team
        assert add(MANAGER, SALES, "PAY").status_code == 400  # engineering teams take engineers
        assert add(MANAGER, FREE, "PAY").status_code == 400  # already in a team

        assert add(MANAGER, new, "PAY").json() == {"ok": True, "team": "PAY"}
        seen = team(http)
        joined = next(x for x in seen["loads"] if x["employee_id"] == new)
        assert len(seen["loads"]) == 9 and joined["capacity_points"] > 0 and joined["open_issues"] == []
        assert new in [p["id"] for p in http.get("/api/people").json()["people"]]  # and can now work in TeamLens
        assert http.get("/api/me", params={"actor": new}).json()["level"] == "engineer"
        assert new not in [p["id"] for p in http.get("/api/staff/pool", params={"actor": MANAGER}).json()["people"]]

        key = seen["backlog"][0]
        assert http.post("/api/issues/assign", json={"actor": new, "issue_key": key, "assignee_id": new}).json()["ok"]
        refused = remove(MANAGER, new)  # nobody leaves a team with work still in their name
        assert refused.status_code == 400 and key in refused.json()["detail"] and "Hand them over first" in refused.json()["detail"]
        http.post("/api/issues/assign", json={"actor": MANAGER, "issue_key": key, "assignee_id": FREE})

        http.post("/api/roles/define", json={"actor": MANAGER, "name": "tech-lead", "rights": ["issues.assign"]})
        http.post("/api/roles/grant", json={"actor": MANAGER, "employee_id": new, "role": "tech-lead"})
        assert remove(MANAGER, new).json() == {"ok": True}
        assert len(team(http)["loads"]) == 8 and http.get("/api/me", params={"actor": new}).json()["rights"] == []
        assert remove(MANAGER, MANAGER).status_code == 403  # the manager is HR's appointment
        assert remove(MANAGER, AWAY).status_code == 400  # away, and still owns work

        log = http.get("/api/audit", params={"actor": MANAGER}).json()
        kinds = [e["kind"] for e in log["entries"]][::-1]
        assert log["intact"] and kinds.count("team_joined") == 1 and kinds[-2:] == ["roles_ended", "team_left"]


def test_leadership_places_people_in_any_team(tmp_path):
    new = HIRES[1]
    with client(tmp_path) as http:
        assert http.post("/api/staff/add", json={"actor": CTO, "employee_id": new, "team": "KYC"}).json()["ok"]
        assert new in [x["employee_id"] for x in team(http, CTO, team="KYC")["loads"]] and team(http, CTO, team="KYC")["can_staff"]
        assert not team(http, FREE)["can_staff"] and team(http, KYC_MANAGER)["can_staff"]
        assert http.post("/api/staff/add", json={"actor": CTO, "employee_id": HIRES[2], "team": "NOPE"}).status_code == 400
        assert http.post("/api/staff/remove", json={"actor": CTO, "employee_id": new}).json()["ok"]

        assert http.post("/api/demo/reset", json={}).json() == {"ok": True}  # the demo goes back to the teams it started with
        http.post("/api/staff/add", json={"actor": CTO, "employee_id": new, "team": "KYC"})
        http.post("/api/demo/reset", json={})
        assert len(team(http, CTO, team="KYC")["loads"]) == 8


def test_leadership_creates_a_team_staffs_it_and_names_its_manager(tmp_path):
    lead, second = HIRES[0], HIRES[1]
    with client(tmp_path) as http:
        make = lambda actor, key, name: http.post("/api/teams", json={"actor": actor, "key": key, "name": name})  # noqa: E731
        assert make(MANAGER, "RSK", "Risk").status_code == 403  # a manager does not create teams
        assert make(CTO, "risk!", "Risk").status_code == 400 and make(CTO, "PAY", "Other").status_code == 400
        assert make(CTO, "rsk", "Risk").json() == {"ok": True, "key": "RSK", "name": "Risk"}
        assert make(CTO, "RSK", "Risk again").status_code == 400

        company = http.get("/api/company", params={"actor": CTO}).json()
        new = company["teams"][-1]
        assert company["can_create"] and (new["key"], new["name"], new["people"], new["manager"]) == ("RSK", "Risk", 0, None)
        assert http.get("/api/people").json()["teams"]["RSK"] == "Risk"

        for who in (lead, second):
            assert http.post("/api/staff/add", json={"actor": CTO, "employee_id": who, "team": "RSK"}).json()["ok"]
        assert http.post("/api/roles/grant", json={"actor": CTO, "employee_id": lead, "role": "manager"}).json() == {"ok": True}
        seen = team(http, lead)  # the new manager runs the new team with the same screens and rules as any other
        assert seen["team"] == "RSK" and len(seen["loads"]) == 2 and "issues.assign" in seen["rights"] and seen["acting"][0]["id"] == lead
        made = http.post("/api/issues", json={"actor": lead, "summary": "Map the risk register", "points": 3}).json()
        assert made["issue_key"] == "RSK-1"
        assert http.post("/api/issues/assign", json={"actor": lead, "issue_key": "RSK-1", "assignee_id": second}).json()["ok"]
        assert team(http, second)["issues"]["RSK-1"]["assignee"] == second
        assert http.get("/api/company", params={"actor": CTO}).json()["teams"][-1]["people"] == 2

        kinds = [e["kind"] for e in http.get("/api/audit", params={"actor": CTO, "team": "RSK"}).json()["entries"]][::-1]
        assert kinds[:4] == ["team_created", "team_joined", "team_joined", "role_granted"]
        http.post("/api/demo/reset", json={})
        assert "RSK" not in http.get("/api/people").json()["teams"]


def test_only_the_chief_executive_decides_who_else_is_leadership(tmp_path):
    ceo = next(e.id for e in C.employees if e.role == "Chief Executive Officer")
    chosen = HIRES[3]
    with client(tmp_path) as http:
        give = lambda actor, who: http.post("/api/leadership/give", json={"actor": actor, "employee_id": who})  # noqa: E731
        take = lambda actor, who: http.post("/api/leadership/take", json={"actor": actor, "employee_id": who})  # noqa: E731
        assert give(CTO, chosen).status_code == 403 and give(MANAGER, chosen).status_code == 403
        assert give(ceo, chosen).json() == {"ok": True} and give(ceo, chosen).status_code == 400
        me = http.get("/api/me", params={"actor": chosen}).json()
        assert me["level"] == "executive" and "company.view" in me["rights"] and "leadership.appoint" not in me["rights"]
        assert me["position"] == "DevOps Engineer"  # the HR record is untouched
        assert len(http.get("/api/company", params={"actor": chosen}).json()["teams"]) == 5
        assert chosen in [p["id"] for p in http.get("/api/people").json()["people"] if p["level"] == "executive"]
        leaders = http.get("/api/company", params={"actor": ceo}).json()
        assert leaders["can_appoint"] and [p["id"] for p in leaders["leaders"] if p["given"]] == [chosen]

        assert take(ceo, CTO).status_code == 400  # a position in the HR record is not taken away here
        assert take(CTO, chosen).status_code == 403 and take(ceo, chosen).json() == {"ok": True}
        assert http.get("/api/company", params={"actor": chosen}).status_code == 403
        kinds = [e["kind"] for e in http.get("/api/audit", params={"actor": ceo}).json()["entries"]][::-1]
        assert kinds == ["leadership_given", "leadership_taken"]
