"""Deterministic generator of a fictional company: HRM records and Jira issues.

The same seed always gives the same company, so tests and evals are reproducible.
Known problems are planted on purpose and listed in ``planted`` as ground truth.
"""

from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path

from pydantic import BaseModel

TODAY = date(2026, 11, 16)  # fixed "now" of the fictional company; the live demo passes the real date

TEAMS = {"PAY": "Payments", "LED": "Ledger", "KYC": "Onboarding", "MOB": "Mobile", "PLT": "Platform"}
OTHER_DEPARTMENTS = ["Sales", "Compliance", "Finance", "People", "Support", "Marketing", "Operations"]
ROLES = ["Backend Engineer", "Backend Engineer", "Frontend Engineer", "QA Engineer",
         "Backend Engineer", "Mobile Engineer", "DevOps Engineer", "Engineering Manager"]
FIRST = ["Dana", "Marek", "Ines", "Tomas", "Leila", "Oskar", "Priya", "Jonas", "Sofia", "Emre",
         "Hana", "Rui", "Nadia", "Viktor", "Clara", "Mateo", "Yuki", "Anton", "Bianca", "Felix"]
LAST = ["Novak", "Silva", "Haddad", "Lindqvist", "Okafor", "Petrov", "Costa", "Meyer", "Tanaka", "Rossi",
        "Kowalski", "Duarte", "Aydin", "Berg", "Moreau"]
WORK = ["Add idempotency keys to", "Fix timeout in", "Refactor", "Write tests for", "Add audit log to",
        "Investigate flaky", "Migrate", "Document", "Add rate limiting to", "Fix rounding in"]
AREAS = ["card payouts", "FX quotes", "ledger postings", "KYC document upload", "push notifications",
         "wallet balance API", "transaction export", "login flow", "deploy pipeline", "fee calculation"]


class Employee(BaseModel):
    id: str
    name: str
    email: str
    department: str
    team: str | None  # Jira project key for engineers
    role: str
    manager_id: str | None
    weekly_hours: int
    status: str  # active | terminated
    salary_eur: int  # RESTRICTED


class Absence(BaseModel):
    employee_id: str
    start: date
    end: date
    reason: str  # RESTRICTED: vacation | sick | parental


class Issue(BaseModel):
    key: str
    project: str
    summary: str
    type: str  # Story | Bug | Task
    status: str  # To Do | In Progress | In Review | Done
    assignee_id: str | None
    story_points: int
    in_sprint: bool
    created: date
    updated: date


class Company(BaseModel):
    today: date
    sprint_start: date
    sprint_end: date
    employees: list[Employee]
    absences: list[Absence]
    issues: list[Issue]
    planted: dict[str, list[str]]  # anomaly name -> employee ids or issue keys


def generate(seed: int = 7, today: date = TODAY) -> Company:
    rng = random.Random(seed)
    names = [f"{first} {last}" for first in FIRST for last in LAST]
    rng.shuffle(names)
    employees: list[Employee] = []

    def add(department: str, team: str | None, role: str, manager_id: str | None) -> Employee:
        n = len(employees) + 1
        name = names[n - 1]
        e = Employee(
            id=f"E{n:03d}", name=name, email=name.lower().replace(" ", ".") + "@halden.example",
            department=department, team=team, role=role, manager_id=manager_id,
            weekly_hours=rng.choice([40, 40, 40, 32, 20]), status="active",
            salary_eur=rng.randrange(38, 110) * 1000,
        )
        employees.append(e)
        return e

    for key in TEAMS:  # 5 teams x 8 engineers = 40
        manager = add("Engineering", key, "Engineering Manager", None)
        for role in ROLES[:-1]:
            add("Engineering", key, role, manager.id)
    while len(employees) < 150:  # everyone else exists in HRM only
        add(rng.choice(OTHER_DEPARTMENTS), None, "Specialist", None)

    engineers = [e for e in employees if e.team and e.role != "Engineering Manager"]
    by_team = {k: [e for e in engineers if e.team == k] for k in TEAMS}

    # Planted people: one per kind, in different teams.
    on_leave, terminated, overloaded, idle = (by_team[k][i] for k, i in
                                              [("PAY", 0), ("LED", 1), ("KYC", 2), ("MOB", 3)])
    terminated.status = "terminated"
    absences = [Absence(employee_id=on_leave.id, start=today - timedelta(days=2),
                        end=today + timedelta(days=5), reason="sick")]
    for e in rng.sample(engineers, 6):  # ordinary absences outside the sprint
        if e not in (on_leave, terminated, overloaded, idle):
            start = today + timedelta(days=rng.randrange(20, 50))
            absences.append(Absence(employee_id=e.id, start=start, end=start + timedelta(days=4),
                                    reason="vacation"))

    issues: list[Issue] = []
    counters = dict.fromkeys(TEAMS, 0)

    def issue(project: str, assignee: Employee | None, status: str, points: int,
              in_sprint: bool = True, stale_days: int | None = None) -> Issue:
        counters[project] += 1
        updated = today - timedelta(days=stale_days if stale_days is not None else rng.randrange(0, 4))
        it = Issue(
            key=f"{project}-{counters[project]}", project=project,
            summary=f"{rng.choice(WORK)} {rng.choice(AREAS)}", type=rng.choice(["Story", "Bug", "Task"]),
            status=status, assignee_id=assignee.id if assignee else None, story_points=points,
            in_sprint=in_sprint, created=updated - timedelta(days=rng.randrange(3, 30)), updated=updated,
        )
        issues.append(it)
        return it

    planted: dict[str, list[str]] = {
        "assigned_while_absent": [], "assigned_to_terminated": [], "overloaded": [overloaded.id],
        "idle": [idle.id], "stale_in_progress": [],
    }
    for key, team in by_team.items():
        for e in team:
            if e is idle:
                continue
            part_time = e.weekly_hours < 32  # part-timers get a load that fits their hours
            for n in range(2 if part_time else rng.randrange(2, 4)):
                status = "In Progress" if n == 0 else rng.choice(["To Do", "In Progress", "In Review", "Done"])
                issue(key, e, status, rng.choice([1, 2] if part_time else [1, 2, 3]))
        for _ in range(6):  # backlog, unassigned
            issue(key, None, "To Do", rng.choice([2, 3, 5]), in_sprint=False)

    planted["assigned_while_absent"] = [issue("PAY", on_leave, "In Progress", 5).key]
    planted["assigned_to_terminated"] = [issue("LED", terminated, "To Do", 3).key]
    for _ in range(5):
        issue("KYC", overloaded, "In Progress", 8)
    planted["stale_in_progress"] = [issue("PLT", by_team["PLT"][0], "In Progress", 3, stale_days=12).key]

    return Company(today=today, sprint_start=today - timedelta(days=7), sprint_end=today + timedelta(days=6),
                   employees=employees, absences=absences, issues=issues, planted=planted)


def write(company: Company, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("employees", "absences", "issues"):
        rows = [r.model_dump(mode="json") for r in getattr(company, name)]
        (out_dir / f"{name}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    (out_dir / "planted.json").write_text(json.dumps(company.planted, indent=1), encoding="utf-8")


if __name__ == "__main__":
    c = generate()
    write(c, Path("data/seed"))
    print(f"{len(c.employees)} employees, {len(c.absences)} absences, {len(c.issues)} issues -> data/seed")
