"""Who works on what, and what looks wrong. Plain code, no LLM.

Agents later explain and act on these findings; they never compute them.
"""

from __future__ import annotations

import sys
from datetime import date

from pydantic import BaseModel

from teamlens.hrm import Availability, Hrm, Person
from teamlens.seed import Issue, generate

POINTS_PER_WEEK = 5  # assumed velocity of one full-time engineer
OVERLOAD_RATIO = 1.5  # open sprint points above 150% of capacity
STALE_DAYS = 7  # "in progress" with no update for a week


class PersonLoad(BaseModel):
    employee_id: str
    name: str
    team: str
    role: str
    open_issues: list[str]
    sprint_points: int
    capacity_points: float
    load: float | None  # sprint_points / capacity_points
    availability: Availability


class Finding(BaseModel):
    kind: str  # assigned_while_absent | assigned_to_terminated | overloaded | idle | stale_in_progress
    employee_id: str | None
    issue_keys: list[str]
    detail: str


class Overview(BaseModel):
    today: date
    loads: list[PersonLoad]
    findings: list[Finding]


def build(people: list[Person], availability: dict[str, Availability], issues: list[Issue],
          today: date) -> Overview:
    """``availability`` must cover the sprint period for every person in ``people``."""
    loads: list[PersonLoad] = []
    findings: list[Finding] = []
    for p in people:
        if not p.team:
            continue  # not in Jira
        av = availability[p.id]
        open_ = [i for i in issues if i.assignee_id == p.id and i.status != "Done"]
        keys = [i.key for i in open_]
        points = sum(i.story_points for i in open_ if i.in_sprint)
        weeks = av.working_days / 5
        capacity = round(POINTS_PER_WEEK * weeks * p.weekly_hours / 40, 1) if av.active else 0.0
        load = round(points / capacity, 2) if capacity else None
        loads.append(PersonLoad(employee_id=p.id, name=p.name, team=p.team, role=p.role, open_issues=keys,
                                sprint_points=points, capacity_points=capacity, load=load, availability=av))

        if not av.active:
            if keys:
                findings.append(Finding(kind="assigned_to_terminated", employee_id=p.id, issue_keys=keys,
                                        detail=f"{p.name} has left the company but still owns {len(keys)} open issue(s)."))
        elif not av.available_today:
            if keys:
                findings.append(Finding(kind="assigned_while_absent", employee_id=p.id, issue_keys=keys,
                                        detail=f"{p.name} is away until {av.unavailable_until} and owns {len(keys)} open issue(s)."))
        elif load is not None and load > OVERLOAD_RATIO:
            findings.append(Finding(kind="overloaded", employee_id=p.id, issue_keys=keys,
                                    detail=f"{p.name} has {points} sprint points against a capacity of {capacity}."))
        elif not keys and p.role != "Engineering Manager":
            findings.append(Finding(kind="idle", employee_id=p.id, issue_keys=[],
                                    detail=f"{p.name} is available and has no open issues."))

    team_keys = {p.team for p in people if p.team}
    for i in issues:
        age = (today - i.updated).days
        if i.project in team_keys and i.status in ("In Progress", "In Review") and age >= STALE_DAYS:
            findings.append(Finding(kind="stale_in_progress", employee_id=i.assignee_id, issue_keys=[i.key],
                                    detail=f"{i.key} is '{i.status}' with no update for {age} days."))
    return Overview(today=today, loads=loads, findings=findings)


def from_company(team: str | None = None) -> Overview:
    c = generate()
    hrm = Hrm(c.employees, c.absences)
    people = hrm.people(team)
    availability = {p.id: hrm.availability(p.id, c.sprint_start, c.sprint_end, c.today) for p in people}
    return build(people, availability, c.issues, c.today)


def show(o: Overview) -> None:
    print(f"{'team':5}{'person':22}{'role':21}{'open':>5}{'pts':>5}{'cap':>6}{'load':>6}  status")
    for x in o.loads:
        av = x.availability
        status = "left" if not av.active else "ok" if av.available_today else f"away until {av.unavailable_until}"
        load = "-" if x.load is None else f"{x.load:.0%}"
        print(f"{x.team:5}{x.name:22}{x.role:21}{len(x.open_issues):>5}{x.sprint_points:>5}"
              f"{x.capacity_points:>6}{load:>6}  {status}")
    print(f"\n{len(o.findings)} finding(s):")
    for f in o.findings:
        print(f"  [{f.kind}] {f.detail}")


if __name__ == "__main__":
    show(from_company(sys.argv[1] if len(sys.argv) > 1 else None))
