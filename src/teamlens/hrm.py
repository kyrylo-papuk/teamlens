"""HRM side: the only module that reads restricted HR fields.

Everything it returns to the outside is a minimised view: ``Person`` has no salary,
``Availability`` says *whether* and *until when* someone is away, never *why*.
"""

from __future__ import annotations

from datetime import date, timedelta

from pydantic import BaseModel

from teamlens.seed import TEAMS, Absence, Employee


class Person(BaseModel):
    id: str
    name: str
    department: str
    team: str | None
    role: str
    manager_id: str | None
    weekly_hours: int
    active: bool


class Availability(BaseModel):
    employee_id: str
    active: bool
    available_today: bool
    unavailable_until: date | None  # last day away, if away today
    away: list[tuple[date, date]] = []  # time away that overlaps the period: dates only, never the reason
    working_days: int  # days the person can work in the period
    period_working_days: int  # working days in the period for a person with no absences


def _weekdays(start: date, end: date) -> list[date]:
    days = (start + timedelta(days=n) for n in range((end - start).days + 1))
    return [d for d in days if d.weekday() < 5]


class Hrm:
    def __init__(self, employees: list[Employee], absences: list[Absence]) -> None:
        self._employees = {e.id: e for e in employees}
        self._absences = absences
        self._first = {e.id: (e.team, e.manager_id) for e in employees}  # teams as the HR system had them at the start
        self.teams: dict[str, str] = dict(TEAMS)  # key -> name; leadership can add to it

    def people(self, team: str | None = None, department: str | None = None) -> list[Person]:
        return [
            Person(id=e.id, name=e.name, department=e.department, team=e.team, role=e.role,
                   manager_id=e.manager_id, weekly_hours=e.weekly_hours, active=e.status == "active")
            for e in self._employees.values()
            if (team is None or e.team == team) and (department is None or e.department == department)
        ]

    def set_team(self, employee_id: str, team: str | None) -> None:
        """Move a person into a team or out of it. The one thing this module changes; see teamlens/staffing.py."""
        e = self._employees[employee_id]
        e.team = team
        e.manager_id = next((m.id for m in self._employees.values() if team and m.team == team
                             and m.role == "Engineering Manager"), None)

    def add_team(self, key: str, name: str) -> None:
        self.teams[key] = name

    def changes(self) -> dict:
        """What differs from the start: teams that were created, and everyone whose team is not the one they began in."""
        return {"created": {k: n for k, n in self.teams.items() if k not in TEAMS},
                "moves": {e.id: e.team for e in self._employees.values() if e.team != self._first[e.id][0]}}

    def replay(self, changes: dict | None = None) -> None:
        """Back to the starting teams, then ``changes`` on top."""
        changes = changes or {"created": {}, "moves": {}}
        self.teams = {**TEAMS, **changes["created"]}
        for e in self._employees.values():
            e.team, e.manager_id = self._first[e.id]
        for employee_id, team in changes["moves"].items():
            self.set_team(employee_id, team)

    def availability(self, employee_id: str, start: date, end: date, today: date) -> Availability:
        e = self._employees[employee_id]
        mine = [a for a in self._absences if a.employee_id == employee_id]
        period = _weekdays(start, end)
        away_now = [a for a in mine if a.start <= today <= a.end]
        return Availability(
            employee_id=employee_id,
            active=e.status == "active",
            available_today=e.status == "active" and not away_now,
            unavailable_until=max(a.end for a in away_now) if away_now else None,
            away=sorted((a.start, a.end) for a in mine if a.start <= end and a.end >= start),
            working_days=sum(1 for d in period if not any(a.start <= d <= a.end for a in mine)),
            period_working_days=len(period),
        )

    def absence_records(self, employee_id: str) -> list[Absence]:
        """RESTRICTED: includes the reason. Must not leave the People side."""
        return [a for a in self._absences if a.employee_id == employee_id]
