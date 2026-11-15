"""HRM side: the only module that reads restricted HR fields.

Everything it returns to the outside is a minimised view: ``Person`` has no salary,
``Availability`` says *whether* and *until when* someone is away, never *why*.
"""

from __future__ import annotations

from datetime import date, timedelta

from pydantic import BaseModel

from teamlens.seed import Absence, Employee


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

    def people(self, team: str | None = None, department: str | None = None) -> list[Person]:
        return [
            Person(id=e.id, name=e.name, department=e.department, team=e.team, role=e.role,
                   manager_id=e.manager_id, weekly_hours=e.weekly_hours, active=e.status == "active")
            for e in self._employees.values()
            if (team is None or e.team == team) and (department is None or e.department == department)
        ]

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
