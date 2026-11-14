"""How the Delivery side sees people: a narrow interface with minimised data only.

``LocalPeople`` reads the HRM in-process. The People agent (A2A) implements the same
interface from another service, so the Delivery agent never touches HR records itself.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol

from teamlens.hrm import Availability, Hrm, Person


class People(Protocol):
    async def person(self, employee_id: str) -> Person | None: ...
    async def team(self, team: str) -> list[Person]: ...
    async def availability(self, employee_ids: list[str]) -> dict[str, Availability]: ...


class LocalPeople:
    def __init__(self, hrm: Hrm, today: date, period_start: date, period_end: date,
                 only: set[str] | None = None) -> None:
        """``only`` limits the directory to these employee ids (people with a Jira account)."""
        self._hrm, self._today, self._start, self._end, self._only = hrm, today, period_start, period_end, only

    def _visible(self, people: list[Person]) -> list[Person]:
        return [p for p in people if self._only is None or p.id in self._only]

    async def person(self, employee_id: str) -> Person | None:
        return next((p for p in self._visible(self._hrm.people()) if p.id == employee_id), None)

    async def team(self, team: str) -> list[Person]:
        return self._visible(self._hrm.people(team))

    async def availability(self, employee_ids: list[str]) -> dict[str, Availability]:
        return {e: self._hrm.availability(e, self._start, self._end, self._today) for e in employee_ids}
