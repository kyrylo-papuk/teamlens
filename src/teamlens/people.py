"""How the Delivery side sees people: a narrow interface with minimised data only.

``LocalPeople`` reads the HRM in-process. The People agent (A2A) implements the same
interface from another service, so the Delivery agent never touches HR records itself.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol

from teamlens.hrm import Availability, Hrm, Person
from teamlens.seed import LEADERSHIP


class People(Protocol):
    async def person(self, employee_id: str) -> Person | None: ...
    async def team(self, team: str) -> list[Person]: ...
    async def department(self, department: str) -> list[Person]: ...
    async def set_team(self, employee_id: str, team: str | None) -> Person: ...
    async def availability(self, employee_ids: list[str]) -> dict[str, Availability]: ...


class LocalPeople:
    def __init__(self, hrm: Hrm, today: date, period_start: date, period_end: date,
                 only: set[str] | None = None) -> None:
        """``only`` limits the directory to these employee ids (people with a Jira account).
        Leadership is always in it: they look at teams, they do not need an account to hold issues."""
        self._hrm, self._today, self._start, self._end, self._only = hrm, today, period_start, period_end, only

    def _visible(self, people: list[Person]) -> list[Person]:
        return _visible(people, self._only)

    async def person(self, employee_id: str) -> Person | None:
        return next((p for p in self._visible(self._hrm.people()) if p.id == employee_id), None)

    async def team(self, team: str) -> list[Person]:
        return self._visible(self._hrm.people(team))

    async def department(self, department: str) -> list[Person]:
        return self._visible(self._hrm.people(department=department))

    async def set_team(self, employee_id: str, team: str | None) -> Person:
        self._hrm.set_team(employee_id, team)
        return next(p for p in self._hrm.people() if p.id == employee_id)

    async def availability(self, employee_ids: list[str]) -> dict[str, Availability]:
        return {e: self._hrm.availability(e, self._start, self._end, self._today) for e in employee_ids}


def _visible(people: list[Person], only: set[str] | None) -> list[Person]:
    return [p for p in people if only is None or p.id in only or p.department == LEADERSHIP]


class PeopleAgentError(Exception):
    pass


class A2APeople:
    """The same interface, answered by the People agent over A2A."""

    def __init__(self, http, today: date, period_start: date, period_end: date,
                 only: set[str] | None = None) -> None:
        """``http`` is an httpx.AsyncClient with the People agent as base URL and the bearer token set."""
        self._http, self._today, self._start, self._end, self._only = http, today, period_start, period_end, only
        self._client = None

    async def _send(self, message):
        from a2a.client import ClientConfig, create_client
        from a2a.types import SendMessageRequest

        if self._client is None:
            self._client = await create_client(str(self._http.base_url),
                                               client_config=ClientConfig(streaming=False, httpx_client=self._http))
        async for event in self._client.send_message(SendMessageRequest(message=message)):
            if event.HasField("message"):
                return event.message
        raise PeopleAgentError("The People agent sent no answer.")

    async def _lookup(self, **request) -> dict:
        from a2a.helpers import get_data_parts, new_data_message
        from a2a.types import Role

        answer = await self._send(new_data_message(request, role=Role.ROLE_USER))
        data = get_data_parts(answer.parts)
        if not data or "error" in data[0]:
            raise PeopleAgentError(data[0]["error"] if data else "The People agent sent no data.")
        return data[0]

    def _visible(self, people: list[Person]) -> list[Person]:
        return _visible(people, self._only)

    async def person(self, employee_id: str) -> Person | None:
        raw = (await self._lookup(skill="person", employee_id=employee_id))["person"]
        found = self._visible([Person(**raw)]) if raw else []
        return found[0] if found else None

    async def team(self, team: str) -> list[Person]:
        return self._visible([Person(**p) for p in (await self._lookup(skill="team", team=team))["people"]])

    async def department(self, department: str) -> list[Person]:
        found = await self._lookup(skill="department", department=department)
        return self._visible([Person(**p) for p in found["people"]])

    async def set_team(self, employee_id: str, team: str | None) -> Person:
        # The People agent is reached with a read-only token, and it has no skill that changes a record.
        raise PermissionError("This deployment reads people from the People agent with a read-only token. "
                              "Who is in which team is changed on the HR side.")

    async def availability(self, employee_ids: list[str]) -> dict[str, Availability]:
        raw = await self._lookup(skill="availability", employee_ids=employee_ids,
                                 start=str(self._start), end=str(self._end))
        return {e: Availability(**a) for e, a in raw["availability"].items()}

    async def ask(self, question: str) -> str:
        """A question in plain language, answered by the People agent's own model."""
        from a2a.helpers import get_message_text, new_text_message
        from a2a.types import Role

        return get_message_text(await self._send(new_text_message(question, role=Role.ROLE_USER)))
