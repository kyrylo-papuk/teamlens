"""The People agent: a separate service that owns HR data and speaks A2A.

Other agents never read the HRM. They ask this agent, and it answers with minimised data:
who exists, and whether and when someone is away. Reasons for absence, pay and contact
details have no skill here at all, so no request, however it is worded, can return them.

Two kinds of requests:
  - a data part  -> answered by plain code (other agents' rules depend on these answers)
  - plain text   -> answered by an LLM agent that can only call the same minimised lookups
"""

from __future__ import annotations

import hmac
from collections.abc import Awaitable, Callable
from datetime import date
from typing import Literal

from a2a.helpers import get_data_parts, new_data_message, new_text_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    HTTPAuthSecurityScheme,
    Role,
    SecurityRequirement,
    SecurityScheme,
    StringList,
)
from pydantic import BaseModel, TypeAdapter, ValidationError
from starlette.applications import Starlette
from starlette.authentication import AuthCredentials, AuthenticationBackend, AuthenticationError, SimpleUser
from starlette.middleware import Middleware
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.responses import JSONResponse

from teamlens.hrm import Hrm

READ = "people:read"  # directory and availability
JSON, TEXT = "application/json", "text/plain"


class Caller(BaseModel):
    name: str
    scopes: set[str]


class PersonRequest(BaseModel):
    skill: Literal["person"]
    employee_id: str


class TeamRequest(BaseModel):
    skill: Literal["team"]
    team: str


class DepartmentRequest(BaseModel):
    skill: Literal["department"]
    department: str


class AvailabilityRequest(BaseModel):
    skill: Literal["availability"]
    employee_ids: list[str]
    start: date
    end: date


Request = TypeAdapter(PersonRequest | TeamRequest | DepartmentRequest | AvailabilityRequest)
Advisor = Callable[[str], Awaitable[str]]


def agent_card(url: str) -> AgentCard:
    return AgentCard(
        name="People agent",
        description="Owns HR data. Answers who works where and whether people are available. "
                    "Never shares reasons for absence, pay or contact details.",
        version="0.1.0",
        supported_interfaces=[AgentInterface(url=url, protocol_binding="JSONRPC")],
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=[JSON, TEXT],
        default_output_modes=[JSON, TEXT],
        security_schemes={"bearer": SecurityScheme(http_auth_security_scheme=HTTPAuthSecurityScheme(scheme="bearer"))},
        security_requirements=[SecurityRequirement(schemes={"bearer": StringList(list=[READ])})],
        skills=[
            AgentSkill(id="directory", name="Employee directory", input_modes=[JSON], output_modes=[JSON],
                       tags=["hr", "directory"],
                       description='Data part {"skill": "person", "employee_id"}, {"skill": "team", "team"} or '
                                   '{"skill": "department", "department"}. '
                                   "Returns id, name, department, team, role, manager, weekly hours, active."),
            AgentSkill(id="availability", name="Availability", input_modes=[JSON], output_modes=[JSON],
                       tags=["hr", "availability"],
                       description='Data part {"skill": "availability", "employee_ids", "start", "end"}. Returns, per '
                                   "person: active, available today, away periods as dates, working days in the period."),
            AgentSkill(id="staffing_question", name="Staffing questions", input_modes=[TEXT], output_modes=[TEXT],
                       tags=["hr", "staffing"], examples=["Who in team PAY works part-time?",
                                                           "Is anyone in PAY away in the next two weeks?"],
                       description="A question in plain language about roles, working hours and time away."),
        ],
    )


class PeopleExecutor(AgentExecutor):
    def __init__(self, hrm: Hrm, today: date, advisor: Advisor | None = None) -> None:
        self._hrm, self._today, self._advisor = hrm, today, advisor

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        def reply(make, payload):
            return event_queue.enqueue_event(make(payload, context_id=context.context_id, role=Role.ROLE_AGENT))

        auth = context.call_context.state.get("auth")
        if auth is None or READ not in auth.scopes:
            await reply(new_data_message, {"error": f"This caller lacks the '{READ}' scope."})
            return
        data = get_data_parts(context.message.parts)
        if data:
            await reply(new_data_message, self._lookup(data[0]))
        elif self._advisor is None:
            await reply(new_text_message, "Questions in plain language are not enabled on this deployment.")
        else:
            await reply(new_text_message, await self._advisor(context.get_user_input()))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        return None  # every request finishes in one step; there is nothing to cancel

    def _lookup(self, raw: dict) -> dict:
        try:
            request = Request.validate_python(raw)
        except ValidationError as exc:
            return {"error": f"Unknown or malformed request: {exc.errors()[0]['msg']}"}
        if isinstance(request, TeamRequest):
            return {"people": [p.model_dump(mode="json") for p in self._hrm.people(request.team)]}
        if isinstance(request, DepartmentRequest):
            return {"people": [p.model_dump(mode="json") for p in self._hrm.people(department=request.department)]}
        known = {p.id: p for p in self._hrm.people()}
        if isinstance(request, PersonRequest):
            person = known.get(request.employee_id)
            return {"person": person.model_dump(mode="json") if person else None}
        return {"availability": {
            e: self._hrm.availability(e, request.start, request.end, self._today).model_dump(mode="json")
            for e in request.employee_ids if e in known}}


class TokenAuth(AuthenticationBackend):
    def __init__(self, tokens: dict[str, Caller]) -> None:
        self._tokens = tokens

    async def authenticate(self, conn):
        if conn.url.path.startswith("/.well-known/"):
            return None  # the agent card is public: it tells callers how to authenticate
        presented = conn.headers.get("authorization", "").removeprefix("Bearer ").strip()
        for token, caller in self._tokens.items():
            if presented and hmac.compare_digest(presented, token):
                return AuthCredentials(sorted(caller.scopes)), SimpleUser(caller.name)
        raise AuthenticationError("A valid bearer token is required.")


def build_app(hrm: Hrm, today: date, tokens: dict[str, Caller], url: str, advisor: Advisor | None = None) -> Starlette:
    card = agent_card(url)
    handler = DefaultRequestHandler(agent_executor=PeopleExecutor(hrm, today, advisor),
                                    task_store=InMemoryTaskStore(), agent_card=card)
    return Starlette(
        routes=[*create_agent_card_routes(card), *create_jsonrpc_routes(handler, "/")],
        middleware=[Middleware(AuthenticationMiddleware, backend=TokenAuth(tokens),
                               on_error=lambda _conn, exc: JSONResponse({"error": str(exc)}, status_code=401))],
    )
