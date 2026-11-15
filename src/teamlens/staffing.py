"""Who is in which team: adding a person to a team and removing one from it.

Plain code, like roles: no model is involved and the agent has no tool for it. The change is one
field of the HR record, the team, made through the People interface and written to the audit log.
Hiring, dismissal and positions stay with HR.

Rules:
  - it takes the right ``team.staff`` in that team: its manager has it, leadership has it in every team
  - only a person who is in no team can be added; a move between teams is a removal, then an addition
  - a person who still owns open issues cannot be removed: their work is handed over first, so no issue
    is left with an owner nobody can see
  - the team's manager is appointed by HR and is not removed here
  - roles a person was given in a team end when they leave it
  - a new team is created by leadership (``teams.create``); it starts with nobody in it and no manager,
    and gets both the same way as any other team
"""

from __future__ import annotations

import re

from teamlens.access import POSITIONS, RIGHTS, Access
from teamlens.delivery.audit import AuditLog
from teamlens.delivery.tools import Toolbox
from teamlens.hrm import Person
from teamlens.people import People

RIGHT = "team.staff"
KEY = re.compile(r"^[A-Z]{2,5}$")
POOL = "Engineering"  # people of this department with no team are waiting to be placed


async def _person(people: People, employee_id: str) -> Person:
    found = await people.person(employee_id)
    if found is None:
        raise ValueError(f"Unknown employee {employee_id}.")
    return found


def _need(access: Access, actor: Person, team: str) -> None:
    if RIGHT not in access.rights(actor, team):
        raise PermissionError(f"{actor.name} does not have the right to do this in team {team}: {RIGHTS[RIGHT].lower()}.")


async def pool(people: People) -> list[Person]:
    """People who can be added to a team: engineers the company has, in no team yet."""
    return [p for p in await people.department(POOL) if p.active and not p.team]


async def add(box: Toolbox, people: People, audit: AuditLog, actor_id: str, employee_id: str, team: str) -> Person:
    actor, target = await _person(people, actor_id), await _person(people, employee_id)
    if team not in box.teams:
        raise ValueError(f"There is no team {team}.")
    _need(box.access, actor, team)
    if not target.active:
        raise ValueError(f"{target.name} no longer works here.")
    if target.team:
        raise ValueError(f"{target.name} is in team {target.team}. Remove them from it first.")
    if target.department != POOL:
        raise ValueError(f"{target.name} works in {target.department}, not in engineering.")
    moved = await people.set_team(employee_id, team)
    audit.record("team_joined", actor_id, "-", employee=employee_id, team=team)
    return moved


async def remove(box: Toolbox, people: People, audit: AuditLog, actor_id: str, employee_id: str) -> Person:
    actor, target = await _person(people, actor_id), await _person(people, employee_id)
    team = target.team
    if not team:
        raise ValueError(f"{target.name} is in no team.")
    _need(box.access, actor, team)
    if POSITIONS.get(target.role) == "manager":
        raise PermissionError(f"{target.name} manages team {team}. A team's manager is appointed by HR and is not removed here.")
    load = next((x for x in (await box.world(actor_id, team)).overview.loads if x.employee_id == employee_id), None)
    if load and load.open_issues:
        raise ValueError(f"{target.name} still owns {len(load.open_issues)} open issue(s): {', '.join(load.open_issues[:5])}. "
                         "Hand them over first.")
    moved = await people.set_team(employee_id, None)
    dropped = box.access.leave(actor, target, team)
    audit.record("team_left", actor_id, "-", employee=employee_id, team=team, roles_dropped=dropped)
    return moved


async def create_team(box: Toolbox, people: People, audit: AuditLog, known: dict[str, str], actor_id: str,
                      key: str, name: str) -> tuple[str, str]:
    """Check a new team. The caller, which owns the list of teams, adds it; ``known`` is that list."""
    actor, key, name = await _person(people, actor_id), key.strip().upper(), " ".join(name.split())
    if "teams.create" not in box.access.rights(actor):
        raise PermissionError(f"{actor.name} does not have the right to do this: {RIGHTS['teams.create'].lower()}.")
    if not KEY.match(key):
        raise ValueError("A team key is 2-5 letters, for example RSK. Issues of the team are numbered with it.")
    if key in known or name.lower() in {n.lower() for n in known.values()}:
        raise ValueError(f"A team with the key {key} or the name '{name}' already exists.")
    if not 2 <= len(name) <= 40:
        raise ValueError("A team name is 2-40 characters.")
    audit.record("team_created", actor_id, "-", team=key, name=name)
    return key, name
