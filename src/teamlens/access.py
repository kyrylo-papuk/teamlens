"""Who may do what: rights, roles made of rights, and roles given to people.

No language model is involved anywhere in this file, on purpose. Access control is the one
place where "the model was talked into it" must be impossible, so roles are managed by
plain code behind buttons, and the agent only ever reads the result.

Model:
  right    one thing a person may do; a fixed list, because each one is tied to a check in code
  role     a named set of rights; three are built in and cannot be edited, teams can add their own
  grant    a role given to a person in a team, on record with who gave it and when

A person's rights in a team are the union of their roles there. Everyone starts with the role
their HR record implies (engineer, or manager of their team); grants only add to that.

Two rules stop rights from spreading by themselves:
  - nobody can give out, or build a role from, a right they do not hold
  - roles are given and taken only inside the giver's own team (administrators excepted)
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from teamlens.hrm import Person

RIGHTS = {
    "overview.view": "See the team's work and load",
    "issues.take": "Take work that has no owner",
    "issues.move_own": "Move your own issues, including to Done",
    "issues.assign": "Reassign any issue in the team",
    "issues.move_any": "Move other people's issues (never to Done)",
    "issues.create": "Create issues",
    "autoassign.configure": "Confirm suggested owners and switch automatic assignment",
    "roles.grant": "Give and take away roles in the team",
    "roles.define": "Create and delete the team's own roles",
}
BUILT_IN = {
    "admin": set(RIGHTS),
    "manager": set(RIGHTS),
    "engineer": {"overview.view", "issues.take", "issues.move_own", "issues.create"},
}
NAME = re.compile(r"^[a-z][a-z0-9-]{1,30}$")


class Role(BaseModel):
    name: str
    rights: list[str]
    built_in: bool = False
    team: str | None = None
    created_by: str | None = None


class Grant(BaseModel):
    employee_id: str
    role: str
    team: str
    by: str
    at: str


class Access:
    def __init__(self, path: Path | None = None, admins: set[str] | None = None, audit=None) -> None:
        """Without a path nothing is saved: every person just has the role their HR record implies."""
        self._path, self._admins, self._audit = path, set(admins or ()), audit
        self._custom: list[Role] = []
        self._grants: list[Grant] = []
        if path and path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            self._custom = [Role(**r) for r in saved["roles"]]
            self._grants = [Grant(**g) for g in saved["grants"]]

    # --- reading ----------------------------------------------------------------------------

    def roles(self, team: str) -> list[Role]:
        built_in = [Role(name=n, rights=sorted(r), built_in=True) for n, r in BUILT_IN.items()]
        return built_in + [r for r in self._custom if r.team == team]

    def _role(self, name: str, team: str) -> Role | None:
        return next((r for r in self.roles(team) if r.name == name), None)

    def base_role(self, person: Person) -> str | None:
        if person.id in self._admins:
            return "admin"
        if not person.team or not person.active:
            return None
        return "manager" if person.role == "Engineering Manager" else "engineer"

    def granted(self, person: Person, team: str) -> list[str]:
        return [g.role for g in self._grants if g.employee_id == person.id and g.team == team]

    def rights(self, person: Person, team: str | None = None) -> set[str]:
        """Rights of ``person`` in ``team`` (their own team by default)."""
        team = team or person.team
        if person.id in self._admins:
            return set(RIGHTS)
        if not team or not person.active:
            return set()
        held = self.granted(person, team) + ([self.base_role(person)] if team == person.team else [])
        return {right for name in held if name and (role := self._role(name, team)) for right in role.rights}

    # --- changing ---------------------------------------------------------------------------

    def _need(self, actor: Person, right: str, team: str) -> set[str]:
        mine = self.rights(actor, team)
        if right not in mine:
            raise PermissionError(f"You do not have the right to do this in team {team}: {RIGHTS[right].lower()}.")
        return mine

    def _save(self, kind: str, actor: Person, **data) -> None:
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps({"roles": [r.model_dump() for r in self._custom],
                                              "grants": [g.model_dump() for g in self._grants]}, indent=1), encoding="utf-8")
        if self._audit:
            self._audit.record(kind, actor.id, "-", **data)

    def grant(self, actor: Person, target: Person, role_name: str) -> None:
        team = target.team
        if not team or not target.active:
            raise ValueError(f"{target.name} is not an active member of a team.")
        mine = self._need(actor, "roles.grant", team)
        role = self._role(role_name, team)
        if role is None:
            raise ValueError(f"Team {team} has no role '{role_name}'.")
        if role.name == "admin":
            raise PermissionError("Administrators are set in configuration, not given as a role.")
        beyond = sorted(set(role.rights) - mine)
        if beyond:
            raise PermissionError(f"You cannot give a role with rights you do not hold yourself: {', '.join(beyond)}.")
        if role_name in self.granted(target, team) or (team == target.team and role_name == self.base_role(target)):
            raise ValueError(f"{target.name} already has the role '{role_name}'.")
        self._grants.append(Grant(employee_id=target.id, role=role_name, team=team, by=actor.id,
                                  at=datetime.now(timezone.utc).isoformat(timespec="seconds")))
        self._save("role_granted", actor, employee=target.id, role=role_name, team=team)

    def revoke(self, actor: Person, target: Person, role_name: str) -> None:
        team = target.team or ""
        self._need(actor, "roles.grant", team)
        mine = [g for g in self._grants if g.employee_id == target.id and g.team == team and g.role == role_name]
        if not mine:
            raise ValueError(f"{target.name} was not given the role '{role_name}'. A role that comes from the "
                             "HR record cannot be taken away here.")
        self._grants.remove(mine[0])
        self._save("role_revoked", actor, employee=target.id, role=role_name, team=team)

    def define(self, actor: Person, name: str, rights: list[str]) -> Role:
        team = actor.team or ""
        mine = self._need(actor, "roles.define", team)
        if not NAME.match(name):
            raise ValueError("A role name is 2-31 characters: lowercase letters, digits and hyphens, starting with a letter.")
        if self._role(name, team):
            raise ValueError(f"A role named '{name}' already exists.")
        unknown = sorted(set(rights) - set(RIGHTS))
        if unknown or not rights:
            raise ValueError(f"Unknown rights: {', '.join(unknown)}." if unknown else "A role needs at least one right.")
        beyond = sorted(set(rights) - mine)
        if beyond:
            raise PermissionError(f"You cannot build a role from rights you do not hold yourself: {', '.join(beyond)}.")
        role = Role(name=name, rights=sorted(set(rights)), team=team, created_by=actor.id)
        self._custom.append(role)
        self._save("role_defined", actor, role=name, rights=role.rights, team=team)
        return role

    def delete(self, actor: Person, name: str) -> None:
        team = actor.team or ""
        self._need(actor, "roles.define", team)
        role = self._role(name, team)
        if role is None or role.built_in:
            raise ValueError("Built-in roles cannot be deleted." if role else f"Team {team} has no role '{name}'.")
        self._custom.remove(role)
        self._grants = [g for g in self._grants if not (g.team == team and g.role == name)]
        self._save("role_deleted", actor, role=name, team=team)
