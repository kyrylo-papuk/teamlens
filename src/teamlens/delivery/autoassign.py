"""Automatic assignment of sprint work that nobody owns. The owner is picked by plain code.

The choice has to be explainable to the person who gets the work and to an auditor, so it is a
rule anyone can check by hand: among the people who are at work, pick the one who ends up
with the lowest load, preferring someone whose role fits the kind of work when triage is sure
what kind it is. The explanation goes into the Jira issue and into the audit log.

Acting without a person's approval is a setting of the team, not a default:
  suggest  the system proposes an owner and someone with the right to assign confirms it
  auto     the system assigns by itself; whoever switched it on is on record and can undo it
In both modes the same policy that guards the agent's plans checks the assignment. On its own
the system also never takes anyone above 100% of capacity: the zone up to the 150% limit is
a judgement call and stays with people.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from teamlens.access import RIGHTS
from teamlens.delivery.audit import AuditLog
from teamlens.delivery.policy import Step, World, check
from teamlens.delivery.tools import Toolbox, ToolFailure
from teamlens.seed import Issue
from teamlens.triage import CONFIDENT, Prediction, Triage
from teamlens.triage.dataset import AREA_ROLE

AUTO_MAX_LOAD = 1.0
SYSTEM_ACTOR = "autoassign"
Mode = Literal["suggest", "auto"]


class Candidate(BaseModel):
    employee_id: str
    name: str
    role: str
    fits_the_work: bool = False  # their role matches the kind of work triage saw in the title
    points_after: int
    capacity: float
    load_after: float


class Suggestion(BaseModel):
    issue_key: str
    summary: str
    points: int
    assignee_id: str | None
    assignee_name: str | None
    reason: str
    candidates: list[Candidate]  # everyone considered, best first
    area: str | None = None  # kind of work, from triage
    area_confidence: float | None = None


class Applied(BaseModel):
    issue_key: str
    assignee_id: str | None
    ok: bool
    detail: str


def suggest(issue: Issue, world: World, already: dict[str, int] | None = None,
            area: Prediction | None = None) -> Suggestion:
    """``already`` holds points handed out earlier in the same pass, so several issues spread out."""
    already = already or {}
    points = issue.story_points if issue.in_sprint else 0
    wanted = AREA_ROLE[area.area] if area and area.confidence >= CONFIDENT else None
    considered = sorted(
        (Candidate(employee_id=x.employee_id, name=x.name, capacity=x.capacity_points, role=x.role,
                   fits_the_work=x.role == wanted,
                   points_after=(after := x.sprint_points + already.get(x.employee_id, 0) + points),
                   load_after=round(after / x.capacity_points, 2))
         for x in world.overview.loads
         if x.availability.active and x.availability.available_today and x.capacity_points > 0
         and x.role != "Engineering Manager"),
        key=lambda c: (c.load_after, c.points_after, c.employee_id))
    fits = [c for c in considered if c.load_after <= AUTO_MAX_LOAD]
    base = {"issue_key": issue.key, "summary": issue.summary, "points": issue.story_points, "candidates": considered,
            "area": area.area if area else None, "area_confidence": area.confidence if area else None}
    if not fits:
        why = "nobody is at work" if not considered else \
              f"it would take everyone above {AUTO_MAX_LOAD:.0%} of capacity; the least loaded would be " \
              f"{considered[0].name} at {considered[0].load_after:.0%}"
        return Suggestion(**base, assignee_id=None, assignee_name=None, reason=f"No automatic owner: {why}.")
    best = fits[0]
    load = f"{best.points_after} of {best.capacity:g} points ({best.load_after:.0%})"
    reason = f"{best.name} is at work and ends up with the lowest load in the team: {load}."
    if wanted:
        # Only a classifier that reports a probability gets a percentage; Claude's answer is a plain label.
        sure = f" ({area.confidence:.0%} sure)" if area.confidence < 1 else ""
        kind = f"This looks like {area.area} work{sure}."
        specialists = [c for c in fits if c.fits_the_work]
        if specialists:
            best = specialists[0]
            load = f"{best.points_after} of {best.capacity:g} points ({best.load_after:.0%})"
            reason = f"{kind} {best.name} is a {best.role} at work and ends up at {load}."
        else:
            reason = f"{kind} No {wanted} can take it without going over capacity, so it goes by load alone. {reason}"
    return Suggestion(**base, assignee_id=best.employee_id, assignee_name=best.name, reason=reason)


class AutoAssigner:
    def __init__(self, box: Toolbox, audit: AuditLog, settings_path: Path, triage: Triage | None = None) -> None:
        self._box, self._audit, self._path, self._triage = box, audit, settings_path, triage

    async def _area(self, issue: Issue) -> Prediction | None:
        """The kind of work, from triage. It only decides who is considered first, never whether a rule holds."""
        return (await self._triage.classify([issue.summary]))[0] if self._triage else None

    # --- team setting -----------------------------------------------------------------------

    def setting(self, team: str) -> dict:
        saved = json.loads(self._path.read_text(encoding="utf-8")) if self._path.exists() else {}
        return saved.get(team, {"mode": "suggest", "set_by": None, "at": None})

    async def set_mode(self, actor_id: str, mode: Mode) -> dict:
        world = await self._box.world(actor_id)
        self._require_assign_rights(world)
        saved = json.loads(self._path.read_text(encoding="utf-8")) if self._path.exists() else {}
        saved[world.actor.team] = {"mode": mode, "set_by": actor_id,
                                   "at": datetime.now(UTC).isoformat(timespec="seconds")}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(saved, indent=1), encoding="utf-8")
        self._audit.record("autoassign_mode", actor_id, "-", team=world.actor.team, mode=mode)
        return saved[world.actor.team]

    @staticmethod
    def _require_assign_rights(world: World) -> None:
        if "autoassign.configure" not in world.rights:
            raise PermissionError(f"{world.actor.name} does not have the right to do this: "
                                  f"{RIGHTS['autoassign.configure'].lower()}.")

    # --- suggestions and assignment ---------------------------------------------------------

    @staticmethod
    def unowned(world: World) -> list[Issue]:
        return sorted((i for i in world.issues.values() if i.in_sprint and i.assignee_id is None and i.status != "Done"),
                      key=lambda i: (-i.story_points, i.key))

    async def suggestions(self, actor_id: str, team: str | None = None) -> list[Suggestion]:
        world = await self._box.world(actor_id, team)
        out, already = [], {}
        for issue in self.unowned(world):
            s = suggest(issue, world, already, await self._area(issue))
            if s.assignee_id:
                already[s.assignee_id] = already.get(s.assignee_id, 0) + issue.story_points
            out.append(s)
        return out

    async def assign(self, actor_id: str, issue_key: str, by_system: bool = False) -> Applied:
        """Recomputes the suggestion on fresh data and applies it. The caller never names the assignee."""
        world = await self._box.world(actor_id)
        if not by_system:
            self._require_assign_rights(world)
        issue = world.issues.get(issue_key)
        if issue is None or issue.assignee_id is not None or issue.status == "Done":
            return Applied(issue_key=issue_key, assignee_id=None, ok=False, detail=f"{issue_key} is not open, unowned work of this team.")
        s = suggest(issue, world, area=await self._area(issue))
        if s.assignee_id is None:
            self._audit.record("auto_assign", SYSTEM_ACTOR if by_system else actor_id, "-", issue=issue_key,
                               assignee=None, on_behalf_of=actor_id, reason=s.reason)
            return Applied(issue_key=issue_key, assignee_id=None, ok=False, detail=s.reason)
        how = "Assigned automatically" if by_system else "Assigned from a suggestion"
        step = Step(tool="assign_issue", issue_key=issue_key, assignee_id=s.assignee_id, reason=f"{how}. {s.reason}")
        verdict = check([step], world)[0]
        if verdict.allowed:
            try:
                await self._box.execute(step, world.actor)
            except ToolFailure as exc:
                verdict.allowed, verdict.reason = False, f"Jira refused: {exc}"
        self._audit.record("auto_assign", SYSTEM_ACTOR if by_system else actor_id, "-", issue=issue_key,
                           assignee=s.assignee_id if verdict.allowed else None, on_behalf_of=actor_id,
                           reason=s.reason if verdict.allowed else verdict.reason, area=s.area,
                           area_confidence=s.area_confidence, candidates=[c.model_dump() for c in s.candidates])
        return Applied(issue_key=issue_key, assignee_id=s.assignee_id if verdict.allowed else None, ok=verdict.allowed,
                       detail=s.reason if verdict.allowed else verdict.reason)

    async def undo(self, actor_id: str, issue_key: str) -> Applied:
        world = await self._box.world(actor_id)
        self._require_assign_rights(world)
        step = Step(tool="assign_issue", issue_key=issue_key, assignee_id=None, reason="Automatic assignment undone.")
        verdict = check([step], world)[0]
        if verdict.allowed:
            await self._box.execute(step, world.actor)
        self._audit.record("auto_assign_undone", actor_id, "-", issue=issue_key, ok=verdict.allowed, reason=verdict.reason)
        return Applied(issue_key=issue_key, assignee_id=None, ok=verdict.allowed, detail=verdict.reason or "Unassigned.")

    async def sweep(self, team: str, manager_id: str) -> list[Applied]:
        """One pass in auto mode: every unowned sprint issue gets an owner or a recorded reason why not."""
        setting = self.setting(team)
        if setting["mode"] != "auto":
            return []
        actor = setting["set_by"] or manager_id  # the system acts for whoever switched auto mode on
        world = await self._box.world(actor)
        return [await self.assign(actor, issue.key, by_system=True) for issue in self.unowned(world)]
