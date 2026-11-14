"""Rules every proposed change must pass. Plain code: the model proposes, this decides.

Checked twice: when a plan is proposed, and again right before each step runs,
because Jira and availability can change while a plan waits for approval.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from teamlens.access import RIGHTS
from teamlens.hrm import Person
from teamlens.jira import STATUSES
from teamlens.overview import OVERLOAD_RATIO, Overview
from teamlens.seed import Issue

MAX_STEPS = 8
MAX_SUMMARY = 200
PlanTool = Literal["assign_issue", "transition_issue", "create_issue"]

# The same rules in words, shown to the model so it can refuse up front instead of learning them by being refused.
# The code below is what decides; this text only saves a round trip.
RULES = f"""- What a person may change depends on their rights in the team. Without the right, the change is refused.
- People change issues of their own team only.
- An issue that is Done cannot be changed. Only the owner of an issue can move it to Done.
- Work can be assigned only to a member of the same team who is at work today.
- An assignment must not take a person above {OVERLOAD_RATIO:.0%} of their sprint capacity.
- New issues are created unassigned, in the backlog.
- A plan has at most {MAX_STEPS} steps."""


class Step(BaseModel):
    tool: PlanTool
    issue_key: str | None = None
    assignee_id: str | None = None
    status: str | None = None
    summary: str | None = None
    reason: str


class Plan(BaseModel):
    summary: str
    steps: list[Step]


class Verdict(BaseModel):
    allowed: bool
    rule: str = ""
    reason: str = ""


class World(BaseModel):
    """What the policy knows at one moment: who asks, the team's load, the team's issues."""
    actor: Person
    rights: set[str] = set()  # what the actor may do in their team, from teamlens.access
    overview: Overview
    issues: dict[str, Issue]


def _deny(rule: str, reason: str) -> Verdict:
    return Verdict(allowed=False, rule=rule, reason=reason)


def check(steps: list[Step], world: World) -> list[Verdict]:
    """One verdict per step, in order. Earlier steps count towards later capacity checks."""
    actor = world.actor
    if len(steps) > MAX_STEPS:
        return [_deny("plan_size", f"A plan may have at most {MAX_STEPS} steps; this one has {len(steps)}.")] * len(steps)
    if not actor.team:
        return [_deny("actor_team", f"{actor.name} is not a member of an engineering team.")] * len(steps)
    rights = world.rights

    def lacks(*any_of: str) -> Verdict | None:
        if rights.intersection(any_of):
            return None
        return _deny("right", f"{actor.name} does not have the right to do this: {RIGHTS[any_of[0]].lower()}.")

    loads = {x.employee_id: x for x in world.overview.loads}
    owner = {key: issue.assignee_id for key, issue in world.issues.items()}  # ownership as the plan unfolds
    added = dict.fromkeys(loads, 0)
    verdicts: list[Verdict] = []

    for step in steps:
        if step.tool == "create_issue":
            if refusal := lacks("issues.create"):
                verdicts.append(refusal)
            elif not (step.summary or "").strip() or len(step.summary) > MAX_SUMMARY:
                verdicts.append(_deny("create_summary", f"A new issue needs a summary of 1-{MAX_SUMMARY} characters."))
            elif step.assignee_id is not None:
                verdicts.append(_deny("create_unassigned", "New issues go to the backlog unassigned; assign in a later step."))
            else:
                verdicts.append(Verdict(allowed=True))
            continue

        issue = world.issues.get(step.issue_key or "")
        if issue is None:
            verdicts.append(_deny("issue_scope", f"{step.issue_key} is not an issue of team {actor.team}."))
        elif issue.status == "Done":
            verdicts.append(_deny("done_is_final", f"{issue.key} is Done and cannot be changed."))
        elif step.tool == "transition_issue":
            if step.status not in STATUSES:
                verdicts.append(_deny("status_unknown", f"Status must be one of: {', '.join(STATUSES)}."))
            elif step.status == issue.status:
                verdicts.append(_deny("no_change", f"{issue.key} is already '{issue.status}'."))
            elif owner[issue.key] == actor.id:  # your own issue: any status, including Done
                verdicts.append(lacks("issues.move_own") or Verdict(allowed=True))
            elif step.status == "Done":
                verdicts.append(_deny("humans_close_work", "Only the person who did the work moves an issue to Done."))
            else:
                verdicts.append(lacks("issues.move_any") or Verdict(allowed=True))
        else:  # assign_issue
            taking = step.assignee_id == actor.id and owner[issue.key] is None
            dropping = step.assignee_id is None and owner[issue.key] == actor.id
            needed = ("issues.take", "issues.assign") if taking else \
                     ("issues.move_own", "issues.assign") if dropping else ("issues.assign",)
            verdicts.append(lacks(*needed) or _assign(step, issue, loads, owner, added))
            if verdicts[-1].allowed:
                points = issue.story_points if issue.in_sprint else 0
                if step.assignee_id in added:
                    added[step.assignee_id] += points
                if owner[issue.key] in added:
                    added[owner[issue.key]] -= points
                owner[issue.key] = step.assignee_id
    return verdicts


def _assign(step: Step, issue: Issue, loads: dict, owner: dict, added: dict) -> Verdict:
    target = step.assignee_id
    if target == owner[issue.key]:
        return _deny("no_change", f"{issue.key} already belongs to {target or 'nobody'}.")
    if target is None:
        return Verdict(allowed=True)  # unassigning is always safe
    load = loads.get(target)
    if load is None:
        return _deny("assignee_team", f"{target} is not a member of this team.")
    if not load.availability.active:
        return _deny("assignee_active", f"{load.name} no longer works here.")
    if not load.availability.available_today:
        return _deny("assignee_available", f"{load.name} is away until {load.availability.unavailable_until}.")
    points = load.sprint_points + added[target] + (issue.story_points if issue.in_sprint else 0)
    limit = OVERLOAD_RATIO * load.capacity_points
    if points > limit:
        return _deny("assignee_capacity", f"{load.name} would have {points} sprint points; the limit is {limit:g} "
                                          f"({OVERLOAD_RATIO:.0%} of a capacity of {load.capacity_points:g}).")
    return Verdict(allowed=True)
