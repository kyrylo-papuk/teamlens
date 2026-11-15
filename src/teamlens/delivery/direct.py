"""A change a person makes with a button instead of asking the agent.

There is one way into Jira, whoever asks: the step is checked by the same policy that checks
the agent's plans, applied by the same code, and written to the same audit log. What a button
skips is the model and the approval step, because the person pressing it is the approval.
"""

from __future__ import annotations

from pydantic import BaseModel

from teamlens.delivery.audit import AuditLog
from teamlens.delivery.policy import Step, check
from teamlens.delivery.tools import Toolbox, ToolFailure


class Done(BaseModel):
    ok: bool
    rule: str = ""  # the rule that refused it, when it was refused
    detail: str


async def apply(box: Toolbox, audit: AuditLog, actor_id: str, step: Step) -> Done:
    world = await box.world(actor_id)
    verdict = check([step], world)[0]
    done = Done(ok=verdict.allowed, rule=verdict.rule, detail=verdict.reason)
    if verdict.allowed:
        try:
            done.detail = await box.execute(step, world.actor)
        except ToolFailure as exc:
            done = Done(ok=False, rule="jira", detail=f"Jira refused: {exc}")
    # Refusals are recorded too: who tried what they were not allowed to do is what an audit is for.
    audit.record("direct_change", actor_id, "-", step=step.model_dump(exclude={"reason"}, exclude_none=True),
                 allowed=done.ok, rule=done.rule, result=done.detail)
    return done
