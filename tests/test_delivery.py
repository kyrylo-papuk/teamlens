import json
from contextlib import asynccontextmanager

from mcp import Client

from teamlens.delivery.agent import DeliveryAgent
from teamlens.delivery.audit import AuditLog
from teamlens.delivery.model import ScriptedModel, call, say
from teamlens.delivery.policy import Step, check
from teamlens.delivery.tools import Toolbox
from teamlens.hrm import Hrm
from teamlens.jira import FakeJira
from teamlens.mcp_servers import jira_server
from teamlens.people import LocalPeople
from teamlens.seed import generate

C = generate()
PAY = [e for e in C.employees if e.team == "PAY"]
MANAGER, AWAY, FREE = PAY[0].id, PAY[1].id, PAY[2].id  # AWAY is the planted absent engineer
AWAY_ISSUE = C.planted["assigned_while_absent"][0]
OVERLOADED = C.planted["overloaded"][0]  # works in team KYC


@asynccontextmanager
async def env(tmp_path):
    jira = FakeJira(C.issues, C.today)
    people = LocalPeople(Hrm(C.employees, C.absences), C.today, C.sprint_start, C.sprint_end)
    async with Client(jira_server(jira)) as client:
        box = Toolbox(client, people, C.today)
        audit = AuditLog(tmp_path / "audit.jsonl")

        def agent(*replies):
            model = ScriptedModel(list(replies))
            return DeliveryAgent(model, box, audit, C.today), model
        yield jira, box, audit, agent


def step(**kw):
    return Step(reason="test", **kw)


def plan(*steps, _id="plan_1"):
    full = [{"issue_key": None, "assignee_id": None, "status": None, "summary": None, "reason": "test", **s} for s in steps]
    return call("propose_plan", _id=_id, summary="test plan", steps=full)


# --- policy -----------------------------------------------------------------------------------

async def test_policy_rules(tmp_path):
    async with env(tmp_path) as (_, box, _, _):
        world = await box.world(MANAGER)
        done = next(i.key for i in world.issues.values() if i.status == "Done")
        theirs = next(i.key for i in world.issues.values() if i.assignee_id == FREE and i.status != "Done")
        cases = {
            "ok": step(tool="assign_issue", issue_key=AWAY_ISSUE, assignee_id=FREE),
            "assignee_available": step(tool="assign_issue", issue_key=theirs, assignee_id=AWAY),
            "assignee_team": step(tool="assign_issue", issue_key=AWAY_ISSUE, assignee_id=OVERLOADED),
            "issue_scope": step(tool="assign_issue", issue_key="KYC-1", assignee_id=FREE),
            "done_is_final": step(tool="assign_issue", issue_key=done, assignee_id=FREE),
            "humans_close_work": step(tool="transition_issue", issue_key=AWAY_ISSUE, status="Done"),
            "create_summary": step(tool="create_issue", summary=" "),
        }
        for expected, s in cases.items():
            verdict = check([s], world)[0]
            if expected == "ok":
                assert verdict.allowed
            else:
                assert not verdict.allowed and verdict.rule == expected, (expected, verdict)


async def test_policy_counts_earlier_steps_towards_capacity(tmp_path):
    async with env(tmp_path) as (_, box, _, _):
        world = await box.world(MANAGER)
        target = next(x for x in world.overview.loads if x.employee_id == FREE)
        movable = [i for i in world.issues.values() if i.in_sprint and i.status != "Done" and i.assignee_id not in (FREE, None)]
        steps = [step(tool="assign_issue", issue_key=i.key, assignee_id=FREE) for i in movable]
        verdicts = check(steps[:8], world)
        assert verdicts[0].allowed and any(v.rule == "assignee_capacity" for v in verdicts)
        allowed = sum(i.story_points for i, v in zip(movable, verdicts) if v.allowed)
        assert target.sprint_points + allowed <= 1.5 * target.capacity_points


async def test_only_the_manager_can_change_issues(tmp_path):
    async with env(tmp_path) as (_, box, _, _):
        verdict = check([step(tool="assign_issue", issue_key=AWAY_ISSUE, assignee_id=FREE)], await box.world(FREE))[0]
        assert verdict.rule == "right" and "reassign any issue" in verdict.reason


# --- audit ------------------------------------------------------------------------------------

def test_audit_chain_detects_tampering(tmp_path):
    log = AuditLog(tmp_path / "a.jsonl")
    for n in range(3):
        log.record("request", "E001", "t1", text=f"q{n}")
    assert log.verify() is None
    lines = (tmp_path / "a.jsonl").read_text(encoding="utf-8").splitlines()
    lines[1] = lines[1].replace("q1", "something else")
    (tmp_path / "a.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert log.verify() == 2


# --- agent ------------------------------------------------------------------------------------

async def test_question_is_answered_from_tools(tmp_path):
    async with env(tmp_path) as (_, _, audit, agent):
        a, model = agent(call("team_overview"), say("Anton is away; his issues need an owner."))
        result = await a.ask(MANAGER, "Who is at risk?", "t1")
        assert result.outcome == "answered" and "Anton" in result.answer
        shown = model.seen[1][-1]["content"][0]["content"]  # tool result the model saw
        assert "assigned_while_absent" in shown and "sick" not in shown and "salary" not in shown
        assert [e["kind"] for e in audit.entries()] == ["request", "outcome"]


async def test_model_is_never_offered_write_tools(tmp_path):
    async with env(tmp_path) as (_, _, _, agent):
        a, model = agent(call("assign_issue", key=AWAY_ISSUE, assignee_id=FREE), say("I need to use a plan."))
        await a.ask(MANAGER, "Reassign it", "t1")
        assert {t["name"] for t in model.tools} == {"team_overview", "sprint_forecast", "get_issue", "search_issues", "propose_plan"}
        refused = model.seen[1][-1]["content"][0]
        assert refused["is_error"] and "propose_plan" in refused["content"]


async def test_approved_plan_changes_jira_and_is_audited(tmp_path):
    async with env(tmp_path) as (jira, _, audit, agent):
        a, _ = agent(plan({"tool": "assign_issue", "issue_key": AWAY_ISSUE, "assignee_id": FREE}))
        waiting = await a.ask(MANAGER, "Move Anton's work", "t1")
        assert waiting.outcome == "needs_approval" and waiting.plan["steps"][0]["issue_key"] == AWAY_ISSUE
        assert (await jira.get(AWAY_ISSUE)).assignee_id == AWAY  # nothing happens before approval

        result = await a.decide("t1", approved=True, approver_id=MANAGER)
        assert result.outcome == "executed"
        assert (await jira.get(AWAY_ISSUE)).assignee_id == FREE
        assert "on behalf of" in jira.comments[AWAY_ISSUE][0]
        assert [e["kind"] for e in audit.entries()] == ["request", "plan_proposed", "approval", "step", "outcome"]
        assert audit.verify() is None


async def test_rejection_and_foreign_approver_change_nothing(tmp_path):
    async with env(tmp_path) as (jira, _, _, agent):
        for thread, approved, approver in (("t1", False, MANAGER), ("t2", True, FREE)):
            a, _ = agent(plan({"tool": "assign_issue", "issue_key": AWAY_ISSUE, "assignee_id": FREE}))
            await a.ask(MANAGER, "Move Anton's work", thread)
            result = await a.decide(thread, approved=approved, approver_id=approver)
            assert result.outcome == "rejected"
            assert (await jira.get(AWAY_ISSUE)).assignee_id == AWAY


async def test_refused_plan_goes_back_to_the_model_with_reasons(tmp_path):
    async with env(tmp_path) as (jira, _, _, agent):
        a, model = agent(
            plan({"tool": "assign_issue", "issue_key": AWAY_ISSUE, "assignee_id": OVERLOADED}, _id="p1"),
            plan({"tool": "assign_issue", "issue_key": AWAY_ISSUE, "assignee_id": FREE}, _id="p2"),
        )
        waiting = await a.ask(MANAGER, "Move Anton's work", "t1")
        feedback = model.seen[1][-1]["content"][0]
        assert feedback["is_error"] and "not a member of this team" in feedback["content"]
        assert waiting.outcome == "needs_approval" and waiting.plan["steps"][0]["assignee_id"] == FREE


async def test_plan_that_stays_wrong_is_denied(tmp_path):
    async with env(tmp_path) as (jira, _, _, agent):
        bad = {"tool": "transition_issue", "issue_key": AWAY_ISSUE, "status": "Done"}
        a, _ = agent(plan(bad, _id="p1"), plan(bad, _id="p2"), plan(bad, _id="p3"))
        result = await a.ask(MANAGER, "Close it", "t1")
        assert result.outcome == "denied" and "Only the person who did the work" in result.answer
        assert (await jira.get(AWAY_ISSUE)).status == "In Progress"


async def test_step_is_rechecked_when_the_world_changed_after_approval(tmp_path):
    async with env(tmp_path) as (jira, _, audit, agent):
        a, _ = agent(plan({"tool": "assign_issue", "issue_key": AWAY_ISSUE, "assignee_id": FREE}))
        await a.ask(MANAGER, "Move Anton's work", "t1")
        await jira.transition(AWAY_ISSUE, "Done")  # someone finished it while the plan waited
        result = await a.decide("t1", approved=True, approver_id=MANAGER)
        assert result.outcome == "failed" and "Done and cannot be changed" in result.answer
        assert (await jira.get(AWAY_ISSUE)).assignee_id == AWAY
        assert json.dumps(audit.entries()[-2]["data"]).count("refused at execution time") == 1


# --- rights decide who may change what --------------------------------------------------------

async def test_engineer_takes_unowned_work_and_closes_their_own(tmp_path):
    async with env(tmp_path) as (jira, box, _, agent):
        world = await box.world(FREE)
        unowned = next(i.key for i in world.issues.values() if i.assignee_id is None)
        own = next(i.key for i in world.issues.values() if i.assignee_id == FREE and i.status != "Done")
        theirs = next(i.key for i in world.issues.values() if i.assignee_id not in (None, FREE) and i.status != "Done")
        allowed = [step(tool="assign_issue", issue_key=unowned, assignee_id=FREE),
                   step(tool="transition_issue", issue_key=own, status="Done"),
                   step(tool="create_issue", summary="Follow-up")]
        assert all(v.allowed for v in check(allowed, world))
        refused = {
            "right": step(tool="assign_issue", issue_key=theirs, assignee_id=FREE),          # not theirs to take
            "humans_close_work": step(tool="transition_issue", issue_key=theirs, status="Done"),
        }
        for rule, s in refused.items():
            assert check([s], world)[0].rule == rule
        assert check([step(tool="transition_issue", issue_key=theirs, status="To Do")], world)[0].rule == "right"

        a, _ = agent(plan({"tool": "assign_issue", "issue_key": unowned, "assignee_id": FREE}))
        await a.ask(FREE, "I'll take it", "t1")
        assert (await a.decide("t1", True, FREE)).outcome == "executed"
        assert (await jira.get(unowned)).assignee_id == FREE


async def test_a_granted_role_changes_what_the_policy_allows(tmp_path):
    async with env(tmp_path) as (_, box, _, _):
        manager, engineer = (await box.world(MANAGER)).actor, (await box.world(FREE)).actor
        world = await box.world(FREE)
        theirs = next(i.key for i in world.issues.values() if i.assignee_id not in (None, FREE, AWAY) and i.status != "Done")
        reassign = step(tool="assign_issue", issue_key=theirs, assignee_id=FREE)
        assert check([reassign], world)[0].rule == "right"

        box.access.define(manager, "tech-lead", ["issues.assign"])
        box.access.grant(manager, engineer, "tech-lead")
        assert check([reassign], await box.world(FREE))[0].allowed

        box.access.revoke(manager, engineer, "tech-lead")
        assert check([reassign], await box.world(FREE))[0].rule == "right"
