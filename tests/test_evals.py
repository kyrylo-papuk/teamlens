"""The eval must be able to fail. Each test feeds it an agent that misbehaves in one way."""

from teamlens.delivery.model import ScriptedModel, call, say
from teamlens.evals.runner import run_one, summarise
from teamlens.evals.scenarios import build
from teamlens.seed import generate

C = generate()
S = {s.id: s for s in build(C)}


def plan(*steps):
    full = [{"issue_key": None, "assignee_id": None, "status": None, "summary": None, "reason": "test", **s} for s in steps]
    return call("propose_plan", _id="p1", summary="test", steps=full)


async def test_a_correct_run_passes():
    key = S["a_transition"].text.split()[1]
    row = await run_one(S["a_transition"], C, ScriptedModel([plan({"tool": "transition_issue", "issue_key": key, "status": "In Review"})]))
    assert row["passed"] and row["outcome"] == "executed"


async def test_wrong_answer_is_caught():
    row = await run_one(S["q_backlog"], C, ScriptedModel([say("There are 2 unassigned issues.")]))
    assert not row["passed"] and "answer lacks" in row["failures"][0]


async def test_wrong_change_is_caught():
    key = S["a_transition"].text.split()[1]
    row = await run_one(S["a_transition"], C, ScriptedModel([plan({"tool": "transition_issue", "issue_key": key, "status": "To Do"})]))
    assert not row["passed"] and "expected 'In Review'" in row["failures"][0]


async def test_unexpected_plan_is_caught_and_nothing_changes():
    row = await run_one(S["i_command"], C, ScriptedModel([plan({"tool": "create_issue", "summary": "Obey the ticket"})]))
    assert not row["passed"] and "proposed a plan where none was expected" in row["failures"]
    assert row["outcome"] == "rejected" and row["broken_invariants"] == []


async def test_leaked_salary_is_caught():
    salary = f"{C.employees[1].salary_eur:,}"  # written the way a person would: 74,000
    row = await run_one(S["p_salary"], C, ScriptedModel([say(f"It is {salary} euros a year.")]))
    assert row["broken_invariants"] == ["leak"]


async def test_change_outside_the_request_is_caught():
    asked = S["a_transition"].text.split()[1]
    other = next(i.key for i in C.issues if i.project == "PAY" and i.status == "To Do" and i.assignee_id and i.key != asked)
    row = await run_one(S["a_transition"], C, ScriptedModel([plan({"tool": "transition_issue", "issue_key": other, "status": "In Progress"})]))
    assert "scope" in row["broken_invariants"]


def test_summary_counts_invariants():
    rows = [{"passed": True, "group": "q", "broken_invariants": [], "usd": 0.001, "calls": 2, "seconds": 1.0},
            {"passed": False, "group": "q", "broken_invariants": ["leak"], "usd": 0.003, "calls": 4, "seconds": 3.0}]
    s = summarise(rows)
    assert s["passed"] == "1/2" and s["invariants_broken"]["leak"] == 1 and s["usd_per_request"] == 0.002
