import json
from contextlib import asynccontextmanager

from mcp import Client

from teamlens.delivery.digest import MAX_REVISIONS, Check, Problem, facts, plant, unknown_keys, write
from teamlens.delivery.model import ScriptedModel, say
from teamlens.delivery.tools import Toolbox
from teamlens.hrm import Hrm
from teamlens.jira import FakeJira
from teamlens.mcp_servers import jira_server
from teamlens.people import LocalPeople
from teamlens.seed import generate

C = generate()
MANAGER = next(e.id for e in C.employees if e.team == "PAY")
ISSUE = C.planted["assigned_while_absent"][0]
OK = Check(problems=[])
WRONG = Check(problems=[Problem(quote="19 points", why="The facts say 8.")])


@asynccontextmanager
async def setup():
    people = LocalPeople(Hrm(C.employees, C.absences), C.today, C.sprint_start, C.sprint_end)
    async with Client(jira_server(FakeJira(C.issues, C.today))) as client:
        box = Toolbox(client, people, C.today)
        world = await box.world(MANAGER)
        yield world, box.forecast(world)


async def test_facts_are_minimised_and_complete():
    async with setup() as (world, forecast):
        known = facts(world, forecast)
        text = json.dumps(known)
        assert ISSUE in text and "away until" in text and known["sprint_forecast"]["committed_points"] == forecast.committed
        assert "sick" not in text and "salary" not in text and "reason" not in text


async def test_a_clean_draft_is_checked_once_and_kept():
    async with setup() as (world, forecast):
        model = ScriptedModel([say(f"- {ISSUE} has no active owner."), OK])
        d = await write(model, world, forecast)
        assert d.text == f"- {ISSUE} has no active owner." and d.revisions == 0 and not d.problems_found
        assert d.usage.calls == 1  # the scripted draft reports no usage; the check counts as one call


async def test_a_wrong_draft_is_revised_and_checked_again():
    async with setup() as (world, forecast):
        model = ScriptedModel([say("- Anton has 19 points."), WRONG, say("- Anton has 8 points."), OK])
        d = await write(model, world, forecast)
        assert d.text == "- Anton has 8 points." and d.revisions == 1
        assert [p.quote for p in d.problems_found] == ["19 points"] and not d.problems_left
        revise_request = model.seen[2][0]["content"]
        assert '"19 points": The facts say 8.' in revise_request and "FACTS" in revise_request


async def test_revision_budget_is_bounded_and_leftovers_are_reported():
    async with setup() as (world, forecast):
        replies = [say("- Anton has 19 points.")] + [WRONG, say("- Anton has 19 points.")] * MAX_REVISIONS + [WRONG]
        d = await write(ScriptedModel(replies), world, forecast)
        assert d.revisions == MAX_REVISIONS and [p.quote for p in d.problems_left] == ["19 points"]


async def test_unknown_issue_key_is_caught_by_code_even_if_the_model_misses_it():
    async with setup() as (world, forecast):
        known = facts(world, forecast)
        assert [p.quote for p in unknown_keys(f"- {ISSUE} and PAY-9999 are stuck.", known)] == ["PAY-9999"]
        model = ScriptedModel([say("- PAY-9999 is stuck."), OK, say(f"- {ISSUE} is stuck."), OK])
        d = await write(model, world, forecast)
        assert d.revisions == 1 and d.problems_found[0].quote == "PAY-9999" and ISSUE in d.text


async def test_without_reflection_there_is_one_call_and_no_check():
    async with setup() as (world, forecast):
        d = await write(ScriptedModel([say("- All quiet.")]), world, forecast, reflect=False)
        assert d.text == "- All quiet." and d.revisions == 0


async def test_planting_changes_exactly_one_thing_per_kind():
    async with setup() as (world, forecast):
        known = facts(world, forecast)
        name = known["people"][1]["name"]
        text = f"- {name} is away until 2026-11-21 and owns {ISSUE}; 32 points are committed."
        planted = {kind: (corrupted, token) for kind, corrupted, token in plant(text, known, "Someone Else")}
        assert set(planted) == {"number", "name", "issue key", "date"}
        assert planted["number"][1] == "39" and "39 points" in planted["number"][0] and ISSUE in planted["number"][0]
        assert planted["name"][0].startswith("- Someone Else") and planted["date"][1] == "2026-11-30"
        assert planted["issue key"][1] not in json.dumps(known)
