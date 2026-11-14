from contextlib import asynccontextmanager

import pytest
from mcp import Client

from teamlens.delivery.audit import AuditLog
from teamlens.delivery.autoassign import AUTO_MAX_LOAD, AutoAssigner, suggest
from teamlens.delivery.tools import Toolbox
from teamlens.hrm import Hrm
from teamlens.jira import FakeJira
from teamlens.mcp_servers import jira_server
from teamlens.people import LocalPeople
from teamlens.seed import generate

C = generate()
PAY = [e for e in C.employees if e.team == "PAY"]
MANAGER, AWAY, ENGINEER = PAY[0].id, PAY[1].id, PAY[2].id


@asynccontextmanager
async def env(tmp_path):
    jira = FakeJira(C.issues, C.today)
    people = LocalPeople(Hrm(C.employees, C.absences), C.today, C.sprint_start, C.sprint_end)
    async with Client(jira_server(jira)) as client:
        box = Toolbox(client, people, C.today)
        audit = AuditLog(tmp_path / "audit.jsonl")
        yield jira, box, audit, AutoAssigner(box, audit, tmp_path / "settings.json")


async def test_picks_the_person_who_ends_up_least_loaded(tmp_path):
    async with env(tmp_path) as (jira, box, _, auto):
        issue = await jira.create("PAY", "New payout bug", story_points=2, in_sprint=True)
        world = await box.world(MANAGER)
        s = suggest(issue, world)
        ids = [c.employee_id for c in s.candidates]
        assert AWAY not in ids and MANAGER not in ids  # away people and the manager are never candidates
        assert s.assignee_id == ids[0] and s.candidates[0].load_after == min(c.load_after for c in s.candidates)
        assert s.candidates[0].load_after <= AUTO_MAX_LOAD and s.assignee_name in s.reason
        assert [f.kind for f in world.overview.findings if f.issue_keys == [issue.key]] == ["unowned_in_sprint"]


async def test_never_goes_above_capacity_on_its_own(tmp_path):
    async with env(tmp_path) as (jira, box, _, auto):
        issue = await jira.create("PAY", "Huge migration", story_points=40, in_sprint=True)
        s = suggest(issue, await box.world(MANAGER))
        assert s.assignee_id is None and "above 100% of capacity" in s.reason
        applied = await auto.assign(MANAGER, issue.key)
        assert not applied.ok and (await jira.get(issue.key)).assignee_id is None


async def test_several_issues_are_spread_not_piled_on_one_person(tmp_path):
    async with env(tmp_path) as (jira, _, _, auto):
        for n in range(3):
            await jira.create("PAY", f"Task {n}", story_points=3, in_sprint=True)
        owners = [s.assignee_id for s in await auto.suggestions(MANAGER)]
        assert len(owners) == 3 and len(set(owners)) > 1


async def test_confirming_a_suggestion_assigns_explains_and_records(tmp_path):
    async with env(tmp_path) as (jira, _, audit, auto):
        issue = await jira.create("PAY", "New payout bug", story_points=2, in_sprint=True)
        applied = await auto.assign(MANAGER, issue.key)
        assert applied.ok and (await jira.get(issue.key)).assignee_id == applied.assignee_id
        assert "lowest load in the team" in jira.comments[issue.key][0]
        entry = audit.entries()[-1]
        assert entry["kind"] == "auto_assign" and entry["actor"] == MANAGER
        assert entry["data"]["assignee"] == applied.assignee_id and len(entry["data"]["candidates"]) >= 2
        assert not (await auto.assign(MANAGER, issue.key)).ok  # it has an owner now


async def test_only_the_manager_can_confirm_or_switch_modes(tmp_path):
    async with env(tmp_path) as (jira, _, _, auto):
        issue = await jira.create("PAY", "New payout bug", story_points=2, in_sprint=True)
        with pytest.raises(PermissionError):
            await auto.assign(ENGINEER, issue.key)
        with pytest.raises(PermissionError):
            await auto.set_mode(ENGINEER, "auto")
        assert (await jira.get(issue.key)).assignee_id is None and auto.setting("PAY")["mode"] == "suggest"


async def test_auto_mode_assigns_by_itself_and_can_be_undone(tmp_path):
    async with env(tmp_path) as (jira, _, audit, auto):
        first = await jira.create("PAY", "Task A", story_points=2, in_sprint=True)
        assert await auto.sweep("PAY", MANAGER) == []  # suggest mode: the system does nothing on its own
        assert (await auto.set_mode(MANAGER, "auto"))["set_by"] == MANAGER
        second = await jira.create("PAY", "Task B", story_points=2, in_sprint=True)
        applied = await auto.sweep("PAY", MANAGER)
        assert [a.ok for a in applied] == [True, True]
        assert (await jira.get(first.key)).assignee_id and (await jira.get(second.key)).assignee_id
        entry = audit.entries()[-1]
        assert entry["actor"] == "autoassign" and entry["data"]["on_behalf_of"] == MANAGER
        assert "Assigned automatically" in jira.comments[second.key][0]
        assert await auto.sweep("PAY", MANAGER) == []  # nothing left without an owner

        assert (await auto.undo(MANAGER, second.key)).ok and (await jira.get(second.key)).assignee_id is None
        assert audit.verify() is None
