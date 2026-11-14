"""The morning digest: what needs attention today, written by the model and then checked.

This is the Reflection pattern, bounded:

    draft  ->  check  ->  (revise -> check) at most MAX_REVISIONS times

  draft   the model writes a few bullets from a JSON of facts (the same minimised data as everywhere)
  check   two layers. Code first: every issue key in the text must exist in the facts. Then a second
          model call lists claims the facts do not support: wrong numbers, names, dates.
  revise  only if the check found something; the problems are handed back with the facts

Reflection costs extra calls on every digest, so its value has to be shown, not assumed.
``python -m teamlens.delivery.digest`` plants known errors into digests and reports how many the
check catches, how often it objects to an untouched digest, and what the extra calls cost.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path

from pydantic import BaseModel

from teamlens.delivery.model import Model, Usage
from teamlens.delivery.policy import World
from teamlens.forecast import Forecast

MAX_REVISIONS = 2
KEY = re.compile(r"\b[A-Z][A-Z0-9]+-\d+\b")
OUT = Path("evals/results/digest.json")

DRAFT = """You write the morning delivery digest for {name}, {role} of team {team}.

You get the facts as JSON. Use only those facts: every name, issue key, number and date you write must be in them.
Write at most six short bullets, the most urgent first: what is at risk today and what to do about it. Refer to \
people by name and issues by key. You know when someone is away, never why; do not guess a reason.
Plain text, each bullet starting with "- ". No heading, no closing remark."""

CHECK = """You verify a delivery digest against the facts it was written from.

List every statement in the digest that the facts contradict or do not contain: a wrong number, a wrong name, a \
person or issue that is not in the facts, a wrong date, a stated reason for someone's absence. Quote the exact \
words from the digest. Advice, priorities and opinions are not statements of fact; do not list them. Arithmetic \
that follows from the facts (a sum, a difference, a percentage) counts as supported. If everything is supported, \
return an empty list."""

REVISE = """You wrote a delivery digest and a check found statements the facts do not support. Rewrite the digest \
so that every listed problem is fixed, using only the facts. Change nothing else. Same format: plain text, at most \
six bullets starting with "- "."""


class Problem(BaseModel):
    quote: str
    why: str


class Check(BaseModel):
    problems: list[Problem]


class Digest(BaseModel):
    text: str
    revisions: int
    problems_found: list[Problem]  # everything the checks objected to along the way
    problems_left: list[Problem]  # what was still unsupported when the revision budget ran out
    usage: Usage = Usage()


def facts(world: World, forecast: Forecast, unowned: list[dict] | None = None) -> dict:
    """Everything the digest may talk about. Nothing here is restricted data."""
    return {
        "today": str(world.overview.today), "team": world.actor.team,
        "sprint_forecast": {"committed_points": forecast.committed, "done_points": forecast.done,
                            "expected_points": forecast.expected, "interval_90": [forecast.low, forecast.high],
                            "at_risk": forecast.at_risk},
        "people": [{"name": x.name, "role": x.role, "open_issues": x.open_issues, "sprint_points": x.sprint_points,
                    "capacity_points": x.capacity_points, "load_percent": None if x.load is None else round(x.load * 100),
                    "status": "left the company" if not x.availability.active else "at work"
                    if x.availability.available_today else f"away until {x.availability.unavailable_until}"}
                   for x in world.overview.loads],
        "problems": [{"kind": f.kind, "issues": f.issue_keys, "detail": f.detail} for f in world.overview.findings],
        "sprint_issues_without_owner": unowned or [],
        "unassigned_backlog_issues": sum(1 for i in world.issues.values()
                                         if i.assignee_id is None and not i.in_sprint and i.status != "Done"),
    }


def unknown_keys(text: str, known: dict) -> list[Problem]:
    """The free check: an issue key that is nowhere in the facts cannot be right."""
    present = set(KEY.findall(json.dumps(known)))
    return [Problem(quote=k, why="This issue key is not in the facts.") for k in sorted(set(KEY.findall(text)) - present)]


async def check(model: Model, text: str, known: dict) -> tuple[list[Problem], Usage]:
    by_code = unknown_keys(text, known)
    answer, usage = await model.structured(CHECK, f"FACTS\n{json.dumps(known, ensure_ascii=False)}\n\nDIGEST\n{text}", Check)
    return by_code + answer.problems, usage


async def write(model: Model, world: World, forecast: Forecast, unowned: list[dict] | None = None,
                reflect: bool = True) -> Digest:
    known = facts(world, forecast, unowned)
    payload = json.dumps(known, ensure_ascii=False)
    system = DRAFT.format(name=world.actor.name, role=world.actor.role, team=world.actor.team)
    text, usage = await model.text(system, payload)
    if not reflect:
        return Digest(text=text, revisions=0, problems_found=[], problems_left=[], usage=usage)

    found: list[Problem] = []
    for revision in range(MAX_REVISIONS + 1):
        problems, spent = await check(model, text, known)
        usage = usage.plus(spent)
        found += problems
        if not problems or revision == MAX_REVISIONS:
            return Digest(text=text, revisions=revision, problems_found=found, problems_left=problems, usage=usage)
        listed = "\n".join(f'- "{p.quote}": {p.why}' for p in problems)
        text, spent = await model.text(REVISE, f"FACTS\n{payload}\n\nDIGEST\n{text}\n\nPROBLEMS\n{listed}")
        usage = usage.plus(spent)
    raise AssertionError("unreachable")


# --- does the check earn its cost? plant known errors and see ---------------------------------

def plant(text: str, known: dict, outsider: str) -> list[tuple[str, str, str]]:
    """Corrupted copies of a digest as (kind, corrupted text, the wrong token now in it). One error each."""
    out = []
    number = re.search(r"(?<![\w.-])(\d{2,3})(?![\w.-])(?!-)", text)  # a two- or three-digit count, not part of a key or date
    if number:
        wrong = str(int(number.group(1)) + 7)
        out.append(("number", text[:number.start(1)] + wrong + text[number.end(1):], wrong))
    name = next((p["name"] for p in known["people"] if p["name"] in text), None)
    if name:
        out.append(("name", text.replace(name, outsider, 1), outsider))
    key = KEY.search(text)
    if key:
        wrong = key.group(0).split("-")[0] + "-9" + key.group(0).split("-")[1]
        out.append(("issue key", text.replace(key.group(0), wrong, 1), wrong))
    day = re.search(r"\d{4}-\d{2}-\d{2}", text)
    if day:
        wrong = str(date.fromisoformat(day.group(0)) + timedelta(days=9))
        out.append(("date", text.replace(day.group(0), wrong, 1), wrong))
    return out


async def evaluate(model: Model) -> dict:
    from mcp import Client

    from teamlens.delivery.tools import Toolbox
    from teamlens.hrm import Hrm
    from teamlens.jira import FakeJira
    from teamlens.mcp_servers import jira_server
    from teamlens.people import LocalPeople
    from teamlens.seed import TEAMS, generate

    c = generate()
    people = LocalPeople(Hrm(c.employees, c.absences), c.today, c.sprint_start, c.sprint_end)
    cases, clean, cost = [], [], {"draft_only": [], "with_check": []}
    async with Client(jira_server(FakeJira(c.issues, c.today))) as client:
        box = Toolbox(client, people, c.today)
        for team in TEAMS:
            manager = next(e.id for e in c.employees if e.team == team and e.role == "Engineering Manager")
            other = next(e.name for e in c.employees if e.team not in (team, None) and e.role != "Engineering Manager")
            world = await box.world(manager)
            known = facts(world, box.forecast(world))

            started = time.perf_counter()
            plain = await write(model, world, box.forecast(world), reflect=False)
            seconds_plain = time.perf_counter() - started
            started = time.perf_counter()
            checked = await write(model, world, box.forecast(world))
            seconds_checked = time.perf_counter() - started
            cost["draft_only"].append((plain.usage.usd, seconds_plain, plain.usage.calls))
            cost["with_check"].append((checked.usage.usd, seconds_checked, checked.usage.calls))
            clean.append({"team": team, "revisions": checked.revisions,
                          "objections_to_first_draft": [p.model_dump() for p in checked.problems_found],
                          "left_unresolved": len(checked.problems_left), "digest": checked.text})

            for kind, corrupted, token in plant(checked.text, known, other):
                problems, _ = await check(model, corrupted, known)
                caught = any(token in p.quote or token in p.why for p in problems)
                cases.append({"team": team, "kind": kind, "planted": token, "caught": caught,
                              "by_code": any(p.quote == token for p in unknown_keys(corrupted, known))})

    def average(rows, index):
        return round(sum(r[index] for r in rows) / len(rows), 5)

    kinds = sorted({k["kind"] for k in cases})
    return {
        "planted_errors": len(cases), "caught": sum(k["caught"] for k in cases),
        "by_kind": {kind: f"{sum(k['caught'] for k in cases if k['kind'] == kind)}/{sum(k['kind'] == kind for k in cases)}"
                    for kind in kinds},
        "caught_by_code_alone": sum(k["by_code"] for k in cases),
        "digests_written": len(clean), "first_drafts_objected_to": sum(bool(d["objections_to_first_draft"]) for d in clean),
        "per_digest": {name: {"usd": average(rows, 0), "seconds": round(average(rows, 1), 1), "model_calls": average(rows, 2)}
                       for name, rows in cost.items()},
        "cases": cases, "digests": clean,
    }


if __name__ == "__main__":
    from teamlens.config import Settings
    from teamlens.delivery.model import ClaudeModel

    sys.stdout.reconfigure(encoding="utf-8")
    result = asyncio.run(evaluate(ClaudeModel(api_key=Settings().anthropic_api_key)))
    print(f"planted errors caught: {result['caught']}/{result['planted_errors']}   " +
          "  ".join(f"{k} {v}" for k, v in result["by_kind"].items()))
    print(f"  of these, caught by the code check alone: {result['caught_by_code_alone']}")
    print(f"first drafts the check objected to: {result['first_drafts_objected_to']}/{result['digests_written']}")
    for name, r in result["per_digest"].items():
        print(f"  {name:11} ${r['usd']}  {r['seconds']}s  {r['model_calls']} model calls per digest")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"saved to {OUT}")
