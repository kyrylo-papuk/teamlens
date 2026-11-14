"""Live demo wiring: the fictional company's HRM joined with a real Jira Cloud project.

  python -m teamlens.live seed      fill the Jira project with one team's sprint and start it
  python -m teamlens.live overview  who works on what, read from the real Jira

Jira Cloud hides e-mail addresses of other accounts, so people are joined to Jira accounts
through a mapping file written by ``seed`` (employee id -> Jira account id).
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import date, datetime, time, timezone
from pathlib import Path

from teamlens.config import Settings
from teamlens.hrm import Hrm
from teamlens.jira_cloud import JiraCloud
from teamlens.overview import Overview, build, show
from teamlens.seed import Company, Issue, generate

USERS_FILE = Path("data/seed/jira_users.json")
EXTRA_LOAD = ["Reconcile failed payouts for October", "Add retry policy to payout webhooks",
              "Cover refund flow with contract tests"]  # makes one person visibly overloaded


def company() -> Company:
    return generate(today=date.today())


def load_users() -> dict[str, str]:
    return json.loads(USERS_FILE.read_text(encoding="utf-8"))


async def seed() -> None:
    s = Settings()
    c = company()
    jira = JiraCloud(s)
    if await jira.search():
        sys.exit(f"Project {s.jira_project_key} already has issues. Seeding only fills an empty project.")

    # The requesting account plays the engineering manager; other accounts are engineers.
    me = (await jira._call("GET", "/rest/api/3/myself"))["accountId"]
    accounts = await jira._call("GET", "/rest/api/3/user/assignable/search", params={"project": s.jira_project_key})
    others = sorted(a["displayName"] for a in accounts if a["accountId"] != me)
    by_name = {a["displayName"]: a["accountId"] for a in accounts}
    team = [e for e in c.employees if e.team == s.jira_team_key]
    manager = next(e for e in team if e.role == "Engineering Manager")
    engineers = [e for e in team if e is not manager][:len(others)]
    users = {manager.id: me} | {e.id: by_name[n] for e, n in zip(engineers, others)}
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    USERS_FILE.write_text(json.dumps(users, indent=1), encoding="utf-8")
    jira = JiraCloud(s, users)

    wanted = [i for i in c.issues if i.project == s.jira_team_key and (i.assignee_id is None or i.assignee_id in users)]
    busy = engineers[1]
    wanted += [Issue(key="", project=s.jira_team_key, summary=summary, type="Story", status="In Progress",
                     assignee_id=busy.id, story_points=5, in_sprint=True, created=c.today, updated=c.today)
               for summary in EXTRA_LOAD]

    sprint_keys = []
    for w in wanted:
        made = await jira.create(s.jira_team_key, w.summary, w.type, w.story_points, w.assignee_id)
        for status in ("In Progress", "In Review", "Done")[:("To Do", "In Progress", "In Review", "Done").index(w.status)]:
            await jira.transition(made.key, status)
        if w.in_sprint:
            sprint_keys.append(made.key)
        print(f"  {made.key}  {w.status:12} {w.assignee_id or '-':5} {w.summary}")

    boards = await jira._call("GET", "/rest/agile/1.0/board", params={"projectKeyOrId": s.jira_project_key})
    board = boards["values"][0]["id"]
    sprints = await jira._call("GET", f"/rest/agile/1.0/board/{board}/sprint", params={"state": "future,active"})
    sprint = sprints["values"][0]
    await jira._call("POST", f"/rest/agile/1.0/sprint/{sprint['id']}/issue", json={"issues": sprint_keys})
    if sprint["state"] != "active":
        def stamp(d: date) -> str:
            return datetime.combine(d, time(9), timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        await jira._call("POST", f"/rest/agile/1.0/sprint/{sprint['id']}",
                         json={"state": "active", "startDate": stamp(c.sprint_start), "endDate": stamp(c.sprint_end)})
    await jira.aclose()
    print(f"{len(wanted)} issues created, {len(sprint_keys)} in the active sprint, {len(users)} people mapped.")


async def overview() -> Overview:
    s = Settings()
    c = company()
    users = load_users()
    jira = JiraCloud(s, users)
    hrm = Hrm(c.employees, c.absences)
    # Only people who have an account on this Jira site; the free plan caps it at ten.
    people = [p for p in hrm.people(s.jira_team_key) if p.id in users]
    availability = {p.id: hrm.availability(p.id, c.sprint_start, c.sprint_end, c.today) for p in people}
    issues = await jira.search()
    await jira.aclose()
    return build(people, availability, issues, c.today)


if __name__ == "__main__":
    if sys.argv[1:] == ["seed"]:
        asyncio.run(seed())
    else:
        show(asyncio.run(overview()))
