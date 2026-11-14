"""Talk to the Delivery agent from a terminal.

  python -m teamlens.delivery "Who is overloaded?"            real Jira, real Claude
  python -m teamlens.delivery --fake "Who is overloaded?"     in-memory Jira (team PAY of the fictional company)
  python -m teamlens.delivery --as E003 "Reassign PAY-1"      ask as another employee
  python -m teamlens.delivery --a2a "..."                     people data from the People agent (start it first:
                                                              python -m teamlens.people_agent)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

import httpx
from mcp import Client

from teamlens import live
from teamlens.config import Settings
from teamlens.delivery.agent import DeliveryAgent
from teamlens.delivery.audit import AuditLog
from teamlens.delivery.model import ClaudeModel
from teamlens.delivery.tools import Toolbox
from teamlens.hrm import Hrm
from teamlens.jira import FakeJira
from teamlens.jira_cloud import JiraCloud
from teamlens.mcp_servers import jira_server
from teamlens.people import A2APeople, LocalPeople
from teamlens.seed import generate


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--as", dest="actor")
    parser.add_argument("--yes", action="store_true", help="approve the plan without asking")
    parser.add_argument("--a2a", action="store_true", help="get people data from the People agent over A2A")
    args = parser.parse_args()
    s = Settings()

    if args.fake:
        c, users = generate(), None
        jira = FakeJira(c.issues, c.today)
    else:
        c, users = live.company(), live.load_users()
        jira = JiraCloud(s, users)
    only, http = (set(users) if users else None), None
    if args.a2a:
        http = httpx.AsyncClient(base_url=s.people_agent_url, timeout=120,
                                 headers={"Authorization": f"Bearer {s.people_agent_token}"})
        people = A2APeople(http, c.today, c.sprint_start, c.sprint_end, only=only)
    else:
        people = LocalPeople(Hrm(c.employees, c.absences), c.today, c.sprint_start, c.sprint_end, only=only)
    manager = next(e.id for e in c.employees if e.team == s.jira_team_key and e.role == "Engineering Manager")
    actor, thread = args.actor or manager, uuid.uuid4().hex[:8]

    async with Client(jira_server(jira)) as client:
        audit = AuditLog(Path("data/audit.jsonl"))
        agent = DeliveryAgent(ClaudeModel(api_key=s.anthropic_api_key), Toolbox(client, people, c.today), audit, c.today)
        result = await agent.ask(actor, args.text, thread)
        if result.outcome == "needs_approval":
            print(f"\nPLAN: {result.plan['summary']}")
            for n, step in enumerate(result.plan["steps"], start=1):
                what = {"assign_issue": f"assign {step['issue_key']} to {step['assignee_id'] or 'nobody'}",
                        "transition_issue": f"move {step['issue_key']} to {step['status']}",
                        "create_issue": f"create \"{step['summary']}\""}[step["tool"]]
                print(f"  {n}. {what}\n     why: {step['reason']}")
            approved = args.yes or input("\nApprove? [y/N] ").strip().lower() == "y"
            result = await agent.decide(thread, approved, approver_id=actor)
        u = result.usage
        print(f"\n[{result.outcome}] {result.answer}")
        print(f"\n({u.calls} model calls, {u.input + u.cache_read + u.cache_write} in / {u.output} out tokens, "
              f"${u.usd:.4f}; audit chain {'intact' if audit.verify() is None else 'BROKEN'})")
    if not args.fake:
        await jira.aclose()
    if http:
        await http.aclose()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
