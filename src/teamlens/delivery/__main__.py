"""Talk to the Delivery agent from a terminal.

  python -m teamlens.delivery "Who is overloaded?"            real Jira, real Claude
  python -m teamlens.delivery --fake "Who is overloaded?"     in-memory Jira (team PAY of the fictional company)
  python -m teamlens.delivery --as E003 "Reassign PAY-1"      ask as another employee
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

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
from teamlens.people import LocalPeople
from teamlens.seed import generate


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--as", dest="actor")
    parser.add_argument("--yes", action="store_true", help="approve the plan without asking")
    args = parser.parse_args()
    s = Settings()

    if args.fake:
        c, users = generate(), None
        jira = FakeJira(c.issues, c.today)
    else:
        c, users = live.company(), live.load_users()
        jira = JiraCloud(s, users)
    hrm = Hrm(c.employees, c.absences)
    people = LocalPeople(hrm, c.today, c.sprint_start, c.sprint_end, only=set(users) if users else None)
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


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
