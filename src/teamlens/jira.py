"""Jira behind one async interface: an in-memory fake now, Jira Cloud next.

The fake backs tests and evals (same data every run); the Cloud client backs the live demo.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol

from teamlens.seed import Issue

STATUSES = ("To Do", "In Progress", "In Review", "Done")


class JiraError(Exception):
    pass


class Jira(Protocol):
    async def search(self, project: str | None = None, assignee_id: str | None = None,
                     status: str | None = None, sprint_only: bool = False) -> list[Issue]: ...
    async def get(self, key: str) -> Issue: ...
    async def assign(self, key: str, assignee_id: str | None) -> Issue: ...
    async def transition(self, key: str, status: str) -> Issue: ...
    async def comment(self, key: str, text: str) -> None: ...
    async def create(self, project: str, summary: str, type: str = "Task", story_points: int = 1,
                     assignee_id: str | None = None) -> Issue: ...


class FakeJira:
    def __init__(self, issues: list[Issue], today: date) -> None:
        self._issues = {i.key: i.model_copy() for i in issues}
        self._today = today
        self.comments: dict[str, list[str]] = {}

    async def search(self, project=None, assignee_id=None, status=None, sprint_only=False) -> list[Issue]:
        return [
            i for i in self._issues.values()
            if (project is None or i.project == project)
            and (assignee_id is None or i.assignee_id == assignee_id)
            and (status is None or i.status == status)
            and (not sprint_only or i.in_sprint)
        ]

    async def get(self, key: str) -> Issue:
        if key not in self._issues:
            raise JiraError(f"Issue {key} does not exist.")
        return self._issues[key]

    async def assign(self, key: str, assignee_id: str | None) -> Issue:
        issue = await self.get(key)
        issue.assignee_id, issue.updated = assignee_id, self._today
        return issue

    async def transition(self, key: str, status: str) -> Issue:
        if status not in STATUSES:
            raise JiraError(f"Unknown status '{status}'. Use one of: {', '.join(STATUSES)}.")
        issue = await self.get(key)
        issue.status, issue.updated = status, self._today
        return issue

    async def comment(self, key: str, text: str) -> None:
        await self.get(key)
        self.comments.setdefault(key, []).append(text)

    async def create(self, project, summary, type="Task", story_points=1, assignee_id=None) -> Issue:
        numbers = [int(k.split("-")[1]) for k, i in self._issues.items() if i.project == project]
        if not numbers:
            raise JiraError(f"Project {project} does not exist.")
        issue = Issue(key=f"{project}-{max(numbers) + 1}", project=project, summary=summary, type=type,
                      status="To Do", assignee_id=assignee_id, story_points=story_points, in_sprint=False,
                      created=self._today, updated=self._today)
        self._issues[issue.key] = issue
        return issue
