"""Live test against a real Jira Cloud site. Opt-in: TEAMLENS_TEST_JIRA=1 and a filled .env."""

import os

import pytest

from teamlens.config import Settings
from teamlens.jira import JiraError
from teamlens.jira_cloud import JiraCloud

pytestmark = pytest.mark.skipif(os.environ.get("TEAMLENS_TEST_JIRA") != "1", reason="needs a real Jira site")


async def test_issue_lifecycle_on_real_jira():
    settings = Settings()
    jira = JiraCloud(settings)
    me = (await jira._call("GET", "/rest/api/3/myself"))["accountId"]
    issue = await jira.create(settings.jira_team_key, "TeamLens live test", type="Bug", story_points=3)
    try:
        assert issue.status == "To Do" and issue.type == "Bug" and issue.story_points == 3
        assert issue.assignee_id is None and issue.project == settings.jira_team_key

        assert (await jira.assign(issue.key, me)).assignee_id == me
        for status in ("In Progress", "In Review", "Done"):
            assert (await jira.transition(issue.key, status)).status == status
        await jira.comment(issue.key, "Checked by the live test.")

        found = await jira.search(assignee_id=me, status="Done")
        assert issue.key in [i.key for i in found]
        assert (await jira.assign(issue.key, None)).assignee_id is None

        with pytest.raises(JiraError, match="Unknown status"):
            await jira.transition(issue.key, "Shipped")
    finally:
        await jira.delete(issue.key)

    with pytest.raises(JiraError, match="does not exist"):
        await jira.get(issue.key)
    await jira.aclose()
