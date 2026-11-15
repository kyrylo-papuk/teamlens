# Data: what we hold, where, and how strictly

TeamLens joins two systems with very different sensitivity. Jira holds work items that the whole
company may read. The HRM holds facts about people, some of which almost nobody may read.
The design rule is simple: **an agent gets the least data that lets it do its job, and the limit is
enforced by what the code can reach, not by what the prompt asks for.**

## Data classes

| Class | Examples | Lives in | Who can read it | May enter an LLM prompt |
|---|---|---|---|---|
| **Internal** | issue key, title, status, story points, sprint | Jira | any employee | yes |
| **Confidential** | who owns which issue, load against capacity, weekly hours, role, manager, availability as dates | Jira; HRM (served by the People agent) | members of the team, and leadership for every team | yes, for a team the requester may see |
| **Restricted** | reason for absence, salary, e-mail, anything else in the HR record | HRM only | the People side only | **never** |

Availability is the interesting case. "Away until 19 November" is needed to plan work, so it is
Confidential. "Sick" is the same HR record one field over, and it is Restricted. The system splits the
record at that line: `Availability` carries dates, and no type on the Delivery side has a field for the reason.

## Where each rule is enforced

| Rule | Enforced by | Proved by |
|---|---|---|
| Restricted fields never leave the People side | `hrm.py` returns only `Person` and `Availability`; neither type has the fields | `test_minimised_views_carry_no_restricted_fields` |
| No agent-to-agent request can return Restricted data | the People agent has no skill for it; unknown requests are rejected | `test_there_is_no_skill_for_restricted_data` |
| The People agent's own model cannot read Restricted data | its two tools wrap only the minimised lookups; the HRM tool that returns reasons is not wrapped | `people_agent/advisor.py` |
| Only known callers reach the People agent | bearer token per calling agent, with scopes; the agent card is the only public route | `test_requests_need_a_token_and_the_right_scope` |
| The Delivery model cannot change Jira | write tools are not in its tool list; it can only submit a plan | `test_model_is_never_offered_write_tools` |
| Changes follow company rules | `delivery/policy.py`, plain code, run at proposal time and again before every step | `test_policy_rules`, `test_step_is_rechecked_when_the_world_changed_after_approval` |
| Who may change what is decided by rights, not job titles; nobody can hand out more than they hold | `access.py`, plain code, no model anywhere near it | `test_nobody_can_hand_out_more_than_they_hold`, `test_a_granted_role_changes_what_the_policy_allows` |
| The model reads only issues of teams the requester may see | `Toolbox.run` checks the project of every Jira read, whatever the model asked for | `test_the_model_reads_only_issues_of_teams_the_person_may_see` |
| A button and the agent have one way into Jira: the same policy, the same code, the same log | `delivery/direct.py` builds one step and hands it to `policy.check` | `test_buttons_go_through_the_same_rules_and_log_as_plans` |
| A higher position widens the view, not the rights: leadership sees every team and changes no issue | `access.py`: the `executive` role holds no right over issues; `Toolbox.world` refuses a team the person has no right to see | `test_leadership_sees_every_team_and_changes_nothing_but_who_manages`, `test_leadership_gets_every_team_from_the_same_tools_and_changes_no_issue` |
| The audit log and the list of who holds which role are returned by right, not hidden by the page | `web/app.py` checks `audit.view` and the role rights before it answers | `test_audit_log_and_role_holders_are_shown_by_right` |
| Who is in which team changes only by plain code, with the right for that team, on record; nobody leaves a team with issues still in their name | `staffing.py`; the agent has no tool for it | `test_a_manager_adds_a_new_person_to_the_team_and_removes_one`, `test_leadership_creates_a_team_staffs_it_and_names_its_manager` |
| Only the chief executive decides who else has leadership access; a position in the HR record is never changed here | `access.py`: `leadership.appoint` | `test_only_the_chief_executive_decides_who_else_is_leadership` |
| Every instance of the public demo shows the same company | `web/shared.py`: version check before a request, save under a lock after a change | `test_a_change_made_through_one_instance_shows_in_the_other`, `test_a_plan_asked_on_one_instance_is_approved_on_another` |
| A person approves every change the agent proposes, and only the requester can | `approve` node pauses the graph | `test_rejection_and_foreign_approver_change_nothing` |
| The system assigns work on its own only where someone with the right switched that on; each assignment is explained, recorded and can be undone | `delivery/autoassign.py` | `test_auto_mode_assigns_by_itself_and_can_be_undone` |
| Every decision is on record and tampering is visible | `delivery/audit.py`, hash-chained | `test_audit_chain_detects_tampering` |
| No person can be identified in a trace | `tracing.py`: names become employee ids in the client's mask function, which the SDK applies to every value it exports | `test_no_name_reaches_a_trace_and_ids_do`, `test_answer_text_from_the_model_is_masked_too` |

## What is stored

| Store | Content | Class | Notes |
|---|---|---|---|
| Jira | issues, plus a comment on every change the agent made: who asked and why | Internal | system of record for work |
| HRM | employees, absences | up to Restricted | system of record for people; in the two-service deployment only the People agent's process opens it |
| Audit log | request text, proposed plan, verdict of every rule, approval, each executed step, outcome, token usage; every automatic assignment with the candidates considered; every role defined, given or taken away | Confidential | append-only; each entry holds the hash of the previous one |
| Conversation state | messages and tool results of a request, so a paused plan can resume | Confidential | holds the same minimised data the model saw; no Restricted data can be in it |
| Roles and grants | the team's own roles, who was given which role, by whom and when | Confidential | changed only through `access.py`; every change is also in the audit log |
| Traces (Langfuse, optional) | each model call with tokens and cost, tool calls and results, policy verdicts, outcomes | Confidential, pseudonymised | off unless keys are set; names are replaced by employee ids before export and long values are cut |
| Model provider | prompts and outputs of each call | Confidential at most | Restricted data is never sent, so it cannot be retained there |

## Why two agents instead of one

A single agent with "do not reveal the reason for absence" in its prompt holds the reason in its
context, and a prompt is a request, not a guarantee. Here the reason never reaches the model, and in the
two-service deployment (the compose file) it is not in the Delivery process at all, so there is nothing to leak.
Tests and quick local runs use a single-process mode with the same interface; the HRM is then loaded
in-process and the boundary is the interface in `people.py`, not a process. The People agent is a separate service with its own
credentials and its own rules about what it answers; the Delivery agent is one of its callers, not its owner.

That is also the reason the two sides use different protocols. Jira is a tool: it does what it is told, so
the Delivery agent reaches it over **MCP**. The People agent decides for itself what to answer and
may refuse, so the Delivery agent reaches it over **A2A**. Rules on the Delivery side depend only on the
People agent's structured answers, produced by plain code; its language model answers free-form
questions and never feeds a policy decision.

## Not built yet

Stated plainly, because a reviewer should not have to find these:

- **Encryption at rest and retention.** The audit log and conversation state are local files and memory.
  Production needs a database with encryption, a retention period per store, and erasure of a person's
  data on request (the hash chain then needs redaction that keeps the chain verifiable).
- **Per-person access rules inside the People agent.** Any caller with the `people:read` scope can read
  the directory and availability of any team. A manager should be limited to their own reports; this needs
  the caller to pass who it acts for, and the People agent to check it.
- **Token management.** Tokens are static and read from configuration. Production needs short-lived
  tokens issued per agent (OAuth client credentials or mTLS), which the agent card can already declare.
- **Free text in Jira.** An issue title can contain personal data typed by a person, such as a customer's name. Nothing scans for it, and the trace mask only knows employees.
