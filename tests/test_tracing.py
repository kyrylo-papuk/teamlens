import json
from contextlib import asynccontextmanager

from mcp import Client
from opentelemetry import trace as otel
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from teamlens.delivery.agent import DeliveryAgent
from teamlens.delivery.audit import AuditLog
from teamlens.delivery.model import ScriptedModel, call, say
from teamlens.delivery.tools import Toolbox
from teamlens.hrm import Hrm
from teamlens.jira import FakeJira
from teamlens.mcp_servers import jira_server
from teamlens.people import LocalPeople
from teamlens.seed import generate
from teamlens.tracing import MAX_CHARS, Masker, Tracing, client

C = generate()
PEOPLE = Hrm(C.employees, C.absences).people("PAY")
MANAGER, AWAY, FREE = PEOPLE[0], PEOPLE[1], PEOPLE[2]
ISSUE = C.planted["assigned_while_absent"][0]


def test_masker_replaces_names_with_ids_everywhere():
    mask = Masker()
    assert mask(data="Anton Lindqvist is away") == "Anton Lindqvist is away"  # nobody is known yet
    mask.learn(PEOPLE)
    assert mask(data=f"{AWAY.name} is away; ask {AWAY.name.split()[-1]}.") == f"{AWAY.id} is away; ask {AWAY.id}."
    assert mask(data={"who": [AWAY.name, {"deep": FREE.name}], "n": 3}) == {"who": [AWAY.id, {"deep": FREE.id}], "n": 3}
    shared = MANAGER.name.split()[0]  # three people in this team are called Ines
    assert sum(p.name.split()[0] == shared for p in PEOPLE) > 1 and mask(data=f"{shared} approved it") == "[person] approved it"
    unique = AWAY.name.split()[0]
    assert mask(data=f"{unique} is back") == f"{AWAY.id} is back"
    assert mask(data="Antonia is not Anton's namesake").startswith("Antonia")  # whole words only
    long = mask(data="x" * (MAX_CHARS + 500))
    assert len(long) < MAX_CHARS + 60 and "500 more characters" in long


def test_tracing_is_a_no_op_without_keys():
    tracing = Tracing.from_settings("", "", "")
    assert not tracing.enabled
    with tracing.request("ask", "t1", "E001", input="hello") as trace, tracing.observe("tool", "tool", input={}) as span:
        trace.update(output="x")
        span.update(output="y")
    tracing.flush()


@asynccontextmanager
async def traced(tmp_path):
    exporter, masker = InMemorySpanExporter(), Masker()
    # The SDK keeps one client per public key, so each test uses its own key to get its own exporter.
    tracing = Tracing(client(f"pk-{tmp_path.name}", "sk-test", "", masker, span_exporter=exporter), masker)
    people = LocalPeople(Hrm(C.employees, C.absences), C.today, C.sprint_start, C.sprint_end)
    async with Client(jira_server(FakeJira(C.issues, C.today))) as mcp:
        box = Toolbox(mcp, people, C.today, tracing=tracing)

        def exported() -> tuple[list[str], str]:
            tracing.flush()
            spans = exporter.get_finished_spans()
            return [s.name for s in spans], json.dumps([dict(s.attributes) for s in spans], default=str, ensure_ascii=False)
        yield box, AuditLog(tmp_path / "audit.jsonl"), exported


async def test_no_name_reaches_a_trace_and_ids_do(tmp_path):
    async with traced(tmp_path) as (box, audit, exported):
        step = {"tool": "assign_issue", "issue_key": ISSUE, "assignee_id": FREE.id, "status": None, "summary": None,
                "reason": f"{AWAY.name} is away, {FREE.name} has room"}
        model = ScriptedModel([call("team_overview"), call("propose_plan", _id="p1", summary=f"Cover for {AWAY.name}", steps=[step])])
        agent = DeliveryAgent(model, box, audit, C.today)
        waiting = await agent.ask(MANAGER.id, f"Who covers for {AWAY.name}?", "thread-1")
        assert waiting.outcome == "needs_approval"
        assert (await agent.decide("thread-1", True, MANAGER.id)).outcome == "executed"

        names, text = exported()
        assert {"ask", "decide", "team_overview", "policy", "jira:assign_issue"} <= set(names)
        assert "thread-1" in text and AWAY.id in text and FREE.id in text and ISSUE in text
        for person in PEOPLE:
            assert person.name not in text and person.name.split()[-1] not in text, person.name
        assert "sick" not in text and "salary" not in text


async def test_answer_text_from_the_model_is_masked_too(tmp_path):
    async with traced(tmp_path) as (box, audit, exported):
        agent = DeliveryAgent(ScriptedModel([say(f"{AWAY.name} is away until Friday; {FREE.name} can help.")]), box, audit, C.today)
        result = await agent.ask(MANAGER.id, "Status?", "thread-2")
        assert AWAY.name in result.answer  # the person asking still gets names; only the trace is masked
        _, text = exported()
        assert AWAY.name not in text and f"{AWAY.id} is away until Friday; {FREE.id} can help." in text


async def test_spans_from_other_libraries_are_not_exported(tmp_path):
    """The mask cannot see a span it did not create, so such a span must not leave the process at all."""
    async with traced(tmp_path) as (box, _, exported):
        with box.tracing.request("ask", "thread-3", MANAGER.id, input="hello"):
            with otel.get_tracer("some.library").start_as_current_span("library.call") as foreign:
                foreign.set_attribute("gen_ai.prompt", f"{AWAY.name} is away")  # what an LLM library might record
            with box.tracing.observe("ours", "tool", input="x"):
                pass
        names, text = exported()
        assert "ours" in names and "library.call" not in names and AWAY.name not in text
