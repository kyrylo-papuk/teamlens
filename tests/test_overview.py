from teamlens.hrm import Hrm
from teamlens.overview import from_company
from teamlens.seed import generate


def test_finds_exactly_the_planted_problems():
    c = generate()
    o = from_company()
    by_kind: dict[str, list] = {}
    for f in o.findings:
        by_kind.setdefault(f.kind, []).append(f)

    assert {k: len(v) for k, v in by_kind.items()} == dict.fromkeys(c.planted, 1)
    for kind in ("assigned_while_absent", "assigned_to_terminated", "stale_in_progress"):
        assert c.planted[kind][0] in by_kind[kind][0].issue_keys
    for kind in ("overloaded", "idle"):
        assert by_kind[kind][0].employee_id == c.planted[kind][0]


def test_team_filter_only_shows_that_team():
    o = from_company("PAY")
    assert {x.team for x in o.loads} == {"PAY"}
    assert [f.kind for f in o.findings] == ["assigned_while_absent"]


def test_minimised_views_carry_no_restricted_fields():
    c = generate()
    hrm = Hrm(c.employees, c.absences)
    away = c.absences[0].employee_id
    av = hrm.availability(away, c.sprint_start, c.sprint_end, c.today)
    assert not av.available_today and av.unavailable_until is not None
    dumped = av.model_dump_json() + "".join(p.model_dump_json() for p in hrm.people())
    assert "sick" not in dumped and "reason" not in dumped and "salary" not in dumped

    text = from_company().model_dump_json()
    assert "sick" not in text and "salary" not in text
