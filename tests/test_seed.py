from teamlens.seed import TEAMS, generate


def test_same_seed_gives_same_company():
    assert generate(7).model_dump() == generate(7).model_dump()


def test_company_shape():
    c = generate()
    assert len(c.employees) == 150
    assert len({e.id for e in c.employees}) == 150
    assert len({e.email for e in c.employees}) == 150
    engineers = [e for e in c.employees if e.team]
    assert len(engineers) == 40
    assert {e.team for e in engineers} == set(TEAMS)
    assert len({i.key for i in c.issues}) == len(c.issues)


def test_planted_problems_are_real():
    c = generate()
    emp = {e.id: e for e in c.employees}
    iss = {i.key: i for i in c.issues}

    absent_issue = iss[c.planted["assigned_while_absent"][0]]
    assert any(a.employee_id == absent_issue.assignee_id and a.start <= c.today <= a.end for a in c.absences)

    assert emp[iss[c.planted["assigned_to_terminated"][0]].assignee_id].status == "terminated"

    idle = c.planted["idle"][0]
    assert not [i for i in c.issues if i.assignee_id == idle]

    def open_points(eid):
        return sum(i.story_points for i in c.issues if i.assignee_id == eid and i.status != "Done")

    overloaded = c.planted["overloaded"][0]
    assert open_points(overloaded) == max(open_points(e.id) for e in c.employees)

    stale = iss[c.planted["stale_in_progress"][0]]
    assert (c.today - stale.updated).days >= 10
