import pytest

from teamlens.access import BUILT_IN, RIGHTS, Access
from teamlens.delivery.audit import AuditLog
from teamlens.hrm import Hrm
from teamlens.seed import generate

C = generate()
PEOPLE = {p.id: p for p in Hrm(C.employees, C.absences).people()}
PAY = [p for p in PEOPLE.values() if p.team == "PAY"]
MANAGER, DANA, VIKTOR = PAY[0], PAY[3], PAY[4]
OTHER_MANAGER = next(p for p in PEOPLE.values() if p.team == "KYC" and p.role == "Engineering Manager")
OUTSIDER = next(p for p in PEOPLE.values() if p.team == "KYC" and p.role != "Engineering Manager")
LEFT = next(p for p in PEOPLE.values() if not p.active)


def access(tmp_path, **kw) -> Access:
    return Access(tmp_path / "access.json", audit=AuditLog(tmp_path / "audit.jsonl"), **kw)


def test_everyone_starts_with_the_role_their_hr_record_implies(tmp_path):
    a = access(tmp_path)
    assert a.base_role(MANAGER) == "manager" and a.rights(MANAGER) == set(RIGHTS)
    assert a.base_role(DANA) == "engineer" and a.rights(DANA) == BUILT_IN["engineer"]
    assert a.rights(DANA, "KYC") == set()  # rights are per team
    assert a.rights(LEFT) == set()  # someone who left has none
    assert a.rights(next(p for p in PEOPLE.values() if not p.team)) == set()


def test_manager_defines_a_role_and_gives_it_in_the_team(tmp_path):
    a = access(tmp_path)
    a.define(MANAGER, "tech-lead", ["issues.assign", "issues.move_any"])
    a.grant(MANAGER, DANA, "tech-lead")
    assert a.rights(DANA) == BUILT_IN["engineer"] | {"issues.assign", "issues.move_any"}
    assert a.granted(DANA, "PAY") == ["tech-lead"]

    reloaded = Access(tmp_path / "access.json")  # survives a restart
    assert "issues.assign" in reloaded.rights(DANA)

    a.revoke(MANAGER, DANA, "tech-lead")
    assert a.rights(DANA) == BUILT_IN["engineer"]
    kinds = [e["kind"] for e in AuditLog(tmp_path / "audit.jsonl").entries()]
    assert kinds == ["role_defined", "role_granted", "role_revoked"]


def test_nobody_can_hand_out_more_than_they_hold(tmp_path):
    a = access(tmp_path)
    a.define(MANAGER, "delegate", ["roles.grant", "issues.take"])
    a.grant(MANAGER, DANA, "delegate")  # Dana may now give roles, but only ones within her own rights
    a.grant(DANA, VIKTOR, "delegate")
    with pytest.raises(PermissionError, match="rights you do not hold"):
        a.grant(DANA, VIKTOR, "manager")
    with pytest.raises(PermissionError, match="do not have the right"):
        a.define(DANA, "sneaky", ["issues.assign"])  # she cannot define roles at all
    with pytest.raises(PermissionError, match="set in configuration"):
        a.grant(MANAGER, DANA, "admin")
    assert "issues.assign" not in a.rights(VIKTOR)


def test_a_role_cannot_be_built_from_rights_the_author_lacks(tmp_path):
    a = access(tmp_path)
    a.define(MANAGER, "role-author", ["roles.define", "issues.take"])
    a.grant(MANAGER, DANA, "role-author")
    with pytest.raises(PermissionError, match="rights you do not hold"):
        a.define(DANA, "power", ["issues.assign", "roles.grant"])
    assert a.define(DANA, "taker", ["issues.take"]).created_by == DANA.id


def test_roles_stay_inside_the_team(tmp_path):
    a = access(tmp_path)
    with pytest.raises(PermissionError):
        a.grant(MANAGER, OUTSIDER, "engineer")  # PAY's manager has no say in KYC
    with pytest.raises(PermissionError):
        a.grant(DANA, VIKTOR, "engineer")  # an engineer cannot give roles
    a.define(MANAGER, "tech-lead", ["issues.assign"])
    with pytest.raises(ValueError, match="no role"):
        a.grant(OTHER_MANAGER, OUTSIDER, "tech-lead")  # PAY's custom role does not exist in KYC


def test_bad_input_and_protected_roles(tmp_path):
    a = access(tmp_path)
    for name, rights in (("Bad Name", ["issues.take"]), ("manager", ["issues.take"]), ("ok-name", ["no.such.right"]), ("ok-name", [])):
        with pytest.raises(ValueError):
            a.define(MANAGER, name, rights)
    with pytest.raises(ValueError, match="Built-in"):
        a.delete(MANAGER, "engineer")
    with pytest.raises(ValueError, match="HR record"):
        a.revoke(MANAGER, DANA, "engineer")
    with pytest.raises(ValueError, match="already has"):
        a.grant(MANAGER, DANA, "engineer")
    with pytest.raises(ValueError, match="not an active member"):
        a.grant(MANAGER, LEFT, "engineer")


def test_deleting_a_role_takes_it_from_everyone(tmp_path):
    a = access(tmp_path)
    a.define(MANAGER, "tech-lead", ["issues.assign"])
    a.grant(MANAGER, DANA, "tech-lead")
    a.delete(MANAGER, "tech-lead")
    assert a.rights(DANA) == BUILT_IN["engineer"] and [r.name for r in a.roles("PAY")] == list(BUILT_IN)


def test_administrator_has_every_right_everywhere(tmp_path):
    a = access(tmp_path, admins={DANA.id})
    assert a.rights(DANA, "KYC") == set(RIGHTS)
    a.grant(DANA, OUTSIDER, "manager")
    assert "roles.grant" in a.rights(OUTSIDER)
    with pytest.raises(PermissionError, match="set in configuration"):
        a.grant(DANA, VIKTOR, "admin")
