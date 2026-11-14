from teamlens.forecast import HELD_OUT, PAST_SPRINTS, Fit, default_fit, evaluate, forecast, history
from teamlens.overview import from_company
from teamlens.seed import TEAMS, generate

C = generate()


def test_history_is_reproducible_and_plausible():
    records = history(C)
    assert records == history(C) and len(records) == PAST_SPRINTS * len(TEAMS)
    assert all(0 <= r.completed <= r.committed for r in records)


def test_regression_beats_average_velocity_on_sprints_it_never_saw():
    result = evaluate(history(C))
    error = result["mean_abs_error_points"]
    assert result["scored_on_sprints"] == HELD_OUT * len(TEAMS)
    assert error["regression"] < error["average_velocity"]
    assert result["share_inside_90_interval"] >= 0.8  # the interval means what it says


def test_more_capacity_means_more_gets_done():
    model = default_fit()
    assert model.per_capacity_point > 0 and model.predict(60, 50) > model.predict(40, 50)


def test_forecast_flags_a_team_that_took_on_too_much():
    model = default_fit()
    safe = forecast(from_company("PAY"), [i for i in C.issues if i.project == "PAY"], model)
    stretched = forecast(from_company("KYC"), [i for i in C.issues if i.project == "KYC"], model)  # the overloaded team
    assert not safe.at_risk and "reachable" in safe.summary
    assert stretched.at_risk and stretched.expected < stretched.committed and "likely to slip" in stretched.summary
    for f in (safe, stretched):
        assert f.done <= f.low <= f.expected <= f.high <= f.committed


def test_forecast_never_promises_less_than_is_already_done():
    pessimist = Fit(intercept=-100, per_capacity_point=0, per_committed_point=0, sigma=1, sprints=10)
    f = forecast(from_company("PAY"), [i for i in C.issues if i.project == "PAY"], pessimist)
    assert f.expected == f.done and f.at_risk
