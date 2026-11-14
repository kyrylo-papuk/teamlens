"""Will the team finish what it committed to this sprint? A linear regression, not a model's guess.

    completed points  ~  capacity of the people at work  +  points committed

It is fitted on past sprints and gives a number with an interval, so "at risk" has a definition:
the forecast is below the commitment. The language model never produces these numbers; it can
only read them through a tool, the same way it reads the team overview.

About the data: a fresh Jira project has no closed sprints, so the history here is generated for
the fictional company (``history``), with noise and a per-team bias the regression does not know.
On a real site the same records come from closed sprints (GET /rest/agile/1.0/board/{id}/sprint
with state=closed, then the issues of each). The honest way to read the numbers below is as a check
that the method and its evaluation are sound, not as accuracy on a real team.

  python -m teamlens.forecast     fit, compare with the usual "average velocity" estimate, print the result
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from teamlens.overview import Overview
from teamlens.seed import TEAMS, Company, Issue, generate

PAST_SPRINTS = 12
HELD_OUT = 3  # most recent sprints of every team, never used for fitting when the model is scored
Z_90 = 1.645  # a 90% interval under normal errors
OUT = Path("evals/results/forecast.json")


class SprintRecord(BaseModel):
    team: str
    sprint: int  # 1 is the oldest
    capacity: float
    committed: int
    completed: int


class Fit(BaseModel):
    intercept: float
    per_capacity_point: float
    per_committed_point: float
    sigma: float  # standard deviation of the errors on the sprints it was fitted on
    sprints: int

    def predict(self, capacity: float, committed: float) -> float:
        return self.intercept + self.per_capacity_point * capacity + self.per_committed_point * committed


class Forecast(BaseModel):
    committed: int
    done: int
    capacity: float
    expected: float
    low: float
    high: float
    at_risk: bool
    summary: str


def history(c: Company, seed: int = 11) -> list[SprintRecord]:
    """Past sprints of every team. What drives the outcome here is deliberately not a clean line."""
    rng = random.Random(seed)
    records = []
    for team in TEAMS:
        people = [e for e in c.employees if e.team == team and e.role != "Engineering Manager" and e.status == "active"]
        full = sum(10 * e.weekly_hours / 40 for e in people)  # points per two-week sprint with everyone present
        bias = rng.uniform(-2.5, 2.5)  # some teams are simply faster; the regression is not told which
        for n in range(1, PAST_SPRINTS + 1):
            capacity = round(full * rng.uniform(0.78, 1.0), 1)  # holidays and sickness
            committed = round(capacity * rng.uniform(0.85, 1.45))  # teams routinely take on too much
            done = 0.80 * capacity + 0.12 * committed + bias + rng.gauss(0, 2.2)
            records.append(SprintRecord(team=team, sprint=n, capacity=capacity, committed=committed,
                                        completed=max(0, min(committed, round(done)))))
    return records


def fit(records: list[SprintRecord]) -> Fit:
    x = np.array([[1.0, r.capacity, r.committed] for r in records])
    y = np.array([r.completed for r in records], dtype=float)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    errors = y - x @ coef
    sigma = float(np.sqrt((errors ** 2).sum() / max(1, len(records) - 3)))
    return Fit(intercept=float(coef[0]), per_capacity_point=float(coef[1]), per_committed_point=float(coef[2]),
               sigma=sigma, sprints=len(records))


def evaluate(records: list[SprintRecord]) -> dict:
    """Fit on the older sprints, score on the most recent ones, against the estimate teams usually use."""
    train = [r for r in records if r.sprint <= PAST_SPRINTS - HELD_OUT]
    test = [r for r in records if r.sprint > PAST_SPRINTS - HELD_OUT]
    model = fit(train)
    velocity = {t: float(np.mean([r.completed for r in train if r.team == t])) for t in TEAMS}
    regression = [abs(model.predict(r.capacity, r.committed) - r.completed) for r in test]
    average = [abs(velocity[r.team] - r.completed) for r in test]
    inside = [abs(model.predict(r.capacity, r.committed) - r.completed) <= Z_90 * model.sigma for r in test]
    return {"fitted_on_sprints": len(train), "scored_on_sprints": len(test),
            "mean_abs_error_points": {"regression": round(float(np.mean(regression)), 2),
                                      "average_velocity": round(float(np.mean(average)), 2)},
            "share_inside_90_interval": round(sum(inside) / len(inside), 2),
            "fit": model.model_dump()}


def forecast(overview: Overview, issues: list[Issue], model: Fit) -> Forecast:
    sprint = [i for i in issues if i.in_sprint]
    committed = sum(i.story_points for i in sprint)
    done = sum(i.story_points for i in sprint if i.status == "Done")
    capacity = round(sum(x.capacity_points for x in overview.loads if x.role != "Engineering Manager"), 1)
    expected = max(float(done), min(float(committed), model.predict(capacity, committed)))
    low = max(float(done), expected - Z_90 * model.sigma)
    high = min(float(committed), expected + Z_90 * model.sigma)
    at_risk = expected < committed - 0.5
    summary = (f"{committed} points committed, {done} done. With {capacity:g} points of capacity the forecast is "
               f"{expected:.0f} points finished (90% interval {low:.0f}-{high:.0f}). ")
    summary += f"About {committed - expected:.0f} points are likely to slip." if at_risk else "The commitment looks reachable."
    return Forecast(committed=committed, done=done, capacity=capacity, expected=round(expected, 1), low=round(low, 1),
                    high=round(high, 1), at_risk=at_risk, summary=summary)


def default_fit() -> Fit:
    """The model used at run time: fitted on all past sprints of the fictional company."""
    return fit(history(generate()))


if __name__ == "__main__":
    result = evaluate(history(generate()))
    error = result["mean_abs_error_points"]
    print(f"fitted on {result['fitted_on_sprints']} sprints, scored on the {result['scored_on_sprints']} most recent\n")
    print(f"  mean absolute error, regression        {error['regression']} points")
    print(f"  mean absolute error, average velocity  {error['average_velocity']} points")
    print(f"  actual result inside the 90% interval  {result['share_inside_90_interval']:.0%} of sprints")
    f = result["fit"]
    print(f"\n  completed = {f['intercept']:.2f} + {f['per_capacity_point']:.2f} x capacity + "
          f"{f['per_committed_point']:.2f} x committed   (sigma {f['sigma']:.2f})")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\nsaved to {OUT}")
