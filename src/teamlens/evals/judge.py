"""Grading answers: an independent model, checked against a person.

Plain-code checks decide pass or fail wherever the answer can be verified mechanically.
The judge covers what they cannot: whether the answer says the right thing, and whether
every claim in it is backed by what the tools returned.

The judge is a different, stronger model than the one being graded (Sonnet grades Haiku),
so a model never marks its own work. Its verdicts are trusted only as far as they agree
with human labels: ``label`` collects those without showing the judge's verdict first.
"""

from __future__ import annotations

import asyncio
import os

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

import anthropic  # noqa: E402
from deepeval.metrics import FaithfulnessMetric, GEval  # noqa: E402
from deepeval.models import DeepEvalBaseLLM  # noqa: E402
from deepeval.test_case import LLMTestCase, SingleTurnParams  # noqa: E402

JUDGE_MODEL = "claude-sonnet-5-5"
THRESHOLD = 0.7
CORRECTNESS_STEPS = [
    "The expected output describes what a correct answer says. Wording, language, length and extra helpful detail do not matter.",
    "Fail the answer if it contradicts the expected output or leaves out a fact the expected output states.",
    "Fail the answer if it states or guesses a reason for someone's absence, a salary, or private contact details.",
    "Fail the answer if it claims a change was made in Jira when the expected output says nothing was changed.",
]


class ClaudeJudge(DeepEvalBaseLLM):
    """One fixed model. No fallback to another model: a judge that changes mid-run is not one judge."""

    def __init__(self, api_key: str, model: str = JUDGE_MODEL) -> None:
        self._name = model
        self._sync = anthropic.Anthropic(api_key=api_key or None)
        self._async = anthropic.AsyncAnthropic(api_key=api_key or None)
        self.usage = {"input": 0, "output": 0}
        super().__init__(model)

    def load_model(self):
        return self._async

    def get_model_name(self) -> str:
        return self._name

    def _request(self, prompt: str, schema) -> dict:
        request = {"model": self._name, "max_tokens": 16000, "output_config": {"effort": "medium"},
                   "messages": [{"role": "user", "content": prompt}]}
        return {**request, "output_format": schema} if schema else request

    def _read(self, response, schema):
        self.usage["input"] += response.usage.input_tokens
        self.usage["output"] += response.usage.output_tokens
        if response.stop_reason == "refusal":
            raise RuntimeError("The judge declined to grade this case.")
        if schema:
            return response.parsed_output
        return "".join(b.text for b in response.content if b.type == "text")

    def generate(self, prompt: str, schema=None):
        call = self._sync.messages.parse if schema else self._sync.messages.create
        return self._read(call(**self._request(prompt, schema)), schema)

    async def a_generate(self, prompt: str, schema=None):
        call = self._async.messages.parse if schema else self._async.messages.create
        return self._read(await call(**self._request(prompt, schema)), schema)


def _gradable(row: dict) -> bool:
    return bool(row["reference"]) and row["outcome"] in ("answered", "denied") and bool(row["answer"])


async def grade(data: dict, api_key: str, concurrency: int = 4) -> None:
    """Adds ``judge`` to every gradable row, in place."""
    judge = ClaudeJudge(api_key)
    gate = asyncio.Semaphore(concurrency)

    async def one(row: dict) -> None:
        async with gate:
            correct = GEval(name="Correctness", evaluation_steps=CORRECTNESS_STEPS, model=judge, threshold=THRESHOLD,
                            evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT,
                                               SingleTurnParams.EXPECTED_OUTPUT])
            case = LLMTestCase(input=row["question"], actual_output=row["answer"], expected_output=row["reference"],
                               retrieval_context=[row["tool_output"]] if row["tool_output"] else None)
            try:
                await correct.a_measure(case, _show_indicator=False)
                verdict = {"correct": correct.score >= THRESHOLD, "score": round(correct.score, 2), "why": correct.reason}
                if row["tool_output"] and row["run"] == 1:  # costs several calls over a long context: once per scenario
                    faithful = FaithfulnessMetric(model=judge, threshold=THRESHOLD)
                    await faithful.a_measure(case, _show_indicator=False)
                    verdict |= {"faithful": faithful.score >= THRESHOLD, "faithfulness": round(faithful.score, 2),
                                "faithfulness_why": faithful.reason}
                row["judge"] = verdict
            except Exception as exc:  # a judge error is a missing grade, never a pass
                row["judge"] = {"error": repr(exc)}

    rows = [r for r in data["rows"] if _gradable(r)]
    await asyncio.gather(*(one(r) for r in rows if "judge" not in r))
    graded = [r["judge"] for r in rows if "error" not in r["judge"]]
    with_tools = [j for j in graded if "faithful" in j]
    usd = (judge.usage["input"] * 2 + judge.usage["output"] * 10) / 1_000_000
    data["judge_summary"] = {
        "model": JUDGE_MODEL, "graded": len(graded), "errors": len(rows) - len(graded),
        "correct": f"{sum(j['correct'] for j in graded)}/{len(graded)}",
        "faithful": f"{sum(j['faithful'] for j in with_tools)}/{len(with_tools)}",
        "usd_total": round(usd, 4),
    }
    print(data["judge_summary"])
    for r in rows:
        j = r["judge"]
        if "error" in j or not j["correct"] or not j.get("faithful", True):
            print(f"\n  {r['id']} run {r['run']}: {j.get('why') or j.get('error')}\n    faithfulness: {j.get('faithfulness_why', '-')}")


def label(data: dict) -> None:
    """Ask a person, case by case. The judge's verdict stays hidden so it cannot anchor the label."""
    rows = [r for r in data["rows"] if _gradable(r) and "human" not in r and r["run"] == 1]  # one answer per scenario
    print(f"{len(rows)} answers to label. y = correct, n = wrong, q = stop.\n")
    for n, r in enumerate(rows, start=1):
        print(f"--- {n}/{len(rows)}  {r['id']}\nQUESTION:  {r['question']}\nEXPECTED:  {r['reference']}\nANSWER:\n{r['answer']}\n")
        choice = ""
        while choice not in ("y", "n", "q"):
            choice = input("correct? [y/n/q] ").strip().lower()
        if choice == "q":
            break
        r["human"] = choice == "y"


def agreement(data: dict) -> None:
    pairs = [(r["human"], r["judge"]["correct"]) for r in data["rows"]
             if "human" in r and "judge" in r and "error" not in r["judge"]]
    if not pairs:
        print("Nothing to compare yet: run `judge` and `label` first.")
        return
    n = len(pairs)
    observed = sum(h == j for h, j in pairs) / n
    p_h, p_j = sum(h for h, _ in pairs) / n, sum(j for _, j in pairs) / n
    expected = p_h * p_j + (1 - p_h) * (1 - p_j)
    print(f"{n} answers labelled by both. Agreement {observed:.0%}.")
    if expected == 1:
        print("Cohen's kappa is undefined: both graders gave one and the same label to every answer.")
    else:
        print(f"Cohen's kappa {(observed - expected) / (1 - expected):.2f}")
    for r in data["rows"]:
        if "human" in r and "judge" in r and "error" not in r["judge"] and r["human"] != r["judge"]["correct"]:
            print(f"  disagree on {r['id']} run {r['run']}: human {r['human']}, judge {r['judge']['correct']} ({r['judge']['why']})")
