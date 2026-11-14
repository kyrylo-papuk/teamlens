"""Evaluate the Delivery agent.

  python -m teamlens.evals run [--model claude-haiku-5-5] [--effort medium] [--repeat 1] [--only q_away,a_cover]
  python -m teamlens.evals judge  RESULTS.json     grade the answers with an independent model
  python -m teamlens.evals label  RESULTS.json     grade them yourself, without seeing the judge
  python -m teamlens.evals agreement RESULTS.json  judge against human
  python -m teamlens.evals patterns                compare the plan and react runs (run both with --mode first)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from teamlens.config import Settings
from teamlens.evals import runner
from teamlens.evals.scenarios import build
from teamlens.seed import generate

RESULTS = Path("evals/results")


def show(result: dict) -> None:
    for r in sorted(result["rows"], key=lambda r: (r["group"], r["id"], r["run"])):
        mark = "ok  " if r["passed"] else "FAIL"
        notes = "; ".join(r["failures"] + [f"INVARIANT {b}" for b in r["broken_invariants"]])
        print(f"  {mark} {r['id']:14} {r['outcome']:10} {r['calls']} calls  ${r['usd']:.4f}  {r['seconds']:5.1f}s  {notes}")
    s = result["summary"]
    print(f"\n{result['model']} / effort {result['effort']} / mode {result.get('mode', 'plan')} / {result['repeat']} run(s) per scenario")
    print(f"  passed            {s['passed']}   " + "  ".join(f"{g} {v}" for g, v in s["by_group"].items()))
    print(f"  invariants broken {s['invariants_broken']}")
    print(f"  per request       ${s['usd_per_request']}, {s['model_calls_per_request']} model calls, "
          f"{s['seconds_p50']}s p50, {s['seconds_p95']}s p95   (run total ${s['usd_total']})")
    print(f"  approvals asked   {s['approvals_asked']} in {len(result['rows'])} requests, {s['tokens_per_request']} tokens per request")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--model", default="claude-haiku-5-5")
    run.add_argument("--effort", default="medium")
    run.add_argument("--repeat", type=int, default=1)
    run.add_argument("--only", default="")
    run.add_argument("--groups", default="", help="comma-separated scenario groups, e.g. action,refusal")
    run.add_argument("--mode", default="plan", choices=["plan", "react"])
    for name in ("judge", "label", "agreement"):
        sub.add_parser(name).add_argument("file", type=Path)
    sub.add_parser("patterns")
    args = parser.parse_args()
    settings = Settings()

    if args.command == "patterns":
        plan = json.loads((RESULTS / "claude-haiku-5-5-medium.json").read_text(encoding="utf-8"))
        react = json.loads((RESULTS / "claude-haiku-5-5-medium-react.json").read_text(encoding="utf-8"))
        table = runner.compare_modes(plan, react)
        print(f"{'':42}{'mode':7}{'passed':>8}{'broken':>8}{'calls':>7}{'tokens':>8}{'$/1000':>8}{'p50 s':>7}{'approvals':>11}")
        for name, modes in table.items():
            for mode, r in modes.items():
                print(f"{name if mode == 'plan' else '':42}{mode:7}{r['passed']:>8}{r['invariants_broken']:>8}{r['model_calls']:>7}"
                      f"{r['tokens']:>8}{r['usd_per_1000']:>8}{r['seconds_p50']:>7}{r['approvals_asked']:>11}")
        (RESULTS / "patterns.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
        print(f"\nsaved to {RESULTS / 'patterns.json'}")
        return

    if args.command == "run":
        c = generate()
        scenarios = [s for s in build(c) if (not args.only or s.id in args.only.split(","))
                     and (not args.groups or s.group in args.groups.split(","))]
        result = asyncio.run(runner.run(scenarios, c, args.model, args.effort, args.repeat, settings.anthropic_api_key,
                                        mode=args.mode))
        show(result)
        if not args.only:
            RESULTS.mkdir(parents=True, exist_ok=True)
            suffix = ("" if args.mode == "plan" else f"-{args.mode}") + (f"-{args.groups.replace(',', '+')}" if args.groups else "")
            out = RESULTS / f"{args.model}-{args.effort}{suffix}.json"
            out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
            print(f"  saved to {out}")
        return

    from teamlens.evals import judge
    data = json.loads(args.file.read_text(encoding="utf-8"))
    if args.command == "judge":
        asyncio.run(judge.grade(data, settings.anthropic_api_key))
    elif args.command == "label":
        judge.label(data)
    else:
        judge.agreement(data)
        return
    args.file.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
