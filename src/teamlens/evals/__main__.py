"""Evaluate the Delivery agent.

  python -m teamlens.evals run [--model claude-haiku-5-5] [--effort medium] [--repeat 1] [--only q_away,a_cover]
  python -m teamlens.evals judge  RESULTS.json     grade the answers with an independent model
  python -m teamlens.evals label  RESULTS.json     grade them yourself, without seeing the judge
  python -m teamlens.evals agreement RESULTS.json  judge against human
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
    print(f"\n{result['model']} / effort {result['effort']} / {result['repeat']} run(s) per scenario")
    print(f"  passed            {s['passed']}   " + "  ".join(f"{g} {v}" for g, v in s["by_group"].items()))
    print(f"  invariants broken {s['invariants_broken']}")
    print(f"  per request       ${s['usd_per_request']}, {s['model_calls_per_request']} model calls, "
          f"{s['seconds_p50']}s p50, {s['seconds_p95']}s p95   (run total ${s['usd_total']})")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--model", default="claude-haiku-5-5")
    run.add_argument("--effort", default="medium")
    run.add_argument("--repeat", type=int, default=1)
    run.add_argument("--only", default="")
    for name in ("judge", "label", "agreement"):
        sub.add_parser(name).add_argument("file", type=Path)
    args = parser.parse_args()
    settings = Settings()

    if args.command == "run":
        c = generate()
        scenarios = [s for s in build(c) if not args.only or s.id in args.only.split(",")]
        result = asyncio.run(runner.run(scenarios, c, args.model, args.effort, args.repeat, settings.anthropic_api_key))
        show(result)
        if not args.only:
            RESULTS.mkdir(parents=True, exist_ok=True)
            out = RESULTS / f"{args.model}-{args.effort}.json"
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
