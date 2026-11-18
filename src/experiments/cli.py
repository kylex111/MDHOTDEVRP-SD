"""CLI for single runs, batch planning/execution, tuning plans, and result summaries."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from configuration.experiment import load_config
from .artifacts import save_json

PROJECT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    single = commands.add_parser("single", help="run one instance and one run ID")
    single.add_argument("--config", required=True)
    single.add_argument("--instance", required=True)
    single.add_argument("--run-id", type=int, default=1)
    single.add_argument("--out", required=True)
    for name in ("plan", "tune"):
        plan = commands.add_parser(
            name, help="write a batch plan without running a solver"
        )
        plan.add_argument("--config", required=True)
        plan.add_argument(
            "--dataset", required=True, help="instance file or dataset directory"
        )
        plan.add_argument("--pattern", default="*.txt")
        plan.add_argument("--out", required=True)
    batch = commands.add_parser("batch", help="execute or resume a saved batch")
    batch.add_argument("--batch", required=True)
    batch.add_argument("--workers", type=int, default=1)
    summary = commands.add_parser("analyze", help="summarize saved run.json files")
    summary.add_argument("--results", required=True)
    summary.add_argument("--out", required=True)
    validation = commands.add_parser("scenario-validation", help="replay saved solutions, run fixed evaluation budgets, and replay again")
    validation.add_argument("--source-batch", required=True)
    validation.add_argument("--out", required=True)
    validation.add_argument("--oos-scenarios", type=int, default=1000)
    validation.add_argument("--max-eval", type=int, default=2000)
    validation.add_argument("--workers", type=int, default=8)
    validation.add_argument("--resume", action="store_true")
    validation.add_argument("--prepare-only", action="store_true", help="freeze inputs without starting search or replay")
    scenario = commands.add_parser("scenario-experiment", help="run fresh fixed-budget searches and independent-demand validation")
    scenario.add_argument("--config", required=True)
    scenario.add_argument("--dataset", required=True)
    scenario.add_argument("--out", required=True)
    scenario.add_argument("--oos-scenarios", type=int, default=1000)
    scenario.add_argument("--workers", type=int, default=8)
    scenario.add_argument("--analysis-md", help="append the result table to an existing analysis document")
    scenario.add_argument("--prepare-only", action="store_true")
    scenario.add_argument("--resume", action="store_true")
    mechanism = commands.add_parser("mechanism-ablation", help="paired GB and charging ablations with independent validation")
    mechanism.add_argument("--baseline", required=True)
    mechanism.add_argument("--out", required=True)
    mechanism.add_argument("--workers", type=int, default=8)
    mechanism.add_argument("--phase", choices=("all", "prepare", "search", "replay", "report"), default="all")
    mechanism.add_argument("--resume", action="store_true")
    mechanism.add_argument("--analysis-md")
    args = parser.parse_args(argv)
    if args.command == "mechanism-ablation":
        from .mechanism_ablation import entry
        entry(args, PROJECT)
    elif args.command == "single":
        from .run import run_record

        result = run_record(
            args.instance, args.out, load_config(args.config), args.run_id
        )
        print(
            json.dumps(
                {"status": result["status"], "archive_size": len(result["archive"])},
                ensure_ascii=False,
            )
        )
    elif args.command in ("plan", "tune"):
        from .batch import build_plan, save_plan

        config = load_config(args.config)
        rows = [("main", None, config)]
        if args.command == "tune":
            from .tuning import configurations

            rows = configurations(config)
        output = save_plan(
            build_plan(args.dataset, rows, PROJECT, args.pattern), args.out
        )
        print(str(output / "batch.json"))
    elif args.command == "batch":
        if args.workers < 1:
            parser.error("--workers must be positive")
        from .batch import execute_plan

        execute_plan(args.batch, PROJECT, args.workers)
    elif args.command == "scenario-experiment":
        from .scenario_experiment import entry

        entry(args, PROJECT)
    elif args.command == "scenario-validation":
        from .scenario_validation import entry

        entry(args, PROJECT)
    else:
        from analysis.summary import summarize

        save_json(args.out, summarize(args.results))
