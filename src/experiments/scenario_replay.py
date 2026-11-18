"""Reevaluate fixed archived plans on independent demand without changing solver or causal decoding logic."""
from __future__ import annotations

import collections
import gzip
import json
import math
from pathlib import Path
import time

import numpy as np

from configuration.parameters import ev_params
from domain.instances import load_problem
from evaluation.budget import Measurement
from evaluation.evaluator import Evaluator
from analysis.scenario_validation import summarize_solution
from .artifacts import json_value, sha256_file
from .validation_io import read_json, write_json, content_hash, task_directory


def source_run_path(batch, task, stage):
    return (Path(task["source_run"]) if stage == "A_replay_time"
            else task_directory(batch, task, stage.replace("C_replay_", "B_search_", 1)) / "run.json")


def replay_run(batch, plan, task, stage):
    output = task_directory(batch, task, stage)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source_path = source_run_path(batch, task, stage)
    source = read_json(source_path)
    if source["status"] != "completed" or source["instance_sha256"] != task["instance_sha256"]:
        raise ValueError("Invalid source run")
    if source["run_id"] != task["run_id"] or source["seeds"] != task["seeds"]:
        raise ValueError("Source run identity mismatch")
    source_hash = sha256_file(source_path)
    cfg = source["config"]
    expected_cfg = plan["time_configs" if stage == "A_replay_time" else "eval_configs"][task["config_id"]]
    if cfg != expected_cfg:
        raise ValueError("Source configuration mismatch")
    dataset = plan["inputs"][task["instance"]]
    if sha256_file(dataset["path"]) != task["instance_sha256"]:
        raise ValueError("Instance changed")
    if sha256_file(dataset["oos_path"]) != dataset["oos_sha256"]:
        raise ValueError("Independent demands changed")
    problem = load_problem(dataset["path"], ev_params)
    scenarios = np.load(dataset["oos_path"], allow_pickle=False)
    n = plan["oos_scenarios"]
    if scenarios.shape != (n, len(problem.demand)) or not np.isfinite(scenarios).all() or np.any(scenarios < 0):
        raise ValueError("Invalid independent demand matrix")
    measurement = Measurement()
    measurement.set_phase("replay")
    evaluator = Evaluator(problem, ra_safe=cfg["ra_safe"], ra_risk=cfg["ra_risk"],
        recourse_policy=cfg["recourse_policy"], measurement=measurement,
        capture_scenarios="metrics", ra_tau_min=cfg["ra_tau_min"],
        ra_formula_version=cfg["ra_formula_version"],
        charging_policy=cfg.get("charging_policy", "ra_based"),
        soc_threshold=cfg.get("soc_threshold", 0.20))
    identity = {"stage": stage, "budget_group": "time1200" if stage == "A_replay_time" else f"eval{cfg['max_eval']}",
        "instance": task["instance"], "instance_sha256": task["instance_sha256"],
        "scenario_count": task["scenario_count"], "run_id": task["run_id"],
        "source_task_key": task["task_key"], "source_run": str(source_path),
        "source_run_sha256": source_hash, "oos_seed": dataset["oos_seed"],
        "oos_sha256": dataset["oos_sha256"], "oos_scenarios": n,
        "configuration": cfg, "source_code_sha256": plan["source_code_sha256"]}
    write_json(output / "manifest.json", {**identity, "status": "started"})
    summaries = []
    filename = "scenario_records.jsonl.gz"
    with gzip.open(output / filename, "xt", encoding="utf-8", compresslevel=5) as stream:
        for sol in source["archive"]:
            before = content_hash(sol["routes_by_clusters"])
            routes = {int(k): v for k, v in sol["routes_by_clusters"].items()}
            value = evaluator.evaluate_saa(routes, scenarios)
            if before != content_hash(sol["routes_by_clusters"]):
                raise ValueError("Replay changed the saved customer arrangement")
            observations = value["scenario_observations"]
            for observation in observations:
                row = {"role": "independent_replay", "run_id": task["run_id"],
                    "solution_id": sol["solution_id"],
                    **{k: v for k, v in observation.items() if k not in ("routes", "evaluation")}}
                stream.write(json.dumps(json_value(row), ensure_ascii=False, allow_nan=False,
                                        separators=(",", ":")) + "\n")
            stream.flush()
            summary = summarize_solution(sol, observations)
            for key, expected in (("cost", value["total_cost"]), ("ra", value["total_ra"])):
                if not math.isclose(summary[key]["mean"], expected, rel_tol=1e-10, abs_tol=1e-8):
                    raise ValueError("Replayed objectives do not match raw observations")
            summaries.append(summary)
            write_json(output / "live_progress.json", {**identity, "status": "replaying",
                "completed_solutions": len(summaries), "total_solutions": len(source["archive"]),
                "completed_scenarios": len(summaries) * n, "counts": measurement.counts,
                "elapsed_sec": time.perf_counter() - started, "updated_unix": time.time()})
    record = {**identity, "status": "completed", "solutions": summaries,
        "replay_runtime_sec": time.perf_counter() - started,
        "measurement": measurement.snapshot(), "raw_records": filename,
        "raw_records_sha256": sha256_file(output / filename),
        "raw_record_count": len(summaries) * n,
        "search_measurement": {k: source[k] for k in
            ("candidate_evaluations", "scenario_simulations", "route_simulations", "runtime_sec")}}
    record["measurement"]["candidate_definition"] = "one archived solution re-evaluated; separate from the search budget"
    record["measurement"]["budget_includes_initialization"] = False
    write_json(output / "run.json", record)
    write_json(output / "manifest.json", {**identity, "status": "completed", "result": "run.json",
        "raw_records": filename, "raw_record_count": len(summaries) * n})
    return verify_replay(output, task, plan)


def verify_replay(output, task, plan):
    """Recompute saved metrics only, without decoding or simulation."""
    output = Path(output)
    record = read_json(output / "run.json")
    source = read_json(record["source_run"])
    n = plan["oos_scenarios"]
    if record["status"] != "completed" or record["oos_scenarios"] != n:
        raise ValueError("Replay is incomplete")
    if record["oos_sha256"] != plan["inputs"][task["instance"]]["oos_sha256"]:
        raise ValueError("Replay used a different demand set")
    if sha256_file(record["source_run"]) != record["source_run_sha256"]:
        raise ValueError("Source record changed during replay")
    if sha256_file(output / record["raw_records"]) != record["raw_records_sha256"]:
        raise ValueError("Raw replay records changed")
    sums = collections.defaultdict(lambda: {"ids": set(), "cost": 0.0, "ra": 0.0, "fails": 0,
                                           "success_cost": 0.0, "success_ra": 0.0})
    with gzip.open(output / record["raw_records"], "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["run_id"] != task["run_id"] or not math.isclose(row["probability"], 1/n, abs_tol=1e-14):
                raise ValueError("Raw row identity or probability mismatch")
            state = sums[row["solution_id"]]
            if row["scenario_id"] in state["ids"]:
                raise ValueError("Repeated scenario ID")
            state["ids"].add(row["scenario_id"])
            state["cost"] += row["cost_with_failure_penalty"]
            state["ra"] += row["ra_with_failure_penalty"]
            state["fails"] += not row["completed"]
            if row["completed"]:
                state["success_cost"] += row["cost_with_failure_penalty"]
                state["success_ra"] += row["ra_with_failure_penalty"]
    original = {s["solution_id"]: s for s in source["archive"]}
    saved = {s["solution_id"]: s for s in record["solutions"]}
    if set(sums) != set(original) or set(saved) != set(original):
        raise ValueError("Archive solution IDs were lost")
    for sid, item in sums.items():
        sol = saved[sid]
        if item["ids"] != set(range(n)) or sol["scenario_count"] != n:
            raise ValueError("Incomplete demand coverage")
        if sol["routes_by_clusters"] != original[sid]["routes_by_clusters"]:
            raise ValueError("Stored routes changed")
        if sol["search_feasible"] != original[sid]["search_feasible"]:
            raise ValueError("Original feasibility was overwritten")
        for metric in ("cost", "ra"):
            if not math.isclose(item[metric]/n, sol[metric]["mean"], rel_tol=1e-10, abs_tol=1e-8):
                raise ValueError("Mean cannot be reproduced")
        if sol["failure"]["failures"] != item["fails"] or sol["oos_feasible"] != (item["fails"] == 0):
            raise ValueError("Failure statistics disagree")
        if sol["eligible"] != bool(original[sid]["search_feasible"] and item["fails"] == 0):
            raise ValueError("Strict feasibility rule changed")
    counters = record["measurement"]["counts"]
    if counters.get("candidate_evaluations", 0) != len(original) or counters.get("scenario_simulations", 0) != n*len(original):
        raise ValueError("Replay counters disagree")
    checks = {"status": "passed", "solutions": len(original), "scenario_records": n*len(original),
        "original_infeasible_solutions": sum(not s["search_feasible"] for s in original.values()),
        "files": {str(output / name): sha256_file(output / name) for name in
                  ("run.json", "manifest.json", record["raw_records"])},
        "extra_searches": 0, "extra_scenario_simulations_for_checks": 0}
    write_json(output / "checks.json", checks)
    return record
