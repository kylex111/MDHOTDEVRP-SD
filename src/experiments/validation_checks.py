"""Verify saved search records without calling a solver."""
from __future__ import annotations

import collections
import gzip
import json
import math
from pathlib import Path

from .artifacts import sha256_file
from .validation_io import read_json, write_json


def verify_search(output, task, config, *, expected_evaluations=None, source_scenarios=None):
    output = Path(output)
    record = read_json(output / "run.json")
    if record["status"] != "completed" or record["config"] != config:
        raise ValueError(f"Incomplete search or configuration mismatch: {output}")
    for key in ("run_id", "seeds", "instance_sha256"):
        if record[key] != task[key]:
            raise ValueError(f"Search identity mismatch: {key}")
    if read_json(output / "manifest.json")["status"] != "completed":
        raise ValueError("Search manifest not completed")
    counts = record["measurement"]["counts"]
    for field in ("candidate_evaluations", "scenario_simulations", "route_simulations"):
        if record[field] != counts[field]:
            raise ValueError(f"Counter disagreement: {field}")
    if expected_evaluations is not None:
        if record["candidate_evaluations"] != expected_evaluations or record["stop_reason"] != "max_eval":
            raise ValueError("Search did not stop at the exact candidate evaluation budget")
        if not record["measurement"]["budget_includes_initialization"]:
            raise ValueError("Initialization missing from budget")
    size = task["scenario_count"]
    if record["scenario_simulations"] != record["candidate_evaluations"] * size:
        raise ValueError("Candidate and scenario counters disagree")
    scenes = read_json(output / "search_scenarios.json")
    if len(scenes["generated"]) != 100 or len(scenes["reduced"]) != size:
        raise ValueError("Unexpected search demand counts")
    if scenes["scenario_seed"] != task["seeds"]["search"] or scenes["beta"] != config["beta"]:
        raise ValueError("Search demand provenance mismatch")
    if source_scenarios is not None and scenes != read_json(source_scenarios):
        raise ValueError("Search demand set differs from the matching time-budget run")
    probs = scenes["probabilities"]
    if len(probs) != size or not math.isclose(sum(probs), 1, abs_tol=1e-12):
        raise ValueError("Invalid reduced probabilities")
    source = {s["solution_id"]: s for s in record["archive"]}
    if len(source) != len(record["archive"]):
        raise ValueError("Repeated archive solution ID")
    values = collections.defaultdict(lambda: {"ids": set(), "cost": 0., "ra": 0., "completed": True})
    raw = output / record["observation_files"]["search_scenario_records"]
    with gzip.open(raw, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            sid, index = row["solution_id"], row["scenario_id"]
            if sid not in source or row["run_id"] != task["run_id"] or index not in range(size):
                raise ValueError("Invalid raw search identity")
            state = values[sid]
            if index in state["ids"] or not math.isclose(row["probability"], probs[index], abs_tol=1e-12):
                raise ValueError("Repeated scenario or probability mismatch")
            state["ids"].add(index)
            state["cost"] += row["probability"] * row["cost_with_failure_penalty"]
            state["ra"] += row["probability"] * row["ra_with_failure_penalty"]
            state["completed"] &= bool(row["completed"])
            if not isinstance(row["failure_reason_counts"], dict):
                raise ValueError("Missing failure diagnostics")
    if set(values) != set(source):
        raise ValueError("Missing archive observations")
    for sid, state in values.items():
        if state["ids"] != set(range(size)) or state["completed"] != source[sid]["search_feasible"]:
            raise ValueError("Incomplete observations or feasibility disagreement")
        for key in ("cost", "ra"):
            if not math.isclose(state[key], source[sid][key], rel_tol=1e-10, abs_tol=1e-8):
                raise ValueError("Saved search objective cannot be reproduced")
    expected_rows = size * len(source)
    if record["observation_files"]["search_rows"] != expected_rows:
        raise ValueError("Raw record count mismatch")
    return record


def write_search_checks(output, record):
    output = Path(output)
    names = ("run.json", "manifest.json", "search_scenarios.json", "search_scenario_records.jsonl.gz",
             "checkpoints.jsonl", "initialization.json", "recording_schema.json")
    write_json(output / "checks.json", {"status": "passed", "candidate_evaluations": record["candidate_evaluations"],
        "scenario_simulations": record["scenario_simulations"], "solutions": len(record["archive"]),
        "extra_searches": 0, "extra_scenario_simulations_for_checks": 0,
        "files": {str(output / name): sha256_file(output / name) for name in names}})
