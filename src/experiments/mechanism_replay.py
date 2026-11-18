"""Independent demand replay for fixed plans and their frozen charging policies."""
from __future__ import annotations
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import time
import numpy as np

from analysis.scenario_validation import summarize_solution, mean_interval
from configuration.parameters import ev_params
from domain.instances import load_problem
from evaluation.evaluator import Evaluator
from evaluation.budget import Measurement
from experiments.artifacts import sha256_file, json_value
from experiments.validation_io import read_json, write_json, content_hash
from experiments.mechanism_ablation import search_path, ensure_space

OPERATING = ("num_charge", "low_soc_driving_time_min", "charging_time_min", "charged_energy_kwh")


def replay(output, plan, task):
    directory = Path(output) / "replay" / task["task_key"]
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source_path = search_path(output, plan, task)
    source = read_json(source_path)
    source_hash = sha256_file(source_path)
    cfg = plan["configs"][task["variant"]]
    old_cfg = source["config"]
    if old_cfg.get("charging_policy", "ra_based") != cfg["charging_policy"]:
        raise ValueError("Saved charging policy disagrees with replay configuration")
    item = plan["inputs"][task["instance"]]
    problem = load_problem(item["path"], ev_params)
    demands = np.load(item["oos_path"], allow_pickle=False)
    n = plan["oos_scenarios"]
    if sha256_file(item["oos_path"]) != item["oos_sha256"] or demands.shape != (n, len(problem.demand)):
        raise ValueError("Validation demand matrix changed")
    measurement = Measurement()
    measurement.set_phase("replay")
    evaluator = Evaluator(problem, cfg["ra_safe"], cfg["ra_risk"],
        recourse_policy=cfg["recourse_policy"], ra_tau_min=cfg["ra_tau_min"],
        ra_formula_version=cfg["ra_formula_version"], charging_policy=cfg["charging_policy"],
        soc_threshold=cfg["soc_threshold"], collect_low_soc=True,
        measurement=measurement, capture_scenarios="metrics")
    identity = {k: task[k] for k in ("task_key", "instance", "instance_sha256", "variant", "run_id", "seeds")}
    identity.update(source_run=str(source_path), source_run_sha256=source_hash,
                    configuration=cfg, oos_scenarios=n, oos_sha256=item["oos_sha256"], oos_seed=item["oos_seed"],
                    low_soc_diagnostic_threshold=.20)
    write_json(directory / "manifest.json", {**identity, "status": "started"})
    summaries = []
    raw = directory / "scenario_records.jsonl.gz"
    with gzip.open(raw, "xt", encoding="utf-8", compresslevel=5) as stream:
        for sol in source["archive"]:
            ensure_space(output)
            routes = {int(k): v for k,v in sol["routes_by_clusters"].items()}
            before = content_hash(routes)
            value = evaluator.evaluate_saa(routes, demands)
            if content_hash(routes) != before:
                raise ValueError("Replay mutated a saved customer arrangement")
            observations = value["scenario_observations"]
            if len(observations) != n or {o["scenario_id"] for o in observations} != set(range(n)):
                raise ValueError("Incomplete replay")
            for obs in observations:
                if any(obs.get(k) is None for k in (*OPERATING, "energy_failure_event")):
                    raise ValueError("Missing operating measurements")
                row = {"solution_id": sol["solution_id"], "run_id": task["run_id"],
                       **{k:v for k,v in obs.items() if k not in ("routes", "evaluation")}}
                stream.write(json.dumps(json_value(row), allow_nan=False, separators=(",", ":")) + "\n")
            stream.flush()
            summary = summarize_solution(sol, observations, requested_prefixes=())
            for metric, expected in (("cost", value["total_cost"]), ("ra", value["total_ra"])):
                if not math.isclose(summary[metric]["mean"], expected, rel_tol=1e-10, abs_tol=1e-8):
                    raise ValueError("Scenario mean disagrees with evaluation")
            for metric in OPERATING:
                summary[metric] = mean_interval([o[metric] for o in observations])
                summary[metric + "_conditional_on_success"] = mean_interval([o[metric] for o in observations if o["completed"]])
            summaries.append(summary)
            write_json(directory / "live_progress.json", {"status": "replaying", "completed_solutions": len(summaries),
                "total_solutions": len(source["archive"]), "completed_scenarios": len(summaries)*n,
                "counts": measurement.counts, "elapsed_sec": time.perf_counter()-started, "updated_unix": time.time()})
    record = {**identity, "status": "completed", "solutions": summaries,
              "replay_runtime_sec": time.perf_counter()-started, "measurement": measurement.snapshot(),
              "raw_records": raw.name, "raw_records_sha256": sha256_file(raw), "raw_record_count": n*len(summaries),
              "search_measurement": {k:source[k] for k in ("runtime_sec", "candidate_evaluations", "scenario_simulations", "route_simulations")}}
    record["measurement"]["candidate_definition"] = "one saved solution replay; separate from search budget"
    record["measurement"]["budget_includes_initialization"] = False
    write_json(directory / "run.json", record)
    write_json(directory / "manifest.json", {**identity, "status": "completed", "result": "run.json"})


def verify_replay(directory, task, plan):
    directory = Path(directory)
    r = read_json(directory / "run.json")
    source = read_json(r["source_run"])
    n = plan["oos_scenarios"]
    if r["status"] != "completed" or r["configuration"] != plan["configs"][task["variant"]]:
        raise ValueError("Incomplete replay or strategy mismatch")
    for k in ("run_id", "instance", "variant", "seeds", "task_key", "instance_sha256"):
        if r[k] != task[k]:
            raise ValueError(f"Replay identity mismatch: {k}")
    if r["oos_scenarios"] != n or r["oos_sha256"] != plan["inputs"][task["instance"]]["oos_sha256"]:
        raise ValueError("Replay demand provenance mismatch")
    if sha256_file(r["source_run"]) != r["source_run_sha256"] or sha256_file(directory/r["raw_records"]) != r["raw_records_sha256"]:
        raise ValueError("Replay/source fingerprint changed")
    sums = defaultdict(lambda: {"ids":set(), "cost":0., "ra":0., "fail":0, "energy":0,
                               "success_cost":0., "success_ra":0., **{k:0. for k in OPERATING}})
    with gzip.open(directory/r["raw_records"], "rt", encoding="utf-8") as stream:
        for line in stream:
            o = json.loads(line)
            s = sums[o["solution_id"]]
            if o["scenario_id"] in s["ids"] or o["run_id"] != task["run_id"] or not math.isclose(o["probability"],1/n,abs_tol=1e-14):
                raise ValueError("Duplicate/invalid scenario identity")
            s["ids"].add(o["scenario_id"])
            s["cost"] += o["cost_with_failure_penalty"]
            s["ra"] += o["ra_with_failure_penalty"]
            s["fail"] += not o["completed"]
            s["energy"] += o["energy_failure_event"]
            if o["completed"]:
                s["success_cost"] += o["cost_with_failure_penalty"]
                s["success_ra"] += o["ra_with_failure_penalty"]
            for k in OPERATING:
                s[k] += o[k]
    originals = {s["solution_id"]:s for s in source["archive"]}
    summaries = {s["solution_id"]:s for s in r["solutions"]}
    if set(sums) != set(originals) or set(summaries) != set(originals) or len(summaries)!=len(r["solutions"]):
        raise ValueError("Lost/duplicate saved solution")
    for sid, s in sums.items():
        summary, original = summaries[sid], originals[sid]
        if s["ids"] != set(range(n)) or summary["routes_by_clusters"] != original["routes_by_clusters"]:
            raise ValueError("Demand coverage or saved routes changed")
        if summary["search_feasible"] != original["search_feasible"] or summary["eligible"] != bool(original["search_feasible"] and s["fail"]==0):
            raise ValueError("Feasibility convention changed")
        if summary["failure"]["failures"] != s["fail"] or summary["energy_failure"]["failures"] != s["energy"]:
            raise ValueError("Failure counts disagree")
        for k in ("cost", "ra", *OPERATING):
            if not math.isclose(summary[k]["mean"],s[k]/n,rel_tol=1e-10,abs_tol=1e-8):
                raise ValueError(f"Cannot reproduce {k}")
        for k in ("cost", "ra"):
            expected = s["success_"+k]/(n-s["fail"]) if s["fail"]<n else None
            saved = summary[k+"_conditional_on_success"]["mean"]
            if (saved is None) != (expected is None) or expected is not None and not math.isclose(saved,expected,rel_tol=1e-10,abs_tol=1e-8):
                raise ValueError("Conditional mean mismatch")
    counts = r["measurement"]["counts"]
    if counts.get("candidate_evaluations",0)!=len(originals) or counts.get("scenario_simulations",0)!=n*len(originals) or r["raw_record_count"] != n*len(originals):
        raise ValueError("Replay counters disagree")
    write_json(directory / "checks.json", {"status":"passed", "solutions":len(originals), "scenarios":n*len(originals),
        "extra_simulations":0, "files":{str(directory/name):sha256_file(directory/name) for name in ("run.json","manifest.json",r["raw_records"])}})
    return r
