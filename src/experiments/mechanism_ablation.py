"""Paired mechanism ablations: frozen searches, independent replay, and reporting."""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import csv
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import replace
import hashlib
from pathlib import Path
import shutil
import time
import traceback
import zipfile

from configuration.experiment import ExperimentConfig, stable_seed
from experiments.artifacts import sha256_file, environment_record
from experiments.validation_io import read_json, write_json, write_csv, write_text, source_inventory, verify_hashes
from experiments.validation_checks import verify_search

INSTANCES = ("c103_21", "r103_21", "rc103_21", "c203_21", "r203_21", "rc203_21")
VARIANTS = ("full", "no_gb", "energy_only", "soc_threshold")


def ensure_space(path):
    if shutil.disk_usage(path).free < 5 * 1024**3:
        raise RuntimeError("Less than 5 GiB free; dispatch stopped")


def task_key(instance, variant, run_id):
    return f"{instance}_{variant}_r{run_id:03d}"


def baseline_record(prior, task):
    return Path(prior) / "runs" / f"{task['instance']}_gbga_movns_r{task['run_id']:03d}"


def validate_coverage(record):
    from evaluation.evaluator import Evaluator
    expected = sorted(n for cluster in record["clusters"].values() for n in cluster)
    for sol in record["archive"]:
        nodes = [n for route in sol["routes_by_clusters"].values() for n in Evaluator._flatten_routes(route)]
        if sorted(nodes) != expected or len(set(nodes)) != len(nodes):
            raise ValueError("Archive customer coverage is invalid")


def prepare(root, output, prior, workers, analysis_md):
    """Audit the old baseline before creating the new immutable batch."""
    import numpy as np
    from configuration.parameters import ev_params
    from domain.instances import load_problem
    from evaluation.scenarios import generate_normal_scenarios

    root, output, prior = map(lambda x: Path(x).resolve(), (root, output, prior))
    if output.exists():
        raise FileExistsError("Output exists; use --resume, never overwrite")
    ensure_space(root)
    old = read_json(prior / "plan.json")
    if sha256_file(prior / "plan.json") != (prior / "plan.sha256").read_text().strip():
        raise ValueError("Baseline plan fingerprint changed")
    verify_hashes(old["frozen_files"])
    audit_path = root / "docs" / "mechanism_implementation_audit.json"
    implementation = read_json(audit_path)
    if implementation["checks_status"] != "passed":
        raise ValueError("Implementation checks missing")
    if implementation["deployed_source_sha256"] != source_inventory(root):
        raise ValueError("Implementation source differs from checked source")
    if implementation["baseline_snapshot_sha256"] != sha256_file(prior / "source_snapshot.zip"):
        raise ValueError("Baseline code snapshot differs from audited version")
    expected_hashes = read_json(prior / "reports" / "input_hashes.json")
    original_cfg = old["configs"]["gbga_movns"]
    base = ExperimentConfig(**original_cfg).validate().require_ra_parameters()
    if (base.algorithm != "spr" or base.init_mode != "gbga" or not base.use_movns
            or base.vnd_order != "ra_first" or base.max_time_sec != 1200
            or base.max_eval is not None or base.max_iter is not None
            or base.n_runs != 30 or base.n_scenarios != 100 or base.n_reduced_scenarios != 10
            or base.beta != .2 or base.gb_mutation_rate != .3):
        raise ValueError("Baseline does not match approved design")
    baseline, tasks, used_seeds = {}, [], set()
    for instance in INSTANCES:
        for run_id in range(1, 31):
            source_task = next(t for t in old["tasks"] if t["instance"] == instance
                               and t["variant"] == "gbga_movns" and t["run_id"] == run_id)
            directory = baseline_record(prior, source_task)
            path = directory / "run.json"
            if expected_hashes[str(path)] != sha256_file(path):
                raise ValueError(f"Baseline run changed: {path}")
            check = read_json(directory / "checks.json")
            verify_hashes(check["files"])
            record = verify_search(directory, source_task, original_cfg)
            validate_coverage(record)
            if record["stop_reason"] != "max_time" or not record["measurement"]["budget_includes_initialization"]:
                raise ValueError("Baseline budget mismatch")
            used_seeds.update(record["seeds"].values())
            key = task_key(instance, "full", run_id)
            baseline[key] = {"run": str(path), "files": check["files"], "record_sha256": sha256_file(path)}
            for variant in VARIANTS:
                tasks.append({**{k: source_task[k] for k in ("instance", "instance_sha256", "run_id", "seeds", "scenario_count")},
                              "task_key": task_key(instance, variant, run_id), "variant": variant})
    configs = {
        "full": base.as_dict(),
        "no_gb": replace(base, algorithm="without_gb").validate().as_dict(),
        "energy_only": replace(base, charging_policy="energy_only").validate().as_dict(),
        "soc_threshold": replace(base, charging_policy="soc_threshold", soc_threshold=.20).validate().as_dict(),
    }
    output.mkdir(parents=True)
    frozen = {str(prior / name): sha256_file(prior / name) for name in ("plan.json", "source_snapshot.zip", "reports/input_hashes.json")}
    inputs = {}
    for instance in INSTANCES:
        item = old["inputs"][instance]
        src = Path(item["path"])
        if sha256_file(src) != item["sha256"]:
            raise ValueError("Instance fingerprint changed")
        dest = output / "inputs" / src.name
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(src, dest)
        problem = load_problem(dest, ev_params)
        seed = stable_seed(20261030, item["sha256"], 0, "mechanism_ablation_oos_v1")
        if seed in used_seeds:
            raise ValueError("Validation seed collides with a search seed")
        demands = generate_normal_scenarios(mu=np.asarray(problem.demand, float), beta=.2,
                                            n_scenarios=2000, random_state=seed)
        matrix = output / "demands" / f"{instance}.npy"
        matrix.parent.mkdir(exist_ok=True)
        np.save(matrix, demands, allow_pickle=False)
        inputs[instance] = {"path": str(dest), "sha256": item["sha256"], "oos_path": str(matrix),
                            "oos_sha256": sha256_file(matrix), "oos_seed": seed}
        meta = matrix.with_suffix(".json")
        write_json(meta, {**inputs[instance], "shape": list(demands.shape), "beta": .2,
                          "customer_ids": list(problem.customers), "node_ids": list(range(len(problem.demand))),
                          "mean_demand": list(problem.demand), "generator": "generate_normal_scenarios; no reduction"})
        for path in (dest, matrix, meta):
            frozen[str(path)] = sha256_file(path)
    for variant, cfg in configs.items():
        path = output / "configs" / f"{variant}.json"
        write_json(path, cfg)
        frozen[str(path)] = sha256_file(path)
    for helper in ("launch_mechanism_ablation.py", "start_mechanism_ablation.cmd", "docs/mechanism_implementation_audit.json"):
        frozen[str(root / helper)] = sha256_file(root / helper)
    source = source_inventory(root)
    with zipfile.ZipFile(output / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name in [*source, "launch_mechanism_ablation.py", "start_mechanism_ablation.cmd", "docs/mechanism_implementation_audit.json"]:
            archive.write(root / name, name)
    write_json(output / "implementation_audit.json", implementation)
    frozen[str(output / "implementation_audit.json")] = sha256_file(output / "implementation_audit.json")
    frozen[str(output / "source_snapshot.zip")] = sha256_file(output / "source_snapshot.zip")
    # Rotate variants while interleaving instances and run IDs to reduce ordering effects.
    tasks.sort(key=lambda t: (t["run_id"], INSTANCES.index(t["instance"]),
                             (VARIANTS.index(t["variant"]) - t["run_id"]) % 4))
    plan = {"schema": "mechanism_ablation_v1", "root": str(root), "output": str(output), "prior": str(prior),
            "created_unix": time.time(), "workers": workers, "oos_scenarios": 2000,
            "configs": configs, "baseline_original_config": original_cfg, "inputs": inputs, "tasks": tasks,
            "baseline": baseline, "source_sha256": source, "frozen_files": frozen,
            "analysis_md": str(analysis_md) if analysis_md else None,
            "source_reuse": "180 Table 6 runs, verified hashes and same RA-first/evaluation rules; source differences audited",
            "new_searches": 540, "replay_runs": 720}
    write_json(output / "environment.json", environment_record(root))
    write_json(output / "plan.json", plan)
    write_text(output / "plan.sha256", sha256_file(output / "plan.json"))
    write_json(output / "progress.json", {"stage": "prepared", "new_searches": 540, "reused_runs": 180,
                                         "replay_runs": 720, "workers": workers, "updated_unix": time.time()})
    return plan


def load_plan(output, check_sources=True):
    output = Path(output)
    if sha256_file(output / "plan.json") != (output / "plan.sha256").read_text().strip():
        raise ValueError("Batch plan changed")
    plan = read_json(output / "plan.json")
    verify_hashes(plan["frozen_files"])
    if check_sources and source_inventory(plan["root"]) != plan["source_sha256"]:
        raise ValueError("Source fingerprint changed; stop dispatch")
    return plan


def search_path(output, plan, task):
    return (Path(plan["baseline"][task["task_key"]]["run"]) if task["variant"] == "full"
            else Path(output) / "search" / task["task_key"] / "run.json")


def check_search(output, plan, task):
    path = search_path(output, plan, task)
    if task["variant"] == "full":
        verify_hashes(plan["baseline"][task["task_key"]]["files"])
        cfg = plan["baseline_original_config"]
    else:
        cfg = plan["configs"][task["variant"]]
    record = verify_search(path.parent, task, cfg,
                           source_scenarios=baseline_record(plan["prior"], task) / "search_scenarios.json")
    validate_coverage(record)
    if record["stop_reason"] != "max_time" or record["runtime_sec"] < 1199.99:
        raise ValueError("Unexpected search termination")
    if not record["measurement"]["budget_includes_initialization"]:
        raise ValueError("Initialization omitted from budget")
    if bool(record["use_gb"]) != (task["variant"] != "no_gb"):
        raise ValueError("Granular-ball mode mismatch")
    return record


def result_receipt(output, stage, task, record):
    from experiments.validation_checks import write_search_checks
    if stage == "search":
        directory = Path(output) / "search" / task["task_key"]
        write_search_checks(directory, record)
        files = read_json(directory / "checks.json")["files"]
        row = {"runtime_sec": record["runtime_sec"], "candidate_evaluations": record["candidate_evaluations"],
               "scenario_simulations": record["scenario_simulations"], "solutions": len(record["archive"]),
               "feasible_solutions": sum(s["search_feasible"] for s in record["archive"])}
    else:
        directory = Path(output) / "replay" / task["task_key"]
        files = read_json(directory / "checks.json")["files"]
        row = {"runtime_sec": record["replay_runtime_sec"], "candidate_evaluations": len(record["solutions"]),
               "scenario_simulations": record["raw_record_count"], "solutions": len(record["solutions"]),
               "feasible_solutions": sum(s["eligible"] for s in record["solutions"])}
    receipt = {"task_key": task["task_key"], "variant": task["variant"], "instance": task["instance"],
               "run_id": task["run_id"], "stage": stage, **row, "files": files}
    write_json(directory / "receipt.json", receipt)
    return receipt


def worker(output, stage, key):
    from experiments.validation_io import install_recording_adapter
    install_recording_adapter()
    plan = load_plan(output)
    task = next(t for t in plan["tasks"] if t["task_key"] == key)
    directory = Path(output) / stage / key
    ensure_space(output)
    try:
        if stage == "search":
            if directory.exists():
                raise FileExistsError("Existing unfinished search is not automatically rerun")
            from experiments.run import run_record
            run_record(plan["inputs"][task["instance"]]["path"], directory,
                       ExperimentConfig(**plan["configs"][task["variant"]]), task["run_id"],
                       seed_overrides=task["seeds"], compact=True)
            record = check_search(output, plan, task)
        else:
            from experiments.mechanism_replay import replay, verify_replay
            if directory.exists():
                raise FileExistsError("Existing unfinished replay is not automatically rerun")
            check_search(output, plan, task)
            replay(output, plan, task)
            record = verify_replay(directory, task, plan)
        return result_receipt(output, stage, task, record)
    except BaseException:
        write_json(Path(output) / "failures" / f"{stage}_{key}.json",
                   {"stage": stage, "task": task, "traceback": traceback.format_exc(), "time": time.time()})
        raise


def progress(output, plan, stage, done, active, error, since):
    total = 540 if stage == "search" else 720
    seconds = sum(r["runtime_sec"] for r in done)
    mean = seconds / len(done) if done else None
    active_rows = []
    for key, started in active.values():
        live = Path(output) / stage / key / "live_progress.json"
        active_rows.append({"task_key": key, "elapsed_sec": time.time() - started,
                            "live": read_json(live) if live.exists() else None})
    eta = None if mean is None else max(0., ((total-len(done))*mean - sum(min(time.time()-s, mean) for _,s in active.values())) / plan["workers"])
    write_json(Path(output) / "progress.json", {
        "stage": stage if error is None else "stopping_after_error", "error": error,
        "completed": len(done), "total": total, "reused_baseline_runs": 180, "workers": plan["workers"],
        "active_tasks": active_rows, "completed_candidate_evaluations": sum(r["candidate_evaluations"] for r in done),
        "completed_scenario_simulations": sum(r["scenario_simulations"] for r in done),
        "empty_feasible_front_runs": sum(r["feasible_solutions"] == 0 for r in done),
        "elapsed_this_session_sec": time.time()-since, "elapsed_since_preparation_sec": time.time()-plan["created_unix"],
        "remaining_stage_sec_estimate": eta, "eta_basis": "completed formal task worker-seconds; excludes later stages",
        "scheduler_pid": os.getpid(), "updated_unix": time.time()})
    write_csv(Path(output) / f"completed_{stage}.csv", [{k:v for k,v in r.items() if k != "files"} for r in done])


def execute_stage(output, stage):
    plan = load_plan(output)
    tasks = [t for t in plan["tasks"] if stage != "search" or t["variant"] != "full"]
    if stage == "replay":
        for task in plan["tasks"]:
            check_search(output, plan, task)
    done, pending = [], []
    for task in tasks:
        directory = Path(output) / stage / task["task_key"]
        if (directory / "receipt.json").exists():
            receipt = read_json(directory / "receipt.json")
            verify_hashes(receipt["files"])
            done.append(receipt)
        elif (directory / "run.json").exists():
            # Recover completed files after a scheduler interruption; never repeat scientific work.
            if stage == "search":
                record = check_search(output, plan, task)
            else:
                from experiments.mechanism_replay import verify_replay
                record = verify_replay(directory, task, plan)
            done.append(result_receipt(output, stage, task, record))
        elif directory.exists():
            raise RuntimeError(f"Unfinished task preserved; manual review required: {directory}")
        else:
            pending.append(task["task_key"])
    iterator, active, error, since = iter(pending), {}, None, time.time()
    with ProcessPoolExecutor(max_workers=plan["workers"]) as pool:
        def dispatch():
            nonlocal error
            if error or (Path(output) / "PAUSE").exists():
                return
            try:
                load_plan(output)
                ensure_space(output)
                key = next(iterator, None)
                if key:
                    active[pool.submit(worker, str(output), stage, key)] = (key, time.time())
            except Exception:
                error = traceback.format_exc()
        for _ in range(plan["workers"]):
            dispatch()
        progress(output, plan, stage, done, active, error, since)
        while active:
            completed, _ = wait(active, timeout=30, return_when=FIRST_COMPLETED)
            for future in completed:
                key, _ = active.pop(future)
                try:
                    done.append(future.result())
                    print(f"{stage}: {len(done)}/{len(tasks)} {key}", flush=True)
                except BaseException:
                    error = traceback.format_exc()
            # Inspect every completed future before dispatching replacements.
            for _ in completed:
                dispatch()
            try:
                progress(output, plan, stage, done, active, error, since)
            except Exception:
                error = traceback.format_exc()
        progress(output, plan, stage, done, active, error, since)
    if error:
        raise RuntimeError(error)
    if len(done) != len(tasks):
        write_json(Path(output) / "progress.json", {"stage": "paused", "phase": stage, "completed": len(done), "total": len(tasks)})
        return False
    write_json(Path(output) / f"{stage}_complete.json", {"count": len(done), "completed_unix": time.time()})
    return True


def entry(args, root):
    output = Path(args.out).resolve()
    if args.workers != 8:
        raise ValueError("This approved batch uses eight single-thread workers")
    if not args.resume:
        prepare(root, output, args.baseline, args.workers, args.analysis_md)
    plan = load_plan(output)
    if Path(args.baseline).resolve() != Path(plan["prior"]):
        raise ValueError("Baseline argument differs from frozen plan")
    if args.phase == "prepare":
        return
    import msvcrt
    with (output / "scheduler.lock").open("a+b") as lock:
        lock.write(b"0"); lock.flush(); lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            if args.phase in ("all", "search"):
                if not execute_stage(output, "search"):
                    return
            if args.phase in ("all", "replay"):
                if not execute_stage(output, "replay"):
                    return
            if args.phase in ("all", "report"):
                from analysis.mechanism_ablation import report
                report(output)
                write_json(output / "progress.json", {"stage": "completed", "searches": 540, "reused_runs": 180,
                           "replay_runs": 720, "updated_unix": time.time(), "reports": "reports"})
        except BaseException:
            write_json(output / "scheduler_error.json", {"error": traceback.format_exc(), "time": time.time()})
            raise
