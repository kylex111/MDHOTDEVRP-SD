"""Run independent-demand replay, fixed-evaluation-budget search, and common-demand replay in sequence."""
from __future__ import annotations

import argparse
import collections
from dataclasses import asdict
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from .artifacts import sha256_file
from .validation_io import (read_json, write_json, write_text, source_inventory,
    preserved_project_state, content_hash, task_directory, verify_hashes, install_recording_adapter)

STAGES = ("A_replay_time", "B_search_eval2000", "C_replay_eval2000")
CPU_IDS = (0, 1, 6, 7, 8, 9, 18, 19)
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
INSTANCE_NAMES = {"c105_21", "r105_21", "rc105_21", "c205_21", "r205_21", "rc205_21"}


def free_space(batch):
    if shutil.disk_usage(batch).free < 5 * 1024**3:
        raise RuntimeError("Free disk space is below 5 GiB; dispatch stopped")


def load_plan(batch):
    batch = Path(batch)
    expected = (batch / "plan.sha256").read_text().strip()
    if sha256_file(batch / "plan.json") != expected:
        raise ValueError("Frozen batch plan changed")
    return read_json(batch / "plan.json")


def verify_frozen(batch, plan, *, all_sources=False):
    if source_inventory(plan["project"]) != plan["source_code_sha256"]:
        raise ValueError("Project source fingerprint changed; dispatch stopped")
    verify_hashes(plan["frozen_files"])
    if all_sources:
        for task in plan["tasks"]:
            verify_hashes(task["source_files"])
    if sys.executable.lower() != plan["python"].lower() or sys.version != plan["environment"]["python"]:
        raise ValueError("Python environment changed")
    import importlib.metadata
    for name, version in plan["environment"]["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"Dependency changed: {name}")


def prepare(source_batch, batch, project, *, n_scenarios=1000, max_eval=2000, workers=8):
    """Freeze the protocol and independent demands from saved records without starting search or simulation."""
    import numpy as np
    from .artifacts import environment_record
    from .validation_checks import verify_search
    from configuration.experiment import stable_seed, ExperimentConfig
    from configuration.parameters import ev_params
    from domain.instances import load_problem, read_instance
    from evaluation.scenarios import generate_normal_scenarios

    source_batch, batch, project = map(lambda p: Path(p).resolve(), (source_batch, batch, project))
    if n_scenarios != 1000 or max_eval != 2000 or not 1 <= workers <= len(CPU_IDS):
        raise ValueError("This approved protocol requires 1000 OOS scenarios, 2000 evaluations, and 1..8 workers")
    if batch.exists():
        raise FileExistsError("Output already exists; use --resume to verify and continue")
    old = read_json(source_batch / "plan.json")
    if len(old["tasks"]) != 900 or {t["instance"] for t in old["tasks"]} != INSTANCE_NAMES:
        raise ValueError("Expected the six-instance, 900-run source batch")
    grid = {(t["instance"], t["scenario_count"], t["run_id"]) for t in old["tasks"]}
    if grid != {(i, s, r) for i in INSTANCE_NAMES for s in (5, 10, 20, 50, 100) for r in range(1, 31)}:
        raise ValueError("Source task grid is incomplete or duplicated")
    current_sources = source_inventory(project)
    for name, fingerprint in old["source_sha256"].items():
        if name != "src/experiments/cli.py" and current_sources.get(name) != fingerprint:
            raise ValueError(f"Original algorithm/evaluator source changed: {name}")
    time_configs = old["configs"]
    eval_configs = {}
    for key, config in time_configs.items():
        ExperimentConfig(**config).validate().require_ra_parameters()
        if config["max_time_sec"] != 1200 or config["max_eval"] is not None or config["max_iter"] is not None:
            raise ValueError("Source is not the approved fixed-time configuration")
        if config["beta"] != .2 or config["n_scenarios"] != 100 or config["n_runs"] != 30:
            raise ValueError("Source demand protocol mismatch")
        eval_configs[key] = {**config, "max_eval": max_eval, "max_time_sec": None, "max_iter": None}
        ExperimentConfig(**eval_configs[key]).validate()
    free_space(batch.parent)
    # Verify saved results before creating output, avoiding a partially initialized batch.
    tasks, total_solutions, infeasible_solutions, empty_runs = [], 0, 0, 0
    for index, task in enumerate(old["tasks"]):
        output = Path(task["output"])
        record = verify_search(output, task, time_configs[task["config_id"]])
        count = len(record["archive"])
        total_solutions += count
        infeasible_solutions += sum(not s["search_feasible"] for s in record["archive"])
        empty_runs += not any(s["search_feasible"] for s in record["archive"])
        files = ("run.json", "manifest.json", "search_scenarios.json", "search_scenario_records.jsonl.gz")
        tasks.append({k: task[k] for k in ("task_key", "config_id", "instance", "instance_sha256", "scenario_count", "run_id", "seeds")})
        tasks[-1].update(directory=output.name, source_run=str(output / "run.json"),
            source_solutions=count, source_files={str(output / f): sha256_file(output / f) for f in files})
        if (index + 1) % 150 == 0:
            print(json.dumps({"preparation_verified_runs": index + 1}, ensure_ascii=False), flush=True)
    if (total_solutions, infeasible_solutions, empty_runs) != (8080, 246, 41):
        raise ValueError("Source counts differ from the approved inventory")
    original_root = project.parent / "A_Modules"
    original_state = preserved_project_state(original_root)
    batch.mkdir(parents=True, exist_ok=False)
    frozen = {str(source_batch / "plan.json"): sha256_file(source_batch / "plan.json")}
    inputs = {}
    old_seed_values = {s for t in tasks for s in t["seeds"].values()}
    for data in old["inputs"]:
        name = data["instance"]
        source_path = Path(data["path"])
        if sha256_file(source_path) != data["sha256"]:
            raise ValueError("Source instance bytes changed")
        target = batch / "inputs" / source_path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target)
        problem, mapping = load_problem(str(target), ev_params, return_mappings=True)
        seed = stable_seed(20261017, data["sha256"], 0, "oos_validation_v1")
        if seed in old_seed_values or seed in {x["oos_seed"] for x in inputs.values()}:
            raise ValueError("Independent seed collision")
        demands = generate_normal_scenarios(problem.demand, .2, n_scenarios, random_state=seed)
        oos_path = batch / "demands" / f"{name}.npy"
        oos_path.parent.mkdir(parents=True, exist_ok=True)
        with oos_path.open("xb") as stream:
            np.save(stream, demands, allow_pickle=False)
        # No complete demand vector may duplicate an original search scenario.
        oos_rows = {row.tobytes() for row in demands}
        for task in tasks:
            if task["instance"] == name and task["scenario_count"] == 5:
                old_demands = read_json(Path(task["source_run"]).parent / "search_scenarios.json")["generated"]
                if any(np.asarray(row, dtype=float).tobytes() in oos_rows for row in old_demands):
                    raise ValueError("Independent demand vector overlaps the search sample")
        node_data = [{**asdict(node), "type": node.type.value} for node in read_instance(str(target))]
        meta = oos_path.with_suffix(".json")
        write_json(meta, {"instance": name, "instance_sha256": data["sha256"], "seed": seed,
            "seed_rule": 'stable_seed(20261017, instance_sha256, 0, "oos_validation_v1")',
            "distribution": "left-truncated normal at 0; mu=nominal demand; sigma=0.2*mu",
            "beta": .2, "shape": list(demands.shape), "probability_each": 1/n_scenarios,
            "nominal_demand": problem.demand, "nodes_in_matrix_column_order": node_data,
            "customer_mapping": mapping, "matrix_sha256": sha256_file(oos_path),
            "no_search_seed_collision": True, "no_exact_search_demand_collision": True,
            "scenario_ids": "0..999, matrix row order; shared by all runs and both budgets"})
        inputs[name] = {"path": str(target), "instance_sha256": data["sha256"], "source": str(source_path),
            "oos_path": str(oos_path), "oos_sha256": sha256_file(oos_path), "oos_seed": seed,
            "metadata": str(meta)}
        frozen.update({str(p): sha256_file(p) for p in (target, oos_path, meta, source_path)})
    for rel, fingerprint in current_sources.items():
        target = batch / "source_snapshot" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project / rel, target)
        if sha256_file(target) != fingerprint:
            raise ValueError("Source changed during snapshot")
        frozen[str(target)] = fingerprint
    for group, configs in (("time1200", time_configs), ("eval2000", eval_configs)):
        for name, config in configs.items():
            path = batch / "configs" / group / f"{name}.json"
            write_json(path, config)
            frozen[str(path)] = sha256_file(path)
    plan = {"schema_version": "scenario_validation_v1", "created_unix": time.time(),
        "source_batch": str(source_batch), "project": str(project), "python": sys.executable,
        "environment": environment_record(project), "workers": workers, "cpu_ids": list(CPU_IDS[:workers]),
        "threads_per_worker": 1, "stages": STAGES, "oos_scenarios": n_scenarios, "max_eval": max_eval,
        "source_solutions": total_solutions, "source_infeasible_solutions": infeasible_solutions,
        "source_empty_front_runs": empty_runs, "source_code_sha256": current_sources,
        "inputs": inputs, "time_configs": time_configs, "eval_configs": eval_configs, "tasks": tasks,
        "frozen_files": frozen, "original_project": str(original_root), "original_state": original_state,
        "min_free_gib": 5, "automatic_search_retries": 0, "strict_oos_feasibility": True,
        "notes": ["Original infeasible archives remain excluded from all main fronts.",
            "R105/R205 and RC105/RC205 are effectively duplicated when time windows are ignored.",
            "All replay evaluations are counted separately from the search budget."]}
    write_json(batch / "plan.json", plan)
    write_text(batch / "plan.sha256", sha256_file(batch / "plan.json") + "\n")
    write_json(batch / "preparation_checks.json", {"status": "passed", "source_runs": 900,
        "source_solutions": total_solutions, "source_infeasible_solutions": infeasible_solutions,
        "source_empty_front_runs": empty_runs, "extra_searches": 0, "extra_scenario_simulations": 0})
    write_text(batch / "README.md", README)
    write_progress(batch, plan, None, {}, {}, "prepared", time.time())
    return plan


def worker(batch, stage, task_key, cpu):
    import psutil
    for name in THREAD_VARIABLES:
        os.environ[name] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    psutil.Process().cpu_affinity([cpu])
    plan = load_plan(batch)
    task = next(t for t in plan["tasks"] if t["task_key"] == task_key)
    output = task_directory(batch, task, stage)
    started = time.perf_counter()
    try:
        verify_frozen(batch, plan)
        verify_hashes(task["source_files"])
        free_space(batch)
        install_recording_adapter()
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            if stage == "B_search_eval2000":
                from configuration.experiment import ExperimentConfig
                from .run import run_record
                from .validation_checks import verify_search, write_search_checks
                config = plan["eval_configs"][task["config_id"]]
                run_record(plan["inputs"][task["instance"]]["path"], output,
                    ExperimentConfig(**config), task["run_id"], seed_overrides=task["seeds"], compact=True)
                record = verify_search(output, task, config, expected_evaluations=plan["max_eval"],
                    source_scenarios=Path(task["source_run"]).parent / "search_scenarios.json")
                write_search_checks(output, record)
                solutions = len(record["archive"])
            else:
                if stage == "C_replay_eval2000":
                    receipt_for(batch, task, "B_search_eval2000")
                from .scenario_replay import replay_run
                record = replay_run(batch, plan, task, stage)
                solutions = len(record["solutions"])
        receipt = {"status": "completed", "stage": stage, "task_key": task_key, "cpu": cpu,
            "pid": os.getpid(), "finished_unix": time.time(), "worker_seconds": time.perf_counter()-started,
            "counts": record["measurement"]["counts"], "solutions": solutions,
            "checks_sha256": sha256_file(output / "checks.json"), "oos_sha256": record.get("oos_sha256"),
            "source_run_sha256": record.get("source_run_sha256"), "source_code_sha256": content_hash(plan["source_code_sha256"])}
        write_json(output / "receipt.json", receipt)
        print(json.dumps(receipt), flush=True)
    except BaseException as exc:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "worker_failure.json", {"stage": stage, "task_key": task_key,
            "status": "failed", "error": repr(exc), "traceback": traceback.format_exc(),
            "finished_unix": time.time(), "automatic_retry": False})
        raise


def receipt_for(batch, task, stage):
    output = task_directory(batch, task, stage)
    receipt = read_json(output / "receipt.json")
    if receipt["status"] != "completed" or receipt["task_key"] != task["task_key"] or receipt["stage"] != stage:
        raise ValueError("Invalid completion receipt")
    if sha256_file(output / "checks.json") != receipt["checks_sha256"]:
        raise ValueError("Acceptance record changed")
    checks = read_json(output / "checks.json")
    if checks["status"] != "passed":
        raise ValueError("Task has not passed acceptance")
    verify_hashes(checks["files"])
    return receipt


def is_alive(record):
    import psutil
    try:
        return abs(psutil.Process(record["pid"]).create_time()-record["create_time"]) < .1
    except psutil.NoSuchProcess:
        return False


def write_progress(batch, plan, stage, done, active, status, started, error=None):
    stage_done = list(done.values())
    counts = collections.Counter()
    worker_seconds, solutions = 0., 0
    for receipt in stage_done:
        counts.update(receipt["counts"])
        worker_seconds += receipt["worker_seconds"]
        solutions += receipt["solutions"]
    activity = []
    for item in active.values():
        task = item["task"]
        live = task_directory(batch, task, stage) / "live_progress.json"
        state = read_json(live) if live.exists() else {}
        counts.update(state.get("counts", {}))
        activity.append({"instance": task["instance"], "S": task["scenario_count"], "run_id": task["run_id"],
            "pid": item["process"].pid, "cpu": item["cpu"], "progress": state,
            "elapsed_sec": time.time()-item["started_unix"]})
    if stage == "B_search_eval2000":
        total = sum(plan["max_eval"]*t["scenario_count"] for t in plan["tasks"])
    elif stage == "A_replay_time":
        total = plan["source_solutions"]*plan["oos_scenarios"]
    elif stage == "C_replay_eval2000":
        complete = read_json(Path(batch) / "stages" / "B_search_eval2000.json")
        total = complete["solutions"]*plan["oos_scenarios"]
    else:
        total = None
    finished_work = sum(r["counts"].get("scenario_simulations", 0) for r in stage_done)
    eta = (max(0, total-counts.get("scenario_simulations", 0))*worker_seconds/finished_work/plan["workers"]
           if total is not None and finished_work and worker_seconds else None)
    stages = {}
    for name in STAGES:
        path = Path(batch) / "stages" / f"{name}.json"
        stages[name] = read_json(path) if path.exists() else {"status": "pending"}
    progress = {"status": status, "stage": stage, "updated_unix": time.time(),
        "session_elapsed_sec": time.time()-started, "batch_elapsed_sec": time.time()-plan["created_unix"],
        "completed_tasks": len(done), "total_tasks": len(plan["tasks"]), "completed_solutions": solutions,
        "counts_including_active_checkpoints": dict(counts), "total_stage_scenario_evaluations": total,
        "stage_eta_sec": eta, "eta_basis": "observed completed task worker-seconds per scenario / configured workers",
        "active": activity, "stages": stages, "error": error, "workers": plan["workers"],
        "free_disk_gib": shutil.disk_usage(batch).free/1024**3}
    write_json(Path(batch) / "progress.json", progress)
    labels = {"A_replay_time": "A: Replay existing solutions on 1,000 scenarios", "B_search_eval2000": "B: Search with 2,000 evaluations",
              "C_replay_eval2000": "C: Replay new solutions on shared demands", None: "Prepared"}
    rows = "".join(f"<tr><td>{a['instance']}</td><td>{a['S']}</td><td>{a['run_id']}</td><td>{a['cpu']}</td>"
        f"<td>{a['progress'].get('counts',{}).get('candidate_evaluations', '—')}</td>"
        f"<td>{a['progress'].get('counts',{}).get('scenario_simulations', '—')}</td>"
        f"<td>{a['elapsed_sec']/60:.1f} min</td></tr>" for a in activity)
    links = "".join(f"<p><a href='reports/{directory}/comparison.html'>{label}</a></p>"
        for directory,label in (("phase_A","Phase A interim report"),("final","Final report with shared references"))
        if (Path(batch)/"reports"/directory/"comparison.html").exists())
    eta_text = f"{eta/3600:.2f} hours" if eta is not None else "Waiting for the first completed task"
    body = f"""<!doctype html><html lang='en'><meta charset='utf-8'><meta http-equiv='refresh' content='10'>
<title>Scenario validation progress</title><style>body{{font-family:'Microsoft YaHei',sans-serif;background:#f5f7fa;color:#182638;margin:36px}}main{{max-width:1100px;margin:auto}}section{{background:white;padding:22px;margin:18px 0;border-radius:8px}}table{{width:100%;border-collapse:collapse}}td,th{{padding:10px;border-bottom:1px solid #dde3ec;text-align:left}}p{{line-height:1.8}}a{{color:#185ba5}}</style><main>
<h1>{labels[stage]}</h1><p>Status: {html.escape(status)} · Refreshes every 10 seconds · Updated {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
<section><p>Current phase: <b>{len(done)} / 900</b> run tasks completed; active processes: <b>{len(active)} / {plan['workers']}</b>.</p>
<p>Solution evaluations in this phase: {counts.get('candidate_evaluations',0):,}; scenario evaluations: {counts.get('scenario_simulations',0):,} / {total or "Pending"}.</p>
<p>Estimated time remaining: {eta_text}; scheduler elapsed time: {(time.time()-started)/3600:.2f} hours.</p>
<p>Replay counts are separate from search budgets. Scenario failures are retained and remaining scenarios continue; program errors stop dispatch.</p></section>
<section><table><tr><th>Instance</th><th>S</th><th>Run ID</th><th>CPU</th><th>Solution evaluations</th><th>Scenario evaluations</th><th>Elapsed</th></tr>{rows}</table></section>
<section><p>Phases A → B → C run sequentially. Parallel dispatch begins after the first task in each phase passes record checks.</p>{links}<p><a href='README.md'>Experiment notes</a> · <a href='progress.json'>Machine-readable progress</a></p>
<p>{html.escape(error or '')}</p></section></main></html>"""
    write_text(Path(batch) / "progress.html", body)


def execute(batch, plan):
    import psutil
    batch = Path(batch)
    free_space(batch)
    verify_frozen(batch, plan, all_sources=True)
    lock = batch / "dispatcher.json"
    if lock.exists() and is_alive(read_json(lock)):
        raise RuntimeError("A dispatcher for this batch is already running")
    # Check active child processes before dispatching to avoid duplicate work after a scheduler exit.
    for path in (batch / "launches").glob("*/*.json"):
        launch = read_json(path)
        if is_alive(launch):
            raise RuntimeError(f"An earlier worker is still running: {path}")
    write_json(lock, {"pid": os.getpid(), "create_time": psutil.Process().create_time(), "started_unix": time.time()})
    started = time.time()
    active, done, stage = {}, {}, None
    try:
        for stage in STAGES:
            done, pending = {}, []
            verify_frozen(batch, plan, all_sources=True)
            if preserved_project_state(plan["original_project"]) != plan["original_state"]:
                raise ValueError("Original A_Modules source or Git state changed")
            for task in plan["tasks"]:
                output = task_directory(batch, task, stage)
                launch = batch / "launches" / stage / f"{task['task_key']}.json"
                if (output / "receipt.json").exists():
                    done[task["task_key"]] = receipt_for(batch, task, stage)
                elif output.exists() or launch.exists():
                    raise RuntimeError(f"Unfinished/failed task requires review; it will not be rerun automatically: {output}")
                else:
                    pending.append(task)
            failure, next_check = None, 0.
            while pending or active:
                if failure is None:
                    try:
                        free_space(batch)
                        if (batch / "PAUSE").exists():
                            raise RuntimeError("PAUSE marker found; dispatch stopped and in-flight tasks will drain")
                        if time.time() >= next_check:
                            verify_frozen(batch, plan, all_sources=True)
                            next_check = time.time()+60
                    except Exception as exc:
                        failure = repr(exc)
                for key, item in list(active.items()):
                    code = item["process"].poll()
                    if code is None:
                        continue
                    item["stdout"].close()
                    item["stderr"].close()
                    active.pop(key)
                    if code != 0:
                        failure = failure or f"Worker exited with status {code}: {key}; see worker logs and failure record"
                    else:
                        try:
                            done[key] = receipt_for(batch, item["task"], stage)
                        except Exception as exc:
                            failure = failure or repr(exc)
                limit = plan["workers"] if done else 1
                while failure is None and pending and len(active) < limit:
                    task = pending.pop(0)
                    verify_hashes(task["source_files"])
                    cpu = next(c for c in plan["cpu_ids"] if c not in {i["cpu"] for i in active.values()})
                    env = {**os.environ, "PYTHONPATH": str(Path(plan["project"])/"src"),
                        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8",
                        **{key: "1" for key in THREAD_VARIABLES}}
                    logs = batch / "worker_logs" / stage
                    logs.mkdir(parents=True, exist_ok=True)
                    out = (logs / f"{task['task_key']}.stdout.log").open("x", encoding="utf-8")
                    err = (logs / f"{task['task_key']}.stderr.log").open("x", encoding="utf-8")
                    command = [plan["python"], "-B", "-m", "experiments.scenario_validation", "--worker",
                        "--out", str(batch), "--stage", stage, "--task-key", task["task_key"], "--cpu", str(cpu)]
                    child = subprocess.Popen(command, cwd=plan["project"], env=env, stdout=out, stderr=err,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                    launch = {"pid": child.pid, "create_time": psutil.Process(child.pid).create_time(),
                        "stage": stage, "task_key": task["task_key"], "cpu": cpu, "started_unix": time.time(), "command": command}
                    write_json(batch / "launches" / stage / f"{task['task_key']}.json", launch)
                    active[task["task_key"]] = {"process": child, "task": task, "cpu": cpu,
                        "stdout": out, "stderr": err, "started_unix": launch["started_unix"]}
                write_progress(batch, plan, stage, done, active, "draining_after_stop" if failure else "running", started, failure)
                if failure and not active:
                    raise RuntimeError(failure)
                if active:
                    time.sleep(5)
            verify_frozen(batch, plan, all_sources=True)
            if len(done) != 900:
                raise ValueError("Incomplete stage")
            summary = {"status": "completed", "stage": stage, "tasks": len(done), "finished_unix": time.time(),
                "solutions": sum(r["solutions"] for r in done.values()),
                "candidate_evaluations": sum(r["counts"].get("candidate_evaluations", 0) for r in done.values()),
                "scenario_simulations": sum(r["counts"].get("scenario_simulations", 0) for r in done.values()),
                "route_simulations": sum(r["counts"].get("route_simulations", 0) for r in done.values()),
                "worker_seconds": sum(r["worker_seconds"] for r in done.values()),
                "original_project_preserved": preserved_project_state(plan["original_project"]) == plan["original_state"]}
            if not summary["original_project_preserved"]:
                raise ValueError("Original project changed")
            write_json(batch / "stages" / f"{stage}.json", summary)
            if stage != "B_search_eval2000":
                write_progress(batch, plan, stage, done, active, "building_report", started)
                for checked_stage in (STAGES if stage == "C_replay_eval2000" else STAGES[:1]):
                    for task in plan["tasks"]:
                        receipt_for(batch, task, checked_stage)
                from analysis.scenario_validation import make_report
                make_report(batch, plan, final=stage == "C_replay_eval2000")
        write_json(batch / "completion.json", {"status": "completed", "finished_unix": time.time(),
            "stages": list(STAGES), "final_report": "reports/final/comparison.html"})
        write_progress(batch, plan, stage, done, active, "completed", started)
    except BaseException as exc:
        # Let active tasks finish writing; do not terminate or rerun them. Check remaining processes before resuming.
        write_json(batch / "dispatcher_failure.json", {"status": "stopped", "stage": stage,
            "error": repr(exc), "traceback": traceback.format_exc(), "finished_unix": time.time(),
            "in_flight_pids": [i["process"].pid for i in active.values()]})
        write_progress(batch, plan, stage, done, active, "stopped", started, repr(exc))
        raise


def entry(args, project):
    batch = Path(args.out).resolve()
    for name in THREAD_VARIABLES:
        os.environ[name] = "1"
    if args.resume:
        plan = load_plan(batch)
        if (Path(args.source_batch).resolve() != Path(plan["source_batch"]) or args.oos_scenarios != plan["oos_scenarios"]
                or args.max_eval != plan["max_eval"] or args.workers != plan["workers"]):
            raise ValueError("Resume options differ from frozen protocol")
        verify_frozen(batch, plan, all_sources=True)
    else:
        plan = prepare(args.source_batch, batch, project, n_scenarios=args.oos_scenarios,
                       max_eval=args.max_eval, workers=args.workers)
    if not args.prepare_only:
        execute(batch, plan)


README = """# 1,000-scenario replay and fixed 2,000-evaluation experiment

## Phases and records

A_replay_time: Reevaluate 8,080 archived plans from the original 900 searches on 1,000 independent demands each. Keep customer assignments fixed; do not run search.
B_search_eval2000: Six instances x five scenario counts x 30 runs, with exactly 2,000 candidate evaluations including initialization. No time or iteration budget.
C_replay_eval2000: Reevaluate all B archives on the same demands frozen in A.

progress.html refreshes automatically; progress.json provides machine-readable status. Each task saves run.json, compressed scenario records, checks.json, and receipt.json.
demands/ stores full .npy matrices and JSON metadata for node mapping, seeds, distribution, and hashes. source_snapshot/ stores code; configs/ stores both configurations.
Each phase starts with one formal task and record checks before expanding to eight single-threaded workers on cores 0, 1, 6, 7, 8, 9, 18, and 19.

## Statistics

Main tables require original search feasibility and success in all 1,000 new scenarios. Retain all failures; do not replace seeds or rerun scientific failures.
Record penalized means, success-conditional means, sample standard deviations, approximate Student-t 95% intervals, failure rates, and Wilson 95% intervals per solution.
Execution before failure is diagnostic, not the economic cost of completed delivery.
Between-run mean intervals use 10,000 BCa resamples. Paired differences use common valid run IDs. Fixed demand samples are not independent searches.
Empty fronts have HV/PFS=0; IGD+ and objective statistics are missing with valid n shown. Prefix stability for 100/250/500/1,000 scenarios reuses saved records.
reports/phase_A/comparison.html uses provisional time-group references. reports/final/comparison.html shares normalization and IGD+ references across both groups, with HV reference (1.1, 1.1).

## Pause and resume

Create a PAUSE file in this directory to stop new dispatch and exit after active tasks finish. Move that file away, then repeat the command with --resume to continue.
Verified completed tasks are not repeated. Incomplete tasks, program errors, source changes, or less than 5 GiB free space stop dispatch and preserve records for inspection.
Do not modify code, frozen configurations, demand matrices, or the source batch during execution. Scenario execution failures are normal results; remaining scenarios continue.
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Internal isolated scenario validation worker")
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--task-key", required=True)
    parser.add_argument("--cpu", type=int, required=True)
    arguments = parser.parse_args()
    worker(arguments.out, arguments.stage, arguments.task_key, arguments.cpu)
