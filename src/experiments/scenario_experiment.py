"""Run a fresh scenario-count experiment, followed by common-demand validation."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, replace
import html
import os
from pathlib import Path
import shutil
import sys
import time

SCENARIO_COUNTS = (5, 10, 20, 50, 100)
INSTANCE_NAMES = {"c105_21", "r105_21", "rc105_21", "c205_21", "r205_21", "rc205_21"}
SEARCH = "B_search_eval8000"
REPLAY = "C_replay_eval8000"


def freeze_check(plan):
    from .validation_io import source_inventory, verify_hashes
    if source_inventory(plan["project"]) != plan["source_code_sha256"]:
        raise ValueError("Source changed after preparation; use its original version to resume")
    verify_hashes(plan["frozen_files"])
    if sys.version != plan["environment"]["python"]:
        raise ValueError("Python version changed")
    import importlib.metadata
    for name, version in plan["environment"]["packages"].items():
        if version is not None and importlib.metadata.version(name) != version:
            raise ValueError(f"Dependency changed: {name}")


def prepare(args, project):
    import numpy as np
    from configuration.experiment import load_config, stable_seed
    from configuration.parameters import ev_params
    from domain.instances import load_problem, read_instance
    from evaluation.scenarios import generate_normal_scenarios
    from .artifacts import environment_record, sha256_file
    from .validation_io import source_inventory, write_json, write_text, install_recording_adapter
    install_recording_adapter()
    from .batch import build_plan

    batch = Path(args.out).resolve()
    if batch.exists():
        raise FileExistsError("Output exists; use --resume to validate and continue")
    config = load_config(args.config).require_ra_parameters()
    if config.algorithm != "spr" or config.n_runs != 30 or config.n_scenarios != 100 or config.beta != .2:
        raise ValueError("Require SPR, 30 runs, 100 original scenarios, and beta=0.2")
    if not config.use_movns or config.init_mode != "gbga":
        raise ValueError("Require GBGA initialization and MOVNS")
    rows = [(f"S{s:03d}", None, replace(config, n_reduced_scenarios=s,
             max_eval=8000, max_time_sec=None, max_iter=None, threads_per_run=1))
            for s in SCENARIO_COUNTS]
    search_plan = build_plan(args.dataset, rows, project)
    instances = search_plan["instances"]
    if len(instances) != 6 or {Path(i["path"]).stem for i in instances} != INSTANCE_NAMES:
        raise ValueError("Dataset directory must contain exactly the six specified instance files")
    if args.oos_scenarios != 1000:
        raise ValueError("This protocol uses 1,000 independent scenarios per instance")
    if not 1 <= args.workers <= 8:
        raise ValueError("Use 1..8 single-threaded workers")
    batch.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(batch.parent).free < 5 * 1024**3:
        raise RuntimeError("Free disk space is below 5 GiB")
    batch.mkdir(exist_ok=False)
    configs = {key: cfg.as_dict() for key, _, cfg in rows}
    inputs, frozen, tasks = {}, {}, []
    search_seeds = {seed for instance in instances for run_id in range(1, 31)
                    for seed in config.seeds(instance["sha256"], run_id).values()}
    for instance in instances:
        source = Path(instance["path"])
        target = batch / "inputs" / source.name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, target)
        problem, mapping = load_problem(str(target), ev_params, return_mappings=True)
        seed = stable_seed(20261017, instance["sha256"], 0, "oos_validation_v1")
        if seed in search_seeds:
            raise ValueError("Independent demand seed collides with a search seed")
        demands = generate_normal_scenarios(problem.demand, config.beta, 1000, random_state=seed)
        demand_path = batch / "demands" / f"{source.stem}.npy"
        demand_path.parent.mkdir(exist_ok=True)
        with demand_path.open("xb") as stream:
            np.save(stream, demands, allow_pickle=False)
        metadata = demand_path.with_suffix(".json")
        write_json(metadata, {"seed": seed, "beta": config.beta, "shape": list(demands.shape),
            "distribution": "normal truncated below zero; sigma=beta*nominal demand",
            "probability_each": 1/1000, "matrix_sha256": sha256_file(demand_path),
            "instance_sha256": instance["sha256"], "customer_mapping": mapping,
            "nodes": [{**asdict(node), "type": node.type.value} for node in read_instance(str(target))]})
        inputs[source.stem] = {"path": str(target), "instance_sha256": instance["sha256"],
            "oos_path": str(demand_path), "oos_sha256": sha256_file(demand_path), "oos_seed": seed}
        frozen.update({str(p): sha256_file(p) for p in (target, demand_path, metadata)})
    for old in search_plan["tasks"]:
        cfg = configs[old["config_id"]]
        name = Path(old["instance"]["path"]).stem
        tasks.append({"task_key": old["task_key"], "directory": old["directory"],
            "config_id": old["config_id"], "scenario_count": cfg["n_reduced_scenarios"],
            "instance": name, "instance_sha256": old["instance"]["sha256"],
            "run_id": old["run_id"], "seeds": config.seeds(old["instance"]["sha256"], old["run_id"])})
    plan = {"schema_version": "fresh_scenario_experiment_v1", "project": str(project),
        "created_unix": time.time(), "python": sys.executable, "workers": args.workers,
        "oos_scenarios": 1000, "max_eval": 8000, "eval_configs": configs,
        "inputs": inputs, "tasks": tasks, "frozen_files": frozen,
        "source_code_sha256": source_inventory(project), "environment": environment_record(project),
        "analysis_md": str(Path(args.analysis_md).resolve()) if args.analysis_md else None}
    write_json(batch / "plan.json", plan)
    write_text(batch / "plan.sha256", sha256_file(batch / "plan.json"))
    progress(batch, "prepared", 0, len(tasks))
    return plan


def load_plan(batch):
    from .artifacts import sha256_file
    from .validation_io import read_json
    if sha256_file(batch / "plan.json") != (batch / "plan.sha256").read_text().strip():
        raise ValueError("Frozen plan changed")
    plan = read_json(batch / "plan.json")
    if plan["schema_version"] != "fresh_scenario_experiment_v1":
        raise ValueError("Not a fresh scenario experiment")
    freeze_check(plan)
    return plan


def verify_task(batch, plan, task, stage):
    from .scenario_replay import verify_replay
    from .validation_checks import verify_search, write_search_checks
    from .validation_io import task_directory
    output = task_directory(batch, task, stage)
    if stage == SEARCH:
        record = verify_search(output, task, plan["eval_configs"][task["config_id"]], expected_evaluations=plan["max_eval"])
        write_search_checks(output, record)
        return record
    return verify_replay(output, task, plan)


def worker(batch, stage, task_key):
    from configuration.experiment import ExperimentConfig
    from .scenario_replay import replay_run
    from .validation_io import task_directory, install_recording_adapter
    import numpy as np
    batch = Path(batch)
    plan = load_plan(batch)
    task = next(t for t in plan["tasks"] if t["task_key"] == task_key)
    if shutil.disk_usage(batch).free < 5 * 1024**3:
        raise RuntimeError("Free disk space is below 5 GiB")
    output = task_directory(batch, task, stage)
    if output.exists():
        return verify_task(batch, plan, task, stage)
    install_recording_adapter()
    from .run import run_record
    if stage == SEARCH:
        run_record(plan["inputs"][task["instance"]]["path"], output,
            ExperimentConfig(**plan["eval_configs"][task["config_id"]]), task["run_id"], compact=True)
        verify_task(batch, plan, task, stage)
    else:
        from .validation_io import read_json
        source = task_directory(batch, task, SEARCH)
        scenes = np.asarray(read_json(source / "search_scenarios.json")["generated"], dtype=float)
        demands = np.load(plan["inputs"][task["instance"]]["oos_path"], allow_pickle=False)
        demand_rows = {row.tobytes() for row in demands}
        if any(row.tobytes() in demand_rows for row in scenes):
            raise ValueError("Independent demands overlap the search sample")
        replay_run(batch, plan, task, stage)
    return task_key


def progress(batch, stage, done, total, error=None):
    from .validation_io import write_json, write_text
    write_json(batch / "progress.json", {"stage": stage, "completed_runs": done,
        "total_runs": total, "updated_unix": time.time(), "error": error})
    write_text(batch / "progress.html", "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<meta http-equiv='refresh' content='600'><title>Scenario experiment</title>"
        f"<h1>Scenario experiment</h1><p>Stage: {html.escape(stage)}</p>"
        f"<p>Completed: {done} / {total}</p><p>{html.escape(error or '')}</p>"
        "<p>Search and independent-demand evaluation use separate counts.</p>")


def execute_stage(batch, plan, stage):
    from .validation_io import task_directory
    pending, done = [], 0
    for task in plan["tasks"]:
        if task_directory(batch, task, stage).exists():
            verify_task(batch, plan, task, stage)
            done += 1
        else:
            pending.append(task)
    progress(batch, stage, done, len(plan["tasks"]))
    # The first formal task is retained and checked before enabling parallel work.
    if pending:
        first = pending.pop(0)
        try:
            worker(str(batch), stage, first["task_key"])
        except Exception as exc:
            progress(batch, stage, done, len(plan["tasks"]), f"{first['directory']}: {exc!r}")
            raise
        done += 1
        progress(batch, stage, done, len(plan["tasks"]))
        if (batch / "PAUSE").exists():
            progress(batch, "paused", done, len(plan["tasks"]))
            return False
    iterator = iter(pending)
    with ProcessPoolExecutor(max_workers=plan["workers"]) as pool:
        active = {}
        def dispatch():
            task = next(iterator, None)
            if task is not None:
                active[pool.submit(worker, str(batch), stage, task["task_key"])] = task
        for _ in range(plan["workers"]):
            dispatch()
        while active:
            completed, _ = wait(active, return_when=FIRST_COMPLETED)
            # Resolve all completed futures before sending any further work.
            for future in completed:
                task = active.pop(future)
                try:
                    future.result()
                except Exception as exc:
                    progress(batch, stage, done, len(plan["tasks"]), f"{task['directory']}: {exc!r}")
                    raise
                done += 1
                print(f"{stage}: {done}/{len(plan['tasks'])} {task['directory']}", flush=True)
            progress(batch, stage, done, len(plan["tasks"]))
            if (batch / "PAUSE").exists():
                for future in active:
                    future.result()
                progress(batch, "paused", done + len(active), len(plan["tasks"]))
                return False
            for _ in completed:
                dispatch()
    return True


def report(batch, plan):
    from analysis.scenario_validation import build_contexts, run_metrics, aggregate_runs
    from .validation_io import read_json, task_directory, write_json, write_csv, write_text
    runs = [read_json(task_directory(batch, t, REPLAY) / "run.json") for t in plan["tasks"]]
    contexts = build_contexts(runs)
    rows, prefixes, solutions = [], [], []
    for run in runs:
        row, prefix = run_metrics(run, contexts[run["instance"]])
        rows.append(row)
        prefixes.extend(prefix)
        for sol in run["solutions"]:
            solutions.append({**{k: run[k] for k in ("instance", "scenario_count", "run_id")}, **sol})
    groups = aggregate_runs(rows)
    output = batch / "reports"
    output.mkdir(exist_ok=True)
    write_csv(output / "runs.csv", rows)
    write_csv(output / "solutions.csv", solutions)
    write_csv(output / "front_stability.csv", prefixes)
    write_csv(output / "groups.csv", [{"instance": g["instance"], "scenario_count": g["scenario_count"],
        "metric": key, **value} for g in groups for key, value in g["metrics"].items()])
    write_csv(output / "empty_front_rates.csv", [{"instance": g["instance"],
        "scenario_count": g["scenario_count"], "measure": key, **g[key]}
        for g in groups for key in ("source_empty_rate", "oos_empty_rate")])
    write_json(output / "report.json", {"groups": groups, "runs": rows,
        "oos_scenarios": 1000, "source": "fresh fixed-evaluation search only"})
    write_json(output / "quality_contexts.json", {name: None if context is None else {
        "lower": context.lower, "upper": context.upper, "reference_front": context.reference_front,
        "hv_reference": context.hv_reference, "source": context.source} for name, context in contexts.items()})
    names = sorted(INSTANCE_NAMES)
    index = {(g["instance"], g["scenario_count"]): g for g in groups}
    metrics = ("HV", "IGD_plus", "PFS", "best_cost", "best_ra", "candidate_evaluations",
               "solution_failure_rate", "source_empty", "oos_empty", "front_retention")
    table = ["| Scenarios / metric | " + " | ".join(names) + " |",
             "|---|" + "---:|" * len(names)]
    for count in SCENARIO_COUNTS:
        for metric in metrics:
            cells = []
            for name in names:
                value = index[name, count]["metrics"][metric]
                cells.append("NA" if value["mean"] is None else
                    f"{value['mean']:.6g} ± {value['std']:.4g} (n={value['n']})" if value["std"] is not None
                    else f"{value['mean']:.6g} (n={value['n']})")
            table.append(f"| {count} / {metric} | " + " | ".join(cells) + " |")
    notes = ("Fresh 8,000-evaluation searches; each saved solution is reevaluated on 1,000 shared independent demands. "
        "Main fronts require original search feasibility and success in all OOS scenarios. "
        "Empty fronts have HV/PFS=0; other metrics report valid n. Means and sample standard deviations are shown. "
        "groups.csv includes run-level BCa 95% intervals; solutions.csv includes Student-t and Wilson intervals. "
        "Penalized objectives and success-conditional objectives are separate; no post-failure completion cost is defined. "
        "Each instance uses shared normalization and an empirical pooled OOS reference front; HV reference=(1.1,1.1). "
        "R105/R205 and RC105/RC205 have duplicate effective inputs when time windows are unused.")
    rendered_rows = []
    for line in [table[0], *table[2:]]:
        rendered_rows.append("<tr>" + "".join(f"<td>{html.escape(cell.strip())}</td>"
                             for cell in line.strip("|").split("|")) + "</tr>")
    write_text(output / "comparison.html", "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<title>Scenario experiment results</title><style>body{font-family:Arial;margin:24px}"
        "table{border-collapse:collapse}td{padding:8px;border:1px solid #ccc;white-space:nowrap}"
        "tr:first-child{font-weight:bold;background:#eef2f6}</style><h1>Fixed-budget scenario experiment</h1>"
        f"<p>{html.escape(notes)}</p><table>{''.join(rendered_rows)}</table>"
        "<p><a href='groups.csv'>Group statistics and confidence intervals</a> · "
        "<a href='solutions.csv'>Solution statistics</a> · <a href='front_stability.csv'>Front stability</a></p>")
    if plan["analysis_md"]:
        path = Path(plan["analysis_md"])
        existing = path.read_text(encoding="utf-8-sig") if path.exists() else ""
        marker = f"<!-- scenario-experiment:{batch.name} -->"
        if marker not in existing:
            write_text(path, existing + f"\n\n{marker}\n\n## {batch.name}\n\n{notes}\n\n" + "\n".join(table) + "\n")
    progress(batch, "completed", len(runs), len(plan["tasks"]))


def entry(args, project):
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    batch = Path(args.out).resolve()
    plan = load_plan(batch) if args.resume else prepare(args, Path(project).resolve())
    print(f"Prepared {len(plan['tasks'])} searches and 1,000 independent demands per instance.", flush=True)
    if args.prepare_only:
        return
    if (batch / "PAUSE").exists():
        print("PAUSE exists; no tasks dispatched.", flush=True)
        return
    for stage in (SEARCH, REPLAY):
        freeze_check(plan)
        if not execute_stage(batch, plan, stage):
            return
    report(batch, plan)
