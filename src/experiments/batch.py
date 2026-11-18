"""Plan and schedule batches with dataset selection and run IDs; contains no algorithm logic."""

from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
from configuration.experiment import ExperimentConfig
from .artifacts import save_json, sha256_file
from .run import run_record


def source_fingerprint(project):
    project = Path(project)
    files = [project / "run.py", *sorted((project / "src").rglob("*.py"))]
    digest = hashlib.sha256()
    inventory = {}
    for path in files:
        relative = path.relative_to(project).as_posix()
        value = sha256_file(path)
        inventory[relative] = value
        digest.update(f"{relative}\x00{value}\n".encode())
    return (digest.hexdigest(), inventory)


def list_instances(dataset, pattern="*.txt"):
    path = Path(dataset).resolve()
    files = [path] if path.is_file() else sorted(path.glob(pattern))
    files = [p for p in files if p.is_file()]
    if not files:
        raise FileNotFoundError(f"No instance files: {path} ({pattern})")
    return files


def build_plan(dataset, config_rows, project, pattern="*.txt"):
    code_sha, source = source_fingerprint(project)
    instances = [
        {"path": str(p), "sha256": sha256_file(p)}
        for p in list_instances(dataset, pattern)
    ]
    tasks = []
    configs = []
    for config_id, change, config in config_rows:
        config.validate().require_ra_parameters()
        configs.append(
            {"config_id": config_id, "change": change, "settings": config.as_dict()}
        )
    for instance_index, instance in enumerate(instances):
        for run_id in range(1, max((row[2].n_runs for row in config_rows)) + 1):
            offset = (run_id - 1) % len(config_rows)
            for config_id, _, config in config_rows[offset:] + config_rows[:offset]:
                if run_id > config.n_runs:
                    continue
                key = hashlib.sha256(
                    json.dumps(
                        [
                            code_sha,
                            config.fingerprint,
                            instance["sha256"],
                            run_id,
                            config_id,
                            instance_index,
                        ],
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                tasks.append(
                    {
                        "task_key": key,
                        "directory": f"i{instance_index:04d}_{config_id}_r{run_id:03d}_{key[:10]}",
                        "instance": instance,
                        "config_id": config_id,
                        "run_id": run_id,
                    }
                )
    return {
        "schema_version": "evrouting-batch-v1",
        "source_sha256": source,
        "code_sha256": code_sha,
        "instances": instances,
        "configs": configs,
        "tasks": tasks,
    }


def save_plan(plan, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "batch.json", plan)
    return output


def _execute(task, settings, output):
    config = ExperimentConfig(**settings)
    return run_record(
        task["instance"]["path"], output, config, task["run_id"], compact=False
    )["status"]


def execute_plan(batch, project, workers=1):
    batch = Path(batch).resolve()
    plan = json.loads((batch / "batch.json").read_text(encoding="utf-8"))
    if plan["schema_version"] != "evrouting-batch-v1":
        raise ValueError("Use a new batch for the current source layout")
    if source_fingerprint(project)[0] != plan["code_sha256"]:
        raise ValueError("Source changed since this batch was planned")
    settings = {row["config_id"]: row["settings"] for row in plan["configs"]}
    pending = []
    for task in plan["tasks"]:
        if sha256_file(task["instance"]["path"]) != task["instance"]["sha256"]:
            raise ValueError("Instance changed since this batch was planned")
        output = batch / "runs" / task["directory"]
        if output.exists():
            manifest = output / "manifest.json"
            if (
                manifest.is_file()
                and json.loads(manifest.read_text(encoding="utf-8")).get("status")
                == "completed"
            ):
                continue
            raise FileExistsError(
                f"Incomplete run is retained at {output}; choose a new batch directory"
            )
        pending.append((task, output))
    failures = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_execute, task, settings[task["config_id"]], output): task
            for task, output in pending
        }
        for future in as_completed(futures):
            task = futures[future]
            try:
                future.result()
                print(f"completed: {task['directory']}", flush=True)
            except Exception as exc:
                failures.append({"task": task["task_key"], "error": repr(exc)})
    save_json(
        batch / "progress.json",
        {
            "planned": len(plan["tasks"]),
            "submitted": len(pending),
            "failures": failures,
        },
    )
    if failures:
        raise RuntimeError(f"{len(failures)} tasks failed; see progress.json")
