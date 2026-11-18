"""Run one experiment: configure threads, record provenance, call the solver, and save results."""

from __future__ import annotations
import os
import time
import traceback
from pathlib import Path
from .artifacts import environment_record, save_json, sha256_file
from .records import export_search_observations, write_jsonl_row


def run_record(
    instance, output_dir, config, run_id, *, seed_overrides=None, compact=False
):
    config.validate().require_ra_parameters()
    if compact and config.algorithm == "spea2":
        raise ValueError("Compact storage is not supported for SPEA2")
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(config.threads_per_run)
    from threadpoolctl import threadpool_limits, threadpool_info
    from .solvers import solve

    instance = Path(instance).resolve()
    if not 1 <= run_id <= config.n_runs:
        raise ValueError("run_id outside config")
    instance_hash = sha256_file(instance)
    seeds = config.seeds(instance_hash, run_id)
    if seed_overrides is not None:
        for role, value in seed_overrides.items():
            if (
                role not in seeds
                or isinstance(value, bool)
                or (not isinstance(value, int))
                or (not 0 <= value < 2**32)
            ):
                raise ValueError(
                    "Seed overrides require known roles and integers in [0, 2**32)"
                )
        seeds.update(seed_overrides)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "instance_path": str(instance),
        "instance_sha256": instance_hash,
        "run_id": run_id,
        "config": config.as_dict(),
        "config_sha256": config.fingerprint,
        "seeds": seeds,
        "seed_overrides": seed_overrides,
        "environment": environment_record(Path(__file__).resolve().parents[2]),
        "storage_mode": "compact_v1" if compact else "full",
        "algorithm": config.algorithm,
    }
    save_json(output / "manifest.json", {**metadata, "status": "started"})
    started = time.perf_counter()
    try:

        def checkpoint_observation(row):
            if compact:
                from .compact import compact_checkpoint

                if row["label"] == "initialization":
                    save_json(output / "initialization.json", row)
                persisted = compact_checkpoint(row)
            else:
                persisted = row
            write_jsonl_row(
                checkpoints,
                {"run_id": run_id, "instance_sha256": instance_hash, **persisted},
            )
            save_json(
                output / "live_progress.json",
                {
                    "status": "searching",
                    "run_id": run_id,
                    "instance_sha256": instance_hash,
                    "checkpoint_id": row["checkpoint_id"],
                    "label": row["label"],
                    "elapsed_sec": row["elapsed_sec"],
                    "counts": row["counts"],
                    "archive_size": len(row["archive"]),
                    "updated_unix": time.time(),
                },
            )

        with (
            (output / "checkpoints.jsonl").open("x", encoding="utf-8") as checkpoints,
            threadpool_limits(limits=config.threads_per_run),
        ):
            metadata["active_threadpools"] = threadpool_info()
            result = solve(
                instance, config, run_id, seeds, checkpoint_sink=checkpoint_observation
            )
        for index, sol in enumerate(result["archive"]):
            sol["solution_id"] = f"run{run_id:03d}_s{index:04d}"
        result.update(metadata)
        if config.algorithm != "spea2":
            if compact:
                from .compact import compact_search_record, export_compact_search

                result["observation_files"] = export_compact_search(result, output)
                result = compact_search_record(result)
            else:
                result["observation_files"] = export_search_observations(result, output)
        else:
            result["observation_files"] = {}
        result["status"] = "completed"
        save_json(output / "run.json", result)
        save_json(
            output / "live_progress.json",
            {
                "status": "completed",
                "run_id": run_id,
                "candidate_evaluations": result.get("candidate_evaluations"),
                "scenario_simulations": result.get("scenario_simulations"),
                "archive_size": len(result["archive"]),
                "updated_unix": time.time(),
            },
        )
        save_json(
            output / "manifest.json",
            {
                **metadata,
                "status": "completed",
                "result": "run.json",
                "observation_files": result["observation_files"],
                "wall_time_with_artifact_export_sec": time.perf_counter() - started,
            },
        )
        return result
    except Exception as exc:
        save_json(
            output / "live_progress.json",
            {
                "status": "failed",
                "run_id": run_id,
                "error": repr(exc),
                "updated_unix": time.time(),
            },
        )
        save_json(
            output / "failure.json",
            {
                **metadata,
                "status": "failed",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "measurement": getattr(exc, "measurement", None),
                "checkpoints": "checkpoints.jsonl",
                "wall_time_sec": time.perf_counter() - started,
            },
        )
        save_json(
            output / "manifest.json",
            {**metadata, "status": "failed", "error": repr(exc)},
        )
        raise
