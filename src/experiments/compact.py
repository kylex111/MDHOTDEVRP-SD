"""Store existing search results compactly without additional evaluation."""

from __future__ import annotations
import gzip
from experiments.artifacts import save_json
from experiments.records import recording_spec, write_jsonl_row


def compact_search_record(result):
    omitted = {
        "debug_log",
        "cand_points",
        "checkpoints",
        "search_scenarios",
        "reduced_scenarios",
        "reduced_probs",
        "min_cost_sol",
        "min_ra_sol",
    }
    saved = {k: v for k, v in result.items() if k not in omitted}
    saved["storage_mode"] = "compact_v1"
    saved["archive"] = [
        {
            "solution_id": sol["solution_id"],
            "routes_by_clusters": sol["routes_by_clusters"],
            "cost": sol["cost"],
            "ra": sol["ra"],
            "search_feasible": sol.get("cend", {}).get("feasible"),
        }
        for sol in result["archive"]
    ]
    return saved


def compact_checkpoint(row):
    return {
        **{k: v for k, v in row.items() if k != "archive"},
        "archive_size": len(row["archive"]),
        "objectives": [[s["cost"], s["ra"]] for s in row["archive"]],
    }


def export_compact_search(result, output):
    save_json(
        output / "search_scenarios.json",
        {
            "scenario_seed": result["scenario_seed"],
            "beta": result["beta"],
            "generated": result["search_scenarios"],
            "reduced": result["reduced_scenarios"],
            "probabilities": result["reduced_probs"],
            "selected_indices": result["selected_idx"],
        },
    )
    rows = 0
    name = "search_scenario_records.jsonl.gz"
    with gzip.open(output / name, "xt", encoding="utf-8", compresslevel=5) as stream:
        for sol in result["archive"]:
            observations = sol["cend"].get("scenario_observations")
            if observations is None:
                raise ValueError("Missing final search observations")
            for observation in observations:
                write_jsonl_row(
                    stream,
                    {
                        "role": "search",
                        "run_id": result["run_id"],
                        "solution_id": sol["solution_id"],
                        **{
                            k: v
                            for k, v in observation.items()
                            if k not in ("routes", "evaluation")
                        },
                    },
                )
                rows += 1
    save_json(
        output / "recording_schema.json",
        {**recording_spec(), "storage_mode": "compact_v1"},
    )
    return {
        "storage_mode": "compact_v1",
        "analysis_status": "deferred",
        "search_scenario_records": name,
        "search_rows": rows,
        "search_scenarios": "search_scenarios.json",
        "checkpoints": "checkpoints.jsonl",
        "initialization": "initialization.json",
        "definitions": "recording_schema.json",
    }
