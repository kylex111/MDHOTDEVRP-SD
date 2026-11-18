"""Export existing scenario records and snapshots; metrics are extracted by the evaluation layer."""

from __future__ import annotations
import json
from experiments.artifacts import json_value, save_json

SCHEMA = "experiment-observations-v1"


def write_jsonl_row(stream, value):
    stream.write(
        json.dumps(json_value(value), ensure_ascii=False, allow_nan=False) + "\n"
    )
    stream.flush()


def recording_spec():
    return {
        "schema_version": SCHEMA,
        "units": {
            "cost": "yuan",
            "time": "minutes for route operations; seconds for computation",
            "energy": "kWh",
            "distance": "km",
            "soc": "remaining battery / capacity",
        },
        "failure_unit": "one independent demand vector for a fixed first-stage plan",
        "energy_failure_event": "at least one route is stopped for an energy failure",
        "charge_unit": "one actually executed positive-energy charge, including before a failure",
        "unknown_values": "null, never imputed as zero",
        "search_observations": "final archive on reduced search scenarios, with nonuniform probabilities",
        "replay_observations": "each final plan on common independent demand vectors",
        "motion_segments": "executed driving only; split at energy-model time-period boundaries; SoC changes linearly within each constant-speed segment; no unfinished arc extrapolation",
        "risk_metric": "energy-failure events recorded; operational risk selection remains pending; motion segments support later low-SOC driving-time calculations",
        "deferred_analysis": [
            "confidence_intervals",
            "run_mean_std_IQR",
            "statistical_tests",
            "multiplicity_correction",
            "effect_sizes",
            "HV",
            "IGD_plus",
            "front_stability",
            "time_quality_curves",
        ],
    }


def export_search_observations(result, output):
    """Export already evaluated archive data; do not add any solver evaluations."""
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
    with (output / "search_scenario_records.jsonl").open(
        "x", encoding="utf-8"
    ) as stream:
        for sol in result["archive"]:
            observations = sol["cend"].get("scenario_observations")
            if observations is None:
                raise ValueError(
                    "Final solution is missing raw search scenario observations"
                )
            for observation in observations:
                write_jsonl_row(
                    stream,
                    {
                        "role": "search",
                        "run_id": result["run_id"],
                        "solution_id": sol["solution_id"],
                        **observation,
                    },
                )
                rows += 1
    save_json(output / "recording_schema.json", recording_spec())
    return {
        "schema_version": SCHEMA,
        "analysis_status": "deferred",
        "search_scenario_records": "search_scenario_records.jsonl",
        "search_rows": rows,
        "search_scenarios": "search_scenarios.json",
        "checkpoints": "checkpoints.jsonl",
        "definitions": "recording_schema.json",
    }
