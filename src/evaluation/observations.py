"""Extract operating metrics from a completed scenario simulation without saving files."""

from __future__ import annotations
from collections import Counter

SCHEMA = "experiment-observations-v1"


def scenario_metrics(cend):
    """Extract measured values; unavailable legacy diagnostics remain null."""
    diagnostics = list(cend.get("route_diagnostics", {}).values())
    known = bool(diagnostics) and all(
        (d.get("energy_failure") is not None for d in diagnostics)
    )

    def total(field):
        values = [d.get(field) for d in diagnostics]
        return sum(values) if values and all((v is not None for v in values)) else None

    soc = [d.get("min_soc_fraction") for d in diagnostics]
    reasons = Counter(
        (d["failure_reason"] for d in diagnostics if d.get("failure_reason"))
    )
    return {
        "cost_with_failure_penalty": float(cend["total_cost"]),
        "ra_with_failure_penalty": float(cend["total_ra"]),
        "completed": bool(cend["feasible"]),
        "failed_routes": cend["failed_routes"],
        "route_denominator": cend["route_count"],
        "energy_failure_event": any((d["energy_failure"] for d in diagnostics))
        if known
        else None,
        "energy_failed_routes": sum((bool(d["energy_failure"]) for d in diagnostics))
        if known
        else None,
        "failure_reason_counts": dict(reasons),
        "min_soc_fraction": min(soc)
        if soc and all((v is not None for v in soc))
        else None,
        "ief_events": total("ief_events"),
        "sef_events": total("sef_events"),
        "load_failure_events": total("load_failure_events"),
        "num_charge": float(cend["stats"]["num_charge_total"]) if known else None,
        "charging_time_min": float(cend["stats"]["charging_time_total"])
        if known
        else None,
        "charged_energy_kwh": float(cend["stats"]["charged_energy_total"])
        if known
        else None,
        "num_restock": total("num_restock"),
        "actual_travel_time_min": total("actual_travel_time_min"),
        "actual_service_time_min": total("actual_service_time_min"),
        "actual_distance_km": total("actual_distance_km"),
        "actual_executed_ra": total("actual_executed_ra"),
        "low_soc_driving_time_min": total("low_soc_driving_time_min"),
        "charge_breakdown": dict(cend.get("stats", {})) if known else None,
        "cost_components_with_failure_penalty": {
            field: sum(cend.get(field, {}).values())
            for field in (
                "dispatch_cost_list",
                "travel_cost_list",
                "service_cost_list",
                "charging_cost_list",
            )
        },
    }


def scenario_observation(cend, scenario_id, probability):
    routes = []
    for cid, diag in cend.get("route_diagnostics", {}).items():
        routes.append(
            {
                "route_id": str(cid),
                **{
                    k: v
                    for k, v in diag.items()
                    if k not in ("trace", "motion_segments")
                },
                "decoded_route": cend["decoded_routes"].get(cid),
                "cost_with_failure_penalty": cend["routes_cost"].get(cid),
                "ra_with_failure_penalty": cend["routes_ra"].get(cid),
                "charging": cend.get("stats_list", {}).get(cid)
                if diag.get("energy_failure") is not None
                else None,
            }
        )
    return {
        "schema_version": SCHEMA,
        "scenario_id": int(scenario_id),
        "probability": float(probability),
        **scenario_metrics(cend),
        "routes": routes,
    }
