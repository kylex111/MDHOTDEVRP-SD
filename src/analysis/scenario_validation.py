"""Analyze saved common-demand replay values without calling a solver or decoder."""
from __future__ import annotations

import collections
import itertools
import math
from pathlib import Path
import warnings

import numpy as np
from scipy import stats

from configuration.experiment import stable_seed
from experiments.validation_io import read_json, write_json, write_csv, task_directory
from .statistics import QualityContext, nondominated

SCENARIO_COUNTS = (5, 10, 20, 50, 100)
PREFIXES = (100, 250, 500, 1000)


def mean_interval(values):
    x = np.asarray(values, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite observation")
    n = len(x)
    result = {"n": n, "mean": None, "std": None, "ci95": None,
              "ci_method": "Student-t approximate mean interval"}
    if not n:
        return result
    result["mean"] = float(x.mean())
    if n > 1:
        sd = float(x.std(ddof=1))
        half = float(stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n))
        result.update(std=sd, ci95=[result["mean"] - half, result["mean"] + half])
    return result


def wilson_interval(failures, total):
    if total < 0 or not 0 <= failures <= total:
        raise ValueError("Invalid failure counts")
    if not total:
        return {"failures": failures, "n": total, "rate": None, "ci95": None,
                "ci_method": "Wilson binomial interval"}
    z = float(stats.norm.ppf(0.975))
    p = failures / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return {"failures": failures, "n": total, "rate": p,
            "ci95": [max(0.0, center - half), min(1.0, center + half)],
            "ci_method": "Wilson binomial interval"}


def summarize_solution(source, observations, requested_prefixes=PREFIXES):
    c = np.asarray([r["cost_with_failure_penalty"] for r in observations], float)
    a = np.asarray([r["ra_with_failure_penalty"] for r in observations], float)
    completed = np.asarray([r["completed"] for r in observations], bool)
    reasons = collections.Counter()
    for row in observations:
        reasons.update(row["failure_reason_counts"])
    result = {
        "solution_id": source["solution_id"], "routes_by_clusters": source["routes_by_clusters"],
        "search_feasible": bool(source["search_feasible"]),
        "source_cost": float(source["cost"]), "source_ra": float(source["ra"]),
        "scenario_count": len(observations), "oos_feasible": bool(completed.all()),
        "eligible": bool(source["search_feasible"] and completed.all()),
        "cost": mean_interval(c), "ra": mean_interval(a),
        "cost_conditional_on_success": mean_interval(c[completed]),
        "ra_conditional_on_success": mean_interval(a[completed]),
        "failure": wilson_interval(int((~completed).sum()), len(completed)),
        "energy_failure": wilson_interval(sum(bool(r["energy_failure_event"]) for r in observations), len(completed)),
        "failure_reason_route_counts": dict(reasons),
        "cost_gap_penalized": float(c.mean()) - float(source["cost"]),
        "ra_gap_penalized": float(a.mean()) - float(source["ra"]),
        "prefixes": {},
    }
    for n in sorted(set(requested_prefixes) | {len(c)}):
        if 0 < n <= len(c):
            result["prefixes"][str(n)] = {
                "n": n, "cost": float(c[:n].mean()), "ra": float(a[:n].mean()),
                "oos_feasible": bool(completed[:n].all()),
                "failure_rate": float((~completed[:n]).mean()),
            }
    return result


def bootstrap_mean(values, seed):
    x = np.asarray([v for v in values if v is not None], float)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite run statistic")
    result = {"n": len(x), "mean": float(x.mean()) if len(x) else None,
              "std": float(x.std(ddof=1)) if len(x) > 1 else None,
              "ci95": None, "ci_method": "BCa bootstrap, 10000 resamples of independent runs",
              "interval_scope": "conditional on the fixed OOS demand set; pointwise interval",
              "ci_status": "insufficient_n"}
    if len(x) < 2:
        return result
    if np.all(x == x[0]):
        result["ci_status"] = "degenerate_constant_sample"
        return result
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", stats.DegenerateDataWarning)
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = stats.bootstrap((x,), np.mean, n_resamples=10000, method="BCa",
                confidence_level=0.95, rng=np.random.default_rng(seed), batch=512)
    ci = [float(boot.confidence_interval.low), float(boot.confidence_interval.high)]
    result["ci_status"] = "ok" if np.isfinite(ci).all() else "degenerate_bootstrap"
    if result["ci_status"] == "ok":
        result["ci95"] = ci
    return result


def front_ids(points, ids, eps=1e-9):
    """Retain nondominated solution IDs, including distinct IDs with identical objectives."""
    x = np.asarray(points, dtype=float).reshape(-1, 2)
    return {sid for p, sid in zip(x, ids)
            if not np.any(np.all(x <= p + eps, axis=1) & np.any(x < p - eps, axis=1))}


def pooled_front(points, eps=1e-9):
    """Build a shared two-objective reference set with vectorized dominance comparisons."""
    x = np.unique(np.asarray(points, float).reshape(-1, 2), axis=0)
    mask = [not np.any(np.all(x <= p + eps, axis=1) & np.any(x < p - eps, axis=1)) for p in x]
    return x[mask].reshape(-1, 2)


def build_contexts(runs):
    pools = collections.defaultdict(lambda: {"bounds": [], "oos": []})
    for run in runs:
        pool = pools[run["instance"]]
        for sol in run["solutions"]:
            if sol["search_feasible"]:
                pool["bounds"].append([sol["source_cost"], sol["source_ra"]])
            if sol["eligible"]:
                point = [sol["cost"]["mean"], sol["ra"]["mean"]]
                pool["bounds"].append(point)
                pool["oos"].append(point)
            # Use the same scale for prefix diagnostics; do not renormalize each subsample.
            if sol["search_feasible"]:
                pool["bounds"].extend([v["cost"], v["ra"]]
                    for v in sol["prefixes"].values() if v["oos_feasible"])
    contexts = {}
    for instance, pool in pools.items():
        if not pool["bounds"]:
            contexts[instance] = None
            continue
        bounds = np.asarray(pool["bounds"], float)
        lower, upper = bounds.min(axis=0), bounds.max(axis=0)
        upper = np.where(upper > lower, upper, lower + 1.0)
        contexts[instance] = QualityContext(lower.tolist(), upper.tolist(),
            pooled_front(pool["oos"]).tolist(), source="pooled_OOS_strict_feasible_report_scope")
    return contexts


def quality(context, points):
    if not len(points):
        return {"HV": 0.0, "IGD_plus": None, "PFS": 0, "status": "empty_feasible_front"}
    if context is None:
        raise ValueError("Nonempty front without normalization")
    if context.reference_front:
        return context.evaluate(points)
    temporary = QualityContext(context.lower, context.upper, nondominated(points).tolist())
    value = temporary.evaluate(points)
    value.update(IGD_plus=None, status="no_common_OOS_reference")
    return value


def run_metrics(run, context):
    source = [s for s in run["solutions"] if s["search_feasible"]]
    after = [s for s in source if s["oos_feasible"]]
    before_points = [[s["source_cost"], s["source_ra"]] for s in source]
    points = [[s["cost"]["mean"], s["ra"]["mean"]] for s in after]
    before_ids = front_ids(before_points, [s["solution_id"] for s in source])
    after_ids = front_ids(points, [s["solution_id"] for s in after])
    q, before_q = quality(context, points), quality(context, before_points)
    front = nondominated(points)
    row = {k: run[k] for k in ("budget_group", "instance", "scenario_count", "run_id")}
    row.update(**q, source_HV=before_q["HV"], source_PFS=before_q["PFS"],
        source_empty=int(not before_ids), oos_empty=int(not after_ids),
        front_retention=len(before_ids & after_ids) / len(before_ids) if before_ids else None,
        HV_change=q["HV"] - before_q["HV"], PFS_change=q["PFS"] - before_q["PFS"],
        source_feasible_solutions=len(source), archive_size=len(run["solutions"]),
        solution_failure_rate=float(np.mean([s["failure"]["rate"] for s in source])) if source else None,
        best_cost=float(front[:, 0].min()) if len(front) else None,
        best_ra=float(front[:, 1].min()) if len(front) else None,
        mean_cost=float(front[:, 0].mean()) if len(front) else None,
        mean_ra=float(front[:, 1].mean()) if len(front) else None,
        candidate_evaluations=run["search_measurement"]["candidate_evaluations"],
        scenario_simulations=run["search_measurement"]["scenario_simulations"],
        search_runtime_sec=run["search_measurement"]["runtime_sec"],
        replay_runtime_sec=run["replay_runtime_sec"],
        replay_scenario_simulations=run["measurement"]["counts"].get("scenario_simulations", 0))
    prefixes = []
    for prefix in sorted(run["solutions"][0]["prefixes"], key=int) if run["solutions"] else []:
        chosen = [s for s in source if s["prefixes"][prefix]["oos_feasible"]]
        pts = [[s["prefixes"][prefix]["cost"], s["prefixes"][prefix]["ra"]] for s in chosen]
        ids = front_ids(pts, [s["solution_id"] for s in chosen])
        union = ids | after_ids
        prefixes.append({**{k: row[k] for k in ("budget_group", "instance", "scenario_count", "run_id")},
            "prefix_n": int(prefix), **quality(context, pts),
            "jaccard_to_full": len(ids & after_ids) / len(union) if union else None,
            "jaccard_status": "ok" if union else "both_empty"})
    return row, prefixes


GROUP_METRICS = ("HV", "IGD_plus", "PFS", "best_cost", "best_ra", "mean_cost", "mean_ra",
    "source_HV", "source_PFS", "source_empty", "oos_empty", "front_retention", "HV_change",
    "PFS_change", "solution_failure_rate", "candidate_evaluations", "scenario_simulations",
    "search_runtime_sec", "replay_runtime_sec")


def aggregate_runs(rows):
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[(row["budget_group"], row["instance"], row["scenario_count"])].append(row)
    result = []
    for (budget, instance, count), items in sorted(grouped.items()):
        if len(items) != 30 or len({x["run_id"] for x in items}) != 30:
            raise ValueError("Group does not contain 30 independent run IDs")
        result.append({"budget_group": budget, "instance": instance, "scenario_count": count,
            "n_runs": len(items), "metrics": {m: bootstrap_mean([r[m] for r in items],
                stable_seed(20261017, instance, count, f"ci:{budget}:{m}")) for m in GROUP_METRICS},
            "source_empty_rate": wilson_interval(sum(r["source_empty"] for r in items), len(items)),
            "oos_empty_rate": wilson_interval(sum(r["oos_empty"] for r in items), len(items))})
    return result


def paired_differences(rows):
    groups = collections.defaultdict(dict)
    for row in rows:
        groups[(row["budget_group"], row["instance"], row["scenario_count"])][row["run_id"]] = row
    result = []
    for budget, instance in sorted({(k[0], k[1]) for k in groups}):
        for left, right in itertools.combinations(SCENARIO_COUNTS, 2):
            a, b = groups[(budget, instance, left)], groups[(budget, instance, right)]
            for metric in ("HV", "IGD_plus", "PFS", "best_cost", "best_ra", "solution_failure_rate"):
                ids = sorted(i for i in a.keys() & b.keys() if a[i][metric] is not None and b[i][metric] is not None)
                result.append({"budget_group": budget, "instance": instance, "left_S": left, "right_S": right,
                    "metric": metric, "paired_run_ids": ids, "difference": "left minus right",
                    **bootstrap_mean([a[i][metric] - b[i][metric] for i in ids],
                        stable_seed(20261017, instance, left * 1000 + right, f"pair:{budget}:{metric}"))})
    for instance in sorted({r["instance"] for r in rows}):
        for count in SCENARIO_COUNTS:
            a = groups.get(("time1200", instance, count))
            b = groups.get(("eval2000", instance, count))
            if a is None or b is None:
                continue
            for metric in ("HV", "IGD_plus", "PFS", "best_cost", "best_ra", "solution_failure_rate"):
                ids = sorted(i for i in a.keys() & b.keys() if a[i][metric] is not None and b[i][metric] is not None)
                result.append({"budget_group": "time1200_minus_eval2000", "instance": instance,
                    "left_S": count, "right_S": count, "metric": metric, "paired_run_ids": ids,
                    "difference": "time1200 minus eval2000",
                    **bootstrap_mean([a[i][metric]-b[i][metric] for i in ids],
                        stable_seed(20261017, instance, count, f"pair:budgets:{metric}"))})
    return result


def make_report(batch, plan, final=False):
    batch = Path(batch)
    runs = []
    for stage in (("A_replay_time", "C_replay_eval2000") if final else ("A_replay_time",)):
        for task in plan["tasks"]:
            output = task_directory(batch, task, stage)
            record = read_json(output / "run.json")
            if record["status"] != "completed":
                raise ValueError("Cannot report incomplete replay")
            runs.append(record)
    contexts = build_contexts(runs)
    rows, prefixes, solutions = [], [], []
    for run in runs:
        row, prefix = run_metrics(run, contexts[run["instance"]])
        rows.append(row)
        prefixes.extend(prefix)
        for sol in run["solutions"]:
            identity = {k: run[k] for k in ("budget_group", "instance", "scenario_count", "run_id")}
            solutions.append({**identity, "solution_id": sol["solution_id"],
                "search_feasible": sol["search_feasible"], "oos_feasible": sol["oos_feasible"],
                "eligible": sol["eligible"], "source_cost": sol["source_cost"], "source_ra": sol["source_ra"],
                "cost_mean_penalized": sol["cost"]["mean"], "cost_ci95_penalized": sol["cost"]["ci95"],
                "cost_std_penalized": sol["cost"]["std"], "ra_std_penalized": sol["ra"]["std"],
                "ra_mean_penalized": sol["ra"]["mean"], "ra_ci95_penalized": sol["ra"]["ci95"],
                "cost_mean_success": sol["cost_conditional_on_success"]["mean"],
                "cost_ci95_success": sol["cost_conditional_on_success"]["ci95"],
                "ra_mean_success": sol["ra_conditional_on_success"]["mean"],
                "cost_std_success": sol["cost_conditional_on_success"]["std"],
                "ra_std_success": sol["ra_conditional_on_success"]["std"],
                "ra_ci95_success": sol["ra_conditional_on_success"]["ci95"],
                "n_success": sol["cost_conditional_on_success"]["n"], "n_scenarios": sol["scenario_count"],
                "failure_rate": sol["failure"]["rate"], "failure_ci95": sol["failure"]["ci95"],
                "energy_failure_rate": sol["energy_failure"]["rate"],
                "failure_reason_route_counts": sol["failure_reason_route_counts"],
                "cost_gap_penalized": sol["cost_gap_penalized"], "ra_gap_penalized": sol["ra_gap_penalized"]})
    groups = aggregate_runs(rows)
    paired = paired_differences(rows)
    report = batch / "reports" / ("final" if final else "phase_A")
    report.mkdir(parents=True, exist_ok=True)
    context_data = {k: None if v is None else {
        "lower": v.lower, "upper": v.upper, "reference_front": v.reference_front,
        "hv_reference": v.hv_reference, "source": v.source,
        "scope": "both_budget_groups" if final else "time_group_only_provisional",
        "bounds_include": "feasible search, full OOS, and prefix diagnostic objectives",
        "reference_status": "ok" if v.reference_front else "no_common_OOS_reference"}
        for k, v in contexts.items()}
    write_json(report / "quality_contexts.json", context_data)
    write_json(report / "report_data.json", {"final": final, "groups": groups, "runs": rows,
        "prefixes": prefixes, "paired_differences": paired, "n_scenarios": plan["oos_scenarios"]})
    write_csv(report / "runs.csv", rows)
    write_csv(report / "solutions.csv", solutions)
    write_csv(report / "front_stability.csv", prefixes)
    write_csv(report / "paired_differences.csv", paired)
    flat_groups = [{"budget_group": g["budget_group"], "instance": g["instance"],
        "scenario_count": g["scenario_count"], "metric": metric, **value}
        for g in groups for metric, value in g["metrics"].items()]
    write_csv(report / "groups.csv", flat_groups)
    write_csv(report / "empty_front_rates.csv", [{"budget_group": g["budget_group"], "instance": g["instance"],
        "scenario_count": g["scenario_count"], "measure": name, **g[name]}
        for g in groups for name in ("source_empty_rate", "oos_empty_rate")])
    from .scenario_validation_report import render_report
    render_report(report, plan, runs, groups, rows, prefixes, final)
    write_json(report / "completion.json", {"status": "completed", "final": final,
        "run_records": len(runs), "solution_records": len(solutions), "group_records": len(groups),
        "reference_scope": "both_budget_groups" if final else "time_group_only_provisional",
        "extra_searches": 0, "extra_scenario_simulations": 0})
    return report
