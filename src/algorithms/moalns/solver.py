"""Adapt Li et al. (2025), DOI 10.3390/electronics14010103, to shared evaluation."""
from datetime import datetime
from pathlib import Path
import copy
import random
import time
import numpy as np
from configuration.experiment import stable_seed
from configuration.moalns import MOALNSParameters
from configuration.parameters import ev_params
from domain.instances import load_problem
from domain.solutions import pack_solution
from evaluation.budget import Measurement, BudgetExceeded
from evaluation.evaluator import Evaluator
from evaluation.scenarios import generate_normal_scenarios, fast_forward_selection
from algorithms.grouping.clusters import clustering
from algorithms.initialization.build import initialize_routes
from algorithms.search.session import SearchSession
from algorithms.search.pareto import pareto_insert, truncate_by_crowding
from .core import AdaptiveSelector, accept, reward, cool, route_key
from .operators import RoutingOperators, validate_customers


def budget_reason(measurement, max_iter):
    if measurement.max_eval is not None and measurement.counts.get("candidate_evaluations", 0) >= measurement.max_eval:
        return "max_eval"
    if measurement.max_time_sec is not None and time.perf_counter() - measurement.started >= measurement.max_time_sec:
        return "max_time"
    if max_iter is not None and measurement.counts.get("search_iterations", 0) >= max_iter:
        return "max_iter"
    return None


def solve_moalns(instance_path, config, run_id, seed, scenario_seed, checkpoint_sink=None):
    config.validate().require_ra_parameters()
    settings = MOALNSParameters.from_dict(config.moalns)
    measurement = Measurement(max_eval=config.max_eval, max_time_sec=config.max_time_sec)
    started = measurement.started
    random.seed(seed)
    np.random.seed(int(seed) % 2**32)
    problem = load_problem(str(instance_path), ev_params)
    scenarios = generate_normal_scenarios(mu=np.asarray(problem.demand, float), beta=config.beta,
        n_scenarios=config.n_scenarios, random_state=scenario_seed)
    selected_idx, reduced, probabilities = fast_forward_selection(scenarios, config.n_reduced_scenarios)
    evaluator = Evaluator(problem, ra_safe=config.ra_safe, ra_risk=config.ra_risk,
        ra_tau_min=config.ra_tau_min, ra_formula_version=config.ra_formula_version,
        recourse_policy=config.recourse_policy, measurement=measurement, capture_scenarios="metrics")
    session = SearchSession(evaluator, reduced, probabilities, measurement, started, 1e-9,
                            config.archive_max_size, 0, None, checkpoint_sink)

    def guard():
        reason = budget_reason(measurement, config.max_iter)
        if reason:
            raise BudgetExceeded(reason, measurement.snapshot())

    def evaluate(routes):
        guard()
        result = session.evaluate(routes)
        return pack_solution(copy.deepcopy(routes), result)

    clusters = clustering(problem, method="kmeans", feature="location", random_state=seed)
    measurement.set_phase("initialization")
    current = None
    initial_routes = None
    init_cost = init_ra = None
    reason = None
    thresholds = np.zeros(2)
    search_seed = stable_seed(seed, "moalns_li2025", 0, "search")
    rng = np.random.default_rng(search_seed)
    destroy = AdaptiveSelector(3, settings.reaction)
    repair = AdaptiveSelector(2, settings.reaction)
    destroy_names = ("random", "worst_distance", "balance")
    repair_names = ("greedy", "balance")
    accepted_keys = set()
    outcomes = {"accepted": 0, "rejected": 0, "reward_25": 0, "reward_5": 0, "reward_1": 0, "reward_zero": 0}
    segments = []
    try:
        guard()
        built = initialize_routes(problem, clusters, seed, np.random.default_rng(seed),
                                  **config.initialization_kwargs())
        initial_routes = {cid: Evaluator._flatten_routes(parts) for cid, parts in built.items()}
        validate_customers(initial_routes, problem.customers)
        current = evaluate(initial_routes)
        init_cost, init_ra = current["cost"], current["ra"]
        thresholds = np.asarray([init_cost, init_ra], float)
        if not np.isfinite(thresholds).all() or (thresholds < 0).any():
            raise ValueError("Threshold initialization requires finite nonnegative objectives")
        session.archive = [current]
        accepted_keys.add(route_key(initial_routes))
    except BudgetExceeded as exc:
        reason = exc.reason
    session.checkpoint(session.evaluated_frontier, "initialization")
    measurement.set_phase("search")
    operators = RoutingOperators(problem, config.removal_ratio, guard)
    if current is not None:
        try:
            while True:
                guard()
                d, r = destroy.choose(rng), repair.choose(rng)
                partial, removed = operators.destroy(current["routes_by_clusters"], destroy_names[d], rng)
                routes = operators.repair(partial, removed, repair_names[r], rng)
                candidate = evaluate(routes)
                accepted = accept(current, candidate, thresholds)
                key = route_key(routes)
                score = reward(candidate, session.archive, accepted, key in accepted_keys, settings.scores)
                outcomes["accepted" if accepted else "rejected"] += 1
                score_name = f"reward_{int(score)}" if score in (25, 5, 1) else "reward_zero"
                outcomes[score_name] += 1
                destroy.record(d, score)
                repair.record(r, score)
                if accepted:
                    current = candidate
                    accepted_keys.add(key)
                    session.archive, _ = pareto_insert(session.archive, candidate, eps=1e-9)
                    session.archive = truncate_by_crowding(session.archive, config.archive_max_size)
                measurement.increment("search_iterations")
                thresholds = cool(thresholds, settings.threshold_steps)
                iteration = measurement.counts["search_iterations"]
                if iteration % settings.segment_length == 0:
                    destroy.update()
                    repair.update()
                    segments.append({"iteration": iteration, "thresholds": thresholds.tolist(),
                                     "destroy": destroy.snapshot(), "repair": repair.snapshot()})
                if (measurement.counts["candidate_evaluations"] % 100 == 0
                        or time.perf_counter() - started - session.last_checkpoint_sec >= 30):
                    session.checkpoint(session.evaluated_frontier, "evaluated_frontier")
        except BudgetExceeded as exc:
            reason = exc.reason
    reason = reason or budget_reason(measurement, config.max_iter)
    session.checkpoint(session.evaluated_frontier, "final")
    measurement.set_phase("result_assembly")
    snapshot = measurement.snapshot()
    counts = snapshot["counts"]
    archive = session.evaluated_frontier
    best_cost = min(archive, key=lambda s: s["cost"]) if archive else None
    best_ra = min(archive, key=lambda s: s["ra"]) if archive else None
    return {
        **config.as_dict(), "timestamp": datetime.now().isoformat(timespec="seconds"),
        "instance": Path(instance_path).stem, "instance_path": str(Path(instance_path).resolve()),
        "run_id": int(run_id), "seed": int(seed), "scenario_seed": int(scenario_seed),
        "init_cost": init_cost, "init_ra": init_ra, "initial_routes": initial_routes,
        "runtime_sec": snapshot["elapsed_sec"], "measurement": snapshot,
        "time_overrun_sec": max(0.0, snapshot["elapsed_sec"] - config.max_time_sec) if config.max_time_sec else 0.0,
        "candidate_evaluations": counts.get("candidate_evaluations", 0),
        "eval_count": counts.get("candidate_evaluations", 0),
        "scenario_simulations": counts.get("scenario_simulations", 0),
        "route_simulations": counts.get("route_simulations", 0),
        "n_iterations": counts.get("search_iterations", 0), "stop_reason": reason,
        "stop_by_time": reason == "max_time", "stop_by_eval": reason == "max_eval", "stop_by_iter": reason == "max_iter",
        "archive": archive, "archive_size": len(archive),
        "pareto_points": [(s["cost"], s["ra"]) for s in archive],
        "best_cost_in_archive": best_cost["cost"] if best_cost else None,
        "ra_of_best_cost": best_cost["ra"] if best_cost else None,
        "best_ra_in_archive": best_ra["ra"] if best_ra else None,
        "cost_of_best_ra": best_ra["cost"] if best_ra else None,
        "search_scenarios": scenarios, "reduced_scenarios": reduced,
        "reduced_probs": probabilities, "selected_idx": selected_idx,
        "clusters": clusters, "checkpoints": session.checkpoints,
        "analysis_status": "deferred", "hv_scope": "deferred_postprocessing",
        "recording": {"candidate_is_atomic": True, "search_scenarios": True},
        "configuration": {"ev": dict(ev_params), "cost": vars(evaluator.cost_params),
                          "energy": vars(evaluator.energy_model), "ra": vars(evaluator.ra_model)},
        "moalns_details": {
            "implementation": "li2025_pareto_adaptation_v1", "doi": "10.3390/electronics14010103",
            "provenance": "paper-based reimplementation, not author source code; no alns package",
            "parameters": settings.as_dict(), "parameters_tuned": False,
            "search_seed": search_seed, "thresholds_final": thresholds.tolist(),
            "threshold_policy": "raw objectives; T0=f(initial); T=max(0,T-0.02) per completed iteration",
            "paper_interpretations": ["Equation 13 linear cooling overrides Algorithm 3 multiplication",
                                     "Remove dominated, not nondominated, archive entries"],
            "adaptation_defaults": ["reaction=0.2", "removal_ratio=0.3", "end_thresholds=0",
                                    "equal raw threshold steps for cost and RA", "archive_limit=200",
                                    "3 removal and 2 insertion operators; fixed route slots"],
            "destroy_names": destroy_names, "repair_names": repair_names,
            "destroy": destroy.snapshot(), "repair": repair.snapshot(),
            "outcomes": outcomes, "segments": segments,
            "accepted_unique_routes": len(accepted_keys),
            "search_archive": [{k: s[k] for k in ("cost", "ra", "routes_by_clusters")} for s in session.archive],
            "report_archive_policy": "all completed evaluations; does not influence search",
            "repair_proxy": "nearest-depot geometric distance; no realized demand",
            "initialization": "common clustering; GBGA construction or random customer order within clusters",
        },
    }
