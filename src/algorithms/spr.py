"""Shared SPR search flow with configurable granular balls and VNS."""

from __future__ import annotations
from algorithms.search.catalog import (
    GB_SHAKING,
    GB_VND,
    NODE_SHAKING,
    NODE_VND,
)
import os
import random
import time
import numpy as np
from configuration.parameters import ev_params
from domain.instances import load_problem
from domain.solutions import pack_solution
from evaluation.evaluator import Evaluator
from evaluation.scenarios import (
    generate_normal_scenarios,
    fast_forward_selection,
)
from evaluation.budget import Measurement, BudgetExceeded
from .grouping.clusters import clustering
from .initialization.build import initialize_routes
from .search.neighborhood import Neighborhood
from .search.node_neighborhood import NeighborhoodNoGB
from .search.shaking import Shaker
from .search.vnd import VNDRefiner
from .search.session import SearchSession
from .search.multiobjective import mo_shake, mo_vnd
from .search.pareto import dominates, pareto_insert, truncate_by_crowding


def solve_spr(instance_path, config, run_id, seed, scenario_seed, checkpoint_sink=None):
    """Evaluate the initial plan on demand scenarios, then run archive-based MOVNS.

    config is an immutable ExperimentConfig. seed controls search randomness;
    scenario_seed controls demand generation. Archive, budget, and snapshots are
    local to this call; repeated calls do not share mutable search state.
    Return a unified result dictionary."""
    use_gb = config.algorithm != "without_gb"
    use_movns = config.use_movns and config.algorithm != "without_vns"
    vnd_phases_started = {"ra": 0, "cost": 0}
    vnd_phases_completed = {"ra": 0, "cost": 0}
    eps = 1e-09
    checkpoint_every = 100
    checkpoint_interval_sec = 30.0
    record_observations = True
    if config.max_iter is not None and config.max_iter <= 0:
        raise ValueError("max_iter must be a positive integer or None.")
    if config.max_eval is not None and config.max_eval <= 0:
        raise ValueError("max_eval must be a positive integer or None.")
    if config.max_time_sec is not None and config.max_time_sec <= 0:
        raise ValueError("max_time_sec must be a positive number or None.")
    if all(
        (x is None for x in [config.max_iter, config.max_eval, config.max_time_sec])
    ):
        raise ValueError(
            "At least one stopping criterion must be set among max_iter, max_eval, max_time_sec."
        )
    # 1. Create the per-run budget and random state; initialization shares the search budget.
    measurement = Measurement(
        max_eval=config.max_eval, max_time_sec=config.max_time_sec
    )
    start_time = time.perf_counter()
    scenario_seed = seed if scenario_seed is None else int(scenario_seed)
    random.seed(seed)
    np.random.seed(int(seed) % 2**32)
    # 2. Load global node indices, generate and reduce demand, and retain probability weights.
    problem = load_problem(instance_path, ev_params)
    instance_name = os.path.splitext(os.path.basename(instance_path))[0]
    mu = np.asarray(problem.demand, dtype=float)
    scenarios = generate_normal_scenarios(
        mu=mu,
        beta=config.beta,
        n_scenarios=config.n_scenarios,
        random_state=scenario_seed,
    )
    selected_idx, reduced_scenarios, reduced_probs = fast_forward_selection(
        scenarios=scenarios, n_select=config.n_reduced_scenarios
    )
    rng = np.random.default_rng(seed)
    # 3. Evaluate objectives; the session owns the archive and sends snapshots to the caller.
    evaluator = Evaluator(
        problem,
        ra_safe=config.ra_safe,
        ra_risk=config.ra_risk,
        recourse_policy=config.recourse_policy,
        measurement=measurement,
        capture_scenarios="metrics" if record_observations else False,
        ra_tau_min=config.ra_tau_min,
        ra_formula_version=config.ra_formula_version,
        charging_policy=config.charging_policy,
        soc_threshold=config.soc_threshold,
    )
    session = SearchSession(
        evaluator,
        reduced_scenarios,
        reduced_probs,
        measurement,
        start_time,
        eps,
        config.archive_max_size,
        checkpoint_every,
        checkpoint_interval_sec,
        checkpoint_sink,
    )
    eval_fn = session.evaluate
    NS = GB_SHAKING if use_gb else NODE_SHAKING
    NL = GB_VND if use_gb else NODE_VND
    if use_gb:
        nb = Neighborhood(
            problem=problem,
            removal_ratio=config.removal_ratio,
            alpha_boundary=0.5,
            min_gb_len=2,
            rng=rng,
            evaluator=None,
        )
    else:
        nb = NeighborhoodNoGB(
            problem=problem,
            removal_ratio=config.removal_ratio,
            alpha_boundary=0.5,
            rng=rng,
            evaluator=None,
        )
    shaker = Shaker(nb, NS)
    vnd = VNDRefiner(
        nb,
        evaluator=eval_fn,
        eval_proxy=eval_fn,
        dominates_fn=dominates,
        rng=rng,
        eps=eps,
    )
    # 4. Cluster customers, partition balls, and build orders; the no-ball variant changes construction and neighborhoods.
    clusters = clustering(
        problem, method="kmeans", feature="location", random_state=seed
    )
    measurement.set_phase("initialization")
    routes_clusters = initialize_routes(
        problem,
        clusters,
        seed,
        rng,
        init_mode=config.init_mode,
        use_gb=use_gb,
        gb_pop_size=config.gb_pop_size,
        customer_pop_size=config.customer_pop_size,
        gb_num_gen=config.gb_num_gen,
        customer_num_gen=config.customer_num_gen,
        gb_crossover_rate=config.gb_crossover_rate,
        customer_crossover_rate=config.customer_crossover_rate,
        gb_mutation_rate=config.gb_mutation_rate,
        customer_mutation_rate=config.customer_mutation_rate,
        split_k=config.split_k,
    )
    # 5. Evaluate the initial plan and its directions, then initialize the nondominated archive.
    init_eval, routes_clusters = evaluator.evaluate_init_ffs(
        routes_clusters, reduced_scenarios, reduced_probs
    )
    session.eval_counter["count"] = measurement.counts.get("candidate_evaluations", 0)
    init_sol = pack_solution(routes_clusters, init_eval)
    session.evaluated_frontier = [init_sol]
    session.checkpoint(session.evaluated_frontier, "initialization")
    measurement.set_phase("search")
    session.archive = []
    session.archive, _ = pareto_insert(session.archive, init_sol, eps=eps)
    if config.archive_max_size is not None:
        session.archive = truncate_by_crowding(
            session.archive, max_size=config.archive_max_size
        )
    session.cand_points = [(float(init_sol["cost"]), float(init_sol["ra"]))]
    session.ref_cost = max(float(init_sol["cost"]) * 1.2, float(init_sol["cost"]) + 1.0)
    session.ref_ra = max(float(init_sol["ra"]) * 1.2, float(init_sol["ra"]) + 1e-06)
    session.archive_change_count = 0
    session.last_improve_iter = 0
    k_max = len(NS)
    session.stop_by_iter = False
    session.stop_by_eval = False
    session.stop_by_time = False
    session.stop_reason = "not_stopped"
    session.start_history(init_sol, session.cand_points)
    # 6. Shake, apply the configured VND phases, and update the archive.
    try:
        session.it = 0
        if not use_movns:
            session.stop_reason = "init_only"
        else:
            while True:
                elapsed = time.perf_counter() - start_time
                if config.max_time_sec is not None and elapsed >= config.max_time_sec:
                    session.stop_by_time = True
                    session.stop_reason = "max_time"
                    break
                if (
                    config.max_eval is not None
                    and session.eval_counter["count"] >= config.max_eval
                ):
                    session.stop_by_eval = True
                    session.stop_reason = "max_eval"
                    break
                if config.max_iter is not None and session.it >= config.max_iter:
                    session.stop_by_iter = True
                    session.stop_reason = "max_iter"
                    break
                session.it += 1
                archive_changed = False
                iter_t0 = time.perf_counter()
                eval_before_iter = int(session.eval_counter["count"])
                elapsed_before_iter = float(time.perf_counter() - start_time)
                old_points_iter = {
                    (float(s["cost"]), float(s["ra"])) for s in session.archive
                }
                k = 1
                while k <= k_max:
                    elapsed = time.perf_counter() - start_time
                    if (
                        config.max_time_sec is not None
                        and elapsed >= config.max_time_sec
                    ):
                        session.stop_by_time = True
                        session.stop_reason = "max_time"
                        break
                    if (
                        config.max_eval is not None
                        and session.eval_counter["count"] >= config.max_eval
                    ):
                        session.stop_by_eval = True
                        session.stop_reason = "max_eval"
                        break
                    old_points = {
                        (float(s["cost"]), float(s["ra"])) for s in session.archive
                    }
                    archive_shake = mo_shake(
                        archive=session.archive,
                        shaker=shaker,
                        evaluator=eval_fn,
                        k=k,
                        cid=None,
                    )
                    for sol in archive_shake:
                        session.cand_points.append(
                            (float(sol["cost"]), float(sol["ra"]))
                        )
                    first_obj, second_obj = config.vnd_objectives(session.it)
                    vnd_phases_started[first_obj] += 1
                    archive_vnd_first = mo_vnd(
                        shaken_set=archive_shake,
                        vnd=vnd,
                        obj=first_obj,
                        NL=NL,
                        cid=None,
                        eps=eps,
                        max_size=config.archive_max_size,
                    )
                    vnd_phases_completed[first_obj] += 1
                    elapsed = time.perf_counter() - start_time
                    if (
                        config.max_time_sec is not None
                        and elapsed >= config.max_time_sec
                    ):
                        session.stop_by_time = True
                        session.stop_reason = "max_time"
                        break
                    if (
                        config.max_eval is not None
                        and session.eval_counter["count"] >= config.max_eval
                    ):
                        session.stop_by_eval = True
                        session.stop_reason = "max_eval"
                        break
                    archive_vnd_second = archive_vnd_first
                    if second_obj is not None:
                        vnd_phases_started[second_obj] += 1
                        archive_vnd_second = mo_vnd(
                            shaken_set=archive_vnd_first,
                            vnd=vnd,
                            obj=second_obj,
                            NL=NL,
                            cid=None,
                            eps=eps,
                            max_size=config.archive_max_size,
                        )
                        vnd_phases_completed[second_obj] += 1
                    for sol in archive_vnd_second:
                        session.cand_points.append(
                            (float(sol["cost"]), float(sol["ra"]))
                        )
                        session.ref_cost = max(
                            session.ref_cost, float(sol["cost"]) * 1.05
                        )
                        session.ref_ra = max(session.ref_ra, float(sol["ra"]) * 1.05)
                    archive_new = list(session.archive)
                    for sol in archive_vnd_second:
                        archive_new, _ = pareto_insert(archive_new, sol, eps=eps)
                    if config.archive_max_size is not None:
                        archive_new = truncate_by_crowding(
                            archive_new, max_size=config.archive_max_size
                        )
                    new_points = {
                        (float(s["cost"]), float(s["ra"])) for s in archive_new
                    }
                    if new_points != old_points:
                        session.archive = archive_new
                        archive_changed = True
                        k = 1
                    else:
                        k += 1
                if session.stop_by_time or session.stop_by_eval or session.stop_by_iter:
                    break
                if archive_changed:
                    session.archive_change_count += 1
                    session.last_improve_iter = session.it
                session.record_iteration(
                    session.it,
                    session.cand_points,
                    archive_changed,
                    iter_t0,
                    eval_before_iter,
                    elapsed_before_iter,
                    old_points_iter,
                    (session.ref_cost, session.ref_ra),
                )
    # Stop at a candidate boundary and retain the frontier of completed evaluations.
    except BudgetExceeded as exc:
        session.stop_reason = exc.reason
        session.stop_by_eval = exc.reason == "max_eval"
        session.stop_by_time = exc.reason == "max_time"
    if session.stop_by_eval or session.stop_by_time:
        for candidate in session.evaluated_frontier:
            session.archive, _ = pareto_insert(session.archive, candidate, eps=eps)
        if config.archive_max_size is not None:
            session.archive = truncate_by_crowding(
                session.archive, config.archive_max_size
            )
    session.checkpoint(session.archive, "final_archive")
    measurement.set_phase("result_assembly")
    # 7. Assemble existing records; the session defines result fields and statistics.
    result = session.result(
        config=config,
        instance_name=instance_name,
        run_id=run_id,
        seed=seed,
        scenario_seed=scenario_seed,
        initial_solution=init_sol,
        generated_scenarios=scenarios,
        selected_indices=selected_idx,
        clusters=clusters,
    )
    result["vnd_order"] = config.vnd_order
    result["vnd_phases_started"] = vnd_phases_started
    result["vnd_phases_completed"] = vnd_phases_completed
    return result
