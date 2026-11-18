"""Run pymoo baselines with the same budget and observation contract as SPR."""

from __future__ import annotations
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np
from pymoo.core.sampling import Sampling
from pymoo.core.callback import Callback
from pymoo.core.termination import Termination
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.moead import MOEAD
from .pymoo_encoding import (
    generate_init_population, build_flat_encoding_state,
    GBClusterRoutingProblemFlat, GBHeuristicSamplingFlat,
)
from .grouping.clusters import clustering
from .initialization.build import initialize_routes
from .search.session import SearchSession
from configuration.parameters import ev_params
from domain.instances import load_problem
from evaluation.evaluator import Evaluator
from evaluation.budget import Measurement, BudgetExceeded
from evaluation.scenarios import generate_normal_scenarios, fast_forward_selection


class SharedTermination(Termination):
    """Check all enabled limits; the evaluator enforces exact candidate limits."""

    def __init__(self, measurement, max_iter):
        super().__init__()
        self.measurement = measurement
        self.max_iter = max_iter
        self.reason = None

    def _update(self, algorithm):
        m = self.measurement
        if m.max_eval is not None and m.counts.get("candidate_evaluations", 0) >= m.max_eval:
            self.reason = "max_eval"
        elif m.max_time_sec is not None and time.perf_counter() - m.started >= m.max_time_sec:
            self.reason = "max_time"
        elif self.max_iter is not None and algorithm.n_gen >= self.max_iter:
            self.reason = "max_iter"
        return float(self.reason is not None)


class ProgressCallback(Callback):
    def __init__(self, session):
        super().__init__()
        self.session = session
        self.history = []
        self.initial_population_size = None

    def notify(self, algorithm):
        session = self.session
        if self.initial_population_size is None:
            self.initial_population_size = len(algorithm.pop)
            session.checkpoint(session.evaluated_frontier, "initialization")
            session.measurement.set_phase("search")
        self.history.append({
            "gen": int(algorithm.n_gen),
            "candidate_evaluations": session.measurement.counts.get("candidate_evaluations", 0),
            "framework_evaluations": int(algorithm.evaluator.n_eval),
            "elapsed_sec": time.perf_counter() - session.started,
            "population_size": len(algorithm.pop),
            "archive_size": len(session.evaluated_frontier),
        })


def build_algorithm_nsga2(pop_size: int, sampling: Sampling):
    from pymoo.operators.crossover.ox import OrderCrossover
    from pymoo.operators.mutation.inversion import InversionMutation

    return NSGA2(
        pop_size=pop_size,
        sampling=sampling,
        crossover=OrderCrossover(),
        mutation=InversionMutation(),
        eliminate_duplicates=True,
    )


def build_algorithm_moead(pop_size: int, sampling: Sampling):
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.operators.crossover.ox import OrderCrossover
    from pymoo.operators.mutation.inversion import InversionMutation

    ref_dirs = get_reference_directions(
        "uniform", 2, n_partitions=pop_size - 1
    )
    return MOEAD(
        ref_dirs=ref_dirs,
        sampling=sampling,
        crossover=OrderCrossover(),
        mutation=InversionMutation(),
        n_neighbors=min(15, len(ref_dirs)),
        prob_neighbor_mating=0.9,
    )


def build_algorithm_nsga3(pop_size: int, sampling: Sampling):
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.util.ref_dirs import get_reference_directions

    ref_dirs = get_reference_directions(
        "uniform", 2, n_partitions=pop_size - 1
    )
    return NSGA3(ref_dirs=ref_dirs, pop_size=len(ref_dirs), sampling=sampling)


def build_algorithm_mopso(
    pop_size: int, sampling: Sampling, seed: Optional[int] = None
):
    from pymoo.algorithms.moo.cmopso import CMOPSO

    return CMOPSO(
        pop_size=pop_size,
        max_velocity_rate=0.2,
        elite_size=max(10, pop_size // 5),
        mutation_rate=0.5,
        sampling=sampling,
        seed=seed,
    )


def build_algorithm(
    name: str, pop_size: int, sampling: Sampling, seed: Optional[int] = None
):
    if pop_size < 2:
        raise ValueError("Population size must be at least 2")
    name = name.lower()
    if name == "nsga2":
        return build_algorithm_nsga2(pop_size=pop_size, sampling=sampling)
    if name == "nsga3":
        return build_algorithm_nsga3(pop_size=pop_size, sampling=sampling)
    if name == "moead":
        return build_algorithm_moead(pop_size=pop_size, sampling=sampling)
    if name in {"mopso", "cmopso"}:
        return build_algorithm_mopso(pop_size=pop_size, sampling=sampling, seed=seed)
    raise ValueError(f"Unknown algorithm: {name}")


def solve_pymoo(instance_path, config, run_id, seed, scenario_seed, checkpoint_sink=None):
    """Include preprocessing and initialization in time and evaluation budgets.

    Store completed evaluations in a reporting archive. This archive does not
    participate in pymoo selection, mating, or particle updates.
    """
    config.validate().require_ra_parameters()
    measurement = Measurement(max_eval=config.max_eval, max_time_sec=config.max_time_sec)
    started = measurement.started
    random.seed(seed)
    np.random.seed(int(seed) % 2**32)
    scenario_seed = seed if scenario_seed is None else int(scenario_seed)
    problem = load_problem(str(instance_path), ev_params)
    scenarios = generate_normal_scenarios(
        mu=np.asarray(problem.demand, dtype=float), beta=config.beta,
        n_scenarios=config.n_scenarios, random_state=scenario_seed,
    )
    selected_idx, reduced, probabilities = fast_forward_selection(
        scenarios, config.n_reduced_scenarios
    )
    evaluator = Evaluator(
        problem, ra_safe=config.ra_safe, ra_risk=config.ra_risk,
        ra_tau_min=config.ra_tau_min, ra_formula_version=config.ra_formula_version,
        recourse_policy=config.recourse_policy, measurement=measurement,
        capture_scenarios="metrics",
    )
    session = SearchSession(
        evaluator, reduced, probabilities, measurement, started, 1e-9,
        config.archive_max_size, 0, None, checkpoint_sink,
    )

    def evaluate(routes):
        values = session.evaluate(routes)
        session.archive = session.evaluated_frontier
        if (measurement.counts["candidate_evaluations"] % 100 == 0
                or time.perf_counter() - started - session.last_checkpoint_sec >= 30.0):
            session.checkpoint(session.archive, "evaluated_frontier")
        return values

    def objective(routes):
        values = evaluate(routes)
        return float(values["total_cost"]), float(values["total_ra"])

    clusters = clustering(problem, method="kmeans", feature="location", random_state=seed)
    measurement.set_phase("initialization")
    callback = ProgressCallback(session)
    termination = SharedTermination(measurement, config.max_iter)
    algorithm = None
    initial_values = None
    stop_reason = None
    try:
        initial = initialize_routes(
            problem, clusters, seed, np.random.default_rng(seed),
            **config.initialization_kwargs(),
        )
        initial_values = evaluate(initial)
        _, population = generate_init_population(initial, pop_size=config.population_size, seed=seed)
        state = build_flat_encoding_state(population[0])
        mo_problem = GBClusterRoutingProblemFlat(state=state, evaluator_fn=objective)
        sampling = GBHeuristicSamplingFlat(
            state=state, seed_solutions=population, random_fill=True, seed=seed
        )
        algorithm = build_algorithm(config.algorithm, config.population_size, sampling, seed=seed)
        algorithm.setup(mo_problem, termination=termination, seed=seed,
                        verbose=False, callback=callback, save_history=False)
        while algorithm.has_next():
            algorithm.next()
        stop_reason = termination.reason or "algorithm_termination"
    except BudgetExceeded as exc:
        # A population batch can stop partway through; completed candidates remain recorded.
        stop_reason = exc.reason

    if callback.initial_population_size is None:
        session.checkpoint(session.evaluated_frontier, "initialization")
    session.archive = session.evaluated_frontier
    session.checkpoint(session.archive, "final")
    measurement.set_phase("result_assembly")
    archive = session.archive
    best_cost = min(archive, key=lambda s: s["cost"]) if archive else None
    best_ra = min(archive, key=lambda s: s["ra"]) if archive else None
    snapshot = measurement.snapshot()
    counts = snapshot["counts"]
    return {
        **config.as_dict(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "instance": Path(instance_path).stem,
        "instance_path": str(Path(instance_path).resolve()),
        "run_id": int(run_id), "seed": int(seed), "scenario_seed": scenario_seed,
        "init_cost": initial_values["total_cost"] if initial_values else None,
        "init_ra": initial_values["total_ra"] if initial_values else None,
        "runtime_sec": snapshot["elapsed_sec"],
        "measurement": snapshot,
        "candidate_evaluations": counts.get("candidate_evaluations", 0),
        "eval_count": counts.get("candidate_evaluations", 0),
        "scenario_simulations": counts.get("scenario_simulations", 0),
        "route_simulations": counts.get("route_simulations", 0),
        "framework_evaluations": int(algorithm.evaluator.n_eval) if algorithm else 0,
        "framework_evaluation_note": "Diagnostic only; excludes seed evaluation and may omit an interrupted population batch",
        "stop_reason": stop_reason,
        "stop_by_eval": stop_reason == "max_eval",
        "stop_by_time": stop_reason == "max_time",
        "stop_by_iter": stop_reason == "max_iter",
        "population_size_requested": config.population_size,
        "population_size_constructed": int(algorithm.pop_size) if algorithm else None,
        "initial_population_size": callback.initial_population_size,
        "n_iterations": len(callback.history),
        "n_gen": len(callback.history),
        "iteration_definition": "completed pymoo generations, including initial population",
        "archive_size": len(archive), "archive": archive,
        "archive_policy": "nondominated completed evaluations, bounded by archive_max_size; reporting only",
        "pareto_points": [(s["cost"], s["ra"]) for s in archive],
        "best_cost_in_archive": best_cost["cost"] if best_cost else None,
        "ra_of_best_cost": best_cost["ra"] if best_cost else None,
        "best_ra_in_archive": best_ra["ra"] if best_ra else None,
        "cost_of_best_ra": best_ra["cost"] if best_ra else None,
        "checkpoints": session.checkpoints,
        "debug_log": callback.history,
        "search_scenarios": scenarios, "reduced_scenarios": reduced,
        "reduced_probs": probabilities, "selected_idx": selected_idx,
        "clusters": clusters,
        "analysis_status": "deferred", "hv_scope": "deferred_postprocessing",
        "recording": {
            "search_scenarios": True,
            "checkpoint_every_evaluations": 100,
            "checkpoint_interval_sec": 30.0,
            "checkpoint_timing": "after completed candidate evaluations; in-flight candidates are atomic",
        },
        "configuration": {
            "ev": dict(ev_params), "cost": vars(evaluator.cost_params),
            "energy": vars(evaluator.energy_model), "ra": vars(evaluator.ra_model),
        },
    }
