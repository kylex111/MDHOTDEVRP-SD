"""Assemble SPEA2, keeping route adaptation separate from the evolution engine."""

from __future__ import annotations
import time
import numpy as np
from .heuristic import RLSHConstructor
from .initializer import HybridInitializer, InitialPopulationConfig
from .operators import CrossoverManager, MutationManager
from .engine import SPEAOptimizer
from domain.instances import load_problem
from configuration.parameters import ev_params
from evaluation.evaluator import Evaluator
from evaluation.scenarios import generate_normal_scenarios


def solve_spea(instance_path, config, run_id, seed, scenario_seed):
    started = time.perf_counter()
    problem = load_problem(str(instance_path), ev_params)
    rng = np.random.default_rng(seed)
    constructor = RLSHConstructor(problem)
    initializer = HybridInitializer(
        problem,
        constructor,
        InitialPopulationConfig(
            population_size=config.population_size,
            heuristic_ratio=0.3,
            shuffle_customers_before_rlsh=True,
            randomize_rlsh_lambda=False,
        ),
    )
    mutations = MutationManager(
        constructor,
        rrm_times=1,
        rnem_times=10,
        rntm_times=10,
        raem_times=10,
        ratm_times=10,
    )
    crossovers = CrossoverManager(constructor, hic_times=2, ric_times=2, inherit_num=2)
    scenarios = generate_normal_scenarios(
        mu=np.asarray(problem.demand, dtype=float),
        beta=config.beta,
        n_scenarios=config.n_scenarios,
        random_state=scenario_seed,
    )
    evaluator = Evaluator(
        problem,
        ra_safe=config.ra_safe,
        ra_risk=config.ra_risk,
        ra_tau_min=config.ra_tau_min,
        ra_formula_version=config.ra_formula_version,
    )
    optimizer = SPEAOptimizer(
        initializer=initializer,
        evaluator=evaluator,
        crossover_manager=crossovers,
        mutation_manager=mutations,
        scenarios=scenarios,
        pop_size=config.population_size,
        archive_size=config.archive_max_size,
        max_generations=config.max_iter,
        rng=rng,
        crossover_prob=config.crossover_probability,
        mutation_prob=config.mutation_probability,
        tournament_size=2,
        verbose=False,
    )
    archive = optimizer.run()
    return {
        "run_id": run_id,
        "seed": seed,
        "algorithm": "spea2",
        "runtime_sec": time.perf_counter() - started,
        "pareto_points": [tuple(ind.objectives) for ind in archive.individuals],
        "archive": [
            {
                "cost": float(ind.objectives[0]),
                "ra": float(ind.objectives[1]),
                "routes": [
                    {
                        "start_depot_id": r.start_depot_id,
                        "end_depot_id": r.end_depot_id,
                        "customers": r.customers,
                    }
                    for r in ind.solution
                ],
            }
            for ind in archive.individuals
        ],
    }
