"""Own search state; evaluators compute objectives and experiment code receives snapshots."""

from __future__ import annotations
import numpy as np
from datetime import datetime
from configuration.parameters import ev_params
from .pareto import archive_stats, hypervolume_2d_min
from .catalog import GB_SHAKING, GB_VND, NODE_SHAKING, NODE_VND
import copy
import time
from domain.solutions import pack_solution
from .pareto import pareto_insert, truncate_by_crowding


class SearchSession:
    """Per-run archive, evaluation counts, and snapshots, recreated for each solve call."""

    def __init__(
        self,
        evaluator,
        scenarios,
        probabilities,
        measurement,
        started,
        eps,
        archive_max_size,
        checkpoint_every,
        checkpoint_interval_sec,
        checkpoint_sink,
    ):
        self.evaluator = evaluator
        self.scenarios, self.probabilities = (scenarios, probabilities)
        self.measurement, self.started = (measurement, started)
        self.eps, self.archive_max_size = (eps, archive_max_size)
        self.checkpoint_every, self.checkpoint_interval_sec = (
            checkpoint_every,
            checkpoint_interval_sec,
        )
        self.checkpoint_sink = checkpoint_sink
        self.archive = []
        self.evaluated_frontier = []
        self.checkpoints = []
        self.last_checkpoint_sec = 0.0
        self.eval_counter = {"count": 0}

    def checkpoint(self, front, label):
        snapshot = {
            "checkpoint_id": len(self.checkpoints),
            "label": label,
            **self.measurement.snapshot(),
            "archive": [
                {
                    "routes_by_clusters": copy.deepcopy(sol["routes_by_clusters"]),
                    "cost": float(sol["cost"]),
                    "ra": float(sol["ra"]),
                    "feasible": sol.get("cend", {}).get("feasible"),
                }
                for sol in front
            ],
        }
        self.checkpoints.append(snapshot)
        self.last_checkpoint_sec = snapshot["elapsed_sec"]
        if self.checkpoint_sink is not None:
            self.checkpoint_sink(snapshot)

    def evaluate(self, routes):
        result = self.evaluator.evaluate_ffs(routes, self.scenarios, self.probabilities)
        self.eval_counter["count"] = self.measurement.counts.get(
            "candidate_evaluations", 0
        )
        sol = pack_solution(copy.deepcopy(routes), result)
        self.evaluated_frontier, _ = pareto_insert(
            self.evaluated_frontier, sol, eps=self.eps
        )
        if self.archive_max_size is not None:
            self.evaluated_frontier = truncate_by_crowding(
                self.evaluated_frontier, self.archive_max_size
            )
        by_count = (
            self.checkpoint_every
            and self.eval_counter["count"] % self.checkpoint_every == 0
        )
        by_time = (
            self.checkpoint_interval_sec is not None
            and time.perf_counter() - self.started - self.last_checkpoint_sec
            >= self.checkpoint_interval_sec
        )
        if by_count or by_time:
            self.checkpoint(self.evaluated_frontier, "evaluated_frontier")
            self.checkpoint(self.archive, "solver_archive")
        return result

    def record_iteration(
        self,
        it,
        cand_points,
        archive_changed,
        iter_t0,
        eval_before_iter,
        elapsed_before_iter,
        old_points_iter,
        reference,
    ):
        """Collect completed-iteration metrics without generating or evaluating candidates."""
        ref_cost, ref_ra = reference
        compute_diagnostic_hv = False
        iter_time = time.perf_counter() - iter_t0
        best_cost_sol = min(self.archive, key=lambda s: s["cost"])
        best_ra_sol = min(self.archive, key=lambda s: s["ra"])
        arch_stat = archive_stats(self.archive)
        hv = (
            hypervolume_2d_min(self.archive, ref_point=(ref_cost, ref_ra))
            if compute_diagnostic_hv
            else float("nan")
        )
        new_points_iter = {(float(s["cost"]), float(s["ra"])) for s in self.archive}
        n_new_to_archive = len(new_points_iter - old_points_iter)
        n_removed_from_archive = len(old_points_iter - new_points_iter)
        best_cost_improve_abs = float(self.prev_best_cost - best_cost_sol["cost"])
        best_ra_improve_abs = float(self.prev_best_ra - best_ra_sol["ra"])
        hv_improve_abs = float(hv - self.prev_hv)
        if abs(best_cost_improve_abs) < 1e-12:
            best_cost_improve_abs = 0.0
        if abs(best_ra_improve_abs) < 1e-12:
            best_ra_improve_abs = 0.0
        if abs(hv_improve_abs) < 1e-12:
            hv_improve_abs = 0.0
        if archive_changed:
            self.stall_iters_so_far = 0
            self.last_improve_iter_so_far = int(it)
        else:
            self.stall_iters_so_far += 1
        eval_count_now = int(self.eval_counter["count"])
        eval_count_iter = int(eval_count_now - eval_before_iter)
        elapsed_now = float(time.perf_counter() - self.started)
        elapsed_iter = float(elapsed_now - elapsed_before_iter)
        n_candidates_now = int(len(cand_points))
        n_candidates_iter = int(n_candidates_now - self.prev_n_candidates)
        eval_per_sec = (
            float(eval_count_iter / iter_time) if iter_time > 1e-12 else np.nan
        )
        cand_per_sec = (
            float(n_candidates_iter / iter_time) if iter_time > 1e-12 else np.nan
        )
        hv_per_sec = float(hv_improve_abs / iter_time) if iter_time > 1e-12 else np.nan
        hv_per_eval = (
            float(hv_improve_abs / eval_count_iter) if eval_count_iter > 0 else np.nan
        )
        self.debug_log.append(
            {
                "iter": int(it),
                "elapsed_sec": elapsed_now,
                "elapsed_iter_sec": elapsed_iter,
                "eval_count": eval_count_now,
                "eval_count_iter": eval_count_iter,
                "n_candidates_so_far": n_candidates_now,
                "n_candidates_iter": n_candidates_iter,
                "best_cost_in_archive": float(best_cost_sol["cost"]),
                "ra_of_best_cost": float(best_cost_sol["ra"]),
                "best_ra_in_archive": float(best_ra_sol["ra"]),
                "cost_of_best_ra": float(best_ra_sol["cost"]),
                "worst_cost_in_archive": float(arch_stat["worst_cost"]),
                "worst_ra_in_archive": float(arch_stat["worst_ra"]),
                "spread_cost": float(arch_stat["spread_cost"]),
                "spread_ra": float(arch_stat["spread_ra"]),
                "archive_size": int(len(self.archive)),
                "archive_changed": bool(archive_changed),
                "n_new_to_archive": int(n_new_to_archive),
                "n_removed_from_archive": int(n_removed_from_archive),
                "hv": float(hv),
                "hv_ref_cost": float(ref_cost),
                "hv_ref_ra": float(ref_ra),
                "best_cost_improve_abs": float(best_cost_improve_abs),
                "best_ra_improve_abs": float(best_ra_improve_abs),
                "hv_improve_abs": float(hv_improve_abs),
                "best_cost_improve_rel": float(
                    best_cost_improve_abs / self.prev_best_cost
                )
                if abs(self.prev_best_cost) > 1e-12
                else np.nan,
                "best_ra_improve_rel": float(best_ra_improve_abs / self.prev_best_ra)
                if abs(self.prev_best_ra) > 1e-12
                else np.nan,
                "hv_improve_rel": float(hv_improve_abs / self.prev_hv)
                if abs(self.prev_hv) > 1e-12
                else np.nan,
                "last_improve_iter_so_far": int(self.last_improve_iter_so_far),
                "stall_iters_so_far": int(self.stall_iters_so_far),
                "iter_time": float(iter_time),
                "eval_per_sec": float(eval_per_sec),
                "cand_per_sec": float(cand_per_sec),
                "hv_per_sec": float(hv_per_sec),
                "hv_per_eval": float(hv_per_eval),
            }
        )
        self.prev_best_cost = float(best_cost_sol["cost"])
        self.prev_best_ra = float(best_ra_sol["ra"])
        self.prev_hv = float(hv)
        self.prev_n_candidates = int(len(cand_points))

    def start_history(self, initial_solution, candidate_points):
        """Per-run statistical context, never shared with another solve call."""
        self.debug_log = []
        self.prev_best_cost = float(initial_solution["cost"])
        self.prev_best_ra = float(initial_solution["ra"])
        self.prev_hv = 0.0
        self.prev_n_candidates = len(candidate_points)
        self.stall_iters_so_far = 0
        self.last_improve_iter_so_far = 0

    def result(
        self,
        *,
        config,
        instance_name,
        run_id,
        seed,
        scenario_seed,
        initial_solution,
        generated_scenarios,
        selected_indices,
        clusters,
    ):
        """Assemble objectives, counts, configuration, and records from completed evaluations."""
        use_gb = config.algorithm != "without_gb"
        use_movns = config.use_movns and config.algorithm != "without_vns"
        NS = GB_SHAKING if use_gb else NODE_SHAKING
        NL = GB_VND if use_gb else NODE_VND
        checkpoint_every = self.checkpoint_every
        checkpoint_interval_sec = self.checkpoint_interval_sec
        record_observations = True
        compute_diagnostic_hv = False
        runtime = time.perf_counter() - self.started
        min_cost_sol = min(self.archive, key=lambda s: s["cost"])
        min_ra_sol = min(self.archive, key=lambda s: s["ra"])
        init_stats = copy.deepcopy(initial_solution.get("stats", {}))
        best_cost_stats = copy.deepcopy(
            min_cost_sol.get("stats", min_cost_sol.get("cend", {}).get("stats", {}))
        )
        best_ra_stats = copy.deepcopy(
            min_ra_sol.get("stats", min_ra_sol.get("cend", {}).get("stats", {}))
        )
        final_arch_stat = archive_stats(self.archive)
        hv_final = (
            hypervolume_2d_min(self.archive, ref_point=(self.ref_cost, self.ref_ra))
            if compute_diagnostic_hv
            else float("nan")
        )
        improve_cost_abs = float(initial_solution["cost"] - min_cost_sol["cost"])
        improve_ra_abs = float(initial_solution["ra"] - min_ra_sol["ra"])
        improve_cost_rel = (
            improve_cost_abs / float(initial_solution["cost"])
            if abs(float(initial_solution["cost"])) > 1e-12
            else np.nan
        )
        improve_ra_rel = (
            improve_ra_abs / float(initial_solution["ra"])
            if abs(float(initial_solution["ra"])) > 1e-12
            else np.nan
        )
        result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "instance": instance_name,
            "beta": float(config.beta),
            "run_id": int(run_id),
            "seed": int(seed),
            "ra_safe_param": float(config.ra_safe),
            "ra_risk_param": float(config.ra_risk),
            "init_cost": float(initial_solution["cost"]),
            "init_ra": float(initial_solution["ra"]),
            "best_cost_in_archive": float(min_cost_sol["cost"]),
            "ra_of_best_cost": float(min_cost_sol["ra"]),
            "best_ra_in_archive": float(min_ra_sol["ra"]),
            "cost_of_best_ra": float(min_ra_sol["cost"]),
            "archive_size": int(len(self.archive)),
            "n_candidates": int(len(self.cand_points)),
            "n_iterations": int(self.it),
            "hv_final": float(hv_final),
            "hv_scope": "within_run_diagnostic_dynamic_reference_not_comparable"
            if compute_diagnostic_hv
            else "deferred_postprocessing",
            "analysis_status": "deferred",
            "recording": {
                "search_scenarios": bool(record_observations),
                "checkpoint_every_evaluations": checkpoint_every,
                "checkpoint_interval_sec": checkpoint_interval_sec,
                "checkpoint_timing": "after completed candidate evaluations; in-flight candidates are atomic",
            },
            "recourse_policy": config.recourse_policy,
            "scenario_seed": scenario_seed,
            "measurement": self.measurement.snapshot(),
            "candidate_evaluations": self.measurement.counts.get(
                "candidate_evaluations", 0
            ),
            "scenario_simulations": self.measurement.counts.get(
                "scenario_simulations", 0
            ),
            "route_simulations": self.measurement.counts.get("route_simulations", 0),
            "checkpoints": self.checkpoints,
            "search_scenarios": generated_scenarios,
            "reduced_scenarios": self.scenarios,
            "reduced_probs": self.probabilities,
            "init_mode": config.init_mode,
            "use_movns": use_movns,
            "use_gb": use_gb,
            "configuration": {
                "ev": dict(ev_params),
                "cost": vars(self.evaluator.cost_params),
                "energy": vars(self.evaluator.energy_model),
                "ra": vars(self.evaluator.ra_model),
                "ra_safe": config.ra_safe,
                "ra_risk": config.ra_risk,
                "gb_ga": {
                    "pop_size": config.gb_pop_size,
                    "num_gen": config.gb_num_gen,
                    "tournament_k": 3,
                    "crossover_rate": config.gb_crossover_rate,
                    "mutation_rate": config.gb_mutation_rate,
                    "inversion_rate": 0.7,
                    "elite_size": 2,
                },
                "customer_ga": {
                    "pop_size": config.customer_pop_size,
                    "num_gen": config.customer_num_gen,
                    "tournament_k": 3,
                    "crossover_rate": config.customer_crossover_rate,
                    "mutation_rate": config.customer_mutation_rate,
                    "inversion_rate": 0.7,
                    "elite_size": 2,
                },
                "split_k": config.split_k,
                "removal_ratio": config.removal_ratio,
                "neighborhood_boundary_weight": 0.5,
                "neighborhoods": {"shaking": list(NS), "vnd": list(NL)},
                "ga_seed_rules": {
                    "gb": "algorithm_seed+1000+cluster_id",
                    "customer": "algorithm_seed+2000+cluster_id",
                },
            },
            "hv_ref_cost": float(self.ref_cost),
            "hv_ref_ra": float(self.ref_ra),
            "spread_cost_final": float(final_arch_stat["spread_cost"]),
            "spread_ra_final": float(final_arch_stat["spread_ra"]),
            "archive_change_count": int(self.archive_change_count),
            "last_improve_iter": int(self.last_improve_iter),
            "improve_cost_abs": float(improve_cost_abs),
            "improve_cost_rel": float(improve_cost_rel),
            "improve_ra_abs": float(improve_ra_abs),
            "improve_ra_rel": float(improve_ra_rel),
            "eval_count": int(self.eval_counter["count"]),
            "runtime_sec": float(runtime),
            "max_time_sec": None
            if config.max_time_sec is None
            else float(config.max_time_sec),
            "saa_S": int(generated_scenarios.shape[0]),
            "n_reduced_scenarios": int(self.scenarios.shape[0]),
            "pareto_points": [
                (float(sol["cost"]), float(sol["ra"])) for sol in self.archive
            ],
            "archive": self.archive,
            "min_cost_sol": min_cost_sol,
            "min_ra_sol": min_ra_sol,
            "cand_points": self.cand_points,
            "clusters": clusters,
            "selected_idx": selected_indices,
            "debug_log": self.debug_log,
            "max_iter_limit": None if config.max_iter is None else int(config.max_iter),
            "max_eval": None if config.max_eval is None else int(config.max_eval),
            "stop_by_iter": bool(self.stop_by_iter),
            "stop_by_eval": bool(self.stop_by_eval),
            "stop_by_time": bool(self.stop_by_time),
            "stop_reason": self.stop_reason,
            "init_stats": init_stats,
            "best_cost_stats": best_cost_stats,
            "best_ra_stats": best_ra_stats,
        }
        return result
