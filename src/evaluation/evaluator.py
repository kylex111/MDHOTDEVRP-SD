"""Decode routes, compute cost and RA, and aggregate scenarios without experiment scheduling or export."""

from __future__ import annotations
from evaluation.costs import CostParams, calculate_cost
import copy
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from evaluation.energy import EnergyModel
from evaluation.anxiety import RangeAnxietyModel
from evaluation.decoder import Decoder
from evaluation.causal import decode_causal
from evaluation.observations import scenario_observation


class Evaluator:
    """Evaluate deterministic routes, causal recourse, scenario-weighted objectives, and initialization directions."""

    def __init__(
        self,
        problem,
        ra_safe,
        ra_risk,
        *,
        margin_ratio: float = 0.5,
        radius_km: float = 3.0,
        recourse_policy: str = "causal_v1",
        measurement=None,
        capture_scenarios: bool | str = False,
        capture_trace: bool = False,
        capture_motion: bool = False,
        ra_tau_min: float | None = None,
        ra_formula_version: str = "paper_exp_norm_v1",
        ra_model=None,
        charging_policy="ra_based",
        soc_threshold=0.20,
        collect_low_soc=False,
    ):
        self.problem = problem
        if recourse_policy != "causal_v1":
            raise ValueError("Unknown recourse policy")
        if not (
            np.isfinite(ra_safe) and np.isfinite(ra_risk) and (0 <= ra_safe < ra_risk)
        ):
            raise ValueError("Require finite 0 <= ra_safe < ra_risk")
        self.recourse_policy = recourse_policy
        self.measurement = measurement
        if capture_scenarios not in (False, True, "metrics"):
            raise ValueError("capture_scenarios must be False, True or 'metrics'")
        self.capture_scenarios = capture_scenarios
        self.capture_trace = capture_trace
        self.capture_motion = capture_motion
        self.energy_model = EnergyModel(clock_mode="relative_07_v1")
        self.ra_model = (
            RangeAnxietyModel(
                problem.soc_max, tau=ra_tau_min, formula_version=ra_formula_version
            )
            if ra_model is None
            else ra_model
        )
        if self.ra_model.formula_version == "paper_exp_norm_v1" and ra_risk > 1:
            raise ValueError("Normalized RA charging thresholds must be <= 1")
        self.decoder = Decoder(
            problem,
            self.energy_model,
            self.ra_model,
            ra_safe,
            ra_risk,
            margin_ratio=margin_ratio,
            radius_km=radius_km,
        )
        if charging_policy not in ("ra_based", "energy_only", "soc_threshold"):
            raise ValueError("Unknown charging policy")
        if not np.isfinite(soc_threshold) or not 0 < soc_threshold < 1:
            raise ValueError("SOC threshold must be in (0, 1)")
        self.decoder.charging_policy = charging_policy
        self.decoder.soc_threshold = float(soc_threshold)
        self.decoder.collect_low_soc = bool(collect_low_soc)
        self.cost_params = CostParams()
        self._nearest_depot_cache = {}

    def _nearest_depot_cached(self, node_id: int):
        if node_id not in self._nearest_depot_cache:
            self._nearest_depot_cache[node_id] = self.problem.nearest_depot(node_id)
        return self._nearest_depot_cache[node_id]

    @staticmethod
    def _flatten_routes(routes):
        """
        Accept either:
        - []
        - [1, 2, 3]
        - [[1, 2], [3, 4]]
        """
        if routes is None or len(routes) == 0:
            return []
        first = routes[0]
        if isinstance(first, (int, np.integer)):
            return list(routes)
        return [n for seg in routes for n in seg]

    @staticmethod
    def _reverse_init_structure(route_struct):
        """
        Support both:
        1) GB segments: [[...], [...], ...]
        2) Flat customer order: [1, 5, 3, 8, ...]
        """
        if not route_struct:
            return route_struct
        if isinstance(route_struct[0], int):
            return list(reversed(route_struct))
        lens = [len(seg) for seg in route_struct]
        flat_route = [n for seg in route_struct for n in seg]
        flat_rev = list(reversed(flat_route))
        new_segments = []
        idx = 0
        for L in lens:
            new_segments.append(flat_rev[idx : idx + L])
            idx += L
        return new_segments

    def _build_full_route(self, flat_route):
        if not flat_route:
            return []
        # SPEA routes include both depots explicitly; preserve their choices without inserting duplicate depots.
        if (
            flat_route[0] in self.problem.depots
            and flat_route[-1] in self.problem.depots
        ):
            return list(flat_route)
        start_depot, _ = self._nearest_depot_cached(flat_route[0])
        end_depot, _ = self._nearest_depot_cached(flat_route[-1])
        return [start_depot] + flat_route + [end_depot]

    def _nominal_route_demand(self, flat_route):
        return float(sum((self.problem.demand[i] for i in flat_route)))

    @staticmethod
    def _empty_route_result(include_restock: bool = False):
        out = {
            "ra": 0.0,
            "cost": 0.0,
            "dist": 0.0,
            "decoded_route": [],
            "num_charge": 0,
            "travel_cost": 0.0,
            "dispatch_cost": 0.0,
            "service_cost": 0.0,
            "charging_cost": 0.0,
        }
        if include_restock:
            out["num_restock"] = 0
            out["stats"] = {
                "num_charge_total": 0,
                "num_charge_opportunistic": 0,
                "num_charge_risk": 0,
                "num_charge_repair": 0,
                "num_charge_natural": 0,
                "charging_time_total": 0.0,
                "charging_time_opportunistic": 0.0,
                "charging_time_risk": 0.0,
                "charging_time_repair": 0.0,
                "charging_time_natural": 0.0,
                "charged_energy_total": 0.0,
                "charged_energy_opportunistic": 0.0,
                "charged_energy_risk": 0.0,
                "charged_energy_repair": 0.0,
                "charged_energy_natural": 0.0,
            }
        return out

    @staticmethod
    def _zero_stats():
        return {
            "num_charge_total": 0,
            "num_charge_opportunistic": 0,
            "num_charge_risk": 0,
            "num_charge_repair": 0,
            "num_charge_natural": 0,
            "charging_time_total": 0.0,
            "charging_time_opportunistic": 0.0,
            "charging_time_risk": 0.0,
            "charging_time_repair": 0.0,
            "charging_time_natural": 0.0,
            "charged_energy_total": 0.0,
            "charged_energy_opportunistic": 0.0,
            "charged_energy_risk": 0.0,
            "charged_energy_repair": 0.0,
            "charged_energy_natural": 0.0,
        }

    @staticmethod
    def _sum_stats(stats_list):
        total = Evaluator._zero_stats()
        for stats in stats_list:
            if not stats:
                continue
            for k in total:
                total[k] += stats.get(k, 0)
        return total

    @staticmethod
    def _pack_routes(per_cluster_results):
        total_cost = 0.0
        total_ra = 0.0
        total_num_charge = 0
        total_num_restock = 0
        decoded_routes = {}
        routes_ra = {}
        routes_cost = {}
        routes_dist = {}
        travel_cost_list = {}
        dispatch_cost_list = {}
        service_cost_list = {}
        charging_cost_list = {}
        num_restock_list = {}
        stats_list = {}
        route_diagnostics = {}
        for cid, res in per_cluster_results:
            total_cost += float(res["cost"])
            total_ra += float(res["ra"])
            total_num_charge += int(res["num_charge"])
            total_num_restock += int(res.get("num_restock", 0))
            decoded_routes[cid] = res["decoded_route"]
            routes_ra[cid] = float(res["ra"])
            routes_cost[cid] = float(res["cost"])
            routes_dist[cid] = float(res["dist"])
            travel_cost_list[cid] = float(res["travel_cost"])
            dispatch_cost_list[cid] = float(res["dispatch_cost"])
            service_cost_list[cid] = float(res["service_cost"])
            charging_cost_list[cid] = float(res["charging_cost"])
            num_restock_list[cid] = int(res.get("num_restock", 0))
            stats_list[cid] = res.get("stats", {})
            route_diagnostics[cid] = res.get("diagnostics", {})
        total_stats = Evaluator._sum_stats(list(stats_list.values()))
        return {
            "total_cost": float(total_cost),
            "total_ra": float(total_ra),
            "decoded_routes": decoded_routes,
            "travel_cost_list": travel_cost_list,
            "dispatch_cost_list": dispatch_cost_list,
            "service_cost_list": service_cost_list,
            "charging_cost_list": charging_cost_list,
            "routes_ra": routes_ra,
            "routes_cost": routes_cost,
            "routes_dist": routes_dist,
            "num_restock_list": num_restock_list,
            "total_num_charge": int(total_num_charge),
            "total_num_restock": int(total_num_restock),
            "stats_list": stats_list,
            "stats": total_stats,
            "route_diagnostics": route_diagnostics,
            "feasible": all(
                (d.get("completed", True) for d in route_diagnostics.values())
            ),
            "failed_routes": sum(
                (not d.get("completed", True) for d in route_diagnostics.values())
            ),
            "route_count": len(per_cluster_results),
        }

    @staticmethod
    def _normalize_probs(probs, K: int):
        if len(probs) != K:
            raise ValueError("Probability length must match number of scenarios.")
        probs = [float(p) for p in probs]
        if any((p < 0 for p in probs)):
            raise ValueError("Probabilities must be nonnegative.")
        prob_sum = sum(probs)
        if prob_sum <= 0:
            raise ValueError("Sum of probabilities must be positive.")
        return [p / prob_sum for p in probs]

    @staticmethod
    def _representative_index(costs, ras, avg_cost, avg_ra):
        """
        Choose representative scenario by 2D closeness to (avg_cost, avg_ra).
        """
        return min(
            range(len(costs)),
            key=lambda i: (costs[i] - avg_cost) ** 2 + (ras[i] - avg_ra) ** 2,
        )

    def evaluate_route(self, flattened_route_global: List[int]) -> Dict[str, Any]:
        """
        Deterministic route evaluation.
        """
        route = list(flattened_route_global)
        if not route:
            return self._empty_route_result(include_restock=False)
        route_demand = self._nominal_route_demand(route)
        full_route = self._build_full_route(route)
        (
            decode_route,
            travel_dist,
            travel_time,
            charging_time,
            service_time,
            num_charge,
            _updated_Q,
            _updated_departure_time,
            _updated_load,
            total_ra,
        ) = self.decoder.decode_deterministic(full_route, route_demand)
        dispatch_ev = 1
        route_cost, cost_detail = calculate_cost(
            dispatch_ev, travel_time, charging_time, service_time, self.cost_params
        )
        return {
            "ra": float(total_ra),
            "cost": float(route_cost),
            "dist": float(travel_dist),
            "decoded_route": decode_route,
            "num_charge": int(num_charge),
            "travel_cost": float(cost_detail.travel_cost),
            "dispatch_cost": float(cost_detail.dispatch_cost),
            "service_cost": float(cost_detail.service_cost),
            "charging_cost": float(cost_detail.charging_cost),
        }

    def evaluate_route_spr(self, flattened_route_global, real_demand) -> Dict[str, Any]:
        """
        SPR route evaluation with unified total statistics.
        """
        route = list(flattened_route_global)
        if not route:
            result = self._empty_route_result(include_restock=True)
            if self.decoder.collect_low_soc:
                result["diagnostics"] = {
                    "completed": True, "failure_reason": None, "energy_failure": False,
                    "ief_events": 0, "sef_events": 0, "load_failure_events": 0,
                    "min_soc_fraction": 1.0, "num_restock": 0,
                    "actual_travel_time_min": 0., "actual_service_time_min": 0.,
                    "actual_distance_km": 0., "actual_executed_ra": 0.,
                    "low_soc_driving_time_min": 0., "low_soc_threshold": .20,
                }
                result["feasible"] = True
            return result
        route_demand = self._nominal_route_demand(route)
        EV_load = min(route_demand, self.problem.vehicle_capacity)
        full_route = self._build_full_route(route)
        if self.measurement is not None:
            self.measurement.increment("route_simulations")
        decoded = decode_causal(
            self.decoder,
            full_route,
            EV_load,
            real_demand,
            capture=self.capture_trace,
            capture_motion=self.capture_motion,
        )
        diagnostics = copy.deepcopy(self.decoder.last_decode_diagnostics)
        (
            decode_route,
            total_dist,
            total_travel_time,
            total_charging_time,
            total_service_time,
            num_charge,
            _Q,
            _departure_time,
            _load,
            total_ra,
            num_restock,
            stats,
        ) = decoded
        dispatch_ev = 1
        route_cost, cost_detail = calculate_cost(
            dispatch_ev,
            total_travel_time,
            total_charging_time,
            total_service_time,
            self.cost_params,
        )
        return {
            "ra": float(total_ra),
            "cost": float(route_cost),
            "dist": float(total_dist),
            "decoded_route": decode_route,
            "num_charge": int(num_charge),
            "num_restock": int(num_restock),
            "travel_cost": float(cost_detail.travel_cost),
            "dispatch_cost": float(cost_detail.dispatch_cost),
            "service_cost": float(cost_detail.service_cost),
            "charging_cost": float(cost_detail.charging_cost),
            "stats": stats,
            "diagnostics": diagnostics,
            "feasible": bool(diagnostics["completed"]),
        }

    def evaluate_clusters(
        self, routes_by_clusters: Dict[int, List[List[int]]]
    ) -> Dict[str, Any]:
        per_cluster_results = []
        for cid, routes in routes_by_clusters.items():
            flat_route = self._flatten_routes(routes)
            per_cluster_results.append((cid, self.evaluate_route(flat_route)))
        return self._pack_routes(per_cluster_results)

    def evaluate_clusters_spr(self, routes_by_clusters, real_demand) -> Dict[str, Any]:
        if self.measurement is not None:
            self.measurement.increment("scenario_simulations")
        if isinstance(routes_by_clusters, list):
            routes_by_clusters = {0: routes_by_clusters}
        per_cluster_results = []
        for cid, routes in routes_by_clusters.items():
            flat_route = self._flatten_routes(routes)
            if not flat_route:
                continue
            per_cluster_results.append(
                (cid, self.evaluate_route_spr(flat_route, real_demand))
            )
        return self._pack_routes(per_cluster_results)

    def _evaluate_init_generic(
        self,
        routes_by_clusters,
        eval_fn,
        *,
        eps: float = 1e-09,
        tie_break_ra: bool = False,
    ):
        """
        Generic forward/reverse direction fixing for initialization.

        eval_fn must accept routes_by_clusters-like dict and return eval_dict with:
        - total_cost
        - total_ra

        Supports both:
        - segmented GB structure: [[...], [...], ...]
        - flat customer order: [...]
        """
        routes_fixed = copy.deepcopy(routes_by_clusters)
        for cid, route_struct in routes_fixed.items():
            cend_fwd = eval_fn({cid: route_struct})
            cost_fwd = float(cend_fwd["total_cost"])
            ra_fwd = float(cend_fwd.get("total_ra", 0.0))
            reversed_struct = self._reverse_init_structure(route_struct)
            cend_bwd = eval_fn({cid: reversed_struct})
            cost_bwd = float(cend_bwd["total_cost"])
            ra_bwd = float(cend_bwd.get("total_ra", 0.0))
            choose_bwd = False
            if cost_bwd < cost_fwd - eps:
                choose_bwd = True
            elif (
                tie_break_ra
                and abs(cost_bwd - cost_fwd) <= eps
                and (ra_bwd < ra_fwd - eps)
            ):
                choose_bwd = True
            if choose_bwd:
                routes_fixed[cid] = reversed_struct
        eval_dict = eval_fn(routes_fixed)
        return (eval_dict, routes_fixed)

    def evaluate_init(
        self, routes_by_clusters: Dict[int, List[List[int]]], eps: float = 1e-09
    ):
        """
        Initialization for deterministic evaluation.
        """
        return self._evaluate_init_generic(
            routes_by_clusters, self.evaluate_clusters, eps=eps, tie_break_ra=False
        )

    def evaluate_init_spr(self, routes_by_clusters, real_demand, eps: float = 1e-09):
        """
        Initialization for SPR evaluation.
        """
        return self._evaluate_init_generic(
            routes_by_clusters,
            lambda x: self.evaluate_clusters_spr(x, real_demand),
            eps=eps,
            tie_break_ra=True,
        )

    @staticmethod
    def _weighted_average_stats(stats_seq, probs):
        avg_stats = Evaluator._zero_stats()
        for s, stats in enumerate(stats_seq):
            if not stats:
                continue
            w = float(probs[s])
            for k in avg_stats:
                avg_stats[k] += w * float(stats.get(k, 0))
        return avg_stats

    # Execute each scenario fully before probability weighting; record candidate and scenario counts separately.
    def _aggregate_scenarios(
        self, routes_by_clusters, scenarios, probs=None
    ) -> Dict[str, Any]:
        S = int(scenarios.shape[0])
        if S <= 0:
            raise ValueError("scenarios must contain at least one scenario.")
        if probs is None:
            probs = [1.0 / S] * S
            tag = "saa"
        else:
            probs = self._normalize_probs(probs, S)
            tag = "ffs"
        if self.measurement is not None:
            self.measurement.before_candidate()
        cends = []
        total_costs = []
        total_ras = []
        total_stats_seq = []
        for s in range(S):
            real_demand = scenarios[s]
            cend = self.evaluate_clusters_spr(routes_by_clusters, real_demand)
            cends.append(cend)
            total_costs.append(float(cend["total_cost"]))
            total_ras.append(float(cend.get("total_ra", 0.0)))
            total_stats_seq.append(cend.get("stats", self._zero_stats()))
        avg_total_cost = sum((probs[s] * total_costs[s] for s in range(S)))
        avg_total_ra = sum((probs[s] * total_ras[s] for s in range(S)))
        avg_total_stats = self._weighted_average_stats(total_stats_seq, probs)
        rep_idx = self._representative_index(
            total_costs, total_ras, avg_total_cost, avg_total_ra
        )
        avg_cend = copy.deepcopy(cends[rep_idx])
        avg_cend["total_cost"] = float(avg_total_cost)
        avg_cend["total_ra"] = float(avg_total_ra)
        avg_cend["stats"] = avg_total_stats
        avg_cend.pop("stats_list", None)
        avg_cend[f"{tag}_rep_scenario_index"] = int(rep_idx)
        if tag == "ffs":
            avg_cend["ffs_probs"] = list(probs)
        avg_cend["feasible"] = all((c["feasible"] for c in cends))
        avg_cend["scenario_failure_probability"] = sum(
            (probs[i] * (not c["feasible"]) for i, c in enumerate(cends))
        )
        avg_cend["scenario_count"] = S
        avg_cend["representative_route_notice"] = (
            "decoded_routes and route_diagnostics belong to one representative scenario"
        )
        for key in (
            "travel_cost_list",
            "dispatch_cost_list",
            "service_cost_list",
            "charging_cost_list",
            "routes_cost",
            "routes_ra",
            "routes_dist",
            "num_restock_list",
        ):
            avg_cend[key] = {
                cid: sum(
                    (probs[i] * c.get(key, {}).get(cid, 0) for i, c in enumerate(cends))
                )
                for cid in cends[0].get(key, {})
            }
        for key in ("total_num_charge", "total_num_restock", "failed_routes"):
            avg_cend[key] = sum((probs[i] * c.get(key, 0) for i, c in enumerate(cends)))
        if self.capture_scenarios == "metrics":
            avg_cend["scenario_observations"] = [
                scenario_observation(c, i, probs[i]) for i, c in enumerate(cends)
            ]
        elif self.capture_scenarios:
            avg_cend["scenario_records"] = [
                dict(scenario_id=i, probability=float(probs[i]), **c)
                for i, c in enumerate(cends)
            ]
        return avg_cend

    def evaluate_saa(self, routes_by_clusters, scenarios):
        return self._aggregate_scenarios(routes_by_clusters, scenarios, probs=None)

    def evaluate_init_saa(self, routes_by_clusters, scenarios, eps: float = 1e-09):
        return self._evaluate_init_generic(
            routes_by_clusters,
            lambda x: self.evaluate_saa(x, scenarios),
            eps=eps,
            tie_break_ra=True,
        )

    def _evaluate_one_saa_scenario(self, routes_by_clusters, real_demand):
        cend = self.evaluate_clusters_spr(routes_by_clusters, real_demand)
        return {
            "cend": cend,
            "total_cost": float(cend["total_cost"]),
            "total_ra": float(cend.get("total_ra", 0.0)),
        }

    def evaluate_saa_n_workers(self, routes_by_clusters, scenarios, n_workers: int = 1):
        S = int(scenarios.shape[0])
        if S <= 0:
            raise ValueError("scenarios must contain at least one scenario.")
        if n_workers <= 1:
            results = [
                self._evaluate_one_saa_scenario(routes_by_clusters, scenarios[s])
                for s in range(S)
            ]
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                futures = [
                    ex.submit(
                        self._evaluate_one_saa_scenario,
                        routes_by_clusters,
                        scenarios[s],
                    )
                    for s in range(S)
                ]
                results = [f.result() for f in futures]
        cends = [r["cend"] for r in results]
        total_costs = [r["total_cost"] for r in results]
        total_ras = [r["total_ra"] for r in results]
        avg_total_cost = sum(total_costs) / S
        avg_total_ra = sum(total_ras) / S
        rep_idx = self._representative_index(
            total_costs, total_ras, avg_total_cost, avg_total_ra
        )
        avg_cend = copy.deepcopy(cends[rep_idx])
        avg_cend["total_cost"] = float(avg_total_cost)
        avg_cend["total_ra"] = float(avg_total_ra)
        avg_cend["saa_rep_scenario_index"] = int(rep_idx)
        return avg_cend

    def evaluate_init_saa_n_workers(
        self, routes_by_clusters, scenarios, eps: float = 1e-09, n_workers: int = 1
    ):
        return self._evaluate_init_generic(
            routes_by_clusters,
            lambda x: self.evaluate_saa_n_workers(x, scenarios, n_workers=n_workers),
            eps=eps,
            tie_break_ra=True,
        )

    def evaluate_ffs(self, routes_by_clusters, reduced_scenarios, reduced_probs):
        return self._aggregate_scenarios(
            routes_by_clusters, reduced_scenarios, probs=reduced_probs
        )

    def evaluate_init_ffs(
        self, routes_by_clusters, reduced_scenarios, reduced_probs, eps: float = 1e-09
    ):
        return self._evaluate_init_generic(
            routes_by_clusters,
            lambda x: self.evaluate_ffs(x, reduced_scenarios, reduced_probs),
            eps=eps,
            tie_break_ra=True,
        )
