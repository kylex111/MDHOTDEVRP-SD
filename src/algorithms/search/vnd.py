"""Deterministic and scenario-based VND with a caller-supplied evaluator."""

from __future__ import annotations
from domain.solutions import freeze_routes
import copy
import time
import numpy as np


class VNDRefiner:
    def __init__(
        self,
        neighborhood,
        evaluator,
        dominates_fn,
        eval_proxy=None,
        rng=None,
        eps=1e-09,
        same_eval=None,
    ):
        """Configure VND components.

        neighborhood supports apply(op, routes_by_clusters, cid=cid).
        evaluator maps routes to a full evaluation result.
        dominates_fn compares results containing cost and ra with tolerance eps.
        eval_proxy is an optional cheaper evaluator; None uses evaluator."""
        self.nb = neighborhood
        self.eval = evaluator
        self.dominates = dominates_fn
        self.eval_proxy = eval_proxy if eval_proxy is not None else evaluator
        self.rng = rng if rng is not None else np.random.default_rng()
        self.eps = float(eps)
        if same_eval is None:
            self.same_eval = self.eval_proxy is self.eval
        else:
            self.same_eval = bool(same_eval)

    @staticmethod
    def _pack(routes, cend):
        return {
            "routes_by_clusters": routes,
            "cost": float(cend["total_cost"]),
            "ra": float(cend["total_ra"]),
            "cend": cend,
        }

    @staticmethod
    def _signature(routes_by_clusters):
        return freeze_routes(routes_by_clusters)

    def refine(
        self, routes_by_clusters, NL=None, cid=None, tries_per_op=3, max_steps=50
    ):
        """Run sequential VND with a cached proxy and a final full evaluation.

        Use proxy dominance to accept moves, then evaluate the final routes once.
        Track second-stage time for proxy misses and the final full evaluation.
        Return final_routes, final_cend_full, and info containing cost, ra,
        t_stage2, and n_cache."""
        eval_cache = {}
        t_stage2 = 0.0

        def _eval_proxy_cached(rbc):
            nonlocal t_stage2
            sig = self._signature(rbc)
            hit = eval_cache.get(sig, None)
            if hit is not None:
                return hit
            t0 = time.time()
            cend = self.eval_proxy(rbc)
            t_stage2 += time.time() - t0
            eval_cache[sig] = cend
            return cend

        cur_routes = copy.deepcopy(routes_by_clusters)
        cur_cend_proxy = _eval_proxy_cached(cur_routes)
        cur = self._pack(cur_routes, cur_cend_proxy)
        NL_list = list(NL) if NL else []
        steps = 0
        idx = 0
        while NL_list and steps < max_steps and (idx < len(NL_list)):
            steps += 1
            op = NL_list[idx]
            improved = False
            best_cand = None
            for _ in range(int(tries_per_op)):
                cand_routes = self.nb.apply(op, cur["routes_by_clusters"], cid=cid)
                cand_cend_proxy = _eval_proxy_cached(cand_routes)
                cand = self._pack(cand_routes, cand_cend_proxy)
                if self.dominates(cand, cur, self.eps):
                    improved = True
                    best_cand = cand
                    break
            if improved:
                cur = best_cand
                idx = 0
            else:
                idx += 1
        final_routes = copy.deepcopy(cur["routes_by_clusters"])
        if self.same_eval:
            final_cend_full = cur["cend"]
        else:
            t0 = time.time()
            final_cend_full = self.eval(final_routes)
            t_stage2 += time.time() - t0
        info = {
            "routes_by_clusters": final_routes,
            "cost": float(final_cend_full["total_cost"]),
            "ra": float(final_cend_full.get("total_ra", 0.0)),
            "cend": final_cend_full,
            "t_stage2": float(t_stage2),
            "n_cache": int(len(eval_cache)),
        }
        return (final_routes, final_cend_full, info)

    def refine_ffs(self, solution, obj="cost", NL=None, cid=None):
        """Run single-objective first-improvement VND without per-operator or step limits.

        Restart at the first neighborhood after an improvement.
        Stop after a full pass without improvement."""
        if obj not in {"cost", "ra"}:
            raise ValueError(f"Unsupported obj={obj}, expected 'cost' or 'ra'")

        def better(cand, cur):
            if obj == "cost":
                return cand["cost"] < cur["cost"] - self.eps
            return cand["ra"] < cur["ra"] - self.eps

        cur = copy.deepcopy(solution)
        NL_list = list(NL) if NL else []
        visited = []
        idx = 0
        while NL_list and idx < len(NL_list):
            op = NL_list[idx]
            cand_routes = self.nb.apply(op, cur["routes_by_clusters"], cid=cid)
            cand_cend = self.eval(cand_routes)
            cand = self._pack(cand_routes, cand_cend)
            visited.append(cand)
            if better(cand, cur):
                cur = cand
                idx = 0
            else:
                idx += 1
        final_routes = copy.deepcopy(cur["routes_by_clusters"])
        final_cend = cur["cend"]
        info = {
            "routes_by_clusters": final_routes,
            "cost": float(cur["cost"]),
            "ra": float(cur["ra"]),
            "cend": final_cend,
            "visited": visited,
        }
        return (final_routes, final_cend, info)
