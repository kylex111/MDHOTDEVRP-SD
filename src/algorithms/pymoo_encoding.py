"""Permutation encoding, repair, and sampling for pymoo; the shared evaluator supplies objectives."""

from __future__ import annotations
import copy
import random
import numpy as np
from typing import Optional
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling


def canonicalize_routes_clusters(routes_clusters):
    return tuple(
        (
            (cid, tuple((tuple(route) for route in routes)))
            for cid, routes in sorted(routes_clusters.items())
        )
    )


def perturb_solution(routes_clusters, rng=None):
    if rng is None:
        rng = random.Random()
    sol = copy.deepcopy(routes_clusters)
    cluster_ids = sorted(sol.keys())
    cid = rng.choice(cluster_ids)
    routes = sol[cid]
    if len(routes) == 0:
        return sol
    move_type = rng.choice(
        [
            "swap_blocks",
            "reverse_block_segment",
            "insert_block",
            "reverse_inside_block",
            "swap_inside_block",
        ]
    )
    if move_type == "swap_blocks" and len(routes) >= 2:
        i, j = sorted(rng.sample(range(len(routes)), 2))
        routes[i], routes[j] = (routes[j], routes[i])
    elif move_type == "reverse_block_segment" and len(routes) >= 2:
        i, j = sorted(rng.sample(range(len(routes)), 2))
        routes[i : j + 1] = list(reversed(routes[i : j + 1]))
    elif move_type == "insert_block" and len(routes) >= 2:
        i, j = rng.sample(range(len(routes)), 2)
        block = routes.pop(i)
        routes.insert(j, block)
    elif move_type == "reverse_inside_block":
        candidate_idx = [k for k, block in enumerate(routes) if len(block) >= 2]
        if candidate_idx:
            k = rng.choice(candidate_idx)
            routes[k] = list(reversed(routes[k]))
    elif move_type == "swap_inside_block":
        candidate_idx = [k for k, block in enumerate(routes) if len(block) >= 2]
        if candidate_idx:
            k = rng.choice(candidate_idx)
            block = routes[k]
            i, j = rng.sample(range(len(block)), 2)
            block[i], block[j] = (block[j], block[i])
    sol[cid] = routes
    return sol


def flatten_routes_clusters(routes_clusters):
    flat = {}
    for cid, blocks in routes_clusters.items():
        seq = []
        for block in blocks:
            seq.extend(block)
        flat[cid] = seq
    return flat


def generate_init_population(core_solution, pop_size, seed=42, max_trials=5000):
    rng = random.Random(seed)
    population = [copy.deepcopy(core_solution)]
    seen = {canonicalize_routes_clusters(core_solution)}
    trials = 0
    while len(population) < pop_size and trials < max_trials:
        new_sol = perturb_solution(core_solution, rng=rng)
        key = canonicalize_routes_clusters(new_sol)
        if key not in seen:
            seen.add(key)
            population.append(new_sol)
        trials += 1
    if len(population) < pop_size:
        print(
            f"[WARN] only generated {len(population)} unique solutions, target={pop_size}"
        )
    population_flat = [flatten_routes_clusters(sol) for sol in population]
    return (population, population_flat)


def build_flat_encoding_state(seed_solution):
    cluster_ids = sorted(seed_solution.keys())
    cluster_lengths = {}
    cluster_nodes = {}
    slices = {}
    start = 0
    for cid in cluster_ids:
        nodes = list(seed_solution[cid])
        length = len(nodes)
        cluster_lengths[cid] = length
        cluster_nodes[cid] = set(nodes)
        slices[cid] = (start, start + length)
        start += length
    return {
        "cluster_ids": cluster_ids,
        "cluster_lengths": cluster_lengths,
        "cluster_nodes": cluster_nodes,
        "slices": slices,
        "n_var": start,
    }


def encode_flat_routes(solution, state):
    x = []
    for cid in state["cluster_ids"]:
        x.extend(list(solution[cid]))
    return np.asarray(x, dtype=int)


def decode_flat_routes(x, state):
    x = np.asarray(x).astype(int)
    sol = {}
    for cid in state["cluster_ids"]:
        s, e = state["slices"][cid]
        sol[cid] = list(map(int, x[s:e]))
    return sol


def repair_flat_routes(sol, state):
    repaired = {}
    for cid in state["cluster_ids"]:
        target_nodes = state["cluster_nodes"][cid]
        seq = sol[cid]
        seen = set()
        filtered = []
        for node in seq:
            if node in target_nodes and node not in seen:
                filtered.append(node)
                seen.add(node)
        missing = [node for node in sorted(target_nodes) if node not in seen]
        filtered.extend(missing)
        repaired[cid] = filtered
    return repaired


def decode_and_repair(x, state):
    sol = decode_flat_routes(x, state)
    sol = repair_flat_routes(sol, state)
    return sol


class GBClusterRoutingProblemFlat(Problem):
    def __init__(self, state, evaluator_fn):
        super().__init__(
            n_var=state["n_var"],
            n_obj=2,
            n_ieq_constr=0,
            xl=np.zeros(state["n_var"], dtype=int),
            xu=np.full(state["n_var"], 10**6, dtype=int),
            vtype=int,
        )
        self.state = state
        self.evaluator_fn = evaluator_fn

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[None, :]
        F = []
        for x in X:
            sol = decode_and_repair(x, self.state)
            f1, f2 = self.evaluator_fn(sol)
            F.append([float(f1), float(f2)])
        out["F"] = np.asarray(F, dtype=float)


class GBHeuristicSamplingFlat(Sampling):
    def __init__(
        self, state, seed_solutions, random_fill=True, seed: Optional[int] = None
    ):
        super().__init__()
        self.state = state
        self.seed_solutions = list(seed_solutions)
        self.random_fill = random_fill
        self.rng = np.random.default_rng(seed)

    def _random_perturb(self, x: np.ndarray) -> np.ndarray:
        y = x.copy()
        cid = int(self.rng.choice(self.state["cluster_ids"]))
        s, e = self.state["slices"][cid]
        seg_len = e - s
        if seg_len >= 2:
            i, j = self.rng.choice(np.arange(s, e), size=2, replace=False)
            y[i], y[j] = (y[j], y[i])
        return y

    def _do(self, problem, n_samples, **kwargs):
        X = []
        for sol in self.seed_solutions[:n_samples]:
            X.append(encode_flat_routes(sol, self.state))
        if len(X) == 0:
            raise ValueError("seed_solutions is empty.")
        while len(X) < n_samples:
            base = X[len(X) % len(X)]
            if self.random_fill:
                X.append(self._random_perturb(base))
            else:
                X.append(base.copy())
        return np.asarray(X, dtype=int)
