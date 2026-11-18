"""Order granular balls with a GA using ball centers and depot locations."""

from __future__ import annotations
from domain.solutions import OrderResult
from typing import List, Optional
import numpy as np
from scipy.spatial.distance import cdist
from configuration.parameters import GAParams


class GbPlannerGA:
    """Return a ball-index permutation for gb_centers (m, 2) and depots_xy (d, 2)."""

    def __init__(self, gb_centers: np.ndarray, depots_xy: np.ndarray, params):
        self.params = params
        self.rng = np.random.default_rng(params.seed)
        centers = np.asarray(gb_centers, dtype=float)
        if centers.ndim != 2 or centers.shape[1] != 2:
            raise ValueError(f"gb_centers must be (m,2), got {centers.shape}")
        depots = np.asarray(depots_xy, dtype=float)
        if depots.ndim != 2 or depots.shape[1] != 2:
            raise ValueError(f"depots_xy must be (d,2), got {depots.shape}")
        self.centers = centers
        self.depots = depots
        self.m = centers.shape[0]
        self.dist = cdist(self.centers, self.centers, metric="euclidean")
        self.attach = (
            cdist(self.centers, self.depots).min(axis=1)
            if self.m > 0
            else np.array([], dtype=float)
        )
        self.best_route: Optional[np.ndarray] = None
        self.best_cost: float = float("inf")

    def evaluate(self, route: np.ndarray) -> float:
        """route: shape (m,), permutation"""
        if route.size == 0:
            return 0.0
        if route.size == 1:
            return float(self.attach[route[0]] + self.attach[route[0]])
        total = float(self.dist[route[:-1], route[1:]].sum())
        total += float(self.attach[route[0]] + self.attach[route[-1]])
        return total

    def init_population(self) -> np.ndarray:
        """Return population as (pop_size, m) permutations."""
        pop = np.empty((self.params.pop_size, self.m), dtype=int)
        base = np.arange(self.m, dtype=int)
        for i in range(self.params.pop_size):
            pop[i] = self.rng.permutation(base)
        return pop

    def tournament_select(self, pop: np.ndarray, costs: np.ndarray) -> np.ndarray:
        """Select one parent by tournament (min cost wins)."""
        k = min(int(self.params.tournament_k), pop.shape[0])
        idx = self.rng.integers(0, pop.shape[0], size=k)
        best = idx[np.argmin(costs[idx])]
        return pop[best].copy()

    def ox_crossover(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Order Crossover (OX) for permutations."""
        n = p1.size
        if n <= 1:
            return p1.copy()
        a, b = sorted(self.rng.choice(n, size=2, replace=False))
        child = np.full(n, -1, dtype=int)
        child[a : b + 1] = p1[a : b + 1]
        fill = [x for x in p2 if x not in child]
        ptr = 0
        for i in range(n):
            if child[i] == -1:
                child[i] = fill[ptr]
                ptr += 1
        return child

    def mutate(self, x: np.ndarray) -> None:
        """In-place mutation: inversion or swap."""
        n = x.size
        if n <= 1:
            return
        if n == 2:
            if self.rng.random() < 0.5:
                x[0], x[1] = (x[1], x[0])
            return
        if self.rng.random() < self.params.inversion_rate:
            i, j = sorted(self.rng.choice(n, size=2, replace=False))
            x[i : j + 1] = x[i : j + 1][::-1]
        else:
            i, j = self.rng.choice(n, size=2, replace=False)
            x[i], x[j] = (x[j], x[i])

    def solve(self) -> OrderResult:
        history: List[float] = []
        if self.m == 0:
            self.best_route = np.array([], dtype=int)
            self.best_cost = 0.0
            history.append(self.best_cost)
            return OrderResult(order=[], cost=self.best_cost, history=history)
        if self.m == 1:
            self.best_route = np.array([0], dtype=int)
            self.best_cost = self.evaluate(self.best_route)
            history.append(self.best_cost)
            return OrderResult(order=[0], cost=self.best_cost, history=history)
        pop = self.init_population()
        costs = np.array([self.evaluate(ind) for ind in pop], dtype=float)
        best_idx = int(np.argmin(costs))
        self.best_route = pop[best_idx].copy()
        self.best_cost = float(costs[best_idx])
        history.append(self.best_cost)
        for _gen in range(self.params.num_gen):
            elite_size = max(0, int(self.params.elite_size))
            elite_size = min(elite_size, self.params.pop_size)
            elite_idx = np.argsort(costs)[:elite_size]
            elites = pop[elite_idx].copy()
            new_pop = []
            if elite_size > 0:
                for e in elites:
                    new_pop.append(e)
            while len(new_pop) < self.params.pop_size:
                p1 = self.tournament_select(pop, costs)
                p2 = self.tournament_select(pop, costs)
                if self.rng.random() < self.params.crossover_rate:
                    c1 = self.ox_crossover(p1, p2)
                    c2 = self.ox_crossover(p2, p1)
                else:
                    c1, c2 = (p1.copy(), p2.copy())
                if self.rng.random() < self.params.mutation_rate:
                    self.mutate(c1)
                if self.rng.random() < self.params.mutation_rate:
                    self.mutate(c2)
                new_pop.append(c1)
                if len(new_pop) < self.params.pop_size:
                    new_pop.append(c2)
            pop = np.array(new_pop, dtype=int)
            costs = np.array([self.evaluate(ind) for ind in pop], dtype=float)
            gen_best_idx = int(np.argmin(costs))
            gen_best_cost = float(costs[gen_best_idx])
            if gen_best_cost < self.best_cost:
                self.best_cost = gen_best_cost
                self.best_route = pop[gen_best_idx].copy()
            history.append(self.best_cost)
        return OrderResult(
            order=self.best_route.tolist() if self.best_route is not None else [],
            cost=self.best_cost,
            history=history,
        )


def plan_gb_order_ga(
    gb_centers: np.ndarray, depots_xy: np.ndarray, params: Optional[GAParams] = None
) -> OrderResult:
    params = params or GAParams()
    return GbPlannerGA(gb_centers, depots_xy, params).solve()
