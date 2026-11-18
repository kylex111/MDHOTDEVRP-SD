"""Build a mixed heuristic and random population for SPEA."""

from __future__ import annotations
from domain.solutions import DepotRoute
from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class InitialPopulationConfig:
    population_size: int
    heuristic_ratio: float = 0.3
    shuffle_customers_before_rlsh: bool = True
    randomize_rlsh_lambda: bool = False
    lambda_low: float = 0.2
    lambda_high: float = 0.8


class HybridInitializer:
    """Adapt the paper's initial population to multiple depots.

    Use RLSH for heuristic routes and capacity-only random construction for the rest.
    Repair the combined population afterward. The original paper assumes one depot."""

    def __init__(self, instance, rlsh_constructor, config: InitialPopulationConfig):
        self.instance = instance
        self.rlsh = rlsh_constructor
        self.config = config

    @property
    def capacity(self) -> float:
        return float(self.instance.vehicle_capacity)

    @property
    def depot_ids(self) -> List[int]:
        return list(self.instance.depots)

    def customer_ids(self) -> List[int]:
        return list(self.instance.customers)

    def demand(self, customer_id: int) -> float:
        return float(self.instance.demand[customer_id])

    def nearest_depot(self, node_id: int) -> int:
        """Read distances from instance.distance_matrix as a NumPy array indexed by [i, j]."""
        best_depot = None
        best_dist = float("inf")
        for depot_id in self.depot_ids:
            d = float(self.instance.distance_matrix[node_id, depot_id])
            if d < best_dist:
                best_dist = d
                best_depot = depot_id
        return int(best_depot)

    def choose_start_depot_for_route(self, first_customer: int) -> int:
        return self.nearest_depot(first_customer)

    def choose_end_depot_for_route(
        self, last_customer: int, start_depot_id: int
    ) -> int:
        return (
            self.nearest_depot(last_customer)
            if last_customer is not None
            else start_depot_id
        )

    def build_heuristic_individual(self, rng: np.random.Generator) -> List[DepotRoute]:
        customer_ids = self.customer_ids()
        if self.config.shuffle_customers_before_rlsh:
            customer_ids = customer_ids[:]
            rng.shuffle(customer_ids)
        old_lam = getattr(self.rlsh, "lam", None)
        if self.config.randomize_rlsh_lambda and old_lam is not None:
            self.rlsh.lam = float(
                rng.uniform(self.config.lambda_low, self.config.lambda_high)
            )
        solution = self.rlsh.construct_solution(customer_ids, rng)
        if self.config.randomize_rlsh_lambda and old_lam is not None:
            self.rlsh.lam = old_lam
        return solution

    def random_customer_order(self, rng: np.random.Generator) -> List[int]:
        ids = self.customer_ids()
        rng.shuffle(ids)
        return ids

    def split_by_capacity_random_only(
        self, ordered_customers: List[int], rng: np.random.Generator
    ) -> List[DepotRoute]:
        """Shuffle customers and fill routes by capacity, ignoring distance in selection.

        Start a new route when capacity is exceeded. Choose the start depot nearest
        the first customer and the end depot nearest the last customer."""
        solution: List[DepotRoute] = []
        current_route: List[int] = []
        current_load = 0.0
        for cid in ordered_customers:
            d = self.demand(cid)
            if d > self.capacity:
                raise ValueError(
                    f"Customer {cid} demand ({d}) exceeds vehicle capacity ({self.capacity})."
                )
            if current_route and current_load + d > self.capacity:
                start_depot_id = self.choose_start_depot_for_route(current_route[0])
                end_depot_id = self.choose_end_depot_for_route(
                    current_route[-1], start_depot_id
                )
                solution.append(
                    DepotRoute(
                        start_depot_id=start_depot_id,
                        end_depot_id=end_depot_id,
                        customers=current_route[:],
                    )
                )
                current_route = [cid]
                current_load = d
            else:
                current_route.append(cid)
                current_load += d
        if current_route:
            start_depot_id = self.choose_start_depot_for_route(current_route[0])
            end_depot_id = self.choose_end_depot_for_route(
                current_route[-1], start_depot_id
            )
            solution.append(
                DepotRoute(
                    start_depot_id=start_depot_id,
                    end_depot_id=end_depot_id,
                    customers=current_route[:],
                )
            )
        return solution

    def build_random_individual(self, rng: np.random.Generator) -> List[DepotRoute]:
        ordered_customers = self.random_customer_order(rng)
        solution = self.split_by_capacity_random_only(ordered_customers, rng)
        return solution

    def initialize_population(self, rng: np.random.Generator) -> List[List[DepotRoute]]:
        pop_size = self.config.population_size
        heuristic_count = int(round(pop_size * self.config.heuristic_ratio))
        heuristic_count = max(0, min(pop_size, heuristic_count))
        random_count = pop_size - heuristic_count
        population: List[List[DepotRoute]] = []
        for _ in range(heuristic_count):
            population.append(self.build_heuristic_individual(rng))
        for _ in range(random_count):
            population.append(self.build_random_individual(rng))
        rng.shuffle(population)
        return population
