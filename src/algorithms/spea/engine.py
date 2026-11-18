"""SPEA2 fitness, selection, and evolution."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import math
import numpy as np


@dataclass
class Individual:
    solution: object
    objectives: Optional[List[float]] = None
    strength: float = 0.0
    raw_fitness: float = 0.0
    density: float = 0.0
    fitness: float = 0.0

    def copy(self) -> "Individual":
        return Individual(
            solution=self.solution,
            objectives=None if self.objectives is None else self.objectives[:],
            strength=self.strength,
            raw_fitness=self.raw_fitness,
            density=self.density,
            fitness=self.fitness,
        )


@dataclass
class Archive:
    individuals: List[Individual] = field(default_factory=list)

    def clear(self) -> None:
        self.individuals.clear()

    def size(self) -> int:
        return len(self.individuals)

    def copy(self) -> "Archive":
        return Archive(individuals=[ind.copy() for ind in self.individuals])


def clone_solution(solution):
    return [
        type(r)(
            start_depot_id=r.start_depot_id,
            end_depot_id=r.end_depot_id,
            customers=r.customers[:],
        )
        for r in solution
    ]


def dominates(a: List[float], b: List[float]) -> bool:
    """A dominates b for minimization if it is no worse in all objectives and strictly better in at least one."""
    no_worse = all((x <= y for x, y in zip(a, b)))
    strictly_better = any((x < y for x, y in zip(a, b)))
    return no_worse and strictly_better


def euclidean_distance(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum(((x - y) ** 2 for x, y in zip(a, b))))


def unique_by_objectives(individuals):
    unique = []
    seen = set()
    for ind in individuals:
        key = tuple(ind.objectives)
        if key not in seen:
            seen.add(key)
            unique.append(ind)
    return unique


def get_nondominated(individuals):
    result = []
    for i, ind_i in enumerate(individuals):
        dominated_flag = False
        for j, ind_j in enumerate(individuals):
            if i == j:
                continue
            if dominates(ind_j.objectives, ind_i.objectives):
                dominated_flag = True
                break
        if not dominated_flag:
            result.append(ind_i)
    return result


class SPEAOptimizer:
    def __init__(
        self,
        initializer,
        evaluator,
        crossover_manager,
        mutation_manager,
        scenarios,
        pop_size: int,
        archive_size: int,
        max_generations: int,
        rng: np.random.Generator,
        crossover_prob: float = 1.0,
        mutation_prob: float = 1.0,
        tournament_size: int = 2,
        k_neighbor: Optional[int] = None,
        verbose: bool = True,
    ):
        """Configure the SPEA2 components.

        initializer provides initialize_population(pop_size, rng).
        evaluator provides evaluate_given_scenarios or evaluate_saa and returns
        total_cost, environmental_cost, and total_satisfaction.
        crossover_manager provides apply_one(parent1, parent2, rng).
        mutation_manager provides apply_all(solution, rng).
        Minimize [total_cost, environmental_cost, -total_satisfaction]."""
        self.initializer = initializer
        self.evaluator = evaluator
        self.crossover_manager = crossover_manager
        self.mutation_manager = mutation_manager
        self.scenarios = scenarios
        self.pop_size = pop_size
        self.archive_size = archive_size
        self.max_generations = max_generations
        self.rng = rng
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.k_neighbor = k_neighbor
        self.verbose = verbose

    def evaluate_solution(self, solution):
        routes = {}
        for i, s in enumerate(solution):
            route = [s.start_depot_id] + list(s.customers) + [s.end_depot_id]
            routes[i] = route
        result = self.evaluator.evaluate_saa(routes, self.scenarios)
        total_cost = float(result["total_cost"])
        total_ra = float(result["total_ra"])
        return [float(total_cost), float(total_ra)]

    def evaluate_population(self, population: List[Individual]) -> None:
        for ind in population:
            if ind.objectives is None:
                ind.objectives = self.evaluate_solution(ind.solution)

    def strength_assignment(self, union: List[Individual]) -> None:
        n = len(union)
        for i in range(n):
            s = 0
            for j in range(n):
                if i == j:
                    continue
                if dominates(union[i].objectives, union[j].objectives):
                    s += 1
            union[i].strength = float(s)
        for i in range(n):
            raw = 0.0
            for j in range(n):
                if i == j:
                    continue
                if dominates(union[j].objectives, union[i].objectives):
                    raw += union[j].strength
            union[i].raw_fitness = raw

    def density_estimation(self, union: List[Individual]) -> None:
        n = len(union)
        if n == 0:
            return
        k = self.k_neighbor
        if k is None:
            k = max(1, int(math.sqrt(n)))
        k = min(k, max(1, n - 1))
        distance_matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = euclidean_distance(union[i].objectives, union[j].objectives)
                distance_matrix[i][j] = d
                distance_matrix[j][i] = d
        for i in range(n):
            distances = [distance_matrix[i][j] for j in range(n) if j != i]
            distances.sort()
            sigma_k = distances[k - 1] if distances else 0.0
            union[i].density = 1.0 / (sigma_k + 2.0)
        for ind in union:
            ind.fitness = ind.raw_fitness + ind.density

    def assign_fitness(self, union: List[Individual]) -> None:
        self.strength_assignment(union)
        self.density_estimation(union)

    def environmental_selection(self, union: List[Individual]) -> Archive:
        """Select raw_fitness < 1 first, fill remaining slots by fitness, and truncate if oversized."""
        archive_candidates = [ind.copy() for ind in union if ind.raw_fitness < 1.0]
        archive_candidates = unique_by_objectives(archive_candidates)
        if len(archive_candidates) < self.archive_size:
            remaining = [ind.copy() for ind in union if ind.raw_fitness >= 1.0]
            remaining.sort(key=lambda x: x.fitness)
            remaining = unique_by_objectives(remaining)
            need = self.archive_size - len(archive_candidates)
            archive_candidates.extend(remaining[:need])
            return Archive(archive_candidates)
        if len(archive_candidates) == self.archive_size:
            return Archive(archive_candidates)
        truncated = self.truncate_archive(archive_candidates, self.archive_size)
        return Archive(truncated)

    def truncate_archive(
        self, archive_inds: List[Individual], target_size: int
    ) -> List[Individual]:
        """Preserve diversity by repeatedly removing the most crowded individual."""
        inds = archive_inds[:]
        while len(inds) > target_size:
            n = len(inds)
            distance_lists = []
            for i in range(n):
                distances = []
                for j in range(n):
                    if i == j:
                        continue
                    d = euclidean_distance(inds[i].objectives, inds[j].objectives)
                    distances.append(d)
                distances.sort()
                distance_lists.append(distances)
            remove_idx = None
            best_signature = None
            for i, dlist in enumerate(distance_lists):
                signature = tuple(dlist)
                if best_signature is None or signature < best_signature:
                    best_signature = signature
                    remove_idx = i
            inds.pop(remove_idx)
        return inds

    def tournament_select_one(self, archive: Archive) -> Individual:
        """Binary tournament selection; prefer lower fitness."""
        if archive.size() == 1:
            return archive.individuals[0]
        idxs = self.rng.choice(
            len(archive.individuals), size=self.tournament_size, replace=False
        )
        candidates = [archive.individuals[int(i)] for i in idxs]
        candidates.sort(key=lambda ind: ind.fitness)
        return candidates[0]

    def mating_selection(self, archive: Archive) -> List[Individual]:
        mating_pool = []
        while len(mating_pool) < self.pop_size:
            mating_pool.append(self.tournament_select_one(archive))
        return mating_pool

    def crossover(self, parent1, parent2):
        if self.rng.random() <= self.crossover_prob:
            c1, c2 = self.crossover_manager.apply_one(parent1, parent2, self.rng)
            return (c1, c2)
        return (clone_solution(parent1), clone_solution(parent2))

    def mutate(self, solution):
        if self.rng.random() <= self.mutation_prob:
            return self.mutation_manager.apply_all(solution, self.rng)
        return clone_solution(solution)

    def reproduce(self, mating_pool: List[Individual]) -> List[Individual]:
        offspring: List[Individual] = []
        order = list(range(len(mating_pool)))
        self.rng.shuffle(order)
        for k in range(0, len(order), 2):
            p1 = mating_pool[order[k]].solution
            if k + 1 < len(order):
                p2 = mating_pool[order[k + 1]].solution
            else:
                p2 = mating_pool[order[0]].solution
            child_sol1, child_sol2 = self.crossover(p1, p2)
            child_sol1 = self.mutate(child_sol1)
            child_sol2 = self.mutate(child_sol2)
            offspring.append(Individual(solution=child_sol1))
            if len(offspring) < self.pop_size:
                offspring.append(Individual(solution=child_sol2))
            if len(offspring) >= self.pop_size:
                break
        return offspring[: self.pop_size]

    def initialize_population(self) -> List[Individual]:
        population = self.initializer.initialize_population(self.rng)
        population = [Individual(solution=sol) for sol in population]
        self.evaluate_population(population)
        return population

    def step(self, population: List[Individual], archive: Archive, generation: int):
        union = [ind.copy() for ind in population] + [
            ind.copy() for ind in archive.individuals
        ]
        self.assign_fitness(union)
        archive = self.environmental_selection(union)
        mating_pool = self.mating_selection(archive)
        offspring = self.reproduce(mating_pool)
        self.evaluate_population(offspring)
        if self.verbose:
            objs = [ind.objectives for ind in archive.individuals]
            _best_f1 = min((obj[0] for obj in objs)) if objs else float("nan")
            _best_f2 = min((obj[1] for obj in objs)) if objs else float("nan")
        return (offspring, archive)

    def run(self) -> Archive:
        population = self.initialize_population()
        archive = Archive([])
        for gen in range(1, self.max_generations + 1):
            population, archive = self.step(population, archive, gen)
        union = [ind.copy() for ind in population] + [
            ind.copy() for ind in archive.individuals
        ]
        self.assign_fitness(union)
        archive = self.environmental_selection(union)
        final_nd = get_nondominated(archive.individuals)
        if self.verbose:
            print(f"Done. Final archive size = {archive.size()}")
            print(f"Final nondominated size = {len(final_nd)}")
        return Archive(final_nd)
