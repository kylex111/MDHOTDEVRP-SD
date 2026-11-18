"""Vehicle parameters and GA/ACO parameter types."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ACOParams:
    num_ants: int = 30
    num_iter: int = 100
    alpha: float = 1.0
    beta: float = 2.0
    rho: float = 0.1
    epsilon: float = 0.1
    greedy_prob: float = 0.3
    seed: Optional[int] = None


@dataclass
class GAParams:
    seed: int = 42
    pop_size: int = 80
    num_gen: int = 300
    tournament_k: int = 3
    crossover_rate: float = 0.9
    mutation_rate: float = 0.2
    inversion_rate: float = 0.7
    elite_size: int = 2


ev_params = {"soc_max": 40.0, "vehicle_capacity": 650.0}
