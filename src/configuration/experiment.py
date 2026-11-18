"""Experiment configuration, validation, and shared or algorithm-specific settings."""

from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
import math

ALGORITHMS = (
    "spr",
    "without_gb",
    "without_vns",
    "nsga2",
    "nsga3",
    "moead",
    "cmopso",
    "spea2",
    "moalns",
)


def stable_seed(base_seed, instance_id, run_id, role):
    payload = json.dumps(
        [int(base_seed), str(instance_id), int(run_id), str(role)],
        separators=(",", ":"),
    )
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "big")


@dataclass(frozen=True)
class ExperimentConfig:
    algorithm: str = "spr"
    n_runs: int = 30
    base_seed: int = 42
    beta: float = 0.2
    n_scenarios: int = 100
    n_reduced_scenarios: int = 10
    threads_per_run: int = 1
    ra_formula_version: str = "paper_exp_norm_v1"
    ra_tau_min: float | None = None
    ra_safe: float = 0.1
    ra_risk: float = 0.7
    recourse_policy: str = "causal_v1"
    charging_policy: str = "ra_based"
    soc_threshold: float = 0.20
    max_time_sec: float | None = 1200.0
    max_eval: int | None = None
    max_iter: int | None = None
    archive_max_size: int = 200
    init_mode: str = "gbga"
    use_movns: bool = True
    vnd_order: str = "ra_first"
    gb_pop_size: int = 30
    customer_pop_size: int = 60
    gb_num_gen: int = 100
    customer_num_gen: int = 100
    gb_crossover_rate: float = 0.9
    customer_crossover_rate: float = 0.9
    gb_mutation_rate: float = 0.2
    customer_mutation_rate: float = 0.3
    split_k: float = 0.6
    removal_ratio: float = 0.3
    alpha: float = 0.95
    candidates_per_neighborhood: int = 1
    patience: int = 50
    population_size: int = 100
    crossover_probability: float = 0.9
    mutation_probability: float = 0.2
    moalns: dict | None = None

    def vnd_objectives(self, iteration):
        """Return phase objectives; alternating changes on outer MOVNS iterations."""
        if self.vnd_order == "alternating":
            return ("ra", "cost") if iteration % 2 else ("cost", "ra")
        return {
            "ra_first": ("ra", "cost"),
            "cost_first": ("cost", "ra"),
            "ra_only": ("ra", None),
            "cost_only": ("cost", None),
        }[self.vnd_order]

    def validate(self):
        if self.charging_policy not in ("ra_based", "energy_only", "soc_threshold"):
            raise ValueError("Unknown charging policy")
        if not math.isfinite(self.soc_threshold) or not 0 < self.soc_threshold < 1:
            raise ValueError("SOC threshold must be in (0, 1)")
        if self.charging_policy != "ra_based" and self.algorithm not in ("spr", "without_gb", "without_vns"):
            raise ValueError("Charging policy ablations currently use the SPR/MOVNS evaluator")
        if self.vnd_order not in ("ra_first", "cost_first", "alternating", "ra_only", "cost_only"):
            raise ValueError("Unknown VND search order")
        if self.algorithm == "moalns":
            from .moalns import MOALNSParameters
            MOALNSParameters.from_dict(self.moalns)
        if self.algorithm not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        for field in (
            "n_runs",
            "n_scenarios",
            "n_reduced_scenarios",
            "threads_per_run",
            "archive_max_size",
            "gb_pop_size",
            "customer_pop_size",
            "gb_num_gen",
            "customer_num_gen",
            "population_size",
            "patience",
            "candidates_per_neighborhood",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if self.threads_per_run != 1:
            raise ValueError("Use one numerical thread per solver process")
        if min(self.gb_pop_size, self.customer_pop_size) < 3:
            raise ValueError("GA populations must accommodate tournament size 3")
        if not 1 <= self.n_reduced_scenarios <= self.n_scenarios:
            raise ValueError(
                "Reduced scenario count must be within the generated count"
            )
        if not math.isfinite(self.beta) or self.beta < 0:
            raise ValueError("beta must be finite and nonnegative")
        for field in (
            "gb_crossover_rate",
            "customer_crossover_rate",
            "gb_mutation_rate",
            "customer_mutation_rate",
            "crossover_probability",
            "mutation_probability",
        ):
            value = getattr(self, field)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{field} must lie in [0, 1]")
        for field in ("split_k", "removal_ratio"):
            value = getattr(self, field)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{field} must lie in (0, 1]")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")
        if (
            self.ra_formula_version != "paper_exp_norm_v1"
            or self.recourse_policy != "causal_v1"
        ):
            raise ValueError("Current experiments use paper_exp_norm_v1 and causal_v1")
        if self.ra_tau_min is not None and (
            not math.isfinite(self.ra_tau_min) or self.ra_tau_min <= 0
        ):
            raise ValueError("ra_tau_min must be positive minutes")
        if not 0 <= self.ra_safe < self.ra_risk <= 1:
            raise ValueError("Require 0 <= ra_safe < ra_risk <= 1")
        if self.init_mode not in ("gbga", "random_cluster"):
            raise ValueError("Unknown initialization mode")
        for field in ("max_eval", "max_iter"):
            value = getattr(self, field)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"{field} must be a positive integer or null")
        if self.max_time_sec is not None and (
            not math.isfinite(self.max_time_sec) or self.max_time_sec <= 0
        ):
            raise ValueError("max_time_sec must be positive or null")
        if self.algorithm in ("ccp", "spea2") and self.max_iter is None:
            raise ValueError(
                "CCP and SPEA require an explicit max_iter; they do not use the SPR time budget"
            )
        if self.algorithm not in ("ccp", "spea2") and all(
            (v is None for v in (self.max_time_sec, self.max_eval, self.max_iter))
        ):
            raise ValueError("At least one stopping budget is required")
        return self

    def require_ra_parameters(self):
        if self.ra_tau_min is None:
            raise ValueError("Set ra_tau_min explicitly (minutes)")
        return self

    def as_dict(self):
        return asdict(self)

    @property
    def fingerprint(self):
        raw = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()

    def seeds(self, instance_id, run_id):
        return {
            role: stable_seed(self.base_seed, instance_id, run_id, role)
            for role in ("algorithm", "search")
        }

    def initialization_kwargs(self):
        return {
            name: getattr(self, name)
            for name in (
                "init_mode",
                "gb_pop_size",
                "customer_pop_size",
                "gb_num_gen",
                "customer_num_gen",
                "gb_crossover_rate",
                "customer_crossover_rate",
                "gb_mutation_rate",
                "customer_mutation_rate",
                "split_k",
            )
        }


def load_config(path):
    with open(path, encoding="utf-8-sig") as stream:
        return ExperimentConfig(**json.load(stream)).validate()
