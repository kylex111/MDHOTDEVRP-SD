"""Generate one-factor tuning configurations and reuse batch scheduling, search, and evaluation."""

from __future__ import annotations
from dataclasses import replace

FACTORS = (
    ("gb_pop_size", (20, 40)),
    ("customer_pop_size", (40, 80)),
    ("gb_num_gen", (50, 150)),
    ("customer_num_gen", (50, 150)),
    ("gb_crossover_rate", (0.5, 0.7)),
    ("customer_crossover_rate", (0.5, 0.7)),
    ("gb_mutation_rate", (0.0, 0.4)),
    ("customer_mutation_rate", (0.1, 0.5)),
    ("split_k", (0.4, 0.8)),
    ("removal_ratio", (0.1, 0.5)),
)


def configurations(base):
    base.validate()
    if base.algorithm != "spr":
        raise ValueError("This factor table applies to SPR")
    result = [("P00", None, base)]
    for factor, alternatives in FACTORS:
        for value in alternatives:
            if value == getattr(base, factor):
                raise ValueError(f"Alternative equals P00: {factor}")
            result.append(
                (
                    f"P{len(result):02d}",
                    {"factor": factor, "value": value},
                    replace(base, **{factor: value}).validate(),
                )
            )
    return result
