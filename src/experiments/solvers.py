"""Select a solver from configuration, loading algorithm dependencies on demand."""

from __future__ import annotations


def solve(instance, config, run_id, seeds, checkpoint_sink=None):
    if config.algorithm == "moalns":
        from algorithms.moalns.solver import solve_moalns
        return solve_moalns(instance, config, run_id, seeds["algorithm"], seeds["search"], checkpoint_sink)
    if config.algorithm in ("spr", "without_gb", "without_vns"):
        from algorithms.spr import solve_spr

        return solve_spr(
            str(instance),
            config,
            run_id,
            seeds["algorithm"],
            seeds["search"],
            checkpoint_sink,
        )
    if config.algorithm == "spea2":
        from algorithms.spea.solver import solve_spea

        return solve_spea(instance, config, run_id, seeds["algorithm"], seeds["search"])
    from algorithms.pymoo_solver import solve_pymoo

    return solve_pymoo(
        instance, config, run_id, seeds["algorithm"], seeds["search"], checkpoint_sink
    )
