"""Shared dominance, archive updates, crowding truncation, and working-solution selection."""

from __future__ import annotations
from typing import Optional
import numpy as np


def pareto_insert(pareto_set, cand, eps=1e-09):
    for sol in pareto_set:
        if dominates(sol, cand, eps) or same_point(sol, cand, eps):
            return (pareto_set, False)
    new_archive = [sol for sol in pareto_set if not dominates(cand, sol, eps)]
    new_archive.append(cand)
    return (new_archive, True)


def truncate_by_crowding(archive, max_size):
    if len(archive) <= max_size:
        return archive
    front = list(range(len(archive)))
    distances = crowding_distance(archive, front)

    def distance_value(index):
        if isinstance(distances, (list, tuple, np.ndarray)):
            return distances[index]
        return distances.get(index, 0.0)

    keep_idx = sorted(front, key=distance_value, reverse=True)[:max_size]
    return [archive[index] for index in keep_idx]


def pick_work_solution(archive, topk=10):
    if not archive:
        return None
    candidates = sorted(archive, key=lambda sol: sol["cost"])[: min(topk, len(archive))]
    idxs = list(range(len(candidates)))
    distances = crowding_distance(candidates, idxs)

    def distance_value(index):
        if isinstance(distances, (list, tuple, np.ndarray)):
            return distances[index]
        return distances.get(index, 0.0)

    return candidates[max(idxs, key=distance_value)]


def pick_work_solution1(archive, rng=None):
    if not archive:
        return None
    rng = rng if rng is not None else np.random.default_rng()
    best_cost = min(archive, key=lambda sol: sol["cost"])
    best_ra = min(archive, key=lambda sol: sol["ra"])
    return best_cost if rng.random() < 0.5 else best_ra


def pick_work_solution2(archive, eps=1e-12):
    if not archive:
        return None
    costs = [sol["cost"] for sol in archive]
    ras = [sol["ra"] for sol in archive]
    cmin, cmax = (min(costs), max(costs))
    rmin, rmax = (min(ras), max(ras))

    def norm(value, lo, hi):
        return 0.0 if abs(hi - lo) < eps else (value - lo) / (hi - lo)

    def score(sol):
        c = norm(sol["cost"], cmin, cmax)
        r = norm(sol["ra"], rmin, rmax)
        return (c * c + r * r) ** 0.5

    return min(archive, key=score)


def build_grid_cells(archive, n_grid=10, eps=1e-12):
    if not archive:
        return {}
    costs = np.asarray([sol["cost"] for sol in archive], dtype=float)
    ras = np.asarray([sol["ra"] for sol in archive], dtype=float)
    cmin, cmax = (float(costs.min()), float(costs.max()))
    rmin, rmax = (float(ras.min()), float(ras.max()))

    def to_bin(value, lo, hi):
        if abs(hi - lo) < eps:
            return 0
        z = (value - lo) / (hi - lo)
        z = min(max(z, 0.0), 1.0)
        idx = int(z * n_grid)
        if idx >= n_grid:
            idx = n_grid - 1
        return idx

    cells = {}
    for index, sol in enumerate(archive):
        cell = (
            to_bin(float(sol["cost"]), cmin, cmax),
            to_bin(float(sol["ra"]), rmin, rmax),
        )
        cells.setdefault(cell, []).append(index)
    return cells


def pick_work_grid(archive, rng=None, n_grid=10, select_pressure=2.0, eps=1e-12):
    if not archive:
        return None
    if len(archive) == 1:
        return archive[0]
    rng = rng if rng is not None else np.random.default_rng()
    cells = build_grid_cells(archive, n_grid=n_grid, eps=eps)
    cell_keys = list(cells.keys())
    densities = np.asarray([len(cells[key]) for key in cell_keys], dtype=float)
    weights = 1.0 / np.power(np.maximum(densities, eps), select_pressure)
    probs = weights / weights.sum()
    chosen_cell = cell_keys[int(rng.choice(len(cell_keys), p=probs))]
    chosen_idx = int(rng.choice(cells[chosen_cell]))
    return archive[chosen_idx]


def pareto_update(archive, new_solutions, eps=1e-06, max_size: Optional[int] = None):
    merged = list(archive)
    for sol in new_solutions:
        merged, _ = pareto_insert(merged, sol, eps=eps)
    if max_size is not None:
        merged = truncate_by_crowding(merged, max_size=max_size)
    return merged


def mo_improvement(archive, new_solutions, eps=1e-06):
    old_points = {(round(sol["cost"], 8), round(sol["ra"], 8)) for sol in archive}
    updated = pareto_update(archive, new_solutions, eps=eps, max_size=None)
    new_points = {(round(sol["cost"], 8), round(sol["ra"], 8)) for sol in updated}
    return new_points != old_points


def archive_stats(archive):
    if not archive:
        return {
            "best_cost": np.nan,
            "best_ra": np.nan,
            "worst_cost": np.nan,
            "worst_ra": np.nan,
            "spread_cost": 0.0,
            "spread_ra": 0.0,
        }
    costs = np.asarray([float(sol["cost"]) for sol in archive], dtype=float)
    ras = np.asarray([float(sol["ra"]) for sol in archive], dtype=float)
    return {
        "best_cost": float(costs.min()),
        "best_ra": float(ras.min()),
        "worst_cost": float(costs.max()),
        "worst_ra": float(ras.max()),
        "spread_cost": float(costs.max() - costs.min()),
        "spread_ra": float(ras.max() - ras.min()),
    }


def dominates(a, b, eps=1e-09):
    no_worse = a["cost"] <= b["cost"] + eps and a["ra"] <= b["ra"] + eps
    strictly_better = a["cost"] < b["cost"] - eps or a["ra"] < b["ra"] - eps
    return no_worse and strictly_better


def same_point(a, b, eps=1e-09):
    if a is None or b is None:
        return False
    return (
        abs(float(a["cost"]) - float(b["cost"])) <= eps
        and abs(float(a["ra"]) - float(b["ra"])) <= eps
    )


def crowding_distance(pop, front, objective_keys=("cost", "ra")):
    dist = {i: 0.0 for i in front}
    if len(front) <= 2:
        for i in front:
            dist[i] = float("inf")
        return dist
    for key in objective_keys:
        front_sorted = sorted(front, key=lambda i: pop[i][key])
        dist[front_sorted[0]] = float("inf")
        dist[front_sorted[-1]] = float("inf")
        min_v = pop[front_sorted[0]][key]
        max_v = pop[front_sorted[-1]][key]
        denom = max_v - min_v if max_v - min_v > 1e-12 else 1e-12
        for idx in range(1, len(front_sorted) - 1):
            prev_v = pop[front_sorted[idx - 1]][key]
            next_v = pop[front_sorted[idx + 1]][key]
            dist[front_sorted[idx]] += (next_v - prev_v) / denom
    return dist


def hypervolume_2d_min(points, ref_point, eps=1e-12):
    """2D hypervolume for minimization points with cost/ra keys or 2-tuples."""
    if not points:
        return 0.0
    rc, rr = (float(ref_point[0]), float(ref_point[1]))
    pts = []
    seen = set()
    for point in points:
        if isinstance(point, dict):
            cost = float(point["cost"])
            ra = float(point["ra"])
        else:
            cost = float(point[0])
            ra = float(point[1])
        if cost <= rc + eps and ra <= rr + eps:
            key = (round(cost, 12), round(ra, 12))
            if key not in seen:
                seen.add(key)
                pts.append((cost, ra))
    if not pts:
        return 0.0
    pts.sort(key=lambda x: (x[0], x[1]))
    hv = 0.0
    prev_ra = rr
    for cost, ra in pts:
        width = max(0.0, rc - cost)
        height = max(0.0, prev_ra - ra)
        hv += width * height
        if ra < prev_ra:
            prev_ra = ra
    return float(hv)
