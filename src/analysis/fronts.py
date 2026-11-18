"""Nondominated filtering, shared normalization, HV, and IGD+ with caller-supplied references."""

from __future__ import annotations
import json
import numpy as np


def dominates_min(a, b, eps=1e-12):
    """
    Bi-objective minimization domination:
    a dominates b iff
    a_i <= b_i for all i, and strictly < in at least one i.
    """
    return (
        a[0] <= b[0] + eps
        and a[1] <= b[1] + eps
        and (a[0] < b[0] - eps or a[1] < b[1] - eps)
    )


def nondominated_filter(points, eps=1e-12):
    """
    Keep only nondominated points for bi-objective minimization.
    Deduplicate first.
    """
    unique_points = []
    seen = set()
    for p in points:
        c = float(p[0])
        r = float(p[1])
        key = (round(c, 12), round(r, 12))
        if key not in seen:
            seen.add(key)
            unique_points.append((c, r))
    nd = []
    for i, p in enumerate(unique_points):
        dominated = False
        for j, q in enumerate(unique_points):
            if i == j:
                continue
            if dominates_min(q, p, eps=eps):
                dominated = True
                break
        if not dominated:
            nd.append(p)
    return nd


def hypervolume_2d_min(points, ref_point=(1.1, 1.1), eps=1e-12):
    """
    2D hypervolume for minimization with a fixed reference point.
    """
    rx, ry = (float(ref_point[0]), float(ref_point[1]))
    filtered = []
    for c, r in points:
        c = float(c)
        r = float(r)
        if c <= rx + eps and r <= ry + eps:
            filtered.append((c, r))
    if not filtered:
        return 0.0
    nd = nondominated_filter(filtered, eps=eps)
    nd = sorted(nd, key=lambda x: (x[0], x[1]))
    hv = 0.0
    prev_r = ry
    for c, r in nd:
        width = max(rx - c, 0.0)
        height = max(prev_r - r, 0.0)
        hv += width * height
        prev_r = min(prev_r, r)
    return float(hv)


def normalize_points(points, cmin, cmax, rmin, rmax, eps=1e-12):
    """
    Instance-wise min-max normalization.
    """
    dc = max(cmax - cmin, eps)
    dr = max(rmax - rmin, eps)
    normed = []
    for c, r in points:
        cn = (float(c) - cmin) / dc
        rn = (float(r) - rmin) / dr
        normed.append((cn, rn))
    return normed


def igd_plus(reference_front, approx_front):
    """
    IGD+ for minimization.

    For each reference point z in reference_front:
        d^+(z, A) = min_{a in A} sqrt(sum_i max(a_i - z_i, 0)^2)

    IGD+(A, R) = average over z in R of d^+(z, A)

    Smaller is better.
    """
    if not reference_front:
        return np.nan
    if not approx_front:
        return np.inf
    ref_arr = np.asarray(reference_front, dtype=float)
    app_arr = np.asarray(approx_front, dtype=float)
    distances = []
    for z in ref_arr:
        diff = np.maximum(app_arr - z, 0.0)
        d = np.sqrt(np.sum(diff * diff, axis=1))
        distances.append(np.min(d))
    return float(np.mean(distances))


def load_points_from_json(json_path):
    """
    Priority:
    1) pareto_points
    2) archive -> [(cost, ra), ...]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pts = data.get("pareto_points", None)
    if pts is not None:
        clean_pts = []
        for item in pts:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                clean_pts.append((float(item[0]), float(item[1])))
        if clean_pts:
            return (data, clean_pts)
    archive = data.get("archive", None)
    if archive is not None:
        clean_pts = []
        for sol in archive:
            if isinstance(sol, dict) and "cost" in sol and ("ra" in sol):
                clean_pts.append((float(sol["cost"]), float(sol["ra"])))
        if clean_pts:
            return (data, clean_pts)
    raise ValueError(f"No valid pareto_points/archive found in: {json_path}")
