"""Result statistics, shared quality metrics, and uncertainty summaries."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import stats


def nondominated(points, eps=1e-09):
    values = np.asarray(points, dtype=float).reshape(-1, 2)
    if not np.isfinite(values).all():
        raise ValueError("Objective values must be finite")
    unique = []
    for p in values:
        if not any((np.max(np.abs(p - q)) <= eps for q in unique)):
            unique.append(p)
    return np.asarray(
        [
            p
            for p in unique
            if not any((np.all(q <= p + eps) and np.any(q < p - eps) for q in unique))
        ]
    ).reshape(-1, 2)


@dataclass
class QualityContext:
    lower: list
    upper: list
    reference_front: list
    hv_reference: tuple = (1.1, 1.1)
    source: str = "pooled_empirical_front"
    eps: float = 1e-09

    @classmethod
    def from_archives(cls, archives, source="pooled_empirical_front"):
        pool = np.asarray([p for a in archives for p in a], dtype=float).reshape(-1, 2)
        if len(pool) == 0 or not np.isfinite(pool).all():
            raise ValueError("A nonempty, finite common pool is required")
        lo, hi = (pool.min(axis=0), pool.max(axis=0))
        hi = np.where(hi > lo, hi, lo + 1.0)
        return cls(lo.tolist(), hi.tolist(), nondominated(pool).tolist(), source=source)

    def normalize(self, points):
        lo, hi = (np.asarray(self.lower), np.asarray(self.upper))
        if (
            lo.shape != (2,)
            or hi.shape != (2,)
            or (not np.all(np.isfinite([lo, hi])))
            or np.any(hi <= lo)
        ):
            raise ValueError("Invalid common normalization bounds")
        return (np.asarray(points, dtype=float).reshape(-1, 2) - lo) / (hi - lo)

    def evaluate(self, points):
        front = nondominated(points, self.eps)
        if len(front) == 0:
            return {
                "HV": 0.0,
                "IGD_plus": None,
                "PFS": 0,
                "status": "empty_feasible_front",
            }
        normalized = self.normalize(front)
        ref = np.asarray(self.hv_reference, dtype=float)
        if (
            ref.shape != (2,)
            or not np.isfinite(ref).all()
            or np.any(normalized > ref + self.eps)
        ):
            raise ValueError(
                "Point exceeds common HV reference; revise the common context, never clip per method"
            )
        points_sorted = normalized[np.argsort(normalized[:, 0])]
        hv, last_y = (0.0, ref[1])
        for x, y in points_sorted:
            hv += (ref[0] - x) * max(0.0, last_y - y)
            last_y = min(last_y, y)
        reference = self.normalize(self.reference_front)
        if len(reference) == 0:
            raise ValueError("IGD+ needs a nonempty empirical reference front")
        distances = np.linalg.norm(
            np.maximum(normalized[None, :, :] - reference[:, None, :], 0), axis=2
        )
        return {
            "HV": float(hv),
            "IGD_plus": float(distances.min(axis=1).mean()),
            "PFS": len(front),
            "status": "ok",
        }


def describe_runs(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or (not np.isfinite(values).all()):
        raise ValueError("Pass one finite scalar per independent unit")
    n = len(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else None
    half = (
        float(stats.t.ppf((1 + confidence) / 2, n - 1) * sd / np.sqrt(n))
        if n > 1
        else None
    )
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    return {
        "n": n,
        "mean": mean,
        "std": sd,
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "confidence": confidence,
        "mean_ci": None if half is None else [mean - half, mean + half],
        "ci_method": "Student-t interval for the mean; assess assumptions",
    }


def paired_comparison(a, b, pair_ids):
    a, b = (np.asarray(a, float), np.asarray(b, float))
    if a.ndim != 1 or a.shape != b.shape or len(pair_ids) != len(a) or (len(a) < 2):
        raise ValueError("Need at least two explicitly matched units")
    if len(set(pair_ids)) != len(pair_ids) or not np.isfinite([a, b]).all():
        raise ValueError("Pair IDs must be unique and values finite")
    difference = a - b
    nonzero = difference[difference != 0]
    if len(nonzero):
        ranks = stats.rankdata(np.abs(nonzero))
        effect = float(
            (ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum()
        )
        p = float(
            stats.wilcoxon(
                difference, zero_method="wilcox", alternative="two-sided"
            ).pvalue
        )
    else:
        effect, p = (0.0, 1.0)
    return {
        "test": "paired_Wilcoxon",
        "n_pairs": len(a),
        "p_raw": p,
        "rank_biserial_a_minus_b": effect,
        "median_difference": float(np.median(difference)),
    }


def holm_adjust(p_values):
    p = np.asarray(p_values, float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid p values")
    order = np.argsort(p)
    result = np.empty_like(p)
    result[order] = np.minimum(
        1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1))
    )
    return result.tolist()
