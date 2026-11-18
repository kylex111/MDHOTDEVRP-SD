"""Select a granular-ball method; customer clusters and balls are modeled separately."""

from __future__ import annotations
import numpy as np
from algorithms.grouping.ball_types import GBClusterResult


def gb_clustering(points, *, method="gb_dbscan", **kwargs) -> GBClusterResult:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must be shape (n, 2), got {pts.shape}")
    if np.any(~np.isfinite(pts)):
        raise ValueError("points contains NaN/Inf")
    if method == "gb_kmeans":
        from .ball_kmeans import cluster_gb_kmeans

        return cluster_gb_kmeans(pts, **kwargs)
    if method == "gb_kmedoids":
        from .ball_kmedoids import cluster_gb_kmedoids

        return cluster_gb_kmedoids(pts, **kwargs)
    if method == "gb_dbscan":
        from .ball_dbscan import cluster_gb_dbscan

        return cluster_gb_dbscan(pts, **kwargs)
    raise ValueError(f"Unknown method: {method}")
