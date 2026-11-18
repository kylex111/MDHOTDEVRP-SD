"""Pack candidate plans and compute structural signatures without running search."""

from __future__ import annotations
from dataclasses import dataclass
import copy


def pack_solution(routes_by_clusters, cend):
    """Deep-copy plans and evaluation results so neighborhood moves cannot mutate archived solutions."""
    return {
        "routes_by_clusters": copy.deepcopy(routes_by_clusters),
        "cost": float(cend["total_cost"]),
        "ra": float(cend.get("total_ra", 0.0)),
        "stats": copy.deepcopy(cend.get("stats", {})),
        "cend": copy.deepcopy(cend),
    }


def freeze_routes(routes_by_clusters):
    """Build stable route signatures for ball-segment and flat customer orders."""
    frozen = []
    for cid, cluster_routes in sorted(routes_by_clusters.items(), key=lambda x: x[0]):
        if len(cluster_routes) == 0:
            frozen_cluster = tuple()
        elif isinstance(cluster_routes[0], int):
            frozen_cluster = tuple(cluster_routes)
        else:
            frozen_cluster = tuple((tuple(route) for route in cluster_routes))
        frozen.append((cid, frozen_cluster))
    return tuple(frozen)


@dataclass
class DepotRoute:
    """Candidate customer order with start/end depots for SPEA; stores no execution state."""

    start_depot_id: int
    end_depot_id: int
    customers: list[int]


@dataclass
class OrderResult:
    """Shared GA/ACO ordering result; cost is a construction heuristic, not final operating cost."""

    order: list[int]
    cost: float
    history: list[float]


@dataclass(frozen=True)
class RouteContext:
    """Connections between the current customer segment, adjacent balls, and depots."""

    depot_mode: str | None = None
    prev_last_customer: int | None = None
    next_gb_center: tuple[float, float] | None = None
