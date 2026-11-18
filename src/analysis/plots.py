"""Plot routes, granular balls, Pareto fronts, and search history."""

from __future__ import annotations
from typing import Dict, List, Optional
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt


def plot_routes(problem, routes, title="EVRP Routes", annotate=True, save_path=None):
    """Plot EVRP routes from a route dictionary or a list of node lists.

    Each dictionary value is a complete merged route.
    problem provides x, y, depots, css, and customers."""
    if routes is None:
        route_list = []
    elif isinstance(routes, dict):
        route_list = [routes[k] for k in sorted(routes.keys())]
    else:
        route_list = list(routes)
    coords = np.column_stack(
        (np.asarray(problem.x, dtype=float), np.asarray(problem.y, dtype=float))
    )
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"problem.x/y must form (N,2) coords, got {coords.shape}")
    n = coords.shape[0]
    dep_idx = np.asarray(getattr(problem, "depots", []), dtype=int)
    cs_idx = np.asarray(getattr(problem, "css", []), dtype=int)
    cust_idx = np.asarray(getattr(problem, "customers", []), dtype=int)

    def _check_idx(name, idx):
        if idx.size == 0:
            return
        if np.any(idx < 0) or np.any(idx >= n):
            bad = idx[(idx < 0) | (idx >= n)]
            raise IndexError(
                f"{name} contains out-of-range indices: {bad.tolist()} (valid: 0..{n - 1})"
            )

    _check_idx("problem.depots", dep_idx)
    _check_idx("problem.css", cs_idx)
    _check_idx("problem.customers", cust_idx)
    fig, ax = plt.subplots(figsize=(8, 7))
    dep = ax.scatter(
        coords[dep_idx, 0],
        coords[dep_idx, 1],
        s=80,
        marker="s",
        c="#D02C1B",
        label="Depot",
        zorder=3,
    )
    cs = ax.scatter(
        coords[cs_idx, 0],
        coords[cs_idx, 1],
        s=70,
        marker="^",
        c="#25A61F",
        label="CS",
        zorder=3,
    )
    cus = ax.scatter(
        coords[cust_idx, 0],
        coords[cust_idx, 1],
        s=35,
        marker="o",
        c="#3179B5",
        label="Customer",
        zorder=3,
    )
    seen_annot = set()
    for r in route_list:
        if r is None:
            continue
        rr = np.asarray(r, dtype=int)
        if rr.size < 2:
            continue
        if np.any(rr < 0) or np.any(rr >= n):
            continue
        pts = coords[rr]
        ax.plot(pts[:, 0], pts[:, 1], linewidth=2, alpha=0.9, zorder=2)
        if annotate:
            for nid in rr:
                nid_int = int(nid)
                if nid_int in seen_annot:
                    continue
                x, y = coords[nid_int]
                ax.text(
                    x + 0.5,
                    y + 0.8,
                    f"{nid_int}",
                    fontsize=9,
                    ha="center",
                    va="center",
                    color="black",
                    zorder=4,
                )
                seen_annot.add(nid_int)
    ax.set_title(title)
    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.3)
    xmn, xmx = (np.nanmin(coords[:, 0]), np.nanmax(coords[:, 0]))
    ymn, ymx = (np.nanmin(coords[:, 1]), np.nanmax(coords[:, 1]))
    dx = max((xmx - xmn) * 0.05, 1e-06)
    dy = max((ymx - ymn) * 0.05, 1e-06)
    ax.set_xlim(xmn - dx, xmx + dx)
    ax.set_ylim(ymn - dy, ymx + dy)
    ax.legend(handles=[cus, cs, dep], loc="best")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    return (fig, ax)


def plot_clusters_gbs(
    problem,
    clusters,
    title="Clusters + Granular Balls",
    show_nodes=True,
    show_circles=True,
    annotate_gb=False,
    annotate_points=False,
    alpha_points=0.85,
    alpha_circle=0.55,
    save_path=None,
):
    """Plot colored clusters, ball boundaries, and member points together.

    clusters maps cluster IDs to lists of ball member lists.
    problem provides x and y, with optional depots, css, and customers."""
    coords = np.column_stack(
        (np.asarray(problem.x, dtype=float), np.asarray(problem.y, dtype=float))
    )
    n = coords.shape[0]
    dep_idx = np.asarray(getattr(problem, "depots", []), dtype=int)
    cs_idx = np.asarray(getattr(problem, "css", []), dtype=int)
    cust_idx = np.asarray(getattr(problem, "customers", []), dtype=int)
    fig, ax = plt.subplots(figsize=(9, 8))
    handles = []
    if show_nodes:
        if dep_idx.size:
            h = ax.scatter(
                coords[dep_idx, 0],
                coords[dep_idx, 1],
                s=50,
                marker="s",
                label="Depot",
                zorder=4,
            )
            handles.append(h)
        if cs_idx.size:
            h = ax.scatter(
                coords[cs_idx, 0],
                coords[cs_idx, 1],
                s=40,
                marker="^",
                label="CS",
                zorder=4,
            )
            handles.append(h)
        if cust_idx.size:
            h = ax.scatter(
                coords[cust_idx, 0],
                coords[cust_idx, 1],
                s=25,
                marker="o",
                alpha=0.25,
                label="Customer (all)",
                zorder=1,
            )
            handles.append(h)
    cluster_keys = sorted(clusters.keys())
    cmap = cm.get_cmap("tab10", max(len(cluster_keys), 1))
    for ci, cid in enumerate(cluster_keys):
        _color = cmap(ci)
        gbs = clusters[cid]
        cluster_points = sorted({p for gb in gbs for p in gb if 0 <= p < n})
        if not cluster_points:
            continue
        pts = coords[cluster_points]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=25,
            alpha=alpha_points,
            zorder=2,
            label=f"Cluster {cid}",
        )
        for gi, gb in enumerate(gbs):
            gb = [p for p in gb if 0 <= p < n]
            if len(gb) < 2:
                if len(gb) == 1:
                    p = coords[gb[0]]
                    ax.scatter([p[0]], [p[1]], s=80, alpha=0.95, zorder=3)
                    if annotate_points:
                        ax.text(
                            p[0] + 0.3,
                            p[1] + 0.3,
                            str(int(gb[0])),
                            fontsize=8,
                            zorder=5,
                        )
                continue
            P = coords[np.asarray(gb, int)]
            center = P.mean(axis=0)
            radius = np.max(np.linalg.norm(P - center, axis=1))
            if show_circles:
                circ = plt.Circle(
                    (center[0], center[1]),
                    radius,
                    fill=False,
                    linewidth=1,
                    alpha=alpha_circle,
                    zorder=3,
                )
                ax.add_patch(circ)
            if annotate_gb:
                ax.text(
                    center[0],
                    center[1],
                    f"{cid}-{gi}",
                    fontsize=9,
                    ha="center",
                    va="center",
                    zorder=5,
                )
            if annotate_points:
                for nid in gb:
                    x, y = coords[int(nid)]
                    ax.text(x + 0.3, y + 0.3, str(int(nid)), fontsize=7, zorder=5)
    ax.set_title(title)
    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.3)
    xmn, xmx = (np.nanmin(coords[:, 0]), np.nanmax(coords[:, 0]))
    ymn, ymx = (np.nanmin(coords[:, 1]), np.nanmax(coords[:, 1]))
    dx = max((xmx - xmn) * 0.05, 1e-06)
    dy = max((ymx - ymn) * 0.05, 1e-06)
    ax.set_xlim(xmn - dx, xmx + dx)
    ax.set_ylim(ymn - dy, ymx + dy)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.show()
    return (fig, ax)


def plot_clusters_gbs_with_gb_routes(
    problem,
    clusters: Dict[int, List[List[int]]],
    gb_routes: Optional[List[List[int]]] = None,
    title: str = "Clusters + Granular Balls + GB Routes",
    show_nodes: bool = True,
    show_circles: bool = True,
    annotate_gb: bool = False,
    annotate_points: bool = False,
    alpha_points: float = 0.85,
    alpha_circle: float = 0.55,
    route_linewidth: float = 2.0,
    route_alpha: float = 0.9,
    save_path: Optional[str] = None,
):
    """Plot clusters, ball boundaries, member points, and routes between ball centers.

    clusters maps cluster IDs to lists of ball member indices.
    gb_routes lists ball-index orders aligned with sorted cluster IDs.
    For example, [2, 1, 0, 3] visits balls 2, 1, 0, then 3."""
    coords = np.column_stack(
        (np.asarray(problem.x, dtype=float), np.asarray(problem.y, dtype=float))
    )
    n = coords.shape[0]
    dep_idx = np.asarray(getattr(problem, "depots", []), dtype=int)
    cs_idx = np.asarray(getattr(problem, "css", []), dtype=int)
    cust_idx = np.asarray(getattr(problem, "customers", []), dtype=int)
    fig, ax = plt.subplots(figsize=(9, 8))
    if show_nodes:
        if dep_idx.size:
            ax.scatter(
                coords[dep_idx, 0],
                coords[dep_idx, 1],
                s=50,
                marker="s",
                label="Depot",
                zorder=4,
            )
        if cs_idx.size:
            ax.scatter(
                coords[cs_idx, 0],
                coords[cs_idx, 1],
                s=40,
                marker="^",
                label="CS",
                zorder=4,
            )
        if cust_idx.size:
            ax.scatter(
                coords[cust_idx, 0],
                coords[cust_idx, 1],
                s=25,
                marker="o",
                alpha=0.25,
                label="Customer (all)",
                zorder=1,
            )
    cluster_keys = sorted(clusters.keys())
    cmap = cm.get_cmap("tab10", max(len(cluster_keys), 1))
    gb_centers_by_cluster: Dict[int, List[Optional[np.ndarray]]] = {}
    for ci, cid in enumerate(cluster_keys):
        _color = cmap(ci)
        gbs = clusters[cid]
        cluster_points = sorted({p for gb in gbs for p in gb if 0 <= p < n})
        if cluster_points:
            pts = coords[cluster_points]
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                s=25,
                alpha=alpha_points,
                zorder=2,
                label=f"Cluster {cid}",
            )
        centers_list: List[Optional[np.ndarray]] = []
        for gi, gb in enumerate(gbs):
            gb = [p for p in gb if 0 <= p < n]
            if len(gb) == 0:
                centers_list.append(None)
                continue
            P = coords[np.asarray(gb, dtype=int)]
            center = P.mean(axis=0)
            centers_list.append(center)
            if len(gb) < 2:
                pxy = coords[int(gb[0])]
                ax.scatter([pxy[0]], [pxy[1]], s=80, alpha=0.95, zorder=3)
                if annotate_points:
                    ax.text(
                        pxy[0] + 0.3,
                        pxy[1] + 0.3,
                        str(int(gb[0])),
                        fontsize=8,
                        zorder=5,
                    )
                continue
            radius = float(np.max(np.linalg.norm(P - center, axis=1)))
            if show_circles:
                circ = plt.Circle(
                    (center[0], center[1]),
                    radius,
                    fill=False,
                    linewidth=1,
                    alpha=alpha_circle,
                    zorder=3,
                )
                ax.add_patch(circ)
            if annotate_gb:
                ax.text(
                    center[0],
                    center[1],
                    f"{cid}-{gi}",
                    fontsize=9,
                    ha="center",
                    va="center",
                    zorder=5,
                )
            if annotate_points:
                for nid in gb:
                    x, y = coords[int(nid)]
                    ax.text(x + 0.3, y + 0.3, str(int(nid)), fontsize=7, zorder=5)
        gb_centers_by_cluster[cid] = centers_list
    if gb_routes is not None:
        for ci, cid in enumerate(cluster_keys):
            if ci >= len(gb_routes):
                continue
            route = gb_routes[ci]
            if not route:
                continue
            centers_list = gb_centers_by_cluster.get(cid, [])
            if not centers_list:
                continue
            route_centers = []
            for gb_idx in route:
                if 0 <= gb_idx < len(centers_list):
                    c = centers_list[gb_idx]
                    if c is not None:
                        route_centers.append(c)
            if len(route_centers) >= 2:
                route_centers = np.vstack(route_centers)
                ax.plot(
                    route_centers[:, 0],
                    route_centers[:, 1],
                    linewidth=route_linewidth,
                    alpha=route_alpha,
                    zorder=6,
                    color=cmap(ci),
                )
    ax.set_title(title)
    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.3)
    xmn, xmx = (np.nanmin(coords[:, 0]), np.nanmax(coords[:, 0]))
    ymn, ymx = (np.nanmin(coords[:, 1]), np.nanmax(coords[:, 1]))
    dx = max((xmx - xmn) * 0.05, 1e-06)
    dy = max((ymx - ymn) * 0.05, 1e-06)
    ax.set_xlim(xmn - dx, xmx + dx)
    ax.set_ylim(ymn - dy, ymx + dy)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.show()
    return (fig, ax)


def plot_pareto_front(archive, title="Pareto Front", save_path=None, show_line=True):
    """
    Plot publication-quality Pareto front.

    Parameters
    ----------
    archive : list[dict]
        Pareto solutions (each dict must contain 'cost' and 'ra')
    title : str
    save_path : str or None
        If provided, save figure
    show_line : bool
        Whether to connect Pareto points
    """
    costs = [s["cost"] for s in archive]
    ras = [s["ra"] for s in archive]
    pairs = sorted(zip(costs, ras))
    costs, ras = zip(*pairs)
    plt.figure(figsize=(6, 5))
    plt.scatter(costs, ras, s=80, color="tab:red", label="Pareto solutions")
    if show_line:
        plt.plot(costs, ras, linewidth=2)
    plt.xlabel("Total cost", fontsize=12)
    plt.ylabel("Total anxiety (RA)", fontsize=12)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_search_evolution(
    cand_points, archive, title="Search Evolution", save_path=None
):
    """
    Visualize search behaviour.

    Parameters
    ----------
    cand_points : list[(cost, ra)]
    work_history : list[(iter, cost, ra)]
    archive : list[dict]
    """
    xs = [c for c, _ in cand_points]
    ys = [a for _, a in cand_points]
    ax = [s["cost"] for s in archive]
    ay = [s["ra"] for s in archive]
    plt.figure(figsize=(7, 5))
    plt.scatter(xs, ys, s=8, alpha=0.2, label="All candidates")
    plt.scatter(ax, ay, s=80, color="red", label="Pareto archive")
    plt.xlabel("Total cost")
    plt.ylabel("Total anxiety (RA)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.show()
