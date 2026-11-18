"""Node, ball, and boundary operators for granular-ball solutions."""

from __future__ import annotations
import copy
import inspect
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Callable

RoutesByClusters = Dict[int, List[List[int]]]


class Neighborhood:
    """
    Neighborhood operators for GB-structured solutions.

    - Input/Output: RoutesByClusters -> RoutesByClusters
    - Each operator performs ONE move (not enumeration).
    - 'cid' semantics:
        cid=None : pick a feasible cluster randomly (if needed)
        cid=int  : apply move only within that cluster
    """

    def __init__(
        self,
        problem,
        alpha_boundary=0.5,
        removal_ratio: float = 0.3,
        rng: Optional[np.random.Generator] = None,
        min_gb_len: int = 2,
        evaluator: Optional[Callable[[List[int]], Any]] = None,
    ):
        self.problem = problem
        self.total_distance_matrix = self.problem.distance_matrix
        self.cap = self.problem.vehicle_capacity
        self.depot_ids = self.problem.depots
        self.removal_ratio = float(removal_ratio)
        self.alpha_boundary = float(alpha_boundary)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.min_gb_len = int(min_gb_len)
        self.evaluator = evaluator

    @staticmethod
    def deep_copy_routes(routes_by_clusters: RoutesByClusters) -> RoutesByClusters:
        return copy.deepcopy(routes_by_clusters)

    @staticmethod
    def _valid_clusters_for_gb_ops(
        sol: RoutesByClusters, cid: Optional[int], min_gb_count: int
    ) -> List[int]:
        if cid is not None:
            return [cid] if cid in sol and len(sol[cid]) >= min_gb_count else []
        return [c for c, gbs in sol.items() if len(gbs) >= min_gb_count]

    @staticmethod
    def _proximity(idx: int, length: int) -> float:
        """Normalized boundary proximity, increasing toward either end as defined in the paper."""
        if length <= 1:
            return 0.0
        return 1.0 - min(idx, length - 1 - idx) / (length - 1)

    def _pick_cluster(self, cands: List[int]) -> Optional[int]:
        if not cands:
            return None
        return int(self.rng.choice(cands))

    def _cluster_load(self, gbs: List[List[int]]) -> float:
        tot = 0.0
        for gb in gbs:
            for n in gb:
                tot += self.problem.demand[n]
        return tot

    def _gb_load(self, gb: List[int]) -> float:
        return sum((self.problem.demand[n] for n in gb))

    def _any_node_positions(self, gbs: List[List[int]]) -> List[Tuple[int, int]]:
        pos = []
        for gi, gb in enumerate(gbs):
            for pi in range(len(gb)):
                pos.append((gi, pi))
        return pos

    @staticmethod
    def _flatten_gbs(gbs: List[List[int]]) -> List[int]:
        return [n for gb in gbs for n in gb]

    @staticmethod
    def _flatpos_to_gbpos(gbs: List[List[int]], flat_pos: int) -> Tuple[int, int]:
        """Map flat_pos to (gb_idx, pos_in_gb).

        Insert before flat_route[flat_pos]; len(flat_route) denotes the end."""
        if not gbs:
            return (0, 0)
        cur = 0
        for gi, gb in enumerate(gbs):
            L = len(gb)
            if flat_pos <= cur + L:
                return (gi, flat_pos - cur)
            cur += L
        return (len(gbs) - 1, len(gbs[-1]))

    def _get_nearest_depot(self, customer):
        """Return the nearest depot ID using depot rows and the customer column of total_distance_matrix."""
        dist_to_depots = self.total_distance_matrix[self.depot_ids, customer]
        nearest_depot = self.depot_ids[int(np.argmin(dist_to_depots))]
        return nearest_depot

    def _distance_increase(self, route: List[int], node: int, pos: int) -> float:
        """Return the distance increase from inserting node at pos in a full customer route.

        Include endpoint connections using the nearest-depot approximation.
        pos=0 inserts at the start; pos=len(route) appends to the route.
        Here, route is the cluster's flattened vehicle route."""
        M = self.total_distance_matrix
        if not route:
            d0 = self._get_nearest_depot(node)
            new_dist = float(M[d0, node] + M[node, d0])
            return new_dist
        node_depot = self._get_nearest_depot(node)
        if pos == 0:
            first = route[0]
            first_depot = self._get_nearest_depot(first)
            old_dist = float(M[first_depot, first])
            new_dist = float(M[node_depot, node] + M[node, first])
        elif pos == len(route):
            last = route[-1]
            last_depot = self._get_nearest_depot(last)
            old_dist = float(M[last, last_depot])
            new_dist = float(M[last, node] + M[node, node_depot])
        else:
            prev_node = route[pos - 1]
            next_node = route[pos]
            old_dist = float(M[prev_node, next_node])
            new_dist = float(M[prev_node, node] + M[node, next_node])
        return float(new_dist - old_dist)

    def relocate_node_intra_gb(
        self, routes_by_clusters: RoutesByClusters, cid: Optional[int] = None
    ) -> RoutesByClusters:
        """ONE random relocate (1-opt) inside a randomly chosen GB of a cluster."""
        sol = self.deep_copy_routes(routes_by_clusters)
        cands = self._valid_clusters_for_gb_ops(sol, cid, min_gb_count=1)
        c = self._pick_cluster(cands)
        if c is None:
            return sol
        gbs = sol[c]
        gb_indices = [i for i, gb in enumerate(gbs)]
        if not gb_indices:
            return sol
        gi = int(self.rng.choice(gb_indices))
        gb = gbs[gi]
        L = len(gb)
        i = int(self.rng.integers(0, L))
        node = gb.pop(i)
        newL = len(gb)
        j = int(self.rng.integers(0, newL + 1))
        gb.insert(j, node)
        return sol

    def relocate_node_inter_gb(
        self,
        routes_by_clusters: RoutesByClusters,
        cid: Optional[int] = None,
        *,
        protect_min_len: bool = False,
    ) -> RoutesByClusters:
        """
        ONE move: pick a node from one GB and insert into another GB within the SAME cluster (ANY GB pair).
        """
        sol = self.deep_copy_routes(routes_by_clusters)
        cands = self._valid_clusters_for_gb_ops(sol, cid, min_gb_count=2)
        c = self._pick_cluster(cands)
        if c is None:
            return sol
        gbs = sol[c]
        src_candidates = []
        for gi, gb in enumerate(gbs):
            if protect_min_len:
                if len(gb) > self.min_gb_len:
                    src_candidates.append(gi)
            elif len(gb) > 0:
                src_candidates.append(gi)
        if not src_candidates:
            return sol
        src_gi = int(self.rng.choice(src_candidates))
        dst_gi = int(self.rng.choice([i for i in range(len(gbs)) if i != src_gi]))
        src = gbs[src_gi]
        dst = gbs[dst_gi]
        i = int(self.rng.integers(0, len(src)))
        node = src.pop(i)
        j = int(self.rng.integers(0, len(dst) + 1))
        dst.insert(j, node)
        if len(src) == 0:
            gbs.pop(src_gi)
        return sol

    def relocate_node_inter_route(
        self,
        routes_by_clusters: RoutesByClusters,
        *,
        src_cid: Optional[int] = None,
        dst_cid: Optional[int] = None,
        protect_min_len: bool = False,
    ) -> RoutesByClusters:
        """
        ONE move: move 1 node from Route(src cluster) -> Route(dst cluster),
        subject to: load(dst) + demand(node) <= cap.
        """
        sol = self.deep_copy_routes(routes_by_clusters)

        def has_any_node(c: int) -> bool:
            return c in sol and any((len(gb) > 0 for gb in sol[c]))

        if src_cid is not None:
            src_cands = [src_cid] if has_any_node(src_cid) else []
        else:
            src_cands = [c for c in sol.keys() if has_any_node(c)]
        if not src_cands:
            return sol
        src = int(self.rng.choice(src_cands))
        src_gbs = sol[src]
        removable_gb = []
        for gi, gb in enumerate(src_gbs):
            if protect_min_len:
                if len(gb) > self.min_gb_len:
                    removable_gb.append(gi)
            elif len(gb) > 0:
                removable_gb.append(gi)
        if not removable_gb:
            return sol
        src_gi = int(self.rng.choice(removable_gb))
        src_gb = src_gbs[src_gi]
        pos = int(self.rng.integers(0, len(src_gb)))
        node = src_gb[pos]
        dem = self.problem.demand[node]

        def cap_ok(dst: int) -> bool:
            if dst == src:
                return False
            return self._cluster_load(sol[dst]) + dem <= self.cap

        if dst_cid is not None:
            dst_cands = [dst_cid] if dst_cid in sol and cap_ok(dst_cid) else []
        else:
            dst_cands = [c for c in sol.keys() if c != src and cap_ok(c)]
        if not dst_cands:
            return sol
        dst = int(self.rng.choice(dst_cands))
        node = src_gb.pop(pos)
        if len(src_gb) == 0:
            src_gbs.pop(src_gi)
        dst_gbs = sol[dst]
        if len(dst_gbs) == 0:
            dst_gbs.append([])
        dst_gi = int(self.rng.integers(0, len(dst_gbs)))
        dst_gb = dst_gbs[dst_gi]
        ins = int(self.rng.integers(0, len(dst_gb) + 1))
        dst_gb.insert(ins, node)
        return sol

    def swap_node_inter_gb(
        self, routes_by_clusters: RoutesByClusters, cid: Optional[int] = None
    ) -> RoutesByClusters:
        """ONE move: swap two nodes anywhere in the same route/cluster (intra-route)."""
        sol = self.deep_copy_routes(routes_by_clusters)
        cands = self._valid_clusters_for_gb_ops(sol, cid, min_gb_count=1)
        c = self._pick_cluster(cands)
        if c is None:
            return sol
        gbs = sol[c]
        positions = self._any_node_positions(gbs)
        if len(positions) < 2:
            return sol
        (g1, p1), (g2, p2) = self.rng.choice(positions, size=2, replace=False)
        g1, p1, g2, p2 = (int(g1), int(p1), int(g2), int(p2))
        gbs[g1][p1], gbs[g2][p2] = (gbs[g2][p2], gbs[g1][p1])
        return sol

    def reverse_node_intra_gb(
        self, routes_by_clusters: RoutesByClusters, cid: Optional[int] = None
    ) -> RoutesByClusters:
        """
        ONE move: reverse the ENTIRE node order of a GB (full reverse).

        Example:
            [n1, n2, n3, n4] -> [n4, n3, n2, n1]
        """
        sol = self.deep_copy_routes(routes_by_clusters)
        cands = self._valid_clusters_for_gb_ops(sol, cid, min_gb_count=1)
        c = self._pick_cluster(cands)
        if c is None:
            return sol
        gbs = sol[c]
        gb_indices = [i for i, gb in enumerate(gbs) if len(gb) >= 2]
        if not gb_indices:
            return sol
        gi = int(self.rng.choice(gb_indices))
        gbs[gi].reverse()
        return sol

    def relocate_gb_intra_route(
        self, routes_by_clusters: RoutesByClusters, cid: Optional[int] = None
    ) -> RoutesByClusters:
        """ONE random relocate of a GB within a cluster."""
        sol = self.deep_copy_routes(routes_by_clusters)
        cands = self._valid_clusters_for_gb_ops(sol, cid, min_gb_count=2)
        c = self._pick_cluster(cands)
        if c is None:
            return sol
        gbs = sol[c]
        i = int(self.rng.integers(0, len(gbs)))
        gb = gbs.pop(i)
        j = int(self.rng.integers(0, len(gbs) + 1))
        gbs.insert(j, gb)
        return sol

    def relocate_gb_inter_route(
        self,
        routes_by_clusters: RoutesByClusters,
        *,
        src_cid: Optional[int] = None,
        dst_cid: Optional[int] = None,
    ) -> RoutesByClusters:
        """
        ONE move: move 1 GB from Route(src) -> Route(dst),
        subject to: load(dst) + load(GB) <= cap.
        """
        sol = self.deep_copy_routes(routes_by_clusters)
        src_cands = (
            [src_cid]
            if src_cid is not None and src_cid in sol and (len(sol[src_cid]) > 0)
            else []
        )
        if src_cid is None:
            src_cands = [c for c, gbs in sol.items() if len(gbs) > 0]
        if not src_cands:
            return sol
        src = int(self.rng.choice(src_cands))
        src_gbs = sol[src]
        gi = int(self.rng.integers(0, len(src_gbs)))
        gb = src_gbs[gi]
        gb_dem = self._gb_load(gb)

        def cap_ok(dst: int) -> bool:
            if dst == src:
                return False
            return self._cluster_load(sol[dst]) + gb_dem <= self.cap

        if dst_cid is not None:
            dst_cands = [dst_cid] if dst_cid in sol and cap_ok(dst_cid) else []
        else:
            dst_cands = [c for c in sol.keys() if c != src and cap_ok(c)]
        if not dst_cands:
            return sol
        dst = int(self.rng.choice(dst_cands))
        gb = src_gbs.pop(gi)
        sol[dst].insert(int(self.rng.integers(0, len(sol[dst]) + 1)), gb)
        return sol

    def boundary_destroy(
        self, routes_by_clusters: RoutesByClusters, cid: Optional[int] = None
    ) -> Tuple[RoutesByClusters, List[int]]:
        """Remove customers by roulette-wheel sampling of combined ball and cluster boundary probabilities.

        Return new_routes_by_clusters and removed_customers."""
        sol = self.deep_copy_routes(routes_by_clusters)
        cluster_ids = [cid] if cid is not None else list(sol.keys())
        removed_all: List[int] = []
        for c in cluster_ids:
            gbs = sol.get(c, [])
            if not gbs:
                continue
            full = [n for gb in gbs for n in gb]
            if len(full) == 0:
                continue
            num_remove = max(1, int(len(full) * self.removal_ratio))
            pos_in_cluster = {nid: idx for idx, nid in enumerate(full)}
            L_cl = len(full)
            candidates = []
            for gb in gbs:
                L_gb = len(gb)
                for i, nid in enumerate(gb):
                    prox_gb = self._proximity(i, L_gb)
                    j = pos_in_cluster.get(nid, 0)
                    prox_cl = self._proximity(j, L_cl)
                    score = (
                        self.alpha_boundary * prox_gb
                        + (1.0 - self.alpha_boundary) * prox_cl
                    )
                    candidates.append((score, nid))
            if not candidates:
                continue
            scores = np.array([s for s, _ in candidates], dtype=float)
            if scores.sum() > 0:
                probs = scores / scores.sum()
            else:
                probs = np.ones(len(candidates), dtype=float) / len(candidates)
            sample_size = min(num_remove, len(candidates))
            picked_idx = self.rng.choice(
                len(candidates), size=sample_size, replace=False, p=probs
            )
            to_remove = {candidates[i][1] for i in picked_idx}
            new_gbs: List[List[int]] = []
            removed_c: List[int] = []
            for gb in gbs:
                new_gb = [n for n in gb if n not in to_remove]
                if len(new_gb) < self.min_gb_len:
                    new_gbs.append(gb[:])
                else:
                    removed_local = [n for n in gb if n in to_remove]
                    removed_c.extend(removed_local)
                    new_gbs.append(new_gb)
            sol[c] = new_gbs
            removed_all.extend(removed_c)
        return (sol, removed_all)

    def greedy_repair(
        self,
        routes_by_clusters: RoutesByClusters,
        removed_customers: List[int],
        cid: Optional[int] = None,
        sort_by_demand_desc: bool = True,
    ) -> RoutesByClusters:
        """Reinsert removed customers by cheapest insertion, including depot connections.

        Each cluster is one vehicle; enforce its total load capacity.
        Evaluate insertion costs on the flat route, then map positions back into balls.
        A specified cid restricts insertion to that cluster; None searches all clusters
        for the cheapest capacity-feasible position."""
        sol = self.deep_copy_routes(routes_by_clusters)
        if not removed_customers:
            return sol
        if sort_by_demand_desc:
            nodes = sorted(
                removed_customers, key=lambda x: self.problem.demand[x], reverse=True
            )
        else:
            nodes = removed_customers[:]
        cl_load = {c: self._cluster_load(gbs) for c, gbs in sol.items()}
        cluster_ids = [cid] if cid is not None else list(sol.keys())
        if not cluster_ids:
            return sol

        def _new_cluster_id(existing_ids) -> int:
            if not existing_ids:
                return 0
            return int(max(existing_ids)) + 1

        for node in nodes:
            dem = self.problem.demand[node]
            best_delta = float("inf")
            best_c = None
            best_pos = None
            for c in cluster_ids:
                gbs = sol.get(c, [])
                if gbs is None:
                    continue
                if cl_load.get(c, 0.0) + dem > self.cap:
                    continue
                flat = self._flatten_gbs(gbs)
                L = len(flat)
                for pos in range(L + 1):
                    delta = float(self._distance_increase(flat, node, pos))
                    if delta < best_delta:
                        best_delta = delta
                        best_c = c
                        best_pos = pos
            if best_c is None:
                continue
            if best_c not in sol:
                sol[best_c] = []
            if len(sol[best_c]) == 0:
                sol[best_c].append([node])
                cl_load[best_c] = cl_load.get(best_c, 0.0) + dem
                continue
            gi, pos_in_gb = self._flatpos_to_gbpos(sol[best_c], int(best_pos))
            sol[best_c][gi].insert(int(pos_in_gb), node)
            cl_load[best_c] = cl_load.get(best_c, 0.0) + dem
        return sol

    def relocate_boundary(self, routes_by_clusters):
        destroyed, removed = self.boundary_destroy(routes_by_clusters, cid=None)
        repaired = self.greedy_repair(destroyed, removed, cid=None)
        return repaired

    def merge_gb_intra_route(
        self, routes_by_clusters: RoutesByClusters, cid: Optional[int] = None
    ) -> RoutesByClusters:
        """
        Merge two adjacent GBs within the same route (cluster).
        """
        sol = self.deep_copy_routes(routes_by_clusters)
        cands = [cid] if cid is not None else [c for c in sol if len(sol[c]) >= 2]
        if not cands:
            return sol
        c = int(self.rng.choice(cands))
        gbs = sol[c]
        if len(gbs) < 2:
            return sol
        i = int(self.rng.integers(0, len(gbs) - 1))
        gbs[i] = gbs[i] + gbs[i + 1]
        del gbs[i + 1]
        return sol

    def apply(self, op, routes_by_clusters, **kwargs):
        fn = getattr(self, op)
        sig = inspect.signature(fn)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return fn(routes_by_clusters, **filtered_kwargs)

    def list_ops(self) -> List[str]:
        ops = []
        for name in dir(self):
            if name.startswith("_"):
                continue
            if name in ("apply", "list_ops", "deep_copy_routes"):
                continue
            attr = getattr(self, name)
            if callable(attr):
                ops.append(name)
        return sorted(ops)
