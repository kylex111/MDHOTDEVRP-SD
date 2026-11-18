"""Perturb the current plan once using neighborhood level k."""

from __future__ import annotations
import copy
from typing import Dict, List

RoutesByClusters = Dict[int, List[List[int]]]


class Shaker:
    def __init__(self, neighborhood, NS):
        self.N = neighborhood
        self.NS = list(NS)

    def shake(self, routes_by_clusters, k, cid=None):
        sol = copy.deepcopy(routes_by_clusters)
        if not self.NS:
            return sol
        kk = max(1, min(int(k), len(self.NS)))
        op = self.NS[kk - 1]
        sol = self.N.apply(op, sol, cid=cid, allow_cross_cluster=True)
        return sol
