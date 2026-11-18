"""Shake archive members, refine each objective, and collect nondominated solutions without experiment I/O."""

from __future__ import annotations
import copy
from domain.solutions import pack_solution, freeze_routes
from .pareto import pareto_insert, truncate_by_crowding


def mo_shake(archive, shaker, evaluator, k, cid=None):
    shaken_set = []
    seen = set()
    for sol in archive:
        base_routes = copy.deepcopy(sol["routes_by_clusters"])
        cand_routes = shaker.shake(base_routes, k=k, cid=cid)
        sig = freeze_routes(cand_routes)
        if sig in seen:
            continue
        seen.add(sig)
        cand_cend = evaluator(cand_routes)
        cand = pack_solution(cand_routes, cand_cend)
        shaken_set.append(cand)
    return shaken_set


def mo_vnd(shaken_set, vnd, obj, NL, cid=None, eps=1e-06, max_size=None):
    vnd_set = []
    for sol in shaken_set:
        final_routes, final_cend, info = vnd.refine_ffs(sol, obj=obj, NL=NL, cid=cid)
        seen_sig = set()
        for cand in info.get("visited", []):
            sig = freeze_routes(cand["routes_by_clusters"])
            if sig in seen_sig:
                continue
            seen_sig.add(sig)
            vnd_set, _ = pareto_insert(vnd_set, cand, eps=eps)
        final_sol = pack_solution(final_routes, final_cend)
        vnd_set, _ = pareto_insert(vnd_set, final_sol, eps=eps)
    if max_size is not None:
        vnd_set = truncate_by_crowding(vnd_set, max_size=max_size)
    return vnd_set
