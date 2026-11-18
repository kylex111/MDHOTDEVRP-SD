"""Shared initialization for the main algorithm and baselines, with explicit parameters."""

from __future__ import annotations
from domain.solutions import RouteContext
import numpy as np
from configuration.parameters import GAParams
from algorithms.grouping.balls import gb_clustering
from .ball_ga import plan_gb_order_ga
from .customer_ga import plan_internal_gbs_order_ga, plan_customers_order


def initialize_routes(
    problem,
    clusters,
    seed,
    rng,
    *,
    init_mode="gbga",
    use_gb=True,
    gb_pop_size=30,
    customer_pop_size=60,
    gb_num_gen=100,
    customer_num_gen=100,
    gb_crossover_rate=0.9,
    customer_crossover_rate=0.9,
    gb_mutation_rate=0.2,
    customer_mutation_rate=0.3,
    split_k=0.6,
):
    """Return cid -> customer order, as ball segments or a flat list without balls.

    Members start as cluster-local indices and are mapped to global node indices.
    GA seeds remain seed+1000+cid and seed+2000+cid."""
    assigned = [node for cluster in clusters.values() for node in cluster]
    if sorted(assigned) != sorted(problem.customers):
        raise ValueError("Customer clusters must cover every customer exactly once")
    depot_coords = np.asarray(
        [(problem.nodes[i].x, problem.nodes[i].y) for i in problem.depots], dtype=float
    )
    routes_clusters = {}
    for cid, cluster in clusters.items():
        customer_ids = list(cluster)
        if init_mode == "random_cluster":
            shuffled_customers = customer_ids.copy()
            rng.shuffle(shuffled_customers)
            routes = [shuffled_customers] if use_gb else shuffled_customers
        elif init_mode == "gbga" and (not use_gb):
            params_customer = GAParams(
                pop_size=customer_pop_size,
                num_gen=customer_num_gen,
                tournament_k=3,
                crossover_rate=customer_crossover_rate,
                mutation_rate=customer_mutation_rate,
                inversion_rate=0.7,
                elite_size=2,
                seed=seed + 2000 + cid,
            )
            routes = plan_customers_order(
                problem,
                customer_ids,
                context=RouteContext(depot_mode="depart & return"),
                params=params_customer,
            ).order
        elif init_mode == "gbga":
            points = np.asarray(
                [(problem.nodes[i].x, problem.nodes[i].y) for i in customer_ids],
                dtype=float,
            )
            gbs_info = gb_clustering(
                points,
                method="gb_kmeans",
                split_k=split_k,
                merge=True,
                random_state=seed + cid,
            )
            members = [j for cl in gbs_info.gbs for j in cl]
            if sorted(members) != list(range(len(customer_ids))):
                raise ValueError(
                    f"Granular balls in cluster {cid} must cover each customer once"
                )
            gbs_customer_id = [[customer_ids[j] for j in cl] for cl in gbs_info.gbs]
            params_gb = GAParams(
                pop_size=gb_pop_size,
                num_gen=gb_num_gen,
                tournament_k=3,
                crossover_rate=gb_crossover_rate,
                mutation_rate=gb_mutation_rate,
                inversion_rate=0.7,
                elite_size=2,
                seed=seed + 1000 + cid,
            )
            gb_order_result = plan_gb_order_ga(
                gbs_info.centers, depot_coords, params_gb
            )
            params_customer = GAParams(
                pop_size=customer_pop_size,
                num_gen=customer_num_gen,
                tournament_k=3,
                crossover_rate=customer_crossover_rate,
                mutation_rate=customer_mutation_rate,
                inversion_rate=0.7,
                elite_size=2,
                seed=seed + 2000 + cid,
            )
            routes = plan_internal_gbs_order_ga(
                problem,
                gb_order_result.order,
                gbs_info.centers,
                gbs_customer_id,
                params=params_customer,
            )
        else:
            raise ValueError(f"Unknown init_mode: {init_mode}")
        routes_clusters[cid] = routes
    return routes_clusters
