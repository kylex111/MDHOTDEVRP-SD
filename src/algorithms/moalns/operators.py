"""Distance-based route proposals; shared decoding owns demand and recourse."""
import math
import numpy as np


def validate_customers(routes, expected, unassigned=()):
    nodes = [n for route in routes.values() for n in route] + list(unassigned)
    if len(nodes) != len(expected) or len(set(nodes)) != len(nodes) or set(nodes) != set(expected):
        raise ValueError("Every customer must occur exactly once")


def choose_best(options, rng, maximize=False):
    target = (max if maximize else min)(row[0] for row in options)
    ties = [row for row in options if abs(row[0] - target) <= 1e-12]
    return ties[int(rng.integers(len(ties)))]


class RoutingOperators:
    def __init__(self, problem, removal_ratio, guard):
        self.customers = tuple(problem.customers)
        self.count = min(len(self.customers), max(1, math.floor(removal_ratio * len(self.customers))))
        self.distance = np.asarray(problem.distance_matrix)
        self.start = {n: float(self.distance[problem.depots, n].min()) for n in self.customers}
        self.end = {n: float(self.distance[n, problem.depots].min()) for n in self.customers}
        self.guard = guard

    def length(self, route):
        return (self.start[route[0]] + self.end[route[-1]]
                + sum(float(self.distance[a, b]) for a, b in zip(route, route[1:]))) if route else 0.0

    def delta(self, route, node, position):
        if not route:
            return self.start[node] + self.end[node]
        if position == 0:
            return self.start[node] + self.distance[node, route[0]] - self.start[route[0]]
        if position == len(route):
            return self.distance[route[-1], node] + self.end[node] - self.end[route[-1]]
        a, b = route[position - 1:position + 1]
        return self.distance[a, node] + self.distance[node, b] - self.distance[a, b]

    def destroy(self, original, kind, rng):
        self.guard()
        validate_customers(original, self.customers)
        routes = {k: list(v) for k, v in original.items()}
        removed = []
        for _ in range(self.count):
            self.guard()
            nonempty = [k for k, r in routes.items() if r]
            if kind == "random":
                choices = [(k, pos) for k in nonempty for pos in range(len(routes[k]))]
                cid, pos = choices[int(rng.integers(len(choices)))]
            else:
                if kind == "balance":
                    _, cid = choose_best([(self.length(routes[k]), k) for k in nonempty], rng, True)
                    nonempty = [cid]
                elif kind != "worst_distance":
                    raise ValueError(kind)
                choices = []
                for k in nonempty:
                    route = routes[k]
                    for pos, node in enumerate(route):
                        shorter = route[:pos] + route[pos + 1:]
                        choices.append((float(self.delta(shorter, node, pos)), k, pos))
                _, cid, pos = choose_best(choices, rng, True)
            removed.append(routes[cid].pop(pos))
        validate_customers(routes, self.customers, removed)
        return routes, removed

    def repair(self, partial, removed, kind, rng):
        routes = {k: list(v) for k, v in partial.items()}
        unassigned = list(removed)
        validate_customers(routes, self.customers, unassigned)
        while unassigned:
            self.guard()
            if kind == "balance":
                _, cid = choose_best([(self.length(r), k) for k, r in routes.items()], rng)
                nodes = [unassigned[int(rng.integers(len(unassigned)))]]
                slots = [cid]
            elif kind == "greedy":
                nodes, slots = unassigned, list(routes)
            else:
                raise ValueError(kind)
            choices = []
            for node in nodes:
                self.guard()
                for cid in slots:
                    route = routes[cid]
                    choices.extend((float(self.delta(route, node, pos)), node, cid, pos)
                                   for pos in range(len(route) + 1))
            _, node, cid, pos = choose_best(choices, rng)
            routes[cid].insert(pos, node)
            unassigned.remove(node)
        validate_customers(routes, self.customers)
        self.guard()
        return routes
