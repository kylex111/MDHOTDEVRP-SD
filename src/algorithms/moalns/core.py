"""Pareto acceptance and segmented operator adaptation from Li et al. (2025)."""
import numpy as np
from algorithms.search.pareto import dominates, same_point


def accept(current, candidate, thresholds):
    if not dominates(current, candidate):
        return True
    return all(abs(candidate[k] - current[k]) < t
               for k, t in zip(("cost", "ra"), thresholds))


def reward(candidate, archive, accepted, previously_accepted, scores):
    if not accepted or previously_accepted or any(same_point(candidate, s) for s in archive):
        return 0.0
    if any(dominates(candidate, s) for s in archive):
        return float(scores[0])
    if not any(dominates(s, candidate) for s in archive):
        return float(scores[1])
    return float(scores[2])


def cool(thresholds, steps):
    # Equation (13), not the contradictory multiplicative pseudocode line.
    return np.maximum(0.0, np.asarray(thresholds) - np.asarray(steps))


def route_key(routes):
    return tuple((k, tuple(v)) for k, v in sorted(routes.items()))


class AdaptiveSelector:
    """Separate removal/insertion roulettes, with local segment statistics."""

    def __init__(self, size, reaction):
        self.weights = np.ones(size)
        self.scores = np.zeros(size)
        self.uses = np.zeros(size, dtype=int)
        self.total_uses = np.zeros(size, dtype=int)
        self.total_scores = np.zeros(size)
        self.reaction = reaction

    def choose(self, rng):
        return int(rng.choice(len(self.weights), p=self.weights / self.weights.sum()))

    def record(self, index, score):
        self.uses[index] += 1
        self.total_uses[index] += 1
        self.scores[index] += score
        self.total_scores[index] += score

    def update(self):
        used = self.uses > 0
        self.weights[used] = ((1 - self.reaction) * self.weights[used]
                             + self.reaction * self.scores[used] / self.uses[used])
        # Prevent numerical extinction after many zero-reward segments.
        self.weights = np.maximum(self.weights, 1e-12)
        self.scores.fill(0)
        self.uses.fill(0)

    def snapshot(self):
        return {name: getattr(self, name).tolist() for name in
                ("weights", "scores", "uses", "total_uses", "total_scores")}
