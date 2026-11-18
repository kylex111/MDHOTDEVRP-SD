"""K-means-based granular-ball splitting and merging."""

from __future__ import annotations
from typing import List
import numpy as np
from sklearn.cluster import KMeans
from algorithms.grouping.ball_types import GranularBall
from algorithms.grouping.ball_types import GBClusterResult


class KMeansCluster:
    def __init__(
        self, points: np.ndarray, split_k: float = 1.0, random_state: int | None = None
    ):
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"points must be shape (n,2), got {pts.shape}")
        if len(pts) and np.any(~np.isfinite(pts)):
            raise ValueError("points contains NaN/Inf")
        if split_k < 0:
            raise ValueError("split_k must be >= 0")
        self.points = pts
        self.split_k = float(split_k)
        if random_state is None:
            random_state = np.random.randint(0, 10**9)
        self.random_state = random_state
        self.threshold = self.split_k * float(np.sqrt(len(self.points)))
        self.gbs: List[GranularBall] = []

    def _split_gb(self, gb: GranularBall) -> List[List[int]]:
        """Split a ball with KMeans(2) and return the two lists of point indices."""
        gb_idx = gb.points_idx
        if len(gb_idx) <= 2:
            return [gb_idx]
        cur_idx = np.asarray(gb_idx, dtype=int)
        pts = self.points[cur_idx]
        km = KMeans(n_clusters=2, random_state=self.random_state, n_init="auto")
        labels = km.fit_predict(pts)
        left = cur_idx[labels == 0].tolist()
        right = cur_idx[labels == 1].tolist()
        return [left, right]

    def generate_gbs(self):
        """Split balls in rounds, starting with one ball.

        Keep balls smaller than the threshold and split the others.
        Stop when the ball count is unchanged; store the result in self.gbs."""
        n = len(self.points)
        self.gbs = []
        if n == 0:
            return
        gbs_list: List[List[int]] = [list(range(n))]
        while True:
            before = len(gbs_list)
            new_list: List[List[int]] = []
            for gb_idx in gbs_list:
                if len(gb_idx) < self.threshold:
                    new_list.append(gb_idx)
                else:
                    tmp_gb = GranularBall(gb_idx, self.points)
                    new_list.extend(self._split_gb(tmp_gb))
            gbs_list = new_list
            after = len(gbs_list)
            if after == before:
                break
        self.gbs = [GranularBall(sorted(set(idx)), self.points) for idx in gbs_list]

    def merge_single_point_gb(self) -> None:
        """Merge singleton balls into the nearest non-singleton ball using pre-merge centers."""
        if not self.gbs:
            return
        singles = [gb for gb in self.gbs if len(gb) == 1]
        others = [gb for gb in self.gbs if len(gb) > 1]
        if not singles or not others:
            return
        other_centers = np.vstack([gb.center for gb in others])
        new_idxs = [list(gb.points_idx) for gb in others]
        for sg in singles:
            p = int(sg.points_idx[0])
            xy = self.points[p]
            j = int(np.argmin(np.linalg.norm(other_centers - xy, axis=1)))
            new_idxs[j].append(p)
        self.gbs = [GranularBall(sorted(set(idx)), self.points) for idx in new_idxs]

    def _merge_contained_gbs(self, merged_gbs):
        """
        merged_gbs: Dict[int, GranularBall]
        return: Dict[int, GranularBall]  (after containment merge)
        """
        centers_radii = {lab: (gb.center, gb.radius) for lab, gb in merged_gbs.items()}
        parents = {lab: lab for lab in merged_gbs}

        def root(lab):
            while parents[lab] != lab:
                parents[lab] = parents[parents[lab]]
                lab = parents[lab]
            return lab

        # Merge each connected containment group once. If A and B contain C, group all three.
        # Do not copy C into both A and B. This also handles containment chains and equal balls.
        for lab, (c, r) in centers_radii.items():
            for other_lab, (oc, or_) in centers_radii.items():
                if lab == other_lab:
                    continue
                d = np.linalg.norm(np.asarray(c) - np.asarray(oc))
                if d + r <= or_:
                    left, right = root(lab), root(other_lab)
                    parents[max(left, right)] = min(left, right)
        new_merged = {}
        for lab, gb in merged_gbs.items():
            new_merged.setdefault(root(lab), []).extend(gb.points_idx)
        out = {
            lab: GranularBall(sorted(idx), self.points)
            for lab, idx in new_merged.items()
        }
        return out

    def merge_gbs(self):
        """
        Combining overlapping or contained balls, using self.gbs only.
        overlap rule: dist(ci,cj) <= ri + rj
        then containment merge
        """
        m = len(self.gbs)
        if m <= 1:
            return self.gbs
        centers = np.vstack([gb.center for gb in self.gbs]).astype(float)
        radii = np.asarray([gb.radius for gb in self.gbs], dtype=float)
        unvisited = list(range(m))
        comp = [-1] * m
        k = -1
        while unvisited:
            p = unvisited[0]
            unvisited.remove(p)
            neighbors = []
            for i in range(m):
                if i == p:
                    continue
                d = np.linalg.norm(centers[i] - centers[p])
                if d <= radii[i] + radii[p]:
                    neighbors.append(i)
            k += 1
            comp[p] = k
            for pi in neighbors:
                if pi in unvisited:
                    unvisited.remove(pi)
                    neighbors_pi = []
                    for j in range(m):
                        if j == pi:
                            continue
                        d_pi = np.linalg.norm(centers[j] - centers[pi])
                        if d_pi <= radii[j] + radii[pi]:
                            neighbors_pi.append(j)
                    for t in neighbors_pi:
                        if t not in neighbors:
                            neighbors.append(t)
                if comp[pi] == -1:
                    comp[pi] = k
        merged_idxs = {}
        for i, lab in enumerate(comp):
            merged_idxs.setdefault(int(lab), []).extend(self.gbs[i].points_idx)
        merged_gbs = {
            lab: GranularBall(sorted(set(idx)), self.points)
            for lab, idx in merged_idxs.items()
        }
        merged_gbs = self._merge_contained_gbs(merged_gbs)
        self.gbs = [merged_gbs[lab] for lab in sorted(merged_gbs.keys())]
        return self.gbs

    def fit(self, merge: bool = True):
        """Generate balls, merge singletons, apply overlap/containment merging, and return groups, centers, and radii."""
        self.generate_gbs()
        self.merge_single_point_gb()
        if merge:
            self.merge_gbs()
        clusters = [sorted(gb.points_idx) for gb in self.gbs]
        # Balls must partition the original customers; route deduplication must not hide membership errors.
        members = [idx for cluster in clusters for idx in cluster]
        if sorted(members) != list(range(len(self.points))):
            raise ValueError("Granular balls must cover every point exactly once")
        centers_arr = (
            np.vstack([gb.center for gb in self.gbs])
            if self.gbs
            else np.zeros((0, 2), dtype=float)
        )
        radii_arr = np.asarray([gb.radius for gb in self.gbs], dtype=float)
        return (clusters, centers_arr, radii_arr)


def cluster_gb_kmeans(
    points: np.ndarray,
    *,
    split_k: float = 1.0,
    merge: bool = False,
    random_state: int = 0,
) -> GBClusterResult:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must be shape (n,2), got {pts.shape}")
    if len(pts) and np.any(~np.isfinite(pts)):
        raise ValueError("points contains NaN/Inf")
    model = KMeansCluster(pts, split_k=split_k, random_state=random_state)
    clusters, centers, radii = model.fit(merge=merge)
    return GBClusterResult(gbs=clusters, centers=centers, radii=radii)
