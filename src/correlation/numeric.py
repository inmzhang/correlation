import collections
import dataclasses
import functools
import itertools
import multiprocessing
from typing import Optional, Iterable, Dict, Set, List, Sequence

import numpy as np
from scipy import optimize

from correlation.utils import HyperEdge, cal_two_points_expects
from correlation.result import CorrelationResult


def cal_high_order_correlations(
    detection_events: np.ndarray,
    hyperedges: Optional[Iterable[HyperEdge]] = None,
    tol: float = 1e-4,
    num_workers: int = 1,
) -> CorrelationResult:
    """Calculate the high order correlation cluster by cluster.

    Args:
        detection_events: The detection events.
        hyperedges: The hyperedges to take into account, excluding the 1st and 2nd
            order edges.
        tol: Tolerance for the legacy `optimize.root()` fallback.
        num_workers: The number of cores to run in parallel.

    Returns:
        The correlation result.
    """
    num_dets = detection_events.shape[1]
    edges = {frozenset({i, j}) for i in range(num_dets) for j in range(i, num_dets)}
    hyperedges = set(edges) if hyperedges is None else set(hyperedges)
    # divide the hyperedges into clusters
    clusters = _divide_into_clusters(hyperedges)
    # calculate the expectations of each hyperedge
    _cal_expectations(detection_events, clusters)
    # solve the clusters
    if num_workers == 1:
        for cluster in clusters:
            cluster.solve(tol=tol)
        solved_clusters = clusters
    else:
        numeric_clusters = []
        for cluster in clusters:
            if not cluster.try_solve_direct():
                numeric_clusters.append(cluster)

        solved_clusters = clusters
        if numeric_clusters:
            with multiprocessing.Pool(min(num_workers, len(numeric_clusters))) as pool:
                solved_numeric = pool.starmap(
                    _solve_cluster,
                    [(cluster, tol) for cluster in numeric_clusters],
                )
            solved_by_root = {cluster.root: cluster for cluster in solved_numeric}
            solved_clusters = [
                solved_by_root.get(cluster.root, cluster)
                for cluster in clusters
            ]
    # adjust the final probabilities
    solved_clusters.sort(key=lambda c: c.weight, reverse=True)
    corr_probs = _adjust_final_probs(solved_clusters, hyperedges)
    return CorrelationResult(corr_probs)


def _divide_into_clusters(hyperedges: Iterable[HyperEdge]) -> List["HyperedgeCluster"]:
    """Divide the hyperedges into clusters."""
    remaining_hyperedges = set(hyperedges)
    hyperedges_sorted = sorted(remaining_hyperedges, key=len, reverse=True)
    clusters = []
    prototypes = {}
    for root in hyperedges_sorted:
        if root not in remaining_hyperedges:
            continue
        cluster = register_cluster(root, prototypes)
        remaining_hyperedges.difference_update(cluster.members)
        clusters.append(cluster)
    # sort the clusters by weights
    clusters.sort(key=lambda c: c.weight, reverse=True)
    return clusters


def _cal_expectations(
    detection_events: np.ndarray, clusters: List["HyperedgeCluster"]
) -> None:
    """Calculate the expectations of each hyperedge."""
    # pre-calculate 2-point expectations using
    # matrix multiply to reduce overhead
    expect_ixj = cal_two_points_expects(detection_events)
    num_dets = detection_events.shape[1]

    expectations = {}
    for i in range(num_dets):
        expectations[frozenset({i})] = expect_ixj[i, i].item()
        for j in range(i):
            expectations[frozenset({i, j})] = expect_ixj[i, j].item()
    for cluster in clusters:
        cluster.register_expectation(detection_events, expectations)


def _adjust_final_probs(
    clusters: List["HyperedgeCluster"],
    hyperedges: Iterable[HyperEdge],
) -> Dict[HyperEdge, float]:
    """Adjust the final correlation probabilities."""
    # cluster roots need not be adjusted
    corr_probs = {c.root: c.solved_probs[c.root] for c in clusters}
    max_weight = clusters[0].weight
    weight_to_adjust = max_weight - 1
    while weight_to_adjust > 0:
        collected_probs = collections.defaultdict(list)
        # all clusters with weight greater than weight_to_adjust
        cluster_related = [
            cluster for cluster in clusters if cluster.weight > weight_to_adjust
        ]
        # adjust the probability of hyperedges with weight
        # weight_to_adjust in each clusters by the probability
        # of the hyperedges with weight greater than that
        for cluster in cluster_related:
            for hyperedge in cluster.with_weight(weight_to_adjust):
                prob_this = cluster.solved_probs[hyperedge]
                supersets = [
                    h for h in corr_probs if hyperedge == h & cluster.root and h not in cluster
                ]
                probs_for_adjust = [corr_probs[h] for h in supersets]
                p_sum = functools.reduce(lambda p, q: p + q - 2 * p * q, probs_for_adjust, 0.0)
                prob_adjusted = (prob_this - p_sum) / (1 - 2*p_sum)
                collected_probs[hyperedge].append(prob_adjusted)
        # average the probabilities of the same hyperedge in different clusters
        collected_probs_mean = {
            hyperedge: np.mean(probs, dtype=np.float64)
            for hyperedge, probs in collected_probs.items()
        }
        corr_probs.update(collected_probs_mean)
        weight_to_adjust -= 1

    # discard those unconcerned hyperedges
    return {
        hyperedge: prob
        for hyperedge, prob in corr_probs.items()
        if hyperedge in hyperedges
    }


@dataclasses.dataclass
class HyperedgeCluster:
    """A cluster of hyperedges to be solved."""
    root: HyperEdge
    members: List[HyperEdge]
    prototype: "ClusterPrototype"
    expectations: List[float] = dataclasses.field(default_factory=list)
    solved_probs: Dict[HyperEdge, float] = dataclasses.field(default_factory=dict)

    @property
    def weight(self) -> int:
        """Return the weight of the cluster."""
        return len(self.root)

    def register_expectation(
        self,
        detection_events: np.ndarray,
        expectations: Dict[HyperEdge, float],
    ) -> None:
        """Calculate the expectation values of each hyperedge from the detection events. """
        for i, hyperedge in enumerate(self.members):
            prob = expectations.get(hyperedge)
            if prob is None:  # pragma: no cover
                prob = np.mean(
                    np.prod(
                        detection_events[:, list(hyperedge)], axis=1, dtype=np.float64
                    ),
                )
                expectations[hyperedge] = prob
            self.expectations.append(prob)

    def __contains__(self, item: HyperEdge):
        return item in self.members

    def with_weight(self, weight: int) -> List[HyperEdge]:
        """Return the hyperedges with the given weight in the cluster."""
        return [hyperedge for hyperedge in self.members if len(hyperedge) == weight]

    def solve(self, tol: float):
        """Solve the cluster."""
        probs = self.prototype.solve_probs(self.expectations, tol)
        self.solved_probs = {
            edge: prob for edge, prob in zip(self.members, probs)
        }

    def try_solve_direct(self) -> bool:
        """Try the closed-form solver and return whether it succeeded."""
        probs = self.prototype.try_solve_probs_direct(self.expectations)
        if probs is None:
            return False
        self.solved_probs = {
            edge: prob for edge, prob in zip(self.members, probs)
        }
        return True
            

def _solve_cluster(cluster: HyperedgeCluster, tol: float):
    """Helper function to make it possible to use multiprocessing."""
    cluster.solve(tol=tol)
    return cluster


class ClusterPrototype:
    """Helper class for cluster computation cache."""
    def __init__(self, size: int) -> None:
        root = frozenset(range(size))
        self.members = list(frozenset(h) for h in powerset(root))
        num_members = len(self.members)
        self.intersections = None
        self.supersets = None
        self.moment_transform = np.zeros((num_members, num_members), dtype=np.float64)
        self.parity_solve_matrix = np.zeros((num_members, num_members), dtype=np.float64)
        for i, target in enumerate(self.members):
            for j, source in enumerate(self.members):
                if source.issubset(target):
                    self.moment_transform[i, j] = (-2.0) ** len(source)
                if len(target & source) % 2 == 1:
                    self.parity_solve_matrix[i, j] = 1.0
        self.parity_solve_matrix = np.linalg.inv(self.parity_solve_matrix)

    def calc_prob(self, vrs: Sequence[float], expectations: List[float]) -> List[float]:
        self._init_numeric_solver_cache()
        eqs = []
        for i, _ in enumerate(self.members):
            intersection = self.intersections[i]
            superset = self.supersets[i]
            eq = -expectations[i]
            for select in superset:
                p = 1.0
                for j, h in enumerate(self.members):
                    if h not in intersection:
                        continue
                    pj = vrs[j]
                    if h in select:
                        p *= pj
                    else:
                        p *= (1.0 - pj)
                eq += p
            eqs.append(eq)
        return eqs

    def solve_probs(self, expectations: Sequence[float], tol: float) -> np.ndarray:
        """Solve the cluster probabilities from the measured expectations."""
        direct_probs = self.try_solve_probs_direct(expectations)
        if direct_probs is not None:
            return direct_probs

        expectations_array = np.asarray(expectations, dtype=np.float64)
        return self._solve_probs_numerically(expectations_array, tol)

    def try_solve_probs_direct(
        self,
        expectations: Sequence[float],
    ) -> Optional[np.ndarray]:
        """Try the closed-form parity-moment solve."""
        expectations_array = np.asarray(expectations, dtype=np.float64)
        moments = 1.0 + self.moment_transform @ expectations_array
        if np.any(moments <= 0.0) or not np.all(np.isfinite(moments)):
            return None

        log_q = self.parity_solve_matrix @ np.log(moments)
        probs = 0.5 * (1.0 - np.exp(log_q))
        if not np.all(np.isfinite(probs)):
            return None
        return probs

    def _init_numeric_solver_cache(self) -> None:
        """Build the combinatorial structures used by the legacy root solver."""
        if self.intersections is not None and self.supersets is not None:
            return

        self.intersections = []
        self.supersets = []
        for hyperedge in self.members:
            intersection = [h for h in self.members if h & hyperedge]
            self.intersections.append(intersection)
            self.supersets.append([
                select
                for select in powerset(intersection)
                if hyperedge.issubset(symmetric_difference(select))
            ])

    def _solve_probs_numerically(
        self,
        expectations: np.ndarray,
        tol: float,
    ) -> np.ndarray:
        """Legacy root solve kept as a fallback for ill-conditioned samples."""
        def equations(vrs):
            eqs = self.calc_prob(vrs, expectations)
            return np.asarray(eqs, dtype=np.float64)

        init_vrs = np.zeros(len(self.members), dtype=np.float64)
        solution = optimize.root(equations, init_vrs, options={"xtol": tol})
        return solution.x

    def instantiate(self, root: HyperEdge) -> HyperedgeCluster:
        """Instantiate a cluster from the prototype."""
        mapping = sorted(root)
        members = [frozenset(mapping[i] for i in hyperedge) for hyperedge in self.members]
        cluster = HyperedgeCluster(root, members, prototype=self)
        return cluster
        

def register_cluster(
    root: HyperEdge,
    prototypes: Dict[int, ClusterPrototype],
) -> HyperedgeCluster:
    """Register a new cluster."""
    size = len(root)
    proto = prototypes.get(size)
    if proto is None:
        proto = ClusterPrototype(size)
        prototypes[size] = proto
    cluster = proto.instantiate(root)
    return cluster


def powerset(iterable: Iterable):
    """powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)

    Reference: https://docs.python.org/3/library/itertools.html#itertools-recipes
    """
    s = list(iterable)  # allows duplicate elements
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(1, len(s) + 1)
    )


def symmetric_difference(iterable_set: Iterable[Set[int]]) -> Set[int]:
    def sym_diff(a: Set[int], b: Set[int]):
        return a.symmetric_difference(b)

    return functools.reduce(sym_diff, iterable_set)


def safe_logistic(x):
    if x < -100:
        return 0.0
    elif x > 100:
        return 1.0
    else:
        return 1 / (1 + np.exp(-x))
