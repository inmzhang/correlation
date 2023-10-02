import collections
import dataclasses
import functools
import itertools
import multiprocessing
from typing import Optional, Iterable, Dict, Set, List, cast

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
    """Calculate the high order correlation numerically.

    Args:
        detection_events: The detection events.
        hyperedges: The hyperedges to take into account, excluding the 1st and 2nd
            order edges.
        tol: Tolerance for the `optimize.root()` subroutine.
        num_workers: The number of cores to run in parallel.

    Returns:
        The correlation result.
    """
    num_dets = detection_events.shape[1]
    edges = {
        frozenset({i, j})
        for i in range(num_dets)
        for j in range(i, num_dets)
    }
    hyperedges = {} if hyperedges is None else set(hyperedges)
    hyperedges.update(edges)
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
        solved_clusters = []
        pool = multiprocessing.Pool(num_workers)
        for cluster in clusters:
            pool.apply_async(_solve_cluster, (cluster, tol), callback=lambda c: solved_clusters.append(c))
        pool.close()
        pool.join()
    # adjust the final probabilities
    solved_clusters.sort(key=lambda c: c.weight, reverse=True)
    corr_probs = _adjust_final_probs(solved_clusters, hyperedges)
    return CorrelationResult(corr_probs)


def _divide_into_clusters(hyperedges: Iterable[HyperEdge]) -> List['HyperedgeCluster']:
    """Divide the hyperedges into clusters."""
    hyperedges_wait_cluster = list(hyperedges)
    hyperedges_wait_cluster.sort(key=len)
    clusters = []
    while hyperedges_wait_cluster:
        hyperedge = hyperedges_wait_cluster.pop()
        cluster = HyperedgeCluster(hyperedge)
        cluster.register(hyperedges_wait_cluster)
        clusters.append(cluster)
    # sort the clusters by weights
    clusters.sort(key=lambda c: c.weight, reverse=True)
    return clusters


def _cal_expectations(
        detection_events: np.ndarray,
        clusters: List['HyperedgeCluster']
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
        cluster.cal_hyperedge_expectations(detection_events, expectations)


def _adjust_final_probs(
        clusters: List['HyperedgeCluster'],
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
            cluster
            for cluster in clusters
            if cluster.weight > weight_to_adjust
        ]
        # adjust the probability of hyperedges with weight
        # weight_to_adjust in each clusters by the probability
        # of the hyperedges with weight greater than that
        for cluster in cluster_related:
            for hyperedge in cluster.with_weight(weight_to_adjust):
                prob_this = cluster.solved_probs[hyperedge]
                supersets = [
                    h for h in corr_probs
                    if hyperedge.issubset(h) and h not in cluster
                ]
                probs_for_adjust = [corr_probs[h] for h in supersets]
                prob_adjusted = functools.reduce(
                    lambda p, q: (p - q) / (1 - 2 * q),
                    probs_for_adjust,
                    prob_this
                )
                collected_probs[hyperedge].append(prob_adjusted)
        # average the probabilities of the same hyperedge in different clusters
        collected_probs_mean = {
            hyperedge: np.mean(probs, dtype=float)
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
    """Dataclass used to store a cluster of hyperedges."""
    root: HyperEdge
    members: List[HyperEdge] = dataclasses.field(default_factory=list)
    expectations: Dict[HyperEdge, float] = dataclasses.field(default_factory=dict)
    solved_probs: Dict[HyperEdge, float] = dataclasses.field(default_factory=dict)

    @property
    def weight(self) -> int:
        """Return the weight of the cluster."""
        return len(self.root)

    def register(self, hyperedges_wait_cluster: List[HyperEdge]):
        """Register the members of the cluster."""
        self.members = list(frozenset(hyperedge) for hyperedge in powerset(self.root))
        self.members.sort(key=len, reverse=True)
        for hyperedge in self.members:
            if hyperedge in hyperedges_wait_cluster:
                hyperedges_wait_cluster.remove(hyperedge)

    def cal_hyperedge_expectations(
            self,
            detection_events: np.ndarray,
            global_expects: Dict[HyperEdge, float],
    ):
        """Calculate the expectation values of each hyperedge
        from the detection events.
        """
        num_shots = detection_events.shape[0]
        dtype_use = 'uint16' if num_shots < 2 * (2 ** 16 - 1) else 'uint32'
        for hyperedge in self.members:
            prob = global_expects.get(hyperedge, None)
            if prob is None:  # pragma: no cover
                prob = np.mean(
                    np.prod(detection_events[:, list(hyperedge)], axis=1, dtype=dtype_use),
                    dtype=float
                )
                global_expects[hyperedge] = cast(float, prob)
            self.expectations[hyperedge] = prob

    def __contains__(self, item: HyperEdge):
        return item in self.members

    def with_weight(self, weight: int) -> List[HyperEdge]:
        """Return the hyperedges with the given weight in the cluster."""
        return [hyperedge for hyperedge in self.members if len(hyperedge) == weight]

    def prob_of_selected_hyperedges(self, selected, all_intersect_with_this, probs):
        """Probability of the selected hyperedges flip simultaneously."""
        prob = 1.0
        for i, hyperedge in enumerate(self.members):
            if hyperedge not in all_intersect_with_this:
                continue
            if hyperedge in selected:
                prob *= probs[i]
            else:
                prob *= (1 - probs[i])
        return prob

    def solve(self, tol: float):
        def equations(vrs):
            eqs = []
            for hyperedge in self.members:
                expect = - self.expectations[hyperedge]
                hyperedges_intersect_with_this = [h for h in self.members if h & hyperedge]
                for selected in powerset(hyperedges_intersect_with_this):
                    sym_diff = symmetric_difference(selected)
                    if hyperedge.issubset(sym_diff):
                        expect += self.prob_of_selected_hyperedges(
                            selected,
                            hyperedges_intersect_with_this,
                            vrs
                        )
                eqs.append(expect)
            return np.array(eqs)

        # solve numerically
        # though weight-2 cluster can be solved analytically
        init_vrs = np.zeros(len(self.members))
        solution = optimize.root(equations, init_vrs, options={'xtol': tol})
        for edge, prob in zip(self.members, solution.x):
            self.solved_probs[edge] = prob


def _solve_cluster(cluster: HyperedgeCluster, tol: float):
    """Helper function to make it possible to use multiprocessing."""
    cluster.solve(tol=tol)
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
