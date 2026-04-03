"""Example for high-order correlation analysis on a surface-code circuit."""
import stim
import numpy as np

import correlation


def main():
    circuit = stim.Circuit.generated(
        code_task='surface_code:rotated_memory_z',
        distance=3,
        rounds=2,
        after_clifford_depolarization=0.01,
        after_reset_flip_probability=0.01,
        before_measure_flip_probability=0.01,
        before_round_data_depolarization=0.01,
    )
    dets = circuit.compile_detector_sampler().sample(shots=250_000)
    dem = circuit.detector_error_model(decompose_errors=True)
    graph = correlation.TannerGraph(dem)
    result = correlation.cal_high_order_correlations(dets, graph.hyperedges)

    prob_from_dem = np.array(
        [graph.hyperedge_probs[hyperedge] for hyperedge in graph.hyperedges],
        dtype=np.float64,
    )
    prob_from_correlation = np.array(
        [result.get(hyperedge) for hyperedge in graph.hyperedges],
        dtype=np.float64,
    )
    full_diff = np.abs(prob_from_dem - prob_from_correlation)

    num_dets = dem.num_detectors
    pair_hyperedges = {
        frozenset({i}) for i in range(num_dets)
    } | {
        frozenset({i, j}) for i in range(num_dets) for j in range(i)
    }
    pair_numeric = correlation.cal_high_order_correlations(dets, pair_hyperedges)
    pair_analytic = correlation.cal_2nd_order_correlations(dets)
    pair_numeric_bdy = np.array([pair_numeric.get([i]) for i in range(num_dets)], dtype=np.float64)
    pair_analytic_bdy = np.array([pair_analytic.get([i]) for i in range(num_dets)], dtype=np.float64)
    pair_numeric_edges = np.array(
        [pair_numeric.get([i, j]) for i in range(num_dets) for j in range(i)],
        dtype=np.float64,
    )
    pair_analytic_edges = np.array(
        [pair_analytic.get([i, j]) for i in range(num_dets) for j in range(i)],
        dtype=np.float64,
    )

    print("Full high-order result vs DEM")
    print(f"  max abs diff:  {full_diff.max():.6f}")
    print(f"  mean abs diff: {full_diff.mean():.6f}")
    print("Pairwise numeric result vs analytic solver")
    print(f"  boundary max abs diff: {np.max(np.abs(pair_numeric_bdy - pair_analytic_bdy)):.6e}")
    print(f"  edge max abs diff:     {np.max(np.abs(pair_numeric_edges - pair_analytic_edges)):.6e}")


if __name__ == '__main__':
    main()
