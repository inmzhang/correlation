import stim
import numpy as np

import correlation


def test_high_order_equals_naive_in_rep_code():
    circuit = stim.Circuit.generated(
        code_task='repetition_code:memory',
        distance=5,
        rounds=7,
        after_clifford_depolarization=0.01,
        before_round_data_depolarization=0.01,
        before_measure_flip_probability=0.01,
        after_reset_flip_probability=0.01,
    )
    dem = circuit.detector_error_model()
    num_dets = dem.num_detectors
    dets = circuit.compile_detector_sampler().sample(shots=50000)
    hyperedges = {
        frozenset({i, j})
        for i in range(num_dets)
        for j in range(i, num_dets)
    }
    result = correlation.cal_high_order_correlations(dets, hyperedges, tol=1e-4)
    bdy = np.zeros(num_dets, dtype=np.float64)
    edges = np.zeros((num_dets, num_dets), dtype=np.float64)
    for i in range(num_dets):
        bdy[i] = result.get([i])
        for j in range(i):
            edges[i, j] = edges[j, i] = result.get([i, j])

    bdy_ideal, edges_ideal = correlation.correlation_from_detector_error_model(dem)
    np.testing.assert_allclose(bdy, bdy_ideal, atol=1e-2)
    np.testing.assert_allclose(edges, edges_ideal, atol=1e-2)


# def test_cal_high_order_surface_code():
#     circuit = stim.Circuit.generated(
#         code_task='surface_code:rotated_memory_z',
#         distance=3,
#         rounds=2,
#         after_clifford_depolarization=0.01,
#         after_reset_flip_probability=0.01,
#         before_measure_flip_probability=0.01,
#         before_round_data_depolarization=0.01,
#     )
#     dets = circuit.compile_detector_sampler().sample(shots=500000)
#     dem = circuit.detector_error_model(decompose_errors=True)
#     graph = correlation.TannerGraph(dem)
#     result = correlation.cal_high_order_correlations(dets, graph.hyperedges, num_workers=8)
#     prob_from_dem = []
#     prob_from_correlation = []
#     for hyperedge, prob in graph.hyperedge_probs.items():
#         prob_from_dem.append(prob)
#         prob_from_correlation.append(result.get(hyperedge))
#     np.testing.assert_allclose(prob_from_dem, prob_from_correlation, atol=1e-2)
