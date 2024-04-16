import stim
import numpy as np

import correlation


def test_2nd_order_correlations():
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
    result = correlation.cal_2nd_order_correlations(dets)
    bdy, edges = result.data
    assert edges.shape == (num_dets, num_dets)
    assert bdy.shape == (num_dets, )
    np.testing.assert_equal(edges, edges.T)
    bdy_ideal, edges_ideal = correlation.correlation_from_detector_error_model(dem)
    np.testing.assert_allclose(bdy, bdy_ideal, atol=0.01)
    np.testing.assert_allclose(edges, edges_ideal, atol=0.01)

