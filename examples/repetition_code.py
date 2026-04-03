"""Example for analytical 2nd-order correlation analysis."""
import stim
import numpy as np

import correlation


def main():
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
    dets = circuit.compile_detector_sampler().sample(shots=500_000)
    result = correlation.cal_2nd_order_correlations(dets)
    bdy, edges = result.data
    bdy_ideal, edges_ideal = correlation.correlation_from_detector_error_model(dem)
    bdy_diff = np.abs(bdy - bdy_ideal)
    edge_diff = np.abs(edges - edges_ideal)
    print("Analytic 2nd-order result vs DEM")
    print(f"  boundary max abs diff:  {bdy_diff.max():.6f}")
    print(f"  boundary mean abs diff: {bdy_diff.mean():.6f}")
    print(f"  edge max abs diff:      {edge_diff.max():.6f}")
    print(f"  edge mean abs diff:     {edge_diff.mean():.6f}")

    
if __name__ == '__main__':
    main()
