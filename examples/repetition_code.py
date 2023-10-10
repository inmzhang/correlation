"""Example for analytical correlation solver for repetition code."""
import stim

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
    print("Boundary correlations:")
    print(f"Calculated: {bdy}")
    print(f"Ideal: {bdy_ideal}")
    print("Edge correlations:")
    print(f"Calculated: {edges}")
    print(f"Ideal: {edges_ideal}")

    
if __name__ == '__main__':
    main()
