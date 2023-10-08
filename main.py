import time

import stim
import correlation


def main():
    # detectors = stim.read_shot_data_file(
    #     path="/home/inm/WorkDir/RustProject/correlation-rs/test_data/rep_code/detectors.b8",
    #     format='b8',
    #     num_detectors=24,
    # )
    # save_dir = "/home/inm/WorkDir/RustProject/correlation-rs/test_data/surface_code"
    # detectors = stim.read_shot_data_file(
    #     path=save_dir + "/detectors.b8",
    #     format="b8",
    #     num_detectors=88,
    # )
    code = "surface_code:rotated_memory_z"
    distance = 3
    rounds = 11
    circuit = stim.Circuit.generated(
        code,
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=0.01,
        after_reset_flip_probability=0.01,
        before_measure_flip_probability=0.01,
        before_round_data_depolarization=0.01,
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    sampler = dem.compile_sampler()
    dets, _, _ = sampler.sample(shots=500000)
    graph = correlation.TannerGraph(dem)
    start = time.perf_counter()
    correlation.cal_high_order_correlations(dets, graph.hyperedges, num_workers=16)
    end = time.perf_counter()
    print(f"Time: {end - start}")


if __name__ == "__main__":
    main()
