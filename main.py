import pathlib
import time
import json

import stim
import correlation


def main():
    save_dir = pathlib.Path(
        "/home/inm/WorkDir/RustProject/correlation-rs/test_data/surface_code"
    )
    dets = stim.read_shot_data_file(
        path=save_dir / "detectors.b8",
        format="b8",
        num_detectors=16,
    )
    # s = time.perf_counter()
    # res = correlation.cal_2nd_order_correlations(dets)
    # e = time.perf_counter()
    # print(f"Analytical time: {(e - s)*1e6}us")
    # print(f"First five elements of analytical boundary: {res.data[0][:5]}")
    # print(f"First five elements of analytical edges: {res.data[1][0, 1:6]}")
    #
    # s = time.perf_counter()
    # res = correlation.cal_high_order_correlations(dets, num_workers=16)
    # e = time.perf_counter()
    # print(f"Numerical time: {(e - s)*1e3}ms")
    # print(f"First five elements of numerical boundary: {[res.get([i]) for i in range(5)]}")
    # print(f"First five elements of numerical edges: {[res.get([0, i]) for i in range(1, 6)]}")

    with open(save_dir / "hyperedges.json", "r") as f:
        save_obj = json.load(f)

    hyperedges = [frozenset(h) for h in save_obj["hyperedges"]]
    analytic_probs = save_obj["probability"]
    s = time.perf_counter()
    res = correlation.cal_high_order_correlations(
        dets, hyperedges, tol=1e-6, num_workers=16
    )
    e = time.perf_counter()
    print(f"Hyperedge solve time: {(e - s)}s")
    show_hyperedges = list(
        (h, p) for h, p in zip(hyperedges, analytic_probs) if len(h) > 2
    )
    print(f"Solved hyperedge probabilities: {[res.get(i[0]) for i in show_hyperedges]}")
    print(f"Analytical hyperedge probabilities: {[i[1] for i in show_hyperedges]}")


if __name__ == "__main__":
    main()
