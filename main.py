import pathlib

import stim

import correlation as cpy


def main():
    data_dir = pathlib.Path("/Users/inm/paper_data/google_qec_d3_5/google_qec3v5_experiment_data/surface_code_bX_d3_r03_center_7_5")
    google_derived_dem_filepath = data_dir / "pij_from_even_for_odd.dem"
    google_derived_dem = stim.DetectorErrorModel.from_file(google_derived_dem_filepath)

    tanner_graph = cpy.TannerGraph(google_derived_dem)
    sampler = google_derived_dem.compile_sampler()
    detectors, _, _ = sampler.sample(500_0000)
    hyperedges = tanner_graph.hyperedges
    # detectors = stim.read_shot_data_file(path=data_dir/"detection_events.b8", format="b8", num_detectors=24)
    # detectors = detectors[::2]

    # correlation_results = cpy.cal_2nd_order_correlations(detectors, hyperedges=tanner_graph.hyperedges)
    correlation_results = cpy.cal_high_order_correlations(detectors, hyperedges, num_workers=8)
    
    # aim = frozenset([17, 5, 7])
    for hyperedge in hyperedges:
        if len(hyperedge) == 1:
        # if 8 in hyperedge:
        # if frozenset([8, 9, 7]).issubset(hyperedge):
            # if any(h != hyperedge and hyperedge.issubset(h) for h in hyperedges):
            #     continue
            print(f"Hyperedge: {hyperedge}, ideal: {tanner_graph.hyperedge_probs.get(hyperedge)}, result: {correlation_results.get(hyperedge)}")
        
    # for k, v in tanner_graph.hyperedge_probs.items():
    #     if len(k) > 2:
    #         print(f"Hyperedge: {k}, ideal: {v}, result: {correlation_results.get(k)}")
    

if __name__ == '__main__':
    main()
    
