# correlation
High order correlation analysis of detector error models

## Installation

```shell
pip install correlation_analysis
```

## Usage

```python
import stim

import correlation

circuit = stim.Circuit.generated(
    code_task='surface_code:rotated_memory_z',
    distance=3,
    rounds=2,
    after_clifford_depolarization=0.01,
    after_reset_flip_probability=0.01,
    before_measure_flip_probability=0.01,
    before_round_data_depolarization=0.01,
)
dets = circuit.compile_detector_sampler().sample(shots=1_000_000)
dem = circuit.detector_error_model(decompose_errors=True)
graph = correlation.TannerGraph(dem)
result = correlation.cal_high_order_correlations(dets, graph.hyperedges, num_workers=16)
prob_from_dem = []
prob_from_correlation = []
for hyperedge, prob in graph.hyperedge_probs.items():
    prob_from_dem.append(prob)
    prob_from_correlation.append(result.get(hyperedge))

print("Probabilities from detector error model:")
print(prob_from_dem)
print("Probabilities from correlation analysis:")
print(prob_from_correlation)
```