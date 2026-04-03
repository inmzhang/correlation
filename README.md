# correlation

High-order correlation analysis for `stim` detector error models.

## Install

Published package:

```shell
pip install correlation-analysis
```

Repository workflow:

```shell
uv sync --dev
```

## Use

```python
import stim
import correlation

circuit = stim.Circuit.generated(
    code_task="surface_code:rotated_memory_z",
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
```

For pairwise-only correlations:

```python
result = correlation.cal_2nd_order_correlations(dets)
boundary, edges = result.data
```

## Repo Shortcuts

```shell
uv run python examples/surface_code.py
uv run python examples/repetition_code.py
uv run --with pytest python -m pytest -q
uv run --with ruff python -m ruff check src examples
```

## More

- Algorithm article: `docs/high_order_correlations.pdf`
- High-order example: `examples/surface_code.py`
- Analytic example: `examples/repetition_code.py`
