# correlation

High-order correlation analysis for detector error models produced by `stim`.

## Solvers

- `correlation.cal_2nd_order_correlations(...)` computes 1st- and 2nd-order correlations analytically.
- `correlation.cal_high_order_correlations(...)` computes higher-order correlations cluster by cluster using a closed-form parity-moment inversion, with a numerical fallback for ill-conditioned sampled moments.
- `correlation.TannerGraph(...)` extracts hyperedges and probabilities from a detector error model.

## Recommended Workflow

This repository includes a checked-in `uv.lock`. Prefer `uv` for reproducible local development and example runs:

```shell
uv sync --dev
uv run python examples/surface_code.py
uv run python examples/repetition_code.py
```

## Installation

```shell
pip install correlation-analysis
```

If you only want the published package, `pip` is sufficient. If you are working from this repository, prefer `uv sync --dev`.

## Quick Start

```python
import numpy as np
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
probs_dem = np.array(
    [graph.hyperedge_probs[hyperedge] for hyperedge in graph.hyperedges],
    dtype=np.float64,
)
probs_num = np.array(
    [result.get(hyperedge) for hyperedge in graph.hyperedges],
    dtype=np.float64,
)

print("max abs diff:", np.max(np.abs(probs_dem - probs_num)))
print("mean abs diff:", np.mean(np.abs(probs_dem - probs_num)))
```

If you only need pairwise correlations, use the analytic solver:

```python
result = correlation.cal_2nd_order_correlations(dets)
boundary, edges = result.data
```

`num_workers=1` is usually sufficient now because the common high-order path is closed form. Additional workers are mainly useful if many clusters fall back to the numerical root solver.

## Documentation And Examples

- Algorithm article: `docs/high_order_correlations.typ`
- Compiled article: `docs/high_order_correlations.pdf`
- High-order example: `examples/surface_code.py`
- Analytic example: `examples/repetition_code.py`

## Development

```shell
uv sync --dev
uv run --with pytest python -m pytest -q
uv run --with ruff python -m ruff check src examples
```
