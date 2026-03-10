# cuvs-bench-elastic

Elasticsearch GPU backend plugin for [cuvs-bench](https://github.com/rapidsai/cuvs).

## Installation

```bash
pip install cuvs-bench[elastic]
```

Or install standalone (requires `cuvs-bench`):

```bash
pip install cuvs-bench-elastic
```

## Usage

Use the `elastic` backend with the cuvs-bench orchestrator:

```python
from cuvs_bench.orchestrator import BenchmarkOrchestrator

orch = BenchmarkOrchestrator(backend_type="elastic")
results = orch.run_benchmark(
    dataset="test-data",
    dataset_path="./datasets",
    host="localhost",
    port=9200,
    basic_auth="elastic:password",
    algorithms="test",
    build=True,
    search=True,
)
```

## Configuration

Build params (index_options): `type`, `m`, `ef_construction`.  
Search params (knn): `num_candidates`, `vector_field`.

See `config/algos/elastic.yaml` for parameter groups.
