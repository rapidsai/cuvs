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

### Convenience API (recommended)

```python
from cuvs_bench_elastic import run_build, run_search, run_benchmark

# Build only
build_results = run_build(
    dataset="test-data",
    dataset_path="./datasets",
    host="localhost",
    port=9200,
)

# Search only (requires existing index)
search_results = run_search(
    dataset="test-data",
    dataset_path="./datasets",
    host="localhost",
    port=9200,
)

# Build and search
results = run_benchmark(
    dataset="test-data",
    dataset_path="./datasets",
    host="localhost",
    port=9200,
)

# With authentication
results = run_benchmark(
    dataset="test-data",
    dataset_path="./datasets",
    host="localhost",
    port=9200,
    username="elastic",
    password="password",
)
```

### Orchestrator API

```python
from cuvs_bench_elastic import ELASTIC, register
from cuvs_bench.orchestrator import BenchmarkOrchestrator

register()
orch = BenchmarkOrchestrator(backend_type=ELASTIC)
results = orch.run_benchmark(
    dataset="test-data",
    dataset_path="./datasets",
    host="localhost",
    port=9200,
    basic_auth="elastic:password",  # or username="elastic", password="password"
    algorithms="test",
    build=True,
    search=True,
)
```

## Configuration

Build params (index_options): `type`, `m`, `ef_construction`.  
Search params (knn): `num_candidates`, `vector_field`.

See `config/algos/elastic.yaml` for parameter groups.
