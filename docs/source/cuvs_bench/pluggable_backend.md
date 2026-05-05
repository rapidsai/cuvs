# Pluggable Backend

cuVS Bench uses a pluggable API so that benchmarks can be run through different execution paths. The default path runs C++ benchmark executables; other backends (e.g. Elasticsearch, Milvus) can be added by implementing the same interface and registering them. Two pieces work together: a **config loader** turns the user's arguments (dataset, algorithms, k, batch_size, and the like) into a structured configuration; a **backend** takes that configuration and runs build and search. Both are registered under a backend type name (e.g. `cpp_gbench`). When `BenchmarkOrchestrator(backend_type="cpp_gbench").run_benchmark(...)` is called, the orchestrator uses the config loader for that type to produce the configuration, then passes it to the backend for that type.

The following shows how the default backend is used:

```python
from cuvs_bench.orchestrator import BenchmarkOrchestrator

orchestrator = BenchmarkOrchestrator(backend_type="cpp_gbench")
results = orchestrator.run_benchmark(
    dataset="deep-image-96-inner",
    algorithms="cuvs_cagra",
    count=10,
    batch_size=10,
    build=True,
    search=True,
)
```

## How a run flows

1. The user calls `orchestrator.run_benchmark(backend_type="...", dataset=..., algorithms=..., count=..., **kwargs)`.

2. The orchestrator looks up the **config loader** for that `backend_type` and calls its **load()** method. The loader reads YAML (or other sources), expands parameter combinations, applies constraints, and returns a **DatasetConfig** and a list of **BenchmarkConfig** (each describing one or more index configs: algorithm, build params, search params).

3. The orchestrator obtains the **backend** for that `backend_type` from the **BackendRegistry** (instantiating it with the config it needs, e.g. executable path, host/port).

4. The orchestrator calls the backend's **build(dataset, indexes, ...)** then **search(dataset, indexes, k, batch_size, ...)**. The backend uses the same config shape that its loader produced.

5. The backend returns **BuildResult** and **SearchResult**; the orchestrator aggregates and returns them.

The config loader and the backend are thus a pair: the loader defines what to run (which algorithms and parameters); the backend defines how it runs (C++ subprocess, HTTP to a service, and so on).

## What the config loader produces

The orchestrator calls the config loader's **load()** method with the same arguments passed to `run_benchmark()` (e.g. `dataset`, `dataset_path`, `algorithms`, `count`, `batch_size`, `groups`, `algo_groups`, and backend-specific options). The loader must return two things:

- **DatasetConfig** – Dataset metadata: `name`, `base_file`, `query_file`, `groundtruth_neighbors_file`, `distance` (e.g. `"euclidean"`), `dims`, and optional `subset_size`. These are used by the orchestrator to build the in-memory `Dataset` and by the backend if it needs file paths.

- **List[BenchmarkConfig]** – Each **BenchmarkConfig** has:
  - **indexes**: a list of **IndexConfig**. Each **IndexConfig** has `name` (e.g. `"my_algo.param1value"`), `algo` (algorithm name), `build_param` (dict of build parameters), `search_params` (list of dicts, one per search parameter combination to benchmark), and `file` (path or identifier where the index is stored).
  - **backend_config**: a dict passed to the backend constructor (e.g. `executable_path` for C++, or `host`, `port`, `index_name` for a network backend). The backend receives this as its `config[in](#in)_init__`.

The following shows how to construct a minimal `DatasetConfig` and one `BenchmarkConfig` (one index, one search param set) so the backend runs a single build and search configuration:

```python
from cuvs_bench.orchestrator.config_loaders import (
    ConfigLoader,
    DatasetConfig,
    BenchmarkConfig,
    IndexConfig,
)

class MyConfigLoader(ConfigLoader):
    @property
    def backend_type(self) -> str:
        return "my_backend"

    def load(self, dataset, dataset_path, algorithms, count=10, batch_size=10000, **kwargs):
        path_to_base = ...  # path to base vectors file
        path_to_queries = ...  # path to query file
        path_to_groundtruth = ...  # path to groundtruth neighbors file
        path_to_index = ...  # path or id where the index is stored
        dataset_config = DatasetConfig(
            name=dataset,
            base_file=path_to_base,
            query_file=path_to_queries,
            groundtruth_neighbors_file=path_to_groundtruth,
            distance="euclidean",
            dims=128,
        )
        index = IndexConfig(
            name=f"{algorithms}.default",
            algo=algorithms,
            build_param={"nlist": 1024},
            search_params=[{"nprobe": 10}],
            file=path_to_index,
        )
        benchmark_config = BenchmarkConfig(
            indexes=[index],
            backend_config={
                "host": ...,  # backend host
                "port": ...,  # backend port
                "index_name": ...,  # name of the index on the backend
            },
        )
        return dataset_config, [benchmark_config]
```

## Adding a new backend

To add a new execution path (e.g. Elasticsearch):

1. Implement a **config loader**. Subclass **ConfigLoader** (from `cuvs_bench.orchestrator.config_loaders`). Implement **load()** to accept the kwargs the orchestrator passes (dataset, dataset_path, algorithms, count, batch_size, and the like) and return `(DatasetConfig, List[BenchmarkConfig])`. Populate **DatasetConfig** with dataset paths and metadata; for each run you want, add an **IndexConfig** (name, algo, build_param, search_params, file) and a **BenchmarkConfig** (indexes, backend_config). The **backend_config** dict is passed to your backend's constructor. Register the loader with **register_config_loader("my_backend", MyConfigLoader)**.

2. Implement the **backend**. Subclass **BenchmarkBackend** (from `cuvs_bench.backends.base`). In **__init__(self, config)**, store the config (this is the **backend_config** produced by the loader). Implement **build(dataset, indexes, force=False, dry_run=False)** to return a **BuildResult** (index_path, build_time_seconds, index_size_bytes, algorithm, build_params, metadata, success). Implement **search(dataset, indexes, k, batch_size, mode=..., ...)** to return a **SearchResult** (neighbors, distances, search_time_ms, queries_per_second, recall, algorithm, search_params, success). Implement the **algo** property (e.g. from `self.config["algo"]`). Set **requires_gpu** or **requires_network** in config if the backend needs them. Register the class with **get_registry().register("my_backend", MyBackend)**.

3. Use the new backend by calling `BenchmarkOrchestrator(backend_type="my_backend").run_benchmark(dataset=..., dataset_path=..., algorithms=..., **kwargs)`. The orchestrator will use your loader to build the configuration and your backend to run build and search.

After implementing your loader and backend, register them as follows:

```python
from cuvs_bench.orchestrator import register_config_loader
from cuvs_bench.backends import get_registry

register_config_loader("my_backend", MyConfigLoader)
get_registry().register("my_backend", MyBackend)
```

## Example: adding an Elasticsearch backend

The following example shows a minimal Elasticsearch-style backend. The config loader builds one dataset config and one benchmark config with a single index; the backend stubs build and search and returns the result types the orchestrator expects. In practice you would replace the stub logic with real Elasticsearch API calls.

Config loader: the **load()** method receives `dataset`, `dataset_path`, `algorithms`, `count`, `batch_size`, and optional kwargs. It returns a **DatasetConfig** (filled from dataset path and name) and a list of one **BenchmarkConfig** containing one **IndexConfig** and a **backend_config** with `host`, `port`, and `index_name` for the backend to use.

```python
from cuvs_bench.orchestrator.config_loaders import (
    ConfigLoader,
    DatasetConfig,
    BenchmarkConfig,
    IndexConfig,
)

class ElasticsearchConfigLoader(ConfigLoader):
    @property
    def backend_type(self) -> str:
        return "elasticsearch"

    def load(self, dataset, dataset_path, algorithms, count=10, batch_size=10000, **kwargs):
        path_to_base = ...  # path to base vectors (e.g. from dataset_path/dataset)
        path_to_queries = ...  # path to query vectors
        path_to_groundtruth = ...  # path to groundtruth file
        path_to_index = ...  # path or id for the index
        dataset_config = DatasetConfig(
            name=dataset,
            base_file=path_to_base,
            query_file=path_to_queries,
            groundtruth_neighbors_file=path_to_groundtruth,
            distance="euclidean",
            dims=kwargs.get("dims", 128),
        )
        index = IndexConfig(
            name=f"{algorithms}.es",
            algo=algorithms,
            build_param={},
            search_params=[{"ef_search": 100}],
            file=path_to_index,
        )
        benchmark_config = BenchmarkConfig(
            indexes=[index],
            backend_config={
                "host": ...,  # Elasticsearch host
                "port": ...,  # Elasticsearch port
                "index_name": ...,  # name of the vector index
                "algo": algorithms,
            },
        )
        return dataset_config, [benchmark_config]
```

Backend: the backend is constructed with **backend_config** (host, port, index_name, algo). **build()** and **search()** return **BuildResult** and **SearchResult** with the required fields; here they are stubbed with minimal values. Replace the stub body with actual Elasticsearch index creation and search calls.

```python
import numpy as np
from cuvs_bench.backends.base import (
    BenchmarkBackend,
    Dataset,
    BuildResult,
    SearchResult,
)
from cuvs_bench.orchestrator.config_loaders import IndexConfig

class ElasticsearchBackend(BenchmarkBackend):
    @property
    def algo(self) -> str:
        return self.config.get("algo", "elasticsearch")

    def build(self, dataset, indexes, force=False, dry_run=False):
        # Stub: in practice, create ES index and bulk-index dataset.base_vectors
        return BuildResult(
            index_path=indexes[0].file if indexes else "",
            build_time_seconds=0.0,
            index_size_bytes=0,
            algorithm=self.algo,
            build_params=indexes[0].build_param if indexes else {},
            metadata={},
            success=True,
        )

    def search(self, dataset, indexes, k, batch_size=10000, mode="latency", force=False, search_threads=None, dry_run=False):
        # Stub: in practice, run ES kNN search and compute recall
        n_queries = dataset.n_queries
        return SearchResult(
            neighbors=np.zeros((n_queries, k), dtype=np.int64),
            distances=np.zeros((n_queries, k), dtype=np.float32),
            search_time_ms=0.0,
            queries_per_second=0.0,
            recall=0.0,
            algorithm=self.algo,
            search_params=indexes[0].search_params if indexes else [],
            success=True,
        )
```

Registration:

```python
from cuvs_bench.orchestrator import register_config_loader
from cuvs_bench.backends import get_registry

register_config_loader("elasticsearch", ElasticsearchConfigLoader)
get_registry().register("elasticsearch", ElasticsearchBackend)
```

The built-in **CppGoogleBenchmarkBackend** (`backend_type="cpp_gbench"`) is one such pair: **CppGBenchConfigLoader** reads the YAML under `config/datasets` and `config/algos`, expands the Cartesian product, and validates with the constraint functions; the backend runs the C++ benchmark executables and merges results. Adding a new C++ algorithm (see [index](index.md)) only adds another executable and config for this backend; it does not add a new backend.

## Components at a glance

```{list-table}
  :header-rows: 1
  :widths: 20 80

* - Component
  - Description

* - ConfigLoader
  - Abstract. **load(**kwargs)** returns `(DatasetConfig, List[BenchmarkConfig])`. Register with **register_config_loader(backend_type, loader_class)**.

* - BenchmarkBackend
  - Abstract. **build(dataset, indexes, force, dry_run)** returns `BuildResult`; **search(dataset, indexes, k, batch_size, mode, ...)** returns `SearchResult`. Optional **initialize()** and **cleanup()**. Properties: **algo**, **requires_gpu**, **requires_network** (from config). Register with **BackendRegistry.register(name, backend_class)**; get an instance with **get_backend(name, config)**.

* - BackendRegistry
  - **get_registry()** returns the singleton. **register(name, backend_class)** and **get_backend(name, config)** tie a backend type name to the class and to instances.
```

