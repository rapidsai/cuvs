# cuVS Bench

cuVS Bench is a reproducible benchmarking tool for ANN search implementations. It is designed for GPU-to-GPU and GPU-to-CPU comparisons, and for capturing useful index configurations that can be reproduced across on-prem and cloud hardware.

Use it to compare build times, search throughput, latency, and recall; find good parameter settings for recall buckets; generate consistent plots; and identify optimization opportunities across index parameters, build time, and search performance.

For dataset file formats, conversion utilities, and ground-truth generation, see [Benchmark Datasets](datasets.md).

For custom benchmark execution paths and backend integrations, see [Backends](pluggable_backend.md).

For setup, see [Installation](install.md). To run benchmark workflows, see [Usage](running.md). To compile the benchmark executables locally, see [Build from Source](install.md#build-from-source).

## Setup options

Most users should start with the pre-built Conda packages or Docker images. Building from source is useful when you are developing benchmark code, testing unreleased algorithms, or changing the benchmark executable targets.

## Creating and customizing algorithm configurations

A YAML configuration defines algorithms and their build/search parameters. Dataset configuration is covered in [Benchmark Datasets](datasets.md#dataset-configurations).

Algorithm configuration files live in `${CUVS_HOME}/python/cuvs_bench/cuvs_bench/config/algos`. Example `cuvs_cagra` configuration:

```yaml
name: cuvs_cagra
constraints:
  build: cuvs_bench.config.algos.constraints.cuvs_cagra_build
  search: cuvs_bench.config.algos.constraints.cuvs_cagra_search
groups:
  base:
    build:
      graph_degree: [32, 64]
      intermediate_graph_degree: [64, 96]
      graph_build_algo: ["NN_DESCENT"]
    search:
      itopk: [32, 64, 128]

  large:
    build:
      graph_degree: [32, 64]
    search:
      itopk: [32, 64, 128]
```

Override default benchmark parameters by creating a custom YAML file for algorithms with a `base` group.

The config has three fields:

1. `name`: algorithm name.
2. `constraints`: optional Python import paths that validate build and search combinations, such as `cuvs_bench.config.algos.constraints.cuvs_cagra_build`. Invalid combinations are skipped.
3. `groups`: run groups. Each group defines a cross-product of `build` and `search` hyper-parameters.

Supported algorithms are listed below. Each algorithm has its own `build` and `search` settings; see the [ANN Algorithm Parameter Tuning Guide](param_tuning.md) for parameter guidance.

| Library | Algorithms |
| --- | --- |
| FAISS_GPU | `faiss_gpu_flat`, `faiss_gpu_ivf_flat`, `faiss_gpu_ivf_pq`, `faiss_gpu_cagra` |
| FAISS_CPU | `faiss_cpu_flat`, `faiss_cpu_ivf_flat`, `faiss_cpu_ivf_pq`, `faiss_cpu_hnsw_flat` |
| GGNN | `ggnn` |
| HNSWLIB | `hnswlib` |
| DiskANN | `diskann_memory`, `diskann_ssd` |
| cuVS | `cuvs_brute_force`, `cuvs_cagra`, `cuvs_ivf_flat`, `cuvs_ivf_pq`, `cuvs_cagra_hnswlib`, `cuvs_vamana` |

### Multi-GPU benchmarks

cuVS includes single-node multi-GPU versions of IVF-Flat, IVF-PQ, and CAGRA.

| Index type | Multi-GPU algo name |
| --- | --- |
| IVF-Flat | `cuvs_mg_ivf_flat` |
| IVF-PQ | `cuvs_mg_ivf_pq` |
| CAGRA | `cuvs_mg_cagra` |

## Adding a new index algorithm

### Implementation and configuration

New algorithms should be C++ classes that inherit `class ANN` from `cpp/bench/ann/src/ann.h` and implement all pure virtual functions.

Define separate build and search parameter structs. The search parameter struct should inherit `struct ANN&lt;T>::AnnSearchParam`. Example:

```c++
template<typename T>
class HnswLib : public ANN<T> {
public:
  struct BuildParam {
    int M;
    int ef_construction;
    int num_threads;
  };

  using typename ANN<T>::AnnSearchParam;
  struct SearchParam : public AnnSearchParam {
    int ef;
    int num_threads;
  };

  // ...
};
```

The benchmark program consumes JSON configuration files for indexes, build parameters, and search parameters. These JSON files are verbose and are generated automatically from YAML. The JSON objects map to YAML `build_param` objects and `search_param` arrays. Example generated JSON for `HnswLib`:

```json
{
  "name" : "hnswlib.M12.ef500.th32",
  "algo" : "hnswlib",
  "build_param": {"M":12, "efConstruction":500, "numThreads":32},
  "file" : "/path/to/file",
  "search_params" : [
    {"ef":10, "numThreads":1},
    {"ef":20, "numThreads":1},
    {"ef":40, "numThreads":1},
  ],
  "search_result_file" : "/path/to/file"
},
```

Build and search params are passed to C++ as JSON objects. Parse them for `HnswLib` as follows:

1. Add functions that parse JSON into `struct BuildParam` and `struct SearchParam`:

```c++
template<typename T>
void parse_build_param(const nlohmann::json& conf,
                       typename cuann::HnswLib<T>::BuildParam& param) {
  param.ef_construction = conf.at("efConstruction");
  param.M = conf.at("M");
  if (conf.contains("numThreads")) {
    param.num_threads = conf.at("numThreads");
  }
}

template<typename T>
void parse_search_param(const nlohmann::json& conf,
                        typename cuann::HnswLib<T>::SearchParam& param) {
  param.ef = conf.at("ef");
  if (conf.contains("numThreads")) {
    param.num_threads = conf.at("numThreads");
  }
}
```

2. Add matching `if` cases to `create_algo()` in `cpp/bench/ann/` and `create_search_param()`. The string literal must match the `algo` value in the configuration file:

```c++
// JSON configuration file contains a line like:  "algo" : "hnswlib"
if (algo == "hnswlib") {
   // ...
}
```

### Adding a CMake target

`cuvs/cpp/bench/ann/CMakeLists.txt` provides a CMake function for new benchmark targets:

```cmake
ConfigureAnnBench(
  NAME <algo_name>
  PATH </path/to/algo/benchmark/source/file>
  INCLUDES <additional_include_directories>
  CXXFLAGS <additional_cxx_flags>
  LINKS <additional_link_library_targets>
)
```

Example target for `HNSWLIB`:

```cmake
ConfigureAnnBench(
  NAME HNSWLIB PATH bench/ann/src/hnswlib/hnswlib_benchmark.cpp INCLUDES
  ${CMAKE_CURRENT_BINARY_DIR}/_deps/hnswlib-src/hnswlib CXXFLAGS "${HNSW_CXX_FLAGS}"
)
```

This creates `HNSWLIB_ANN_BENCH`, which runs `HNSWLIB` benchmarks.

Add an `algos.yaml` entry that maps the algorithm name to its executable and declares GPU requirements:

```yaml
cuvs_ivf_pq:
  executable: CUVS_IVF_PQ_ANN_BENCH
  requires_gpu: true
```

`executable` specifies the binary used to build/search the index and is assumed to be available in `cuvs/cpp/build/`.
`requires_gpu` specifies whether the algorithm requires a GPU.
