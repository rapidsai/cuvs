# cuVS Bench

cuVS Bench is a reproducible benchmarking tool for ANN search implementations. It is designed for GPU-to-GPU and GPU-to-CPU comparisons, and for capturing useful index configurations that can be reproduced across on-prem and cloud hardware.

Use it to compare build times, search throughput, latency, and recall; find good parameter settings for recall buckets; generate consistent plots; and identify optimization opportunities across index parameters, build time, and search performance.

For dataset file formats, conversion utilities, and ground-truth generation, see [cuVS Bench Datasets](datasets.md).

For custom benchmark execution paths and backend integrations, see [Pluggable Backend](pluggable_backend.md).

## Installing the benchmarks

There are two main ways pre-compiled benchmarks are distributed:

- [Conda](#conda): best when you want a Python package without containers. Pip wheels are planned for users who cannot use conda.
- [Docker](#docker): best when you want a containerized workflow. It needs Docker and, for GPU runs, [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker).

### Conda

```bash
conda create --name cuvs_benchmarks
conda activate cuvs_benchmarks

# to install GPU package:
conda install -c rapidsai -c conda-forge cuvs-bench=<rapids_version> cuda-version=13.1*

# to install CPU package for usage in CPU-only systems:
conda install -c rapidsai -c conda-forge  cuvs-bench-cpu
```

Use `rapidsai-nightly` instead of `rapidsai` for nightly benchmarks. The CPU package currently supports HNSW benchmarks.

Please see the [build instructions](build.md) to build the benchmarks from source.

### Docker

Images are available for GPU and CPU-only systems:

- `cuvs-bench`: includes GPU and CPU benchmarks, supports all algorithms, downloads million-scale datasets as needed, and requires the NVIDIA Container Toolkit for GPU algorithms.
- `cuvs-bench-cpu`: includes only CPU benchmarks and is the smallest image for systems without GPUs.

Nightly images are located on [Docker Hub](https://hub.docker.com/r/rapidsai/cuvs-bench/tags).

The following command pulls the nightly container for Python version 3.13, CUDA version 12.9, and cuVS version 26.06:

```bash
docker pull rapidsai/cuvs-bench:26.06a-cuda12-py3.13 # substitute cuvs-bench for the exact desired container.
```

CUDA and Python versions can be changed to supported values:
- Supported CUDA versions: 12, 13
- Supported Python versions: 3.11, 3.13, and 3.14

Exact tags are listed on Docker Hub:
- [cuVS bench images](https://hub.docker.com/r/rapidsai/cuvs-bench/tags)
- [cuVS bench CPU only images](https://hub.docker.com/r/rapidsai/cuvs-bench-cpu/tags)

**Note:** GPU containers use the CUDA toolkit inside the container. The host only needs a compatible driver, so CUDA 12 containers can run on systems with CUDA 13.x-capable drivers. GPU access also requires the NVIDIA Docker runtime from the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker).

## Running the benchmarks

### End-to-end: smaller-scale benchmarks (&lt;1M to 10M)

This example downloads, installs, and runs benchmarks on a 10M-vector subset of Yandex Deep-1B. Datasets are stored under `RAPIDS_DATASET_ROOT_DIR` when set, otherwise under a local `datasets` subdirectory.

```bash
# (1) Prepare dataset.
python -m cuvs_bench.get_dataset --dataset deep-image-96-angular --normalize
```

```python
# (2) Build and search index.
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

```bash
# (3) Export data.
python -m cuvs_bench.run --data-export --dataset deep-image-96-inner

# (4) Plot results.
python -m cuvs_bench.plot --dataset deep-image-96-inner
```

| Dataset name | Train rows | Columns | Test rows | Distance |
| --- | --- | --- | --- | --- |
| `deep-image-96-angular` | 10M | 96 | 10K | Angular |
| `fashion-mnist-784-euclidean` | 60K | 784 | 10K | Euclidean |
| `glove-50-angular` | 1.1M | 50 | 10K | Angular |
| `glove-100-angular` | 1.1M | 100 | 10K | Angular |
| `mnist-784-euclidean` | 60K | 784 | 10K | Euclidean |
| `nytimes-256-angular` | 290K | 256 | 10K | Angular |
| `sift-128-euclidean` | 1M | 128 | 10K | Euclidean |

These datasets include ground-truth test sets with 100 neighbors, so `k` must be less than or equal to 100.

### End-to-end: large-scale benchmarks (>10M vectors)

`cuvs_bench.get_dataset` does not download billion-scale datasets. Use the billion-scale dataset guide to prepare them; after that, the Python commands below work the same way.

To download billion-scale datasets, visit [big-ann-benchmarks](http://big-ann-benchmarks.com/neurips21.html)

The `wiki-all` dataset contains 88M 768-dimensional vectors for realistic RAG/LLM-scale benchmarking, plus 1M and 10M subsets. See the [Wiki-all Dataset Guide](wiki_all_dataset.md) to download it.

The example below runs on a 100M-vector subset of Yandex Deep-1B. Datasets at this scale are best suited to large-memory GPUs such as A100 or H100.

```bash
mkdir -p datasets/deep-1B
# (1) Prepare dataset.
# download manually "Ground Truth" file of "Yandex DEEP"
# suppose the file name is deep_new_groundtruth.public.10K.bin
python -m cuvs_bench.split_groundtruth --groundtruth datasets/deep-1B/deep_new_groundtruth.public.10K.bin
# two files 'groundtruth.neighbors.ibin' and 'groundtruth.distances.fbin' should be produced
```

```python
# (2) Build and search index.
from cuvs_bench.orchestrator import BenchmarkOrchestrator

orchestrator = BenchmarkOrchestrator(backend_type="cpp_gbench")
results = orchestrator.run_benchmark(
    dataset="deep-1B",
    algorithms="cuvs_cagra",
    count=10,
    batch_size=10,
    build=True,
    search=True,
)
```

```bash
# (3) Export data.
python -m cuvs_bench.run --data-export --dataset deep-1B

# (4) Plot results.
python -m cuvs_bench.plot --dataset deep-1B
```

The usage of `python -m cuvs_bench.split_groundtruth` is:

```bash
usage: split_groundtruth.py [-h] --groundtruth GROUNDTRUTH

options:
  -h, --help            show this help message and exit
  --groundtruth GROUNDTRUTH
                        Path to billion-scale dataset groundtruth file (default: None)
```

### Testing on new datasets

Each benchmark dataset needs a descriptor with file names and basic dataset properties. Descriptors for several common datasets are already available in [datasets.yaml](https://github.com/rapidsai/cuvs/blob/branch-25.04/python/cuvs_bench/cuvs_bench/config/datasets/datasets.yaml).

For a new dataset, create a descriptor such as `mydataset.yaml`:

```yaml
- name: mydata-1M
  base_file: mydata-1M/base.100M.u8bin
  subset_size: 1000000
  dims: 128
  query_file: mydata-10M/queries.u8bin
  groundtruth_neighbors_file: mydata-1M/groundtruth.neighbors.ibin
  distance: euclidean
```

Choose any `name` and pass it as `--dataset`. File paths are relative to `--dataset-path`. The optional `subset_size` uses the first `subset_size` vectors, which lets you benchmark multiple subsets without duplicating vectors. Ground truth must be generated separately for each subset.

To run the benchmark on the newly defined `mydata-1M` dataset, you can use the following command line:

```bash
python -m cuvs_bench.run --dataset mydata-1M --dataset-path=/path/to/data/folder --dataset-configuration=mydataset.yaml  --algorithms=cuvs_cagra
```

### Running with Docker containers

Docker supports end-to-end runs and manual execution inside the container.

#### End-to-end run on GPU

Without a custom entrypoint, the container runs the full workflow from [Running the benchmarks](#running-the-benchmarks).

For GPU systems, set `DATA_FOLDER` to a dedicated local folder. Datasets are stored in `$DATA_FOLDER/datasets` and results in `$DATA_FOLDER/result`.

```bash
export DATA_FOLDER=path/to/store/datasets/and/results
docker run --gpus all --rm -it -u $(id -u)                      \
    -v $DATA_FOLDER:/data/benchmarks                            \
    rapidsai/cuvs-bench:26.06a-cuda12-py3.13              \
    "--dataset deep-image-96-angular"                           \
    "--normalize"                                               \
    "--algorithms cuvs_cagra,cuvs_ivf_pq --batch-size 10 -k 10" \
    ""
```

Usage of the above command is as follows:

| Argument | Description |
| --- | --- |
| `rapidsai/cuvs-bench:26.06a-cuda12-py3.13` | Image to use. See "Docker" section for links to lists of available tags. |
| `"--dataset deep-image-96-angular"` | Dataset name |
| `"--normalize"` | Whether to normalize the dataset |
| `"--algorithms cuvs_cagra,hnswlib --batch-size 10 -k 10"` | Arguments passed to the `run` script, such as the algorithms to benchmark, the batch size, and `k` |
| `""` | Additional (optional) arguments that will be passed to the `plot` script. |

***Note about user and file permissions:*** The flag `-u $(id -u)` allows the user inside the container to match the `uid` of the user outside the container, allowing the container to read and write to the mounted volume indicated by the `$DATA_FOLDER` variable.

#### End-to-end run on CPU

Use the same argument pattern with the CPU-only container on systems without a GPU.

***Note:*** Use the `cuvs-bench-cpu` image and omit `--gpus all`:

```bash
export DATA_FOLDER=path/to/store/datasets/and/results
docker run  --rm -it -u $(id -u)                  \
    -v $DATA_FOLDER:/data/benchmarks              \
    rapidsai/cuvs-bench-cpu:26.06a-py3.13     \
     "--dataset deep-image-96-angular"            \
     "--normalize"                                \
     "--algorithms hnswlib --batch-size 10 -k 10" \
     ""
```

#### Manually run the scripts inside the container

All `cuvs-bench` images include the Conda packages, so you can open a shell and run commands directly:

```bash
export DATA_FOLDER=path/to/store/datasets/and/results
docker run --gpus all --rm -it -u $(id -u)          \
    --entrypoint /bin/bash                          \
    --workdir /data/benchmarks                      \
    -v $DATA_FOLDER:/data/benchmarks                \
    rapidsai/cuvs-bench:26.06a-cuda12-py3.13
```

This opens a container shell with the `cuvs-bench` Python package ready to use:

```bash
(base) root@00b068fbb862:/data/benchmarks# python -m cuvs_bench.get_dataset --dataset deep-image-96-angular --normalize
```

Containers can also run in detached mode.

### Evaluating the results

Build benchmarks report these measurements:

| Name | Description |
| --- | --- |
| Benchmark | A name that uniquely identifies the benchmark instance |
| Time | Wall-time spent training the index |
| CPU | CPU time spent training the index |
| Iterations | Number of iterations (this is usually 1) |
| GPU | GPU time spent building |
| index_size | Number of vectors used to train index |

Search benchmarks report these measurements. The most important fields are `Latency`, `items_per_second`, and `end_to_end`.

| Name | Description |
| --- | --- |
| Benchmark | A name that uniquely identifies the benchmark instance |
| Time | The wall-clock time of a single iteration (batch) divided by the number of threads. |
| CPU | The average CPU time (user + sys time). This does not include idle time (which can also happen while waiting for GPU sync). |
| Iterations | Total number of batches. This is going to be `total_queries` / `n_queries`. |
| GPU | GPU latency of a single batch (seconds). In throughput mode this is averaged over multiple threads. |
| Latency | Latency of a single batch (seconds), calculated from wall-clock time. In throughput mode this is averaged over multiple threads. |
| Recall | Proportion of correct neighbors to ground truth neighbors. Note this column is only present if groundtruth file is specified in dataset configuration. |
| items_per_second | Total throughput, a.k.a Queries per second (QPS). This is approximately `total_queries` / `end_to_end`. |
| k | Number of neighbors being queried in each iteration |
| end_to_end | Total time taken to run all batches for all iterations |
| n_queries | Total number of query vectors in each batch |
| total_queries | Total number of vector queries across all iterations ( = `iterations` * `n_queries`) |

Notes:
- `Time` and `end_to_end` are measured slightly differently, so `end_to_end = Time * Iterations` is only approximate.
- Output tables may also include hyper-parameters for each benchmarked configuration.
- Recall can fluctuate when fewer neighbors are processed than are available in the benchmark, because processed query count depends on iteration count.

## Creating and customizing dataset configurations

A YAML configuration defines datasets, algorithms, and their build/search parameters. A single algorithm configuration can often be reused across datasets.

The default `${CUVS_HOME}/python/cuvs_bench/cuvs_bench/config/datasets/datasets.yaml` includes several datasets. Example entry for `sift-128-euclidean`:

```yaml
- name: sift-128-euclidean
  base_file: sift-128-euclidean/base.fbin
  query_file: sift-128-euclidean/query.fbin
  groundtruth_neighbors_file: sift-128-euclidean/groundtruth.neighbors.ibin
  dims: 128
  distance: euclidean
```

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
