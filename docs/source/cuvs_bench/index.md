# cuVS Bench

cuVS bench provides a reproducible benchmarking tool for various ANN search implementations. It's especially suitable for comparing GPU implementations as well as comparing GPU against CPU. One of the primary goals of cuVS is to capture ideal index configurations for a variety of important usage patterns so the results can be reproduced easily on different hardware environments, such as on-prem and cloud.

This tool offers several benefits, including

1.  Making fair comparisons of index build times
2.  Making fair comparisons of index search throughput and/or latency
3.  Finding the optimal parameter settings for a range of recall buckets
4.  Easily generating consistently styled plots for index build and search
5.  Profiling blind spots and potential for algorithm optimization
6.  Investigating the relationship between different parameter settings, index build times, and search performance.

- [Installing the benchmarks](#installing-the-benchmarks)
  - [Conda](#conda)
  - [Docker](#docker)
- [Running the benchmarks](#running-the-benchmarks)
  - [End-to-end: smaller-scale benchmarks (\<1M to 10M)](#end-to-end-smaller-scale-benchmarks-1m-to-10m)
  - [End-to-end: large-scale benchmarks (\>10M vectors)](#end-to-end-large-scale-benchmarks-10m-vectors)
  - [Running with Docker containers](#running-with-docker-containers)
    - [End-to-end run on GPU](#end-to-end-run-on-gpu)
    - [Manually run the scripts inside the container](#manually-run-the-scripts-inside-the-container)
  - [Evaluating the results](#evaluating-the-results)
- [Creating and customizing dataset configurations](#creating-and-customizing-dataset-configurations)
  - [Multi-GPU benchmarks](#multi-gpu-benchmarks)
- [Adding a new index algorithm](#adding-a-new-index-algorithm)
  - [Implementation and configuration](#implementation-and-configuration)
  - [Adding a Cmake target](#adding-a-cmake-target)

## Installing the benchmarks

There are two main ways pre-compiled benchmarks are distributed:

- [Conda](#conda) For users not using containers but want an easy to install and use Python package. Pip wheels are planned to be added as an alternative for users that cannot use conda and prefer to not use containers.
- [Docker](#docker) Only needs docker and [NVIDIA docker](https://github.com/NVIDIA/nvidia-docker) to use. Provides a single docker run command for basic dataset benchmarking, as well as all the functionality of the conda solution inside the containers.

### Conda

``` bash
conda create --name cuvs_benchmarks
conda activate cuvs_benchmarks

# to install GPU package:
conda install -c rapidsai -c conda-forge cuvs-bench=<rapids_version> cuda-version=13.1*

# to install CPU package for usage in CPU-only systems:
conda install -c rapidsai -c conda-forge  cuvs-bench-cpu
```

The channel <span class="title-ref">rapidsai</span> can easily be substituted with <span class="title-ref">rapidsai-nightly</span> if nightly benchmarks are desired. The CPU package currently allows to run the HNSW benchmarks.

Please see the `build instructions <build>` to build the benchmarks from source.

### Docker

We provide images for GPU enabled systems, as well as systems without a GPU. The following images are available:

- \`cuvs-bench\`: Contains GPU and CPU benchmarks, can run all algorithms supported. Will download million-scale datasets as required. Best suited for users that prefer a smaller container size for GPU based systems. Requires the NVIDIA Container Toolkit to run GPU algorithms, can run CPU algorithms without it.
- \`cuvs-bench-cpu\`: Contains only CPU benchmarks with minimal size. Best suited for users that want the smallest containers to reproduce benchmarks on systems without a GPU.

Nightly images are located in [dockerhub](https://hub.docker.com/r/rapidsai/cuvs-bench/tags).

The following command pulls the nightly container for Python version 3.13, CUDA version 12.9, and cuVS version 26.06:

``` bash
docker pull rapidsai/cuvs-bench:26.06a-cuda12-py3.13 # substitute cuvs-bench for the exact desired container.
```

The CUDA and python versions can be changed for the supported values: - Supported CUDA versions: 12, 13 - Supported Python versions: 3.11, 3.11, 3.13, and 3.14

You can see the exact versions as well in the dockerhub site: - [cuVS bench images](https://hub.docker.com/r/rapidsai/cuvs-bench/tags) - [cuVS bench CPU only images](https://hub.docker.com/r/rapidsai/cuvs-bench-cpu/tags)

**Note:** GPU containers use the CUDA toolkit from inside the container, the only requirement is a driver installed on the host machine that supports that version. So, for example, CUDA 12 containers can run in systems with a CUDA 13.x capable driver. Please also note that the Nvidia-Docker runtime from the [Nvidia Container Toolkit](https://github.com/NVIDIA/nvidia-docker) is required to use GPUs inside docker containers.

## Running the benchmarks

### End-to-end: smaller-scale benchmarks (\<1M to 10M)

The steps below demonstrate how to download, install, and run benchmarks on a subset of 10M vectors from the Yandex Deep-1B dataset. By default the datasets will be stored and used from the folder indicated by the <span class="title-ref">RAPIDS_DATASET_ROOT_DIR</span> environment variable if defined, otherwise a datasets sub-folder from where the script is being called.

``` bash
# (1) Prepare dataset.
python -m cuvs_bench.get_dataset --dataset deep-image-96-angular --normalize
```

``` python
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

``` bash
# (3) Export data.
python -m cuvs_bench.run --data-export --dataset deep-image-96-inner

# (4) Plot results.
python -m cuvs_bench.plot --dataset deep-image-96-inner
```

|  |  |  |  |  |
|----|----|----|----|----|
| Dataset name | Train rows | Columns | Test rows | Distance |
| <span class="title-ref">deep-image-96-angular</span> | 10M | 96 | 10K | Angular |
| <span class="title-ref">fashion-mnist-784-euclidean</span> | 60K | 784 | 10K | Euclidean |
| <span class="title-ref">glove-50-angular</span> | 1.1M | 50 | 10K | Angular |
| <span class="title-ref">glove-100-angular</span> | 1.1M | 100 | 10K | Angular |
| <span class="title-ref">mnist-784-euclidean</span> | 60K | 784 | 10K | Euclidean |
| <span class="title-ref">nytimes-256-angular</span> | 290K | 256 | 10K | Angular |
| <span class="title-ref">sift-128-euclidean</span> | 1M | 128 | 10K | Euclidean |

All of the datasets above contain ground test datasets with 100 neighbors. Thus <span class="title-ref">k</span> for these datasets must be less than or equal to 100.

### End-to-end: large-scale benchmarks (\>10M vectors)

<span class="title-ref">cuvs_bench.get_dataset</span> cannot be used to download the billion-scale datasets due to their size. You should instead use our billion-scale datasets guide to download and prepare them. All other python commands mentioned below work as intended once the billion-scale dataset has been downloaded.

To download billion-scale datasets, visit [big-ann-benchmarks](http://big-ann-benchmarks.com/neurips21.html)

We also provide a new dataset called <span class="title-ref">wiki-all</span> containing 88 million 768-dimensional vectors. This dataset is meant for benchmarking a realistic retrieval-augmented generation (RAG)/LLM embedding size at scale. It also contains 1M and 10M vector subsets for smaller-scale experiments. See our `Wiki-all Dataset Guide <wiki_all_dataset>` for more information and to download the dataset.

The steps below demonstrate how to download, install, and run benchmarks on a subset of 100M vectors from the Yandex Deep-1B dataset. Please note that datasets of this scale are recommended for GPUs with larger amounts of memory, such as the A100 or H100.

``` bash
mkdir -p datasets/deep-1B
# (1) Prepare dataset.
# download manually "Ground Truth" file of "Yandex DEEP"
# suppose the file name is deep_new_groundtruth.public.10K.bin
python -m cuvs_bench.split_groundtruth --groundtruth datasets/deep-1B/deep_new_groundtruth.public.10K.bin
# two files 'groundtruth.neighbors.ibin' and 'groundtruth.distances.fbin' should be produced
```

``` python
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

``` bash
# (3) Export data.
python -m cuvs_bench.run --data-export --dataset deep-1B

# (4) Plot results.
python -m cuvs_bench.plot --dataset deep-1B
```

The usage of <span class="title-ref">python -m cuvs_bench.split_groundtruth</span> is:

``` bash
usage: split_groundtruth.py [-h] --groundtruth GROUNDTRUTH

options:
  -h, --help            show this help message and exit
  --groundtruth GROUNDTRUTH
                        Path to billion-scale dataset groundtruth file (default: None)
```

### Testing on new datasets

To run benchmark on a dataset, it is required have a descriptor that defines the file names and a few other properties of that dataset. Descriptors for several popular datasets are already available in <span class="title-ref">datasets.yaml \<https://github.com/rapidsai/cuvs/blob/branch-25.04/python/cuvs_bench/cuvs_bench/config/datasets/datasets.yaml\></span>\`.

Let's consider how to test on a new dataset. First we create a descriptor <span class="title-ref">mydataset.yaml</span>

Here <span class="title-ref">name</span> can be chosen arbitrarily. We pass <span class="title-ref">name</span> as the <span class="title-ref">--dataset</span> argument for the benchmark. The file names are relative to the path given by <span class="title-ref">--dataset-path</span> argument. The <span class="title-ref">subset_size</span><span class="title-ref"> field is optional. This argument defines how many vectors to use from the dataset file, the first \`subset_size</span> vectors will be used. This way you can define benchmarks on multiple subsets of the same dataset without duplicating the dataset vectors. Note that the ground truth vectors have to be generated for each subset separately.

To run the benchmark on the newly defined <span class="title-ref">mydata-1M</span> dataset, you can use the following command line:

### Running with Docker containers

Two methods are provided for running the benchmarks with the Docker containers.

#### End-to-end run on GPU

When no other entrypoint is provided, an end-to-end script will run through all the steps in [Running the benchmarks](#running-the-benchmarks) above.

For GPU-enabled systems, the <span class="title-ref">DATA_FOLDER</span> variable should be a local folder where you want datasets stored in <span class="title-ref">\$DATA_FOLDER/datasets</span> and results in <span class="title-ref">\$DATA_FOLDER/result</span> (we highly recommend <span class="title-ref">\$DATA_FOLDER</span> to be a dedicated folder for the datasets and results of the containers):

``` bash
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

|  |  |
|----|----|
| Argument | Description |
| <span class="title-ref">rapidsai/cuvs-bench:26.06a-cuda12-py3.13</span> | Image to use. See "Docker" section for links to lists of available tags. |
| <span class="title-ref">"--dataset deep-image-96-angular"</span> | Dataset name |
| <span class="title-ref">"--normalize"</span> | Whether to normalize the dataset |
| <span class="title-ref">"--algorithms cuvs_cagra,hnswlib --batch-size 10 -k 10"</span> | Arguments passed to the <span class="title-ref">run</span> script, such as the algorithms to benchmark, the batch size, and <span class="title-ref">k</span> |
| <span class="title-ref">""</span> | Additional (optional) arguments that will be passed to the <span class="title-ref">plot</span> script. |

**\*Note about user and file permissions:**\* The flag <span class="title-ref">-u \$(id -u)</span> allows the user inside the container to match the <span class="title-ref">uid</span> of the user outside the container, allowing the container to read and write to the mounted volume indicated by the <span class="title-ref">\$DATA_FOLDER</span> variable.

#### End-to-end run on CPU

The container arguments in the above section also be used for the CPU-only container, which can be used on systems that don't have a GPU installed.

**\*Note:**\* the image changes to <span class="title-ref">cuvs-bench-cpu</span> container and the <span class="title-ref">--gpus all</span> argument is no longer used:

``` bash
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

All of the <span class="title-ref">cuvs-bench</span> images contain the Conda packages, so they can be used directly by logging directly into the container itself:

``` bash
export DATA_FOLDER=path/to/store/datasets/and/results
docker run --gpus all --rm -it -u $(id -u)          \
    --entrypoint /bin/bash                          \
    --workdir /data/benchmarks                      \
    -v $DATA_FOLDER:/data/benchmarks                \
    rapidsai/cuvs-bench:26.06a-cuda12-py3.13
```

This will drop you into a command line in the container, with the <span class="title-ref">cuvs-bench</span> python package ready to use, as described in the [Running the benchmarks](#running-the-benchmarks) section above:

``` bash
(base) root@00b068fbb862:/data/benchmarks# python -m cuvs_bench.get_dataset --dataset deep-image-96-angular --normalize
```

Additionally, the containers can be run in detached mode without any issue.

### Evaluating the results

The benchmarks capture several different measurements. The table below describes each of the measurements for index build benchmarks:

|            |                                                        |
|------------|--------------------------------------------------------|
| Name       | Description                                            |
| Benchmark  | A name that uniquely identifies the benchmark instance |
| Time       | Wall-time spent training the index                     |
| CPU        | CPU time spent training the index                      |
| Iterations | Number of iterations (this is usually 1)               |
| GPU        | GU time spent building                                 |
| index_size | Number of vectors used to train index                  |

The table below describes each of the measurements for the index search benchmarks. The most important measurements <span class="title-ref">Latency</span>, <span class="title-ref">items_per_second</span>, <span class="title-ref">end_to_end</span>.

|  |  |
|----|----|
| Name | Description |
| Benchmark | A name that uniquely identifies the benchmark instance |
| Time | The wall-clock time of a single iteration (batch) divided by the number of threads. |
| CPU | The average CPU time (user + sys time). This does not include idle time (which can also happen while waiting for GPU sync). |
| Iterations | Total number of batches. This is going to be <span class="title-ref">total_queries</span> / <span class="title-ref">n_queries</span>. |
| GPU | GPU latency of a single batch (seconds). In throughput mode this is averaged over multiple threads. |
| Latency | Latency of a single batch (seconds), calculated from wall-clock time. In throughput mode this is averaged over multiple threads. |
| Recall | Proportion of correct neighbors to ground truth neighbors. Note this column is only present if groundtruth file is specified in dataset configuration. |
| items_per_second | Total throughput, a.k.a Queries per second (QPS). This is approximately <span class="title-ref">total_queries</span> / <span class="title-ref">end_to_end</span>. |
| k | Number of neighbors being queried in each iteration |
| end_to_end | Total time taken to run all batches for all iterations |
| n_queries | Total number of query vectors in each batch |
| total_queries | Total number of vectors queries across all iterations ( = <span class="title-ref">iterations</span> \* <span class="title-ref">n_queries</span>) |

Note the following: - A slightly different method is used to measure <span class="title-ref">Time</span> and <span class="title-ref">end_to_end</span>. That is why <span class="title-ref">end_to_end</span> = <span class="title-ref">Time</span> \* <span class="title-ref">Iterations</span> holds only approximately. - The actual table displayed on the screen may differ slightly as the hyper-parameters will also be displayed for each different combination being benchmarked. - Recall calculation: the number of queries processed per test depends on the number of iterations. Because of this, recall can show slight fluctuations if less neighbors are processed then it is available for the benchmark.

## Creating and customizing dataset configurations

A single configuration will often define a set of algorithms, with associated index and search parameters, that can be generalize across datasets. We use YAML to define dataset specific and algorithm specific configurations.

A default <span class="title-ref">datasets.yaml</span> is provided by CUVS in <span class="title-ref">\${CUVS_HOME}/python/cuvs_bench/cuvs_bench/config/datasets/datasets.yaml</span> with configurations available for several datasets. Here's a simple example entry for the <span class="title-ref">sift-128-euclidean</span> dataset:

``` yaml
- name: sift-128-euclidean
  base_file: sift-128-euclidean/base.fbin
  query_file: sift-128-euclidean/query.fbin
  groundtruth_neighbors_file: sift-128-euclidean/groundtruth.neighbors.ibin
  dims: 128
  distance: euclidean
```

Configuration files for ANN algorithms supported by <span class="title-ref">cuvs-bench</span> are provided in <span class="title-ref">\${CUVS_HOME}/python/cuvs_bench/cuvs_bench/config/algos</span>. <span class="title-ref">cuvs_cagra</span> algorithm configuration looks like:

``` yaml
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

The default parameters for which the benchmarks are run can be overridden by creating a custom YAML file for algorithms with a <span class="title-ref">base</span> group.

The config above has 3 fields:

1.  <span class="title-ref">name</span> - The name of the algorithm for which the parameters are being specified.
2.  <span class="title-ref">constraints</span> - Optional. Python import paths to functions that validate build and search parameter combinations (e.g. `cuvs_bench.config.algos.constraints.cuvs_cagra_build`). Each function returns `True` if the parameters are valid, `False` otherwise; invalid combinations are skipped and not benchmarked.
3.  <span class="title-ref">groups</span> - Run groups, each with a set of parameters. Each group defines a cross-product of all hyper-parameter fields for <span class="title-ref">build</span> and <span class="title-ref">search</span>.

The table below contains all algorithms supported by cuVS. Each unique algorithm will have its own set of <span class="title-ref">build</span> and <span class="title-ref">search</span> settings. The `ANN Algorithm Parameter Tuning Guide <param_tuning>` contains detailed instructions on choosing build and search parameters for each supported algorithm.

|  |  |
|----|----|
| Library | Algorithms |
| FAISS_GPU | <span class="title-ref">faiss_gpu_flat</span>, <span class="title-ref">faiss_gpu_ivf_flat</span>, <span class="title-ref">faiss_gpu_ivf_pq</span>, <span class="title-ref">faiss_gpu_cagra</span> |
| FAISS_CPU | <span class="title-ref">faiss_cpu_flat</span>, <span class="title-ref">faiss_cpu_ivf_flat</span>, <span class="title-ref">faiss_cpu_ivf_pq</span>, <span class="title-ref">faiss_cpu_hnsw_flat</span> |
| GGNN | <span class="title-ref">ggnn</span> |
| HNSWLIB | <span class="title-ref">hnswlib</span> |
| DiskANN | <span class="title-ref">diskann_memory</span>, <span class="title-ref">diskann_ssd</span> |
| cuVS | <span class="title-ref">cuvs_brute_force</span>, <span class="title-ref">cuvs_cagra</span>, <span class="title-ref">cuvs_ivf_flat</span>, <span class="title-ref">cuvs_ivf_pq</span>, <span class="title-ref">cuvs_cagra_hnswlib</span>, <span class="title-ref">cuvs_vamana</span> |

### Multi-GPU benchmarks

cuVS implements single node multi-GPU versions of IVF-Flat, IVF-PQ and CAGRA indexes.

|            |                                                 |
|------------|-------------------------------------------------|
| Index type | Multi-GPU algo name                             |
| IVF-Flat   | <span class="title-ref">cuvs_mg_ivf_flat</span> |
| IVF-PQ     | <span class="title-ref">cuvs_mg_ivf_pq</span>   |
| CAGRA      | <span class="title-ref">cuvs_mg_cagra</span>    |

## Adding a new index algorithm

### Implementation and configuration

Implementation of a new algorithm should be a C++ class that inherits <span class="title-ref">class ANN</span> (defined in <span class="title-ref">cpp/bench/ann/src/ann.h</span>) and implements all the pure virtual functions.

In addition, it should define two <span class="title-ref">struct\`s for building and searching parameters. The searching parameter class should inherit \`struct ANN\<T\>::AnnSearchParam</span>. Take <span class="title-ref">class HnswLib</span> as an example, its definition is:

``` c++
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

The benchmark program uses JSON format natively in a configuration file to specify indexes to build, along with the build and search parameters. However the JSON config files are overly verbose and are not meant to be used directly. Instead, the Python scripts parse YAML and create these json files automatically. It's important to realize that these json objects align with the yaml objects for <span class="title-ref">build_param</span>, whose value is a JSON object, and <span class="title-ref">search_param</span>, whose value is an array of JSON objects. Take the json configuration for <span class="title-ref">HnswLib</span> as an example of the json after it's been parsed from yaml:

``` json
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

The build and search params are ultimately passed to the C++ layer as json objects for each param configuration to benchmark. The code below shows how to parse these params for \`Hnswlib\`:

1.  First, add two functions for parsing JSON object to <span class="title-ref">struct BuildParam</span> and <span class="title-ref">struct SearchParam</span>, respectively:

``` c++
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

2.  Next, add corresponding <span class="title-ref">if</span> case to functions <span class="title-ref">create_algo()</span> (in <span class="title-ref">cpp/bench/ann/) and \`create_search_param()</span> by calling parsing functions. The string literal in <span class="title-ref">if</span> condition statement must be the same as the value of <span class="title-ref">algo</span> in configuration file. For example,

``` c++
// JSON configuration file contains a line like:  "algo" : "hnswlib"
if (algo == "hnswlib") {
   // ...
}
```

### Adding a Cmake target

In <span class="title-ref">cuvs/cpp/bench/ann/CMakeLists.txt</span>, we provide a <span class="title-ref">CMake</span> function to configure a new Benchmark target with the following signature:

``` cmake
ConfigureAnnBench(
  NAME <algo_name>
  PATH </path/to/algo/benchmark/source/file>
  INCLUDES <additional_include_directories>
  CXXFLAGS <additional_cxx_flags>
  LINKS <additional_link_library_targets>
)
```

To add a target for <span class="title-ref">HNSWLIB</span>, we would call the function as:

``` cmake
ConfigureAnnBench(
  NAME HNSWLIB PATH bench/ann/src/hnswlib/hnswlib_benchmark.cpp INCLUDES
  ${CMAKE_CURRENT_BINARY_DIR}/_deps/hnswlib-src/hnswlib CXXFLAGS "${HNSW_CXX_FLAGS}"
)
```

This will create an executable called <span class="title-ref">HNSWLIB_ANN_BENCH</span>, which can then be used to run <span class="title-ref">HNSWLIB</span> benchmarks.

Add a new entry to <span class="title-ref">algos.yaml</span> to map the name of the algorithm to its binary executable and specify whether the algorithm requires GPU support.

``` yaml
cuvs_ivf_pq:
  executable: CUVS_IVF_PQ_ANN_BENCH
  requires_gpu: true
```

<span class="title-ref">executable</span> : specifies the name of the binary that will build/search the index. It is assumed to be available in <span class="title-ref">cuvs/cpp/build/</span>. <span class="title-ref">requires_gpu</span> : denotes whether an algorithm requires GPU to run.

<div class="toctree" maxdepth="4">

build.rst datasets.rst param_tuning.rst pluggable_backend.rst wiki_all_dataset.rst

</div>
