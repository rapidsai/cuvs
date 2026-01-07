~~~~~~~~~~
cuVS Bench
~~~~~~~~~~

cuVS bench provides a reproducible benchmarking tool for approximate nearest-neighbor (ANN) search implementations and is especially suitable for comparing GPU and CPU performance. One of the primary goals of cuVS is to capture optimal index configurations for different usage patterns, enabling consistent results across on-premises and cloud environments. This tool offers several benefits, including:

* Making fair comparisons of index build times

* Making fair comparisons of index search throughput and latency

* Finding the optimal parameter settings for a range of recall buckets

* Generating consistently styled plots for index build and search

* Profiling blind spots and potential for algorithm optimization

* Investigating the relationship between different parameter settings, index build times, and search performance.

Install the Benchmarks
***********************

[ZAR Comment] Need a new section here to give a quick overview of the installation process, as well as any prerequisites.

Installing and Distributing the Benchmarks
==========================================

There are two primary methods for precompiled benchmarks to be distributed:

- `Conda`_: For users who do not use containers but want an easy installation and the use of Python packages. Pip wheels are planned to be added as an alternative for users who cannot use conda and prefer not to use containers.
- `Docker`_: Provides a single Docker run command for basic dataset benchmarking, along with all the functionality of the conda solution inside the containers. Only needs Docker and [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) to use.

Conda
-----

.. code-block:: bash

   conda create --name cuvs_benchmarks
   conda activate cuvs_benchmarks

   # To install the GPU package:
   conda install -c rapidsai -c conda-forge cuvs-bench=<rapids_version> cuda-version=13.0*

   # To install the CPU package for usage in CPU-only systems:
   conda install -c rapidsai -c conda-forge  cuvs-bench-cpu

The channel `rapidsai` can easily be substituted with `rapidsai-nightly` if nightly benchmarks are desired. The CPU package currently allows running the HNSW benchmarks.

Refer to the :doc:`build instructions <build>` to build the benchmarks from source.

Docker
------

We provide images for GPU-enabled systems and for systems without a GPU. The following images are available:

- `cuvs-bench`: Contains GPU and CPU benchmarks, can run all algorithms supported. Will download million-scale datasets as required. Best suited for users who prefer a smaller container size for GPU-based systems. Requires the NVIDIA Container Toolkit to run GPU algorithms; it can run CPU algorithms without it.
- `cuvs-bench-datasets`: Contains the GPU and CPU benchmarks with million-scale datasets already included in the container. Best suited for users who want to run multiple million-scale datasets already included in the image.
- `cuvs-bench-cpu`: Contains only CPU benchmarks with minimal size. Best suited for users who want the smallest containers to reproduce benchmarks on systems without a GPU.

Nightly images are located in `dockerhub <https://hub.docker.com/r/rapidsai/cuvs-bench/tags>`_.

The following command pulls the nightly container for Python version 3.10, CUDA version 12.5, and cuVS version 24.12:

.. code-block:: bash

   docker pull rapidsai/cuvs-bench:25.10a-cuda12.5-py3.10 # substitute cuvs-bench for the exact desired container.

The CUDA and Python versions can be changed for the supported values:
- Supported CUDA versions: 12
- Supported Python versions: 3.10 and 3.11.

You can see the exact versions as well on the Docker Hub site:
- `cuVS bench images <https://hub.docker.com/r/rapidsai/cuvs-bench/tags>`_
- `cuVS bench with pre-loaded million-scale datasets images <https://hub.docker.com/r/rapidsai/cuvs-bench-cpu/tags>`_
- `cuVS bench CPU only images <https://hub.docker.com/r/rapidsai/cuvs-bench-datasets/tags>`_

**Note:** GPU containers use the CUDA toolkit from inside the container; the only requirement is a driver installed on the host machine that supports that version. So, for example, CUDA 11.8 containers can run on systems with CUDA 12. x-capable driver. Also note that the Nvidia-Docker runtime from the `Nvidia Container Toolkit <https://github.com/NVIDIA/nvidia-docker>`_ is required to use GPUs inside Docker containers.

Running the Benchmarks
=======================

[ZAR Comment]: Let's add a little overview about the "why" here.

Running Smaller-scale Benchmarks (<1M to 10M)
---------------------------------------------

The following steps demonstrate how to download, install, and run benchmarks on a subset of 10M vectors from the Yandex Deep-1B dataset. By default, the datasets will be stored and used from the folder indicated by the `RAPIDS_DATASET_ROOT_DIR` environment variable if defined, otherwise a datasets sub-folder from where the script is being called:

.. code-block:: bash


    # (1) prepare dataset.
    python -m cuvs_bench.get_dataset --dataset deep-image-96-angular --normalize

    # (2) build and search index
    python -m cuvs_bench.run --dataset deep-image-96-inner --algorithms cuvs_cagra --batch-size 10 -k 10

    # (3) export data
    python -m cuvs_bench.run --data-export --dataset deep-image-96-inner

    # (4) plot results
    python -m cuvs_bench.plot --dataset deep-image-96-inner


.. list-table::

 * - Dataset name
   - Train rows
   - Columns
   - Test rows
   - Distance

 * - `deep-image-96-angular`
   - 10M
   - 96
   - 10K
   - Angular

 * - `fashion-mnist-784-euclidean`
   - 60K
   - 784
   - 10K
   - Euclidean

 * - `glove-50-angular`
   - 1.1M
   - 50
   - 10K
   - Angular

 * - `glove-100-angular`
   - 1.1M
   - 100
   - 10K
   - Angular

 * - `mnist-784-euclidean`
   - 60K
   - 784
   - 10K
   - Euclidean

 * - `nytimes-256-angular`
   - 290K
   - 256
   - 10K
   - Angular

 * - `sift-128-euclidean`
   - 1M
   - 128
   - 10K
   - Euclidean

All of the datasets above contain ground test datasets with 100 neighbors. Thus, `k` for these datasets must be less than or equal to 100.

Running Large-scale Benchmarks (>10M vectors)
---------------------------------------------

`cuvs_bench.get_dataset` cannot be used to download the billion-scale datasets due to their size. You should instead use our billion-scale datasets guide to download and prepare them.
All other Python commands mentioned below work as intended once the billion-scale dataset has been downloaded.

To download billion-scale datasets, visit `big-ann-benchmarks <http://big-ann-benchmarks.com/neurips21.html>`_

We also provide a new dataset, `wiki-all`, containing 88 million 768-dimensional vectors. This dataset is meant for benchmarking a realistic retrieval-augmented generation (RAG)/LLM embedding size at scale. It also includes 1M and 10M-vector subsets for smaller-scale experiments. Refer to the :doc:`Wiki-all Dataset Guide <wiki_all_dataset>` for more information and to download the dataset.


The following steps demonstrate how to download, install, and run benchmarks on a subset of 100M vectors from the Yandex Deep-1B dataset. Note that datasets of this scale are recommended for GPUs with more memory, such as the A100 or H100.

.. code-block:: bash

    mkdir -p datasets/deep-1B
    # (1) prepare dataset
    # download manually "Ground Truth" file of "Yandex DEEP"
    # suppose the file name is deep_new_groundtruth.public.10K.bin
    python -m cuvs_bench.split_groundtruth --groundtruth datasets/deep-1B/deep_new_groundtruth.public.10K.bin
    # two files 'groundtruth.neighbors.ibin' and 'groundtruth.distances.fbin' should be produced

    # (2) build and search index
    python -m cuvs_bench.run --dataset deep-1B --algorithms cuvs_cagra --batch-size 10 -k 10

    # (3) export data
    python -m cuvs_bench.run --data-export --dataset deep-1B

    # (4) plot results
    python -m cuvs_bench.plot --dataset deep-1B

The usage of `python -m cuvs_bench.split_groundtruth` is:

.. code-block:: bash

    usage: split_groundtruth.py [-h] --groundtruth GROUNDTRUTH

    options:
      -h, --help            show this help message and exit
      --groundtruth GROUNDTRUTH
                            Path to billion-scale dataset groundtruth file (default: None)

Running with Docker Containers
===============================

Two methods are provided for running the benchmarks with the Docker containers: end-to-end running on GPU and end-to-end running on CPU. You can also run scripts manually inside the container.  

Running End-to-end on GPU
--------------------------

When no other entrypoint is provided, an end-to-end script runs through all the steps in `Running the Benchmarks`_.

For GPU-enabled systems, the `DATA_FOLDER` variable should be a local folder where you want datasets stored in `$DATA_FOLDER/datasets` and results in `$DATA_FOLDER/result` (we highly recommend `$DATA_FOLDER` to be a dedicated folder for the datasets and results of the containers):

.. code-block:: bash

    export DATA_FOLDER=path/to/store/datasets/and/results
    docker run --gpus all --rm -it -u $(id -u)                      \
        -v $DATA_FOLDER:/data/benchmarks                            \
        rapidsai/cuvs-bench:25.10-cuda12.9-py3.13              \
        "--dataset deep-image-96-angular"                           \
        "--normalize"                                               \
        "--algorithms cuvs_cagra,cuvs_ivf_pq --batch-size 10 -k 10" \
        ""

Usage of the previous command is as follows:

.. list-table::

 * - Argument
   - Description

 * - `rapidsai/cuvs-bench:25.10-cuda12.9-py3.13`
   - Image to use. Can be either `cuvs-bench` or `cuvs-bench-datasets`.

 * - `"--dataset deep-image-96-angular"`
   - Dataset name

 * - `"--normalize"`
   - Whether to normalize the dataset

 * - `"--algorithms cuvs_cagra,hnswlib --batch-size 10 -k 10"`
   - Arguments passed to the `run` script, such as the algorithms to benchmark, the batch size, and `k`.

 * - `""`
   - Additional (optional) arguments passed to the `plot` script.

The flag `-u $(id -u)` allows the user inside the container to match the `uid` of the user outside the container, allowing the container to read and write to the mounted volume indicated by the `$DATA_FOLDER` variable.

Running End-to-end on CPU
--------------------------

The container arguments from the previous section can also be used for the CPU-only container, which works on systems without a GPU installed.

*Note*: The image changes to `cuvs-bench-cpu` container and the `--gpus all` argument is no longer used:

.. code-block:: bash

    export DATA_FOLDER=path/to/store/datasets/and/results
    docker run  --rm -it -u $(id -u)                  \
        -v $DATA_FOLDER:/data/benchmarks              \
        rapidsai/cuvs-bench-cpu:24.10a-py3.10     \
         "--dataset deep-image-96-angular"            \
         "--normalize"                                \
         "--algorithms hnswlib --batch-size 10 -k 10" \
         ""

Running the Scripts Manually Inside the Container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All of the `cuvs-bench` images contain the Conda packages, so they can be used directly by logging directly into the container itself:

.. code-block:: bash

    export DATA_FOLDER=path/to/store/datasets/and/results
    docker run --gpus all --rm -it -u $(id -u)          \
        --entrypoint /bin/bash                          \
        --workdir /data/benchmarks                      \
        -v $DATA_FOLDER:/data/benchmarks                \
        rapidsai/cuvs-bench:25.10-cuda12.9-py3.13

This drops you into a command line in the container, with the `cuvs-bench` Python package ready to use, as described in the previous [Running the Benchmarks](#running-the-benchmarks) section:

.. code-block:: bash

    (base) root@00b068fbb862:/data/benchmarks# python -m cuvs_bench.get_dataset --dataset deep-image-96-angular --normalize

Additionally, the containers can be run in detached mode without any issue.

Evaluating the Results
-----------------------

The benchmarks capture several different measurements. The following table describes each of the measurements for index build benchmarks:

.. list-table::

 * - Name
   - Description

 * - Benchmark
   - A name that uniquely identifies the benchmark instance

 * - Time
   - Wall-time spent training the index

 * - CPU
   - CPU time spent training the index

 * - Iterations
   - Number of iterations (this is usually 1)

 * - GPU
   - GU time spent building

 * - index_size
   - Number of vectors used to train the index

The following table describes each measurement for the index search benchmarks. The most important measurements are `Latency`, `items_per_second`, and `end_to_end`.

.. list-table::

 * - Name
   - Description

 * - Benchmark
   - A name that uniquely identifies the benchmark instance

 * - Time
   - The wall-clock time of a single iteration (batch) divided by the number of threads.

 * - CPU
   - The average CPU time (user + sys time). This does not include idle time (which can also happen while waiting for GPU sync).

 * - Iterations
   - Total number of batches. This is going to be `total_queries` / `n_queries`.

 * - GPU
   - GPU latency of a single batch (seconds). In throughput mode, this is averaged over multiple threads.

 * - Latency
   - Latency of a single batch (seconds), calculated from wall-clock time. In throughput mode, this is averaged over multiple threads.

 * - Recall
   - Proportion of correct neighbors to ground truth neighbors. Note that this column is only present if the ground truth file is specified in the dataset configuration.

 * - items_per_second
   - Total throughput, also known as queries per second (QPS). This is approximately `total_queries` / `end_to_end`.

 * - k
   - Number of neighbors being queried in each iteration

 * - end_to_end
   - Total time taken to run all batches for all iterations

 * - n_queries
   - Total number of query vectors in each batch

 * - total_queries
   - Total number of vectors queries across all iterations ( = `iterations` * `n_queries`)

Note the following:
 * A slightly different method is used to measure `Time` and `end_to_end`. That is why `end_to_end` = `Time` * `Iterations` holds only approximately.
 * The actual table displayed on the screen may differ slightly, as the hyperparameters will also be displayed for each different combination being benchmarked.
 * Recall calculation: the number of queries processed per test depends on the number of iterations. Because of this, recall can show slight fluctuations if fewer neighbors are processed than are available  for the benchmark.

Create and Customize Dataset Configurations
************************************************

A single configuration often defines a set of algorithms, along with associated index and search parameters, that can be generalized across datasets. We use YAML to define dataset-specific and algorithm-specific configurations.

A default `datasets.yaml` is provided by CUVS in `${CUVS_HOME}/python/cuvs_bench/src/cuvs_bench/run/conf` with configurations available for several datasets. Here's a simple example entry for the `sift-128-euclidean` dataset:

.. code-block:: yaml

    - name: sift-128-euclidean
      base_file: sift-128-euclidean/base.fbin
      query_file: sift-128-euclidean/query.fbin
      groundtruth_neighbors_file: sift-128-euclidean/groundtruth.neighbors.ibin
      dims: 128
      distance: euclidean

Configuration files for ANN algorithms supported by `cuvs-bench` are provided in `${CUVS_HOME}/python/cuvs_bench/cuvs_bench/config/algos`. `cuvs_cagra` algorithm configuration looks like:

.. code-block:: yaml

    name: cuvs_cagra
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

The default parameters used to run the benchmarks can be overridden by creating a custom YAML file for algorithms with a `base` group.

The previous config has two fields:
1. `name` - define the name of the algorithm for which the parameters are being specified.
2. `groups` - define a run group which has a particular set of parameters. Each group helps create a cross-product of all hyper-parameter fields for `build` and `search`.

The following table contains all algorithms supported by cuVS. Each unique algorithm will have its own set of `build` and `search` settings. The :doc:`ANN Algorithm Parameter Tuning Guide <param_tuning>` contains detailed instructions on choosing build and search parameters for each supported algorithm.

.. list-table::

 * - Library
   - Algorithms

 * - FAISS_GPU
   - `faiss_gpu_flat`, `faiss_gpu_ivf_flat`, `faiss_gpu_ivf_pq`, `faiss_gpu_cagra`

 * - FAISS_CPU
   - `faiss_cpu_flat`, `faiss_cpu_ivf_flat`, `faiss_cpu_ivf_pq`, `faiss_cpu_hnsw_flat`

 * - GGNN
   - `ggnn`

 * - HNSWLIB
   - `hnswlib`

 * - DiskANN
   - `diskann_memory`, `diskann_ssd`

 * - cuVS
   - `cuvs_brute_force`, `cuvs_cagra`, `cuvs_ivf_flat`, `cuvs_ivf_pq`, `cuvs_cagra_hnswlib`, `cuvs_vamana`


Multi-GPU Benchmarks
--------------------

cuVS implements single-node multi-GPU versions of IVF-Flat, IVF-PQ, and CAGRA indexes.

.. list-table::

 * - Index type
   - Multi-GPU algo name

 * - IVF-Flat
   - `cuvs_mg_ivf_flat`

 * - IVF-PQ
   - `cuvs_mg_ivf_pq`

 * - CAGRA
   - `cuvs_mg_cagra`


Add a New Index Algorithm
**************************

[ZAR Comment]: Let's add a brief overview regarding the "why" of this section.

Implementing and Configuring
--------------------------------

Implementation of a new algorithm should be a C++ class that inherits `class ANN` (defined in `cpp/bench/ann/src/ann.h`) and implements all the pure virtual functions.

In addition, it should define two `structs` for building and searching parameters. The searching parameter class should inherit `struct ANN<T>::AnnSearchParam`. Take `class HnswLib` as an example, its definition is:

.. code-block:: c++

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


The benchmark program uses JSON format natively in a configuration file to specify indexes to build, along with the build and search parameters. However, the JSON config files are overly verbose and are not meant to be used directly. Instead, the Python scripts parse YAML and automatically create these JSON files. It is important to realize that these JSON objects align with the YAML objects for `build_param`, whose value is a JSON object, and `search_param`, whose value is an array of JSON objects. Take the JSON configuration for `HnswLib` as an example of the json after it is parsed from YAML:

.. code-block:: json

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

The build and search params are ultimately passed to the C++ layer as json objects for each param configuration to benchmark. The code below shows how to parse these params for `Hnswlib`:

1. First, add two functions for parsing JSON objects to `struct BuildParam` and `struct SearchParam`, respectively:

.. code-block:: c++

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



2. Next, add corresponding `if` case to functions `create_algo()` (in `cpp/bench/ann/) and `create_search_param()` by calling parsing functions. The string literal in the `if` condition statement must be the same as the value of `algo` in the configuration file. For example,

.. code-block:: c++

      // JSON configuration file contains a line like:  "algo" : "hnswlib"
      if (algo == "hnswlib") {
         // ...
      }

Adding a Cmake Target
---------------------

In `cuvs/cpp/bench/ann/CMakeLists.txt`, the `CMake` function is available to configure a new Benchmark target with the following signature:


.. code-block:: cmake

    ConfigureAnnBench(
      NAME <algo_name>
      PATH </path/to/algo/benchmark/source/file>
      INCLUDES <additional_include_directories>
      CXXFLAGS <additional_cxx_flags>
      LINKS <additional_link_library_targets>
    )

To add a target for `HNSWLIB`, call the function as:

.. code-block:: cmake

    ConfigureAnnBench(
      NAME HNSWLIB PATH bench/ann/src/hnswlib/hnswlib_benchmark.cpp INCLUDES
      ${CMAKE_CURRENT_BINARY_DIR}/_deps/hnswlib-src/hnswlib CXXFLAGS "${HNSW_CXX_FLAGS}"
    )

This creates an executable called `HNSWLIB_ANN_BENCH`, which can then be used to run `HNSWLIB` benchmarks.

Add a new entry to `algos.yaml` that maps the algorithm name to its binary executable and specifies whether the algorithm requires GPU support.

.. code-block:: yaml

    cuvs_ivf_pq:
      executable: CUVS_IVF_PQ_ANN_BENCH
      requires_gpu: true

`executable`: specifies the name of the binary that builds and searches the index. It is assumed to be available in `cuvs/cpp/build/`.
`requires_gpu`: denotes whether an algorithm requires a GPU to run.


.. toctree::
   :maxdepth: 4

   build.rst
   datasets.rst
   param_tuning.rst
   wiki_all_dataset.rst
