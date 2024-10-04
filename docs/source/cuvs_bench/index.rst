~~~~~~~~~~
cuVS Bench
~~~~~~~~~~

cuVS bench provides a reproducible benchmarking tool for various ANN search implementations. It's especially suitable for comparing GPU implementations as well as comparing GPU against CPU. One of the primary goals of cuVS is to capture ideal index configurations for a variety of important usage patterns so the results can be reproduced easily on different hardware environments, such as on-prem and cloud.

This tool offers several benefits, including

#. Making fair comparisons of index build times

#. Making fair comparisons of index search throughput and/or latency

#. Finding the optimal parameter settings for a range of recall buckets

#. Easily generating consistently styled plots for index build and search

#. Profiling blind spots and potential for algorithm optimization

#. Investigating the relationship between different parameter settings, index build times, and search performance.

- `Installing the benchmarks`_

  * `Conda`_

  * `Docker`_

- `How to run the benchmarks`_

  * `Step 1: Prepare the dataset`_

  * `Step 2: Build and search index`_

  * `Step 3: Data export`_

  * `Step 4: Plot the results`_

- `Running the benchmarks`_

  * `End-to-end: smaller-scale benchmarks (<1M to 10M)`_

  * `End-to-end: large-scale benchmarks (>10M vectors)`_

  * `Running with Docker containers`_

    * `End-to-end run on GPU`_

    * `Manually run the scripts inside the container`_

  * `Evaluating the results`_

- `Creating and customizing dataset configurations`_

- `Adding a new index algorithm`_

  * `Implementation and configuration`_

  * `Adding a Cmake target`_

Installing the benchmarks
=========================

There are two main ways pre-compiled benchmarks are distributed:

- `Conda`_ For users not using containers but want an easy to install and use Python package. Pip wheels are planned to be added as an alternative for users that cannot use conda and prefer to not use containers.
- `Docker`_ Only needs docker and [NVIDIA docker](https://github.com/NVIDIA/nvidia-docker) to use. Provides a single docker run command for basic dataset benchmarking, as well as all the functionality of the conda solution inside the containers.

Conda
-----

.. code-block:: bash

   conda create --name cuvs_benchmarks
   conda activate cuvs_benchmarks

   # to install GPU package:
   conda install -c rapidsai -c conda-forge -c nvidia cuvs-ann-bench=<rapids_version> cuda-version=11.8*

   # to install CPU package for usage in CPU-only systems:
   conda install -c rapidsai -c conda-forge  cuvs-bench-cpu

The channel `rapidsai` can easily be substituted `rapidsai-nightly` if nightly benchmarks are desired. The CPU package currently allows to run the HNSW benchmarks.

Please see the :doc:`build instructions <build>` to build the benchmarks from source.

Docker
------

We provide images for GPU enabled systems, as well as systems without a GPU. The following images are available:

- `cuvs-bench`: Contains GPU and CPU benchmarks, can run all algorithms supported. Will download million-scale datasets as required. Best suited for users that prefer a smaller container size for GPU based systems. Requires the NVIDIA Container Toolkit to run GPU algorithms, can run CPU algorithms without it.
- `cuvs-bench-datasets`: Contains the GPU and CPU benchmarks with million-scale datasets already included in the container. Best suited for users that want to run multiple million scale datasets already included in the image.
- `cuvs-bench-cpu`: Contains only CPU benchmarks with minimal size. Best suited for users that want the smallest containers to reproduce benchmarks on systems without a GPU.

Nightly images are located in `dockerhub <https://hub.docker.com/r/rapidsai/cuvs-ann-bench/tags>`_, meanwhile release (stable) versions are located in `NGC <https://hub.docker.com/r/rapidsai/cuvs_bench>`_, starting with release 24.10.

The following command pulls the nightly container for python version 10, cuda version 12, and CUVS version 23.10:

.. code-block:: bash

   docker pull rapidsai/cuvs_bench:24.10a-cuda12.0-py3.10 #substitute cuvs_bench for the exact desired container.

The CUDA and python versions can be changed for the supported values:
- Supported CUDA versions: 11.4 and 12.x
- Supported Python versions: 3.9 and 3.10.

You can see the exact versions as well in the dockerhub site:
- `cuVS bench images <https://hub.docker.com/r/rapidsai/cuvs_bench/tags>`_
- `cuVS bench with datasets preloaded images <https://hub.docker.com/r/rapidsai/cuvs-bench-cpu/tags>`_
- `cuVS bench CPU only images <https://hub.docker.com/r/rapidsai/cuvs-bench-datasets/tags>`_

**Note:** GPU containers use the CUDA toolkit from inside the container, the only requirement is a driver installed on the host machine that supports that version. So, for example, CUDA 11.8 containers can run in systems with a CUDA 12.x capable driver. Please also note that the Nvidia-Docker runtime from the `Nvidia Container Toolkit <https://github.com/NVIDIA/nvidia-docker>`_ is required to use GPUs inside docker containers.

How to run the benchmarks
=========================

We provide a collection of lightweight Python scripts to run the benchmarks. There are 4 general steps to running the benchmarks and visualizing the results.
#. Prepare Dataset
#. Build Index and Search Index
#. Data Export
#. Plot Results

Step 1: Prepare the dataset
---------------------------

The script `cuvs_bench.get_dataset` will download and unpack the dataset in directory that the user provides. As of now, only million-scale datasets are supported by this script. For more information on :doc:`datasets and formats <datasets>`.

The usage of this script is:

.. code-block:: bash

    usage: get_dataset.py [-h] [--name NAME] [--dataset-path DATASET_PATH] [--normalize]

    options:
      -h, --help            show this help message and exit
      --dataset DATASET     dataset to download (default: glove-100-angular)
      --dataset-path DATASET_PATH
                            path to download dataset (default: ${RAPIDS_DATASET_ROOT_DIR})
      --normalize           normalize cosine distance to inner product (default: False)

When option `normalize` is provided to the script, any dataset that has cosine distances
will be normalized to inner product. So, for example, the dataset `glove-100-angular`
will be written at location `datasets/glove-100-inner/`.

Step 2: Build and search index
------------------------------

The script `cuvs_bench.run` will build and search indices for a given dataset and its
specified configuration.

The usage of the script `cuvs_bench.run` is:

.. code-block:: bash

    usage: __main__.py [-h] [--subset-size SUBSET_SIZE] [-k COUNT] [-bs BATCH_SIZE] [--dataset-configuration DATASET_CONFIGURATION] [--configuration CONFIGURATION] [--dataset DATASET]
                       [--dataset-path DATASET_PATH] [--build] [--search] [--algorithms ALGORITHMS] [--groups GROUPS] [--algo-groups ALGO_GROUPS] [-f] [-m SEARCH_MODE]

    options:
      -h, --help            show this help message and exit
      --subset-size SUBSET_SIZE
                            the number of subset rows of the dataset to build the index (default: None)
      -k COUNT, --count COUNT
                            the number of nearest neighbors to search for (default: 10)
      -bs BATCH_SIZE, --batch-size BATCH_SIZE
                            number of query vectors to use in each query trial (default: 10000)
      --dataset-configuration DATASET_CONFIGURATION
                            path to YAML configuration file for datasets (default: None)
      --configuration CONFIGURATION
                            path to YAML configuration file or directory for algorithms Any run groups found in the specified file/directory will automatically override groups of the same name
                            present in the default configurations, including `base` (default: None)
      --dataset DATASET     name of dataset (default: glove-100-inner)
      --dataset-path DATASET_PATH
                            path to dataset folder, by default will look in RAPIDS_DATASET_ROOT_DIR if defined, otherwise a datasets subdirectory from the calling directory (default:
                            os.getcwd()/datasets/)
      --build
      --search
      --algorithms ALGORITHMS
                            run only comma separated list of named algorithms. If parameters `groups` and `algo-groups` are both undefined, then group `base` is run by default (default: None)
      --groups GROUPS       run only comma separated groups of parameters (default: base)
      --algo-groups ALGO_GROUPS
                            add comma separated <algorithm>.<group> to run. Example usage: "--algo-groups=cuvs_cagra.large,hnswlib.large" (default: None)
      -f, --force           re-run algorithms even if their results already exist (default: False)
      -m SEARCH_MODE, --search-mode SEARCH_MODE
                            run search in 'latency' (measure individual batches) or 'throughput' (pipeline batches and measure end-to-end) mode (default: throughput)
      -t SEARCH_THREADS, --search-threads SEARCH_THREADS
                            specify the number threads to use for throughput benchmark. Single value or a pair of min and max separated by ':'. Example --search-threads=1:4. Power of 2 values between 'min' and 'max' will be used. If only 'min' is
                            specified, then a single test is run with 'min' threads. By default min=1, max=<num hyper threads>. (default: None)
      -r, --dry-run         dry-run mode will convert the yaml config for the specified algorithms and datasets to the json format that's consumed by the lower-level c++ binaries and then print the command to run execute the benchmarks but
                            will not actually execute the command. (default: False)

`dataset`: name of the dataset to be searched in `datasets.yaml`_

`dataset-configuration`: optional filepath to custom dataset YAML config which has an entry for arg `dataset`

`configuration`: optional filepath to YAML configuration for an algorithm or to directory that contains YAML configurations for several algorithms. Refer to `Dataset.yaml config`_ for more info.

`algorithms`: runs all algorithms that it can find in YAML configs found by `configuration`. By default, only `base` group will be run.

`groups`: run only specific groups of parameters configurations for an algorithm. Groups are defined in YAML configs (see `configuration`), and by default run `base` group

`algo-groups`: this parameter is helpful to append any specific algorithm+group combination to run the benchmark for in addition to all the arguments from `algorithms` and `groups`. It is of the format `<algorithm>.<group>`, or for example, `cuvs_cagra.large`

For every algorithm run by this script, it outputs an index build statistics JSON file in `<dataset-path/<dataset>/result/build/<{algo},{group}.json>`
and an index search statistics JSON file in `<dataset-path/<dataset>/result/search/<{algo},{group},k{k},bs{batch_size}.json>`. NOTE: The filenames will not have ",{group}" if `group = "base"`.

For every algorithm run by this script, it outputs an index build statistics JSON file in `<dataset-path/<dataset>/result/build/<{algo},{group}.json>`
and an index search statistics JSON file in `<dataset-path/<dataset>/result/search/<{algo},{group},k{k},bs{batch_size}.json>`. NOTE: The filenames will not have ",{group}" if `group = "base"`.

`dataset-path` :
#. data is read from `<dataset-path>/<dataset>`
#. indices are built in `<dataset-path>/<dataset>/index`
#. build/search results are stored in `<dataset-path>/<dataset>/result`

`build` and `search` : if both parameters are not supplied to the script then it is assumed both are `True`.

`indices` and `algorithms` : these parameters ensure that the algorithm specified for an index is available in `algos.yaml` and not disabled, as well as having an associated executable.

Step 3: Data export
-------------------

The script `cuvs_bench.data_export` will convert the intermediate JSON outputs produced by `cuvs_bench.run` to more easily readable CSV files, which are needed to build charts made by `cuvs_bench.plot`.

.. code-block:: bash

    usage: data_export.py [-h] [--dataset DATASET] [--dataset-path DATASET_PATH]

    options:
      -h, --help            show this help message and exit
      --dataset DATASET     dataset to download (default: glove-100-inner)
      --dataset-path DATASET_PATH
                            path to dataset folder (default: ${RAPIDS_DATASET_ROOT_DIR})

Build statistics CSV file is stored in `<dataset-path/<dataset>/result/build/<{algo},{group}.csv>`
and index search statistics CSV file in `<dataset-path/<dataset>/result/search/<{algo},{group},k{k},bs{batch_size},{suffix}.csv>`, where suffix has three values:
#. `raw`: All search results are exported
#. `throughput`: Pareto frontier of throughput results is exported
#. `latency`: Pareto frontier of latency results is exported

Step 4: Plot the results
------------------------

The script `cuvs_bench.plot` will plot results for all algorithms found in index search statistics CSV files `<dataset-path/<dataset>/result/search/*.csv`.

The usage of this script is:

.. code-block:: bash

    usage:  [-h] [--dataset DATASET] [--dataset-path DATASET_PATH] [--output-filepath OUTPUT_FILEPATH] [--algorithms ALGORITHMS] [--groups GROUPS] [--algo-groups ALGO_GROUPS]
            [-k COUNT] [-bs BATCH_SIZE] [--build] [--search] [--x-scale X_SCALE] [--y-scale {linear,log,symlog,logit}] [--x-start X_START] [--mode {throughput,latency}]
            [--time-unit {s,ms,us}] [--raw]

    options:
      -h, --help            show this help message and exit
      --dataset DATASET     dataset to plot (default: glove-100-inner)
      --dataset-path DATASET_PATH
                            path to dataset folder (default: /home/coder/cuvs/datasets/)
      --output-filepath OUTPUT_FILEPATH
                            directory for PNG to be saved (default: /home/coder/cuvs)
      --algorithms ALGORITHMS
                            plot only comma separated list of named algorithms. If parameters `groups` and `algo-groups are both undefined, then group `base` is plot by default
                            (default: None)
      --groups GROUPS       plot only comma separated groups of parameters (default: base)
      --algo-groups ALGO_GROUPS, --algo-groups ALGO_GROUPS
                            add comma separated <algorithm>.<group> to plot. Example usage: "--algo-groups=cuvs_cagra.large,hnswlib.large" (default: None)
      -k COUNT, --count COUNT
                            the number of nearest neighbors to search for (default: 10)
      -bs BATCH_SIZE, --batch-size BATCH_SIZE
                            number of query vectors to use in each query trial (default: 10000)
      --build
      --search
      --x-scale X_SCALE     Scale to use when drawing the X-axis. Typically linear, logit or a2 (default: linear)
      --y-scale {linear,log,symlog,logit}
                            Scale to use when drawing the Y-axis (default: linear)
      --x-start X_START     Recall values to start the x-axis from (default: 0.8)
      --mode {throughput,latency}
                            search mode whose Pareto frontier is used on the y-axis (default: throughput)
      --time-unit {s,ms,us}
                            time unit to plot when mode is latency (default: ms)
      --raw                 Show raw results (not just Pareto frontier) of mode arg (default: False)

`mode`: plots pareto frontier of `throughput` or `latency` results exported in the previous step

`algorithms`: plots all algorithms that it can find results for the specified `dataset`. By default, only `base` group will be plotted.

`groups`: plot only specific groups of parameters configurations for an algorithm. Groups are defined in YAML configs (see `configuration`), and by default run `base` group

`algo-groups`: this parameter is helpful to append any specific algorithm+group combination to plot results for in addition to all the arguments from `algorithms` and `groups`. It is of the format `<algorithm>.<group>`, or for example, `cuvs_cagra.large`

Running the benchmarks
======================

End-to-end: smaller-scale benchmarks (<1M to 10M)
-------------------------------------------------

The steps below demonstrate how to download, install, and run benchmarks on a subset of 10M vectors from the Yandex Deep-1B dataset By default the datasets will be stored and used from the folder indicated by the `RAPIDS_DATASET_ROOT_DIR` environment variable if defined, otherwise a datasets sub-folder from where the script is being called:

.. code-block:: bash


    # (1) prepare dataset.
    python -m cuvs_bench.get_dataset --dataset deep-image-96-angular --normalize

    # (2) build and search index
    python -m cuvs_bench.run --dataset deep-image-96-inner --algorithms cuvs_cagra --batch-size 10 -k 10

    # (3) export data
    python -m cuvs_bench.data_export --dataset deep-image-96-inner

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

All of the datasets above contain ground test datasets with 100 neighbors. Thus `k` for these datasets must be  less than or equal to 100.

End-to-end: large-scale benchmarks (>10M vectors)
-------------------------------------------------

`cuvs_bench.get_dataset` cannot be used to download the `billion-scale datasets`_ due to their size. You should instead use our billion-scale datasets guide to download and prepare them.
All other python commands mentioned below work as intended once the billion-scale dataset has been downloaded.

To download billion-scale datasets, visit `big-ann-benchmarks <http://big-ann-benchmarks.com/neurips21.html>`_

We also provide a new dataset called `wiki-all` containing 88 million 768-dimensional vectors. This dataset is meant for benchmarking a realistic retrieval-augmented generation (RAG)/LLM embedding size at scale. It also contains 1M and 10M vector subsets for smaller-scale experiments. See our :doc:`Wiki-all Dataset Guide <wiki_all_dataset>` for more information and to download the dataset.


The steps below demonstrate how to download, install, and run benchmarks on a subset of 100M vectors from the Yandex Deep-1B dataset. Please note that datasets of this scale are recommended for GPUs with larger amounts of memory, such as the A100 or H100.

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
    python -m cuvs_bench.data_export --dataset deep-1B

    # (4) plot results
    python -m cuvs_bench.plot --dataset deep-1B

The usage of `python -m cuvs_bench.split_groundtruth` is:

.. code-block:: bash
    usage: split_groundtruth.py [-h] --groundtruth GROUNDTRUTH

    options:
      -h, --help            show this help message and exit
      --groundtruth GROUNDTRUTH
                            Path to billion-scale dataset groundtruth file (default: None)

Running with Docker containers
------------------------------

Two methods are provided for running the benchmarks with the Docker containers.

End-to-end run on GPU
~~~~~~~~~~~~~~~~~~~~~

When no other entrypoint is provided, an end-to-end script will run through all the steps in `Running the benchmarks`_ above.

For GPU-enabled systems, the `DATA_FOLDER` variable should be a local folder where you want datasets stored in `$DATA_FOLDER/datasets` and results in `$DATA_FOLDER/result` (we highly recommend `$DATA_FOLDER` to be a dedicated folder for the datasets and results of the containers):

.. code-block:: bash

    export DATA_FOLDER=path/to/store/datasets/and/results
    docker run --gpus all --rm -it -u $(id -u)                      \
        -v $DATA_FOLDER:/data/benchmarks                            \
        rapidsai/cuvs-bench:24.10a-cuda11.8-py3.10              \
        "--dataset deep-image-96-angular"                           \
        "--normalize"                                               \
        "--algorithms cuvs_cagra,cuvs_ivf_pq --batch-size 10 -k 10" \
        ""

Usage of the above command is as follows:

.. list-table::

 * - Argument
   - Description

 * - `rapidsai/cuvs-bench:24.10a-cuda11.8-py3.10`
   - Image to use. Can be either `cuvs-bench` or `cuvs-bench-datasets`

 * - `"--dataset deep-image-96-angular"`
   - Dataset name

 * - `"--normalize"`
   - Whether to normalize the dataset

 * - `"--algorithms cuvs_cagra,hnswlib --batch-size 10 -k 10"`
   - Arguments passed to the `run` script, such as the algorithms to benchmark, the batch size, and `k`

 * - `""`
   - Additional (optional) arguments that will be passed to the `plot` script.

***Note about user and file permissions:*** The flag `-u $(id -u)` allows the user inside the container to match the `uid` of the user outside the container, allowing the container to read and write to the mounted volume indicated by the `$DATA_FOLDER` variable.

End-to-end run on CPU
~~~~~~~~~~~~~~~~~~~~~

The container arguments in the above section also be used for the CPU-only container, which can be used on systems that don't have a GPU installed.

***Note:*** the image changes to `cuvs-bench-cpu` container and the `--gpus all` argument is no longer used:

.. code-block:: bash

    export DATA_FOLDER=path/to/store/datasets/and/results
    docker run  --rm -it -u $(id -u)                  \
        -v $DATA_FOLDER:/data/benchmarks              \
        rapidsai/cuvs-bench-cpu:24.10a-py3.10     \
         "--dataset deep-image-96-angular"            \
         "--normalize"                                \
         "--algorithms hnswlib --batch-size 10 -k 10" \
         ""

Manually run the scripts inside the container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All of the `cuvs-bench` images contain the Conda packages, so they can be used directly by logging directly into the container itself:

.. code-block:: bash

    export DATA_FOLDER=path/to/store/datasets/and/results
    docker run --gpus all --rm -it -u $(id -u)          \
        --entrypoint /bin/bash                          \
        --workdir /data/benchmarks                      \
        -v $DATA_FOLDER:/data/benchmarks                \
        rapidsai/cuvs-bench:24.10a-cuda11.8-py3.10

This will drop you into a command line in the container, with the `cuvs-bench` python package ready to use, as described in the [Running the benchmarks](#running-the-benchmarks) section above:

.. code-block:: bash

    (base) root@00b068fbb862:/data/benchmarks# python -m cuvs_bench.get_dataset --dataset deep-image-96-angular --normalize

Additionally, the containers can be run in detached mode without any issue.

Evaluating the results
----------------------

The benchmarks capture several different measurements. The table below describes each of the measurements for index build benchmarks:

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
   - Number of vectors used to train index

The table below describes each of the measurements for the index search benchmarks. The most important measurements `Latency`, `items_per_second`, `end_to_end`.

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
   - GPU latency of a single batch (seconds). In throughput mode this is averaged over multiple threads.

 * - Latency
   - Latency of a single batch (seconds), calculated from wall-clock time. In throughput mode this is averaged over multiple threads.

 * - Recall
   - Proportion of correct neighbors to ground truth neighbors. Note this column is only present if groundtruth file is specified in dataset configuration.

 * - items_per_second
   - Total throughput, a.k.a Queries per second (QPS). This is approximately `total_queries` / `end_to_end`.

 * - k
   - Number of neighbors being queried in each iteration

 * - end_to_end
   - Total time taken to run all batches for all iterations

 * - n_queries
   - Total number of query vectors in each batch

 * - total_queries
   - Total number of vectors queries across all iterations ( = `iterations` * `n_queries`)

Note the following:
- A slightly different method is used to measure `Time` and `end_to_end`. That is why `end_to_end` = `Time` * `Iterations` holds only approximately.
- The actual table displayed on the screen may differ slightly as the hyper-parameters will also be displayed for each different combination being benchmarked.
- Recall calculation: the number of queries processed per test depends on the number of iterations. Because of this, recall can show slight fluctuations if less neighbors are processed then it is available for the benchmark.

Creating and customizing dataset configurations
===============================================

A single configuration will often define a set of algorithms, with associated index and search parameters, that can be generalize across datasets. We use YAML to define dataset specific and algorithm specific configurations.

A default `datasets.yaml` is provided by CUVS in `${CUVS_HOME}/python/cuvs-ann-bench/src/cuvs_bench/run/conf` with configurations available for several datasets. Here's a simple example entry for the `sift-128-euclidean` dataset:

.. code-block:: yaml

    - name: sift-128-euclidean
      base_file: sift-128-euclidean/base.fbin
      query_file: sift-128-euclidean/query.fbin
      groundtruth_neighbors_file: sift-128-euclidean/groundtruth.neighbors.ibin
      dims: 128
      distance: euclidean

Configuration files for ANN algorithms supported by `cuvs-bench` are provided in `${CUVS_HOME}/python/cuvs-bench/src/cuvs_bench/run/conf`. `cuvs_cagra` algorithm configuration looks like:

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

The default parameters for which the benchmarks are run can be overridden by creating a custom YAML file for algorithms with a `base` group.

There config above has 2 fields:
1. `name` - define the name of the algorithm for which the parameters are being specified.
2. `groups` - define a run group which has a particular set of parameters. Each group helps create a cross-product of all hyper-parameter fields for `build` and `search`.

The table below contains all algorithms supported by cuVS. Each unique algorithm will have its own set of `build` and `search` settings. The :doc:`ANN Algorithm Parameter Tuning Guide <param_tuning>` contains detailed instructions on choosing build and search parameters for each supported algorithm.

.. list-table::

 * - Library
   - Algorithms

 * - FAISS_GPU
   - `faiss_gpu_flat`, `faiss_gpu_ivf_flat`, `faiss_gpu_ivf_pq`

 * - FAISS_CPU
   - `faiss_cpu_flat`, `faiss_cpu_ivf_flat`, `faiss_cpu_ivf_pq`

 * - GGNN
   - `ggnn`

 * - HNSWLIB
   - `hnswlib`

 * - cuVS
   - `cuvs_brute_force`, `cuvs_cagra`, `cuvs_ivf_flat`, `cuvs_ivf_pq`, `cuvs_cagra_hnswlib`

Adding a new index algorithm
=============================

Implementation and configuration
--------------------------------

Implementation of a new algorithm should be a C++ class that inherits `class ANN` (defined in `cpp/bench/ann/src/ann.h`) and implements all the pure virtual functions.

In addition, it should define two `struct`s for building and searching parameters. The searching parameter class should inherit `struct ANN<T>::AnnSearchParam`. Take `class HnswLib` as an example, its definition is:

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


The benchmark program uses JSON format natively in a configuration file to specify indexes to build, along with the build and search parameters. However the JSON config files are overly verbose and are not meant to be used directly. Instead, the Python scripts parse YAML and create these json files automatically. It's important to realize that these json objects align with the yaml objects for `build_param`, whose value is a JSON object, and `search_param`, whose value is an array of JSON objects. Take the json configuration for `HnswLib` as an example of the json after it's been parsed from yaml:

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

1. First, add two functions for parsing JSON object to `struct BuildParam` and `struct SearchParam`, respectively:

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



2. Next, add corresponding `if` case to functions `create_algo()` (in `cpp/bench/ann/) and `create_search_param()` by calling parsing functions. The string literal in `if` condition statement must be the same as the value of `algo` in configuration file. For example,

.. code-block:: c++
      // JSON configuration file contains a line like:  "algo" : "hnswlib"
      if (algo == "hnswlib") {
         // ...
      }

Adding a Cmake target
---------------------

In `cuvs/cpp/bench/ann/CMakeLists.txt`, we provide a `CMake` function to configure a new Benchmark target with the following signature:


.. code-block:: cmake
    ConfigureAnnBench(
      NAME <algo_name>
      PATH </path/to/algo/benchmark/source/file>
      INCLUDES <additional_include_directories>
      CXXFLAGS <additional_cxx_flags>
      LINKS <additional_link_library_targets>
    )

To add a target for `HNSWLIB`, we would call the function as:

.. code-block:: cmake

    ConfigureAnnBench(
      NAME HNSWLIB PATH bench/ann/src/hnswlib/hnswlib_benchmark.cpp INCLUDES
      ${CMAKE_CURRENT_BINARY_DIR}/_deps/hnswlib-src/hnswlib CXXFLAGS "${HNSW_CXX_FLAGS}"
    )

This will create an executable called `HNSWLIB_ANN_BENCH`, which can then be used to run `HNSWLIB` benchmarks.

Add a new entry to `algos.yaml` to map the name of the algorithm to its binary executable and specify whether the algorithm requires GPU support.

.. code-block:: yaml
    cuvs_ivf_pq:
      executable: CUVS_IVF_PQ_ANN_BENCH
      requires_gpu: true

`executable` : specifies the name of the binary that will build/search the index. It is assumed to be available in `cuvs/cpp/build/`.
`requires_gpu` : denotes whether an algorithm requires GPU to run.