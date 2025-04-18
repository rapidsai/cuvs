~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Build cuVS Bench From Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dependencies
============

CUDA 11 and a GPU with Volta architecture or later are required to run the benchmarks.

Please refer to the  :doc:`installation docs <../build>` for the base requirements to build cuVS.

In addition to the base requirements for building cuVS, additional dependencies needed to build the ANN benchmarks include:
1. FAISS GPU >= 1.7.1
2. Google Logging (GLog)
3. H5Py
4. HNSWLib
5. nlohmann_json
6. GGNN

`rapids-cmake <https://github.com/rapidsai/rapids-cmake>`_ is used to build the ANN benchmarks so the code for dependencies not already supplied in the CUDA toolkit will be downloaded and built automatically.

The easiest (and most reproducible) way to install the dependencies needed to build the ANN benchmarks is to use the conda environment file located in the `conda/environments` directory of the cuVS repository. The following command will use `mamba` (which is preferred over `conda`) to build and activate a new environment for compiling the benchmarks:

.. code-block:: bash

    conda env create --name cuvs_benchmarks -f conda/environments/cuvs_bench_cuda-118_arch-x86_64.yaml
    conda activate cuvs_benchmarks

The above conda environment will also reduce the compile times as dependencies like FAISS will already be installed and not need to be compiled with `rapids-cmake`.

Compiling the Benchmarks
========================

After the needed dependencies are satisfied, the easiest way to compile ANN benchmarks is through the `build.sh` script in the root of the RAFT source code repository. The following will build the executables for all the support algorithms:

.. code-block:: bash

    ./build.sh bench-ann

You can limit the algorithms that are built by providing a semicolon-delimited list of executable names (each algorithm is suffixed with `_ANN_BENCH`):

.. code-block:: bash

    ./build.sh bench-ann -n --limit-bench-ann=HNSWLIB_ANN_BENCH;CUVS_IVF_PQ_ANN_BENCH

Available targets to use with `--limit-bench-ann` are:
- FAISS_GPU_IVF_FLAT_ANN_BENCH
- FAISS_GPU_IVF_PQ_ANN_BENCH
- FAISS_CPU_IVF_FLAT_ANN_BENCH
- FAISS_CPU_IVF_PQ_ANN_BENCH
- FAISS_GPU_FLAT_ANN_BENCH
- FAISS_CPU_FLAT_ANN_BENCH
- GGNN_ANN_BENCH
- HNSWLIB_ANN_BENCH
- CUVS_CAGRA_ANN_BENCH
- CUVS_IVF_PQ_ANN_BENCH
- CUVS_IVF_FLAT_ANN_BENCH

By default, the `*_ANN_BENCH` executables program infer the dataset's datatype from the filename's extension. For example, an extension of `fbin` uses a `float` datatype, `f16bin` uses a `float16` datatype, extension of `i8bin` uses `int8_t` datatype, and `u8bin` uses `uint8_t` type. Currently, only `float`, `float16`, int8_t`, and `unit8_t` are supported.
