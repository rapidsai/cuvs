# Build cuVS Bench From Source

## Dependencies

CUDA 12 and a GPU with Volta architecture or later are required to run the benchmarks.

Please refer to the `installation docs <../build>` for the base requirements to build cuVS.

In addition to the base requirements for building cuVS, additional dependencies needed to build the ANN benchmarks include:

1.  FAISS GPU \>= 1.7.1
2.  Google Logging (GLog)
3.  H5Py
4.  HNSWLib
5.  nlohmann_json
6.  GGNN

[rapids-cmake](https://github.com/rapidsai/rapids-cmake) is used to build the ANN benchmarks so the code for dependencies not already supplied in the CUDA toolkit will be downloaded and built automatically.

The easiest (and most reproducible) way to install the dependencies needed to build the ANN benchmarks is to use the conda environment file located in the <span class="title-ref">conda/environments</span> directory of the cuVS repository. The following command will use <span class="title-ref">mamba</span> (which is preferred over <span class="title-ref">conda</span>) to build and activate a new environment for compiling the benchmarks:

``` bash
conda env create --name cuvs_benchmarks -f conda/environments/bench_ann_cuda-131_arch-$(uname -m).yaml
conda activate cuvs_benchmarks
```

The above conda environment will also reduce the compile times as dependencies like FAISS will already be installed and not need to be compiled with <span class="title-ref">rapids-cmake</span>.

## Compiling the Benchmarks

After the needed dependencies are satisfied, the easiest way to compile ANN benchmarks is through the <span class="title-ref">build.sh</span> script in the root of the RAFT source code repository. The following will build the executables for all the support algorithms:

``` bash
./build.sh bench-ann
```

You can limit the algorithms that are built by providing a semicolon-delimited list of executable names (each algorithm is suffixed with <span class="title-ref">\_ANN_BENCH</span>):

``` bash
./build.sh bench-ann -n --limit-bench-ann=HNSWLIB_ANN_BENCH;CUVS_IVF_PQ_ANN_BENCH
```

Available targets to use with <span class="title-ref">--limit-bench-ann</span> are:

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

By default, the <span class="title-ref">\*\_ANN_BENCH</span> executables program infer the dataset's datatype from the filename's extension. For example, an extension of <span class="title-ref">fbin</span> uses a <span class="title-ref">float</span> datatype, <span class="title-ref">f16bin</span> uses a <span class="title-ref">float16</span> datatype, extension of <span class="title-ref">i8bin</span> uses <span class="title-ref">int8_t</span> datatype, and <span class="title-ref">u8bin</span> uses <span class="title-ref">uint8_t</span> type. Currently, only <span class="title-ref">float</span>, <span class="title-ref">float16</span>, <span class="title-ref">int8_t</span>, and <span class="title-ref">unit8_t</span> are supported.
