# cuVS Bench Parameter Tuning Guide

This guide outlines the various parameter settings that can be specified in `cuVS Benchmarks <index>` yaml configuration files and explains the impact they have on corresponding algorithms to help inform their settings for benchmarking across desired levels of recall.

## Benchmark modes

When you run benchmarks with `BenchmarkOrchestrator.run_benchmark()`, you can choose how parameters are explored:

**Sweep mode (default)**

Pass `mode="sweep"` or omit `mode`. The orchestrator builds the full Cartesian product of all build and search parameter lists defined in the algorithm YAML (see `Creating and customizing dataset configurations <index>`). Every valid combination (after constraint filtering) is run. Use this for exhaustive comparison across the configured parameter grid.

**Tune mode**

Pass `mode="tune"` to perform hyperparameter optimization using Optuna instead of running every combination. You must pass:

- **constraints** (dict): The optimization target and optional bounds. One metric must be `"maximize"` or `"minimize"` (the goal). Others can set hard limits with `{"min": X}` or `{"max": X}`. Examples: `{"recall": "maximize", "latency": {"max": 10}}` or `{"latency": "minimize", "recall": {"min": 0.95}}`.
- **n_trials** (int, optional): Maximum number of Optuna trials (default 100). Ignored in sweep mode.

Example:

``` python
results = orchestrator.run_benchmark(
    mode="tune",
    dataset="deep-image-96-inner",
    algorithms="cuvs_cagra",
    constraints={"recall": "maximize", "latency": {"max": 5.0}},
    n_trials=50,
    count=10,
    batch_size=10,
)
```

The parameter tables below describe the build and search knobs that sweep mode varies and that tune mode can optimize.

## cuVS Indexes

### cuvs_brute_force

Use cuVS brute-force index for exact search. Brute-force has no further build or search parameters.

### cuvs_ivf_flat

IVF-flat uses an inverted-file index, which partitions the vectors into a series of clusters, or lists, storing them in an interleaved format which is optimized for fast distance computation. The searching of an IVF-flat index reduces the total vectors in the index to those within some user-specified nearest clusters called probes.

IVF-flat is a simple algorithm which won't save any space, but it provides competitive search times even at higher levels of recall.

|  |  |  |  |  |  |
|----|----|----|----|----|----|
| Parameter | Type | Required | Data Type | Default | Description |
| <span class="title-ref">nlist</span> | <span class="title-ref">build</span> | Y | Positive integer \>0 | 1024 | Number of clusters to partition the vectors into. Larger values will put less points into each cluster but this will impact index build time as more clusters need to be trained. |
| <span class="title-ref">niter</span> | <span class="title-ref">build</span> | N | Positive integer \>0 | 20 | Number of kmeans iterations to use when training the ivf clusters |
| <span class="title-ref">ratio</span> | <span class="title-ref">build</span> | N | Positive integer \>0 | 2 | <span class="title-ref">1/ratio</span> is the number of training points which should be used to train the clusters. |
| <span class="title-ref">dataset_memory_type</span> | <span class="title-ref">build</span> | N | \[<span class="title-ref">device</span>, <span class="title-ref">host</span>, <span class="title-ref">mmap</span>\] | <span class="title-ref">mmap</span> | Where should the dataset reside? |
| <span class="title-ref">query_memory_type</span> | <span class="title-ref">search</span> | N | \[<span class="title-ref">device</span>, <span class="title-ref">host</span>, <span class="title-ref">mmap</span>\] | <span class="title-ref">device</span> | Where should the queries reside? |
| <span class="title-ref">nprobe</span> | <span class="title-ref">search</span> | Y | Positive integer \>0 |  | The closest number of clusters to search for each query vector. Larger values will improve recall but will search more points in the index. |

### cuvs_ivf_pq

IVF-pq is an inverted-file index, which partitions the vectors into a series of clusters, or lists, in a similar way to IVF-flat above. The difference is that IVF-PQ uses product quantization to also compress the vectors, giving the index a smaller memory footprint. Unfortunately, higher levels of compression can also shrink recall, which a refinement step can improve when the original vectors are still available.

|  |  |  |  |  |  |
|----|----|----|----|----|----|
| Parameter | Type | Required | Data Type | Default | Description |
| <span class="title-ref">nlist</span> | <span class="title-ref">build</span> | Y | Positive integer \>0 | 1024 | Number of clusters to partition the vectors into. Larger values will put less points into each cluster but this will impact index build time as more clusters need to be trained. |
| <span class="title-ref">niter</span> | <span class="title-ref">build</span> | N | Positive integer \>0 | 20 | Number of kmeans iterations to use when training the ivf clusters |
| <span class="title-ref">ratio</span> | <span class="title-ref">build</span> | N | Positive integer \>0 | 2 | <span class="title-ref">1/ratio</span> is the number of training points which should be used to train the clusters. |
| <span class="title-ref">pq_dim</span> | <span class="title-ref">build</span> | N | Positive integer. Multiple of 8. | 0 | Dimensionality of the vector after product quantization. When 0, a heuristic is used to select this value. |
| <span class="title-ref">pq_bits</span> | <span class="title-ref">build</span> | N | Positive integer \[4-8\] | 8 | Bit length of the vector element after quantization. |
| <span class="title-ref">codebook_kind</span> | <span class="title-ref">build</span> | N | \[<span class="title-ref">cluster</span>, <span class="title-ref">subspace</span>\] | <span class="title-ref">subspace</span> | Type of codebook. See `IVF-PQ index overview <../neighbors/ivfpq>` for more detail |
| <span class="title-ref">dataset_memory_type</span> | <span class="title-ref">build</span> | N | \[<span class="title-ref">device</span>, <span class="title-ref">host</span>, <span class="title-ref">mmap</span>\] | <span class="title-ref">mmap</span> | Where should the dataset reside? |
| <span class="title-ref">query_memory_type</span> | <span class="title-ref">search</span> | N | \[<span class="title-ref">device</span>, <span class="title-ref">host</span>, <span class="title-ref">mmap</span>\] | <span class="title-ref">device</span> | Where should the queries reside? |
| <span class="title-ref">nprobe</span> | <span class="title-ref">search</span> | Y | Positive integer \>0 | 20 | The closest number of clusters to search for each query vector. Larger values will improve recall but will search more points in the index. |
| <span class="title-ref">internalDistanceDtype</span> | <span class="title-ref">search</span> | N | \[<span class="title-ref">float</span>, <span class="title-ref">half</span>\] | <span class="title-ref">half</span> | The precision to use for the distance computations. Lower precision can increase performance at the cost of accuracy. |
| <span class="title-ref">smemLutDtype</span> | <span class="title-ref">search</span> | N | \[<span class="title-ref">float</span>, <span class="title-ref">half</span>, <span class="title-ref">fp8</span>\] | <span class="title-ref">half</span> | The precision to use for the lookup table in shared memory. Lower precision can increase performance at the cost of accuracy. |
| <span class="title-ref">refine_ratio</span> | <span class="title-ref">search</span> | N | Positive integer \>0 | 1 | <span class="title-ref">refine_ratio \* k</span> nearest neighbors are queried from the index initially and an additional refinement step improves recall by selecting only the best <span class="title-ref">k</span> neighbors. |

### cuvs_cagra

CAGRA uses a graph-based index, which creates an intermediate, approximate kNN graph using IVF-PQ and then further refining and optimizing to create a final kNN graph. This kNN graph is used by CAGRA as an index for search.

|  |  |  |  |  |  |
|----|----|----|----|----|----|
| Parameter | Type | Required | Data Type | Default | Description |
| <span class="title-ref">graph_degree</span> | <span class="title-ref">build</span> | N | Positive integer \>0 | 64 | Degree of the final kNN graph index. |
| <span class="title-ref">intermediate_graph_degree</span> | <span class="title-ref">build</span> | N | Positive integer \>0 | 128 | Degree of the intermediate kNN graph before the CAGRA graph is optimized |
| <span class="title-ref">graph_build_algo</span> | <span class="title-ref">build</span> | <span class="title-ref">N</span> | \[<span class="title-ref">IVF_PQ</span>, <span class="title-ref">NN_DESCENT</span>, <span class="title-ref">ACE</span>\] | <span class="title-ref">IVF_PQ</span> | Algorithm to use for building the initial kNN graph, from which CAGRA will optimize into the navigable CAGRA graph |
| <span class="title-ref">dataset_memory_type</span> | <span class="title-ref">build</span> | N | \[<span class="title-ref">device</span>, <span class="title-ref">host</span>, <span class="title-ref">mmap</span>\] | <span class="title-ref">mmap</span> | Where should the dataset reside? |
| <span class="title-ref">npartitions</span> | <span class="title-ref">build</span> | N | Positive integer \>0 | 1 | The number of partitions to use for the ACE build. Small values might improve recall but potentially degrade performance and increase memory usage. Partitions should not be too small to prevent issues in KNN graph construction. The partition size is on average 2 \* (n_rows / npartitions) \* dim \* sizeof(T). 2 is because of the core and augmented vectors. Please account for imbalance in the partition sizes (up to 3x in our tests). |
| <span class="title-ref">build_dir</span> | <span class="title-ref">build</span> | N | String | "/tmp/ace_build" | The directory to use for the ACE build. Must be specified when using ACE build. This should be the fastest disk in the system and hold enough space for twice the dataset, final graph, and label mapping. |
| <span class="title-ref">ef_construction</span> | <span class="title-ref">build</span> | Y | Positive integer \>0 | 120 | Controls index time and accuracy when using ACE build. Bigger values increase the index quality. At some point, increasing this will no longer improve the quality. |
| <span class="title-ref">use_disk</span> | <span class="title-ref">build</span> | N | Boolean | <span class="title-ref">false</span> | Whether to use disk-based storage for ACE build. When true, forces ACE to use disk-based storage even if the graph fits in host and GPU memory. When false, ACE will use in-memory storage if the graph fits in host and GPU memory and disk-based storage otherwise. |
| <span class="title-ref">query_memory_type</span> | <span class="title-ref">search</span> | N | \[<span class="title-ref">device</span>, <span class="title-ref">host</span>, <span class="title-ref">mmap</span>\] | <span class="title-ref">device</span> | Where should the queries reside? |
| <span class="title-ref">itopk</span> | <span class="title-ref">search</span> | N | Positive integer \>0 | 64 | Number of intermediate search results retained during the search. Higher values improve search accuracy at the cost of speed |
| <span class="title-ref">search_width</span> | <span class="title-ref">search</span> | N | Positive integer \>0 | 1 | Number of graph nodes to select as the starting point for the search in each iteration. |
| <span class="title-ref">max_iterations</span> | <span class="title-ref">search</span> | N | Positive integer \>=0 | 0 | Upper limit of search iterations. Auto select when 0 |
| <span class="title-ref">algo</span> | <span class="title-ref">search</span> | N | \[<span class="title-ref">auto</span>, <span class="title-ref">single_cta</span>, <span class="title-ref">multi_cta</span>, <span class="title-ref">multi_kernel</span>\] | <span class="title-ref">auto</span> | Algorithm to use for search. It's usually best to leave this to <span class="title-ref">auto</span>. |
| <span class="title-ref">graph_memory_type</span> | <span class="title-ref">search</span> | N | \[<span class="title-ref">device</span>, <span class="title-ref">host_pinned</span>, <span class="title-ref">host_huge_page</span>\] | <span class="title-ref">device</span> | Memory type to store graph |
| <span class="title-ref">internal_dataset_memory_type</span> | <span class="title-ref">search</span> | N | \[<span class="title-ref">device</span>, <span class="title-ref">host_pinned</span>, <span class="title-ref">host_huge_page</span>\] | <span class="title-ref">device</span> | Memory type to store dataset |

The <span class="title-ref">graph_memory_type</span> or <span class="title-ref">internal_dataset_memory_type</span> options can be useful for large datasets that do not fit the device memory. Setting <span class="title-ref">internal_dataset_memory_type</span> other than <span class="title-ref">device</span> has negative impact on search speed. Using <span class="title-ref">host_huge_page</span> option is only supported on systems with Heterogeneous Memory Management or on platforms that natively support GPU access to system allocated memory, for example Grace Hopper.

To fine tune CAGRA index building we can customize IVF-PQ index builder options using the following settings. These take effect only if <span class="title-ref">graph_build_algo == "IVF_PQ"</span>. It is recommended to experiment using a separate IVF-PQ index to find the config that gives the largest QPS for large batch. Recall does not need to be very high, since CAGRA further optimizes the kNN neighbor graph. Some of the default values are derived from the dataset size which is assumed to be \[n_vecs, dim\].

|  |  |  |  |  |  |
|----|----|----|----|----|----|
| Parameter | Type | Required | Data Type | Default | Description |
| <span class="title-ref">ivf_pq_build_nlist</span> | <span class="title-ref">build</span> | N | Positive integer \>0 | sqrt(n_vecs) | Number of clusters to partition the vectors into. Larger values will put less points into each cluster but this will impact index build time as more clusters need to be trained. |
| <span class="title-ref">ivf_pq_build_niter</span> | <span class="title-ref">build</span> | N | Positive integer \>0 | 25 | Number of k-means iterations to use when training the clusters. |
| <span class="title-ref">ivf_pq_build_ratio</span> | <span class="title-ref">build</span> | N | Positive integer \>0 | 10 | <span class="title-ref">1/ratio</span> is the number of training points which should be used to train the clusters. |
| <span class="title-ref">ivf_pq_pq_dim</span> | <span class="title-ref">build</span> | N | Positive integer. Multiple of 8 | dim/2 rounded up to 8 | Dimensionality of the vector after product quantization. When 0, a heuristic is used to select this value. <span class="title-ref">pq_dim</span> \* <span class="title-ref">pq_bits</span> must be a multiple of 8. |
| <span class="title-ref">ivf_pq_build_pq_bits</span> | <span class="title-ref">build</span> | N | Positive integer \[4-8\] | 8 | Bit length of the vector element after quantization. |
| <span class="title-ref">ivf_pq_build_codebook_kind</span> | <span class="title-ref">build</span> | N | \[<span class="title-ref">cluster</span>, <span class="title-ref">subspace</span>\] | <span class="title-ref">subspace</span> | Type of codebook. See `IVF-PQ index overview <../neighbors/ivfpq>` for more detail |
| <span class="title-ref">ivf_pq_build_nprobe</span> | <span class="title-ref">search</span> | N | Positive integer \>0 | min(2\*dim, nlist) | The closest number of clusters to search for each query vector. Larger values will improve recall but will search more points in the index. |
| <span class="title-ref">ivf_pq_build_internalDistanceDtype</span> | <span class="title-ref">search</span> | N | \[<span class="title-ref">float</span>, <span class="title-ref">half</span>\] | <span class="title-ref">half</span> | The precision to use for the distance computations. Lower precision can increase performance at the cost of accuracy. |
| <span class="title-ref">ivf_pq_build_smemLutDtype</span> | <span class="title-ref">search</span> | N | \[<span class="title-ref">float</span>, <span class="title-ref">half</span>, <span class="title-ref">fp8</span>\] | <span class="title-ref">fp8</span> | The precision to use for the lookup table in shared memory. Lower precision can increase performance at the cost of accuracy. |
| <span class="title-ref">ivf_pq_build_refine_ratio</span> | <span class="title-ref">search</span> | N | Positive integer \>0 | 2 | <span class="title-ref">refine_ratio \* k</span> nearest neighbors are queried from the index initially and an additional refinement step improves recall by selecting only the best <span class="title-ref">k</span> neighbors. |

Alternatively, if <span class="title-ref">graph_build_algo == "NN_DESCENT"</span>, then we can customize the following parameters

|  |  |  |  |  |  |
|----|----|----|----|----|----|
| Parameter | Type | Required | Data Type | Default | Description |
| <span class="title-ref">nn_descent_niter</span> | <span class="title-ref">build</span> | N | Positive integer \>0 | 20 | Number of nn-descent iterations |
| <span class="title-ref">nn_descent_intermediate_graph_degree</span> | <span class="title-ref">build</span> | N | Positive integer \>0 | <span class="title-ref">cagra.intermediate_graph_degree</span> \* 1.5 | Intermadiate graph degree during nn-descent iterations |
| nn_descent_termination_threshold | <span class="title-ref">build</span> | N | Positive float \>0 | 1e-4 | Early stopping threshold for nn-descent convergence |

### cuvs_cagra_hnswlib

This is a benchmark that enables interoperability between <span class="title-ref">CAGRA</span> built <span class="title-ref">HNSW</span> search. It uses the <span class="title-ref">CAGRA</span> built graph as the base layer of an <span class="title-ref">hnswlib</span> index to search queries only within the base layer (this is enabled with a simple patch to <span class="title-ref">hnswlib</span>).

<span class="title-ref">build</span> : Same as <span class="title-ref">build</span> of CAGRA

<span class="title-ref">search</span> : Same as <span class="title-ref">search</span> of Hnswlib

### cuvs_vamana

Benchmark for building an in-memory Vamana graph based index on the GPU and interoperability with DiskANN for search.

|  |  |  |  |  |  |
|----|----|----|----|----|----|
| Parameter | Type | Required | Data Type | Default | Description |
| <span class="title-ref">graph_degree</span> | <span class="title-ref">build</span> | N | Positive integer \>0 | 32 | Maximum degree of the graph index |
| <span class="title-ref">visited_size</span> | <span class="title-ref">build</span> | N | Positive integer \>0 | 64 | Maximum number of visited nodes per search corresponds to the L parameter in the Vamana literature |
| <span class="title-ref">alpha</span> | <span class="title-ref">build</span> | N | Positive float \>0 | 1.2 | Alpha for pruning parameter |
| <span class="title-ref">L_search</span> | <span class="title-ref">search</span> | Y | Positive integer \>0 |  | Maximum number of visited nodes per search corresponds to the L parameter in the Vamana literature. Larger values improve recall at the cost of search time. |

## FAISS Indexes

### faiss_gpu_flat

Use FAISS flat index on the GPU, which performs an exact search using brute-force and doesn't have any further build or search parameters.

### faiss_gpu_ivf_flat

IVF-flat uses an inverted-file index, which partitions the vectors into a series of clusters, or lists, storing them in an interleaved format which is optimized for fast distance computation. The searching of an IVF-flat index reduces the total vectors in the index to those within some user-specified nearest clusters called probes.

IVF-flat is a simple algorithm which won't save any space, but it provides competitive search times even at higher levels of recall.

|  |  |  |  |  |  |
|----|----|----|----|----|----|
| Parameter | Type | Required | Data Type | Default | Description |
| <span class="title-ref">nlists</span> | <span class="title-ref">build</span> | Y | Positive integer \>0 |  | Number of clusters to partition the vectors into. Larger values will put less points into each cluster but this will impact index build time as more clusters need to be trained |
| <span class="title-ref">ratio</span> | <span class="title-ref">build</span> | N | Positive integer \>0 | 2 | <span class="title-ref">1/ratio</span> is the number of training points which should be used to train the clusters. |
| <span class="title-ref">nprobe</span> | <span class="title-ref">search</span> | Y | Positive integer \>0 | 20 | The closest number of clusters to search for each query vector. Larger values will improve recall but will search more points in the index. |

### faiss_gpu_ivf_pq

IVF-pq is an inverted-file index, which partitions the vectors into a series of clusters, or lists, in a similar way to IVF-flat above. The difference is that IVF-PQ uses product quantization to also compress the vectors, giving the index a smaller memory footprint. Unfortunately, higher levels of compression can also shrink recall, which a refinement step can improve when the original vectors are still available.

|  |  |  |  |  |  |
|----|----|----|----|----|----|
| Parameter | Type | Required | Data Type | Default | Description |
| <span class="title-ref">nlist</span> | <span class="title-ref">build</span> | Y | Positive integer \>0 |  | Number of clusters to partition the vectors into. Larger values will put less points into each cluster but this will impact index build time as more clusters need to be trained. |
| <span class="title-ref">ratio</span> | <span class="title-ref">build</span> | N | Positive integer \>0 | 2 | <span class="title-ref">1/ratio</span> is the number of training points which should be used to train the clusters. |
| <span class="title-ref">M_ratio</span> | <span class="title-ref">build</span> | Y | Positive integer. Power of 2 \[8-64\] |  | Ratio of number of chunks or subquantizers for each vector. Computed by <span class="title-ref">dims</span> / <span class="title-ref">M_ratio</span> |
| <span class="title-ref">usePrecomputed</span> | <span class="title-ref">build</span> | N | Boolean | <span class="title-ref">false</span> | Use pre-computed lookup tables to speed up search at the cost of increased memory usage. |
| <span class="title-ref">useFloat16</span> | <span class="title-ref">build</span> | N | Boolean | <span class="title-ref">false</span> | Use half-precision floats for clustering step. |
| <span class="title-ref">nprobe</span> | <span class="title-ref">search</span> | Y | Positive integer \>0 |  | The closest number of clusters to search for each query vector. Larger values will improve recall but will search more points in the index. |
| <span class="title-ref">refine_ratio</span> | <span class="title-ref">search</span> | N | Positive number \>=1 | 1 | <span class="title-ref">refine_ratio \* k</span> nearest neighbors are queried from the index initially and an additional refinement step improves recall by selecting only the best <span class="title-ref">k</span> neighbors. |

### faiss_cpu_flat

Use FAISS flat index on the CPU, which performs an exact search using brute-force and doesn't have any further build or search parameters.

|  |  |  |  |  |  |
|----|----|----|----|----|----|
| Parameter | Type | Required | Data Type | Default | Description |
| <span class="title-ref">numThreads</span> | <span class="title-ref">search</span> | N | Positive integer \>0 | 1 | Number of threads to use for queries. |

### faiss_cpu_ivf_flat

Use FAISS IVF-Flat index on CPU

|  |  |  |  |  |  |
|----|----|----|----|----|----|
| Parameter | Type | Required | Data Type | Default | Description |
| <span class="title-ref">nlists</span> | <span class="title-ref">build</span> | Y | Positive integer \>0 |  | Number of clusters to partition the vectors into. Larger values will put less points into each cluster but this will impact index build time as more clusters need to be trained |
| <span class="title-ref">ratio</span> | <span class="title-ref">build</span> | N | Positive integer \>0 | 2 | <span class="title-ref">1/ratio</span> is the number of training points which should be used to train the clusters. |
| <span class="title-ref">nprobe</span> | <span class="title-ref">search</span> | Y | Positive integer \>0 |  | The closest number of clusters to search for each query vector. Larger values will improve recall but will search more points in the index. |
| <span class="title-ref">numThreads</span> | <span class="title-ref">search</span> | N | Positive integer \>0 | 1 | Number of threads to use for queries. |

### faiss_cpu_ivf_pq

Use FAISS IVF-PQ index on CPU

|  |  |  |  |  |  |
|----|----|----|----|----|----|
| Parameter | Type | Required | Data Type | Default | Description |
| <span class="title-ref">nlist</span> | <span class="title-ref">build</span> | Y | Positive integer \>0 |  | Number of clusters to partition the vectors into. Larger values will put less points into each cluster but this will impact index build time as more clusters need to be trained. |
| <span class="title-ref">ratio</span> | <span class="title-ref">build</span> | N | Positive integer \>0 | 2 | <span class="title-ref">1/ratio</span> is the number of training points which should be used to train the clusters. |
| <span class="title-ref">M</span> | <span class="title-ref">build</span> | Y | Positive integer. Power of 2 \[8-64\] |  | Ratio of number of chunks or subquantizers for each vector. Computed by <span class="title-ref">dims</span> / <span class="title-ref">M_ratio</span> |
| <span class="title-ref">usePrecomputed</span> | <span class="title-ref">build</span> | N | Boolean | <span class="title-ref">false</span> | Use pre-computed lookup tables to speed up search at the cost of increased memory usage. |
| <span class="title-ref">bitsPerCode</span> | <span class="title-ref">build</span> | N | Positive integer \[4-8\] | 8 | Number of bits for representing each quantized code. |
| <span class="title-ref">nprobe</span> | <span class="title-ref">search</span> | Y | Positive integer \>0 |  | The closest number of clusters to search for each query vector. Larger values will improve recall but will search more points in the index. |
| <span class="title-ref">refine_ratio</span> | <span class="title-ref">search</span> | N | Positive number \>=1 | 1 | <span class="title-ref">refine_ratio \* k</span> nearest neighbors are queried from the index initially and an additional refinement step improves recall by selecting only the best <span class="title-ref">k</span> neighbors. |
| <span class="title-ref">numThreads</span> | <span class="title-ref">search</span> | N | Positive integer \>0 | 1 | Number of threads to use for queries. |

## HNSW

### cuvs_hnsw

cuVS HNSW builds an HNSW index using the ACE (Augmented Core Extraction) algorithm, which enables GPU-accelerated HNSW index construction for datasets too large to fit in GPU memory.

|  |  |  |  |  |  |
|----|----|----|----|----|----|
| Parameter | Type | Required | Data Type | Default | Description |
| <span class="title-ref">hierarchy</span> | <span class="title-ref">build</span> | N | \[<span class="title-ref">NONE</span>, <span class="title-ref">CPU</span>, <span class="title-ref">GPU</span>\] | <span class="title-ref">NONE</span> | Type of HNSW hierarchy to build. <span class="title-ref">NONE</span> creates a base-layer-only index, <span class="title-ref">CPU</span> builds full hierarchy on CPU, <span class="title-ref">GPU</span> builds full hierarchy on GPU. |
| <span class="title-ref">efConstruction</span> | <span class="title-ref">build</span> | Y | Positive integer \>0 |  | Controls index time and accuracy. Bigger values increase the index quality. At some point, increasing this will no longer improve the quality. |
| <span class="title-ref">M</span> | <span class="title-ref">build</span> | Y | Positive integer. Often between 2-100 |  | Number of bi-directional links create for every new element during construction. Higher values work for higher intrinsic dimensionality and/or high recall, low values can work for datasets with low intrinsic dimensionality and/or low recalls. Also affects the algorithm's memory consumption. |
| <span class="title-ref">numThreads</span> | <span class="title-ref">build</span> | N | Positive integer \>0 | 1 | Number of threads to use to build the index. |
| <span class="title-ref">npartitions</span> | <span class="title-ref">build</span> | N | Positive integer \>0 | 1 | Number of partitions to use for the ACE build. Small values might improve recall but potentially degrade performance and increase memory usage. The partition size is on average 2 \* (n_rows / npartitions) \* dim \* sizeof(T). 2 is because of the core and augmented vectors. Please account for imbalance in the partition sizes (up to 3x in our tests). |
| <span class="title-ref">ef_construction</span> | <span class="title-ref">build</span> | N | Positive integer \>0 | 120 | Controls index time and accuracy when using ACE build. Bigger values increase the index quality. At some point, increasing this will no longer improve the quality. |
| <span class="title-ref">build_dir</span> | <span class="title-ref">build</span> | N | String | "/tmp/ace_build" | The directory to use for the ACE build. This should be the fastest disk in the system and hold enough space for twice the dataset, final graph, and label mapping. |
| <span class="title-ref">use_disk</span> | <span class="title-ref">build</span> | N | Boolean | <span class="title-ref">false</span> | Whether to use disk-based storage for ACE build. When true, forces ACE to use disk-based storage even if the graph fits in host and GPU memory. When false, ACE will use in-memory storage if the graph fits in host and GPU memory and disk-based storage otherwise. |
| <span class="title-ref">ef</span> | <span class="title-ref">search</span> | Y | Positive integer \>0 |  | Size of the dynamic list for the nearest neighbors used for search. Higher value leads to more accurate but slower search. Cannot be lower than <span class="title-ref">k</span>. |
| <span class="title-ref">numThreads</span> | <span class="title-ref">search</span> | N | Positive integer \>0 | 1 | Number of threads to use for queries. |

### hnswlib

|  |  |  |  |  |  |
|----|----|----|----|----|----|
| Parameter | Type | Required | Data Type | Default | Description |
| <span class="title-ref">efConstruction</span> | <span class="title-ref">build</span> | Y | Positive integer \>0 |  | Controls index time and accuracy. Bigger values increase the index quality. At some point, increasing this will no longer improve the quality. |
| <span class="title-ref">M</span> | <span class="title-ref">build</span> | Y | Positive integer. Often between 2-100 |  | Number of bi-directional links create for every new element during construction. Higher values work for higher intrinsic dimensionality and/or high recall, low values can work for datasets with low intrinsic dimensionality and/or low recalls. Also affects the algorithm's memory consumption. |
| <span class="title-ref">numThreads</span> | <span class="title-ref">build</span> | N | Positive integer \>0 | 1 | Number of threads to use to build the index. |
| <span class="title-ref">ef</span> | <span class="title-ref">search</span> | Y | Positive integer \>0 |  | Size of the dynamic list for the nearest neighbors used for search. Higher value leads to more accurate but slower search. Cannot be lower than <span class="title-ref">k</span>. |
| <span class="title-ref">numThreads</span> | <span class="title-ref">search</span> | N | Positive integer \>0 | 1 | Number of threads to use for queries. |

Please refer to [HNSW algorithm parameters guide](https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md) from <span class="title-ref">hnswlib</span> to learn more about these arguments.

## DiskANN

### diskann_memory

Use DiskANN in-memory index for approximate search.

|  |  |  |  |  |  |
|----|----|----|----|----|----|
| Parameter | Type | Required | Data Type | Default | Description |
| <span class="title-ref">R</span> | <span class="title-ref">build</span> | Y | Positive integer \>0 |  | Maximum degree of the graph index |
| <span class="title-ref">L_build</span> | <span class="title-ref">build</span> | Y | Positive integer \>0 |  | number of visited nodes per greedy search during graph construction |
| <span class="title-ref">alpha</span> | <span class="title-ref">build</span> | N | Positive number \>=1 | 1.2 | controls the pruning parameter of the graph construction |
| <span class="title-ref">num_threads</span> | <span class="title-ref">build</span> | N | Positive integer \>0 | omp_get_max_threads() | Number of CPU threads to use to build the index. |
| <span class="title-ref">L_search</span> | <span class="title-ref">search</span> | Y | Positive integer \>0 |  | visited list size during search |
