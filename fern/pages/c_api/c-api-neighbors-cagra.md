---
slug: api-reference/c-api-neighbors-cagra
---

# Cagra

_Source header: `c/include/cuvs/neighbors/cagra.h`_

## C API for CUDA ANN Graph-based nearest neighbor search

_Doxygen group: `cagra_c_index_params`_

### cuvsCagraGraphBuildAlgo

Enum to denote which ANN algorithm is used to build CAGRA graph

```c
enum cuvsCagraGraphBuildAlgo { ... } ;
```

**Values**

| Name | Value |
| --- | --- |
| `AUTO_SELECT` | `0` |
| `IVF_PQ` | `1` |

_Source: `c/include/cuvs/neighbors/cagra.h:29`_

### cuvsCagraHnswHeuristicType

A strategy for selecting the graph build parameters based on similar HNSW index

parameters.

Define how cuvsCagraIndexParamsFromHnswParams should construct a graph to construct a graph that is to be converted to (used by) a CPU HNSW index.

```c
enum cuvsCagraHnswHeuristicType { ... } ;
```

_Source: `c/include/cuvs/neighbors/cagra.h:54`_

### cuvsCagraCompressionParams

Parameters for VPQ compression.

```c
struct cuvsCagraCompressionParams { ... } ;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `pq_bits` | `uint32_t` | The bit length of the vector element after compression by PQ. Possible values: [4, 5, 6, 7, 8]. Hint: the smaller the 'pq_bits', the smaller the index size and the better the search performance, but the lower the recall. |
| `pq_dim` | `uint32_t` | The dimensionality of the vector after compression by PQ. When zero, an optimal value is selected using a heuristic. TODO: at the moment `dim` must be a multiple `pq_dim`. |
| `vq_n_centers` | `uint32_t` | Vector Quantization (VQ) codebook size - number of "coarse cluster centers". When zero, an optimal value is selected using a heuristic. |
| `kmeans_n_iters` | `uint32_t` | The number of iterations searching for kmeans centers (both VQ & PQ phases). |
| `vq_kmeans_trainset_fraction` | `double` | The fraction of data to use during iterative kmeans building (VQ phase). When zero, an optimal value is selected using a heuristic. |
| `pq_kmeans_trainset_fraction` | `double` | The fraction of data to use during iterative kmeans building (PQ phase). When zero, an optimal value is selected using a heuristic. |

_Source: `c/include/cuvs/neighbors/cagra.h:82`_

### cuvsAceParams

Parameters for ACE (Augmented Core Extraction) graph build.

ACE enables building indexes for datasets too large to fit in GPU memory by:

1. Partitioning the dataset in core (closest) and augmented (second-closest) partitions using balanced k-means.
2. Building sub-indexes for each partition independently
3. Concatenating sub-graphs into a final unified index

```c
struct cuvsAceParams { ... } ;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `npartitions` | `size_t` | Number of partitions for ACE (Augmented Core Extraction) partitioned build. When set to 0 (default), the number of partitions is automatically derived based on available host and GPU memory to maximize partition size while ensuring the build fits in memory. Small values might improve recall but potentially degrade performance and increase memory usage. Partitions should not be too small to prevent issues in KNN graph construction. The partition size is on average 2 * (n_rows / npartitions) * dim * sizeof(T). 2 is because of the core and augmented vectors. Please account for imbalance in the partition sizes (up to 3x in our tests). If the specified number of partitions results in partitions that exceed available memory, the value will be automatically increased to fit memory constraints and a warning will be issued. |
| `ef_construction` | `size_t` | The index quality for the ACE build. Bigger values increase the index quality. At some point, increasing this will no longer improve the quality. |
| `build_dir` | `const char*` | Directory to store ACE build artifacts (e.g., KNN graph, optimized graph). Used when `use_disk` is true or when the graph does not fit in host and GPU memory. This should be the fastest disk in the system and hold enough space for twice the dataset, final graph, and label mapping. |
| `use_disk` | `bool` | Whether to use disk-based storage for ACE build. When true, enables disk-based operations for memory-efficient graph construction. |
| `max_host_memory_gb` | `double` | Maximum host memory to use for ACE build in GiB. When set to 0 (default), uses available host memory. When set to a positive value, limits host memory usage to the specified amount. Useful for testing or when running alongside other memory-intensive processes. |
| `max_gpu_memory_gb` | `double` | Maximum GPU memory to use for ACE build in GiB. When set to 0 (default), uses available GPU memory. When set to a positive value, limits GPU memory usage to the specified amount. Useful for testing or when running alongside other memory-intensive processes. |

_Source: `c/include/cuvs/neighbors/cagra.h:136`_

### cuvsCagraIndexParams

Supplemental parameters to build CAGRA Index

```c
struct cuvsCagraIndexParams { ... } ;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `metric` | `cuvsDistanceType` | Distance type. |
| `intermediate_graph_degree` | `size_t` | Degree of input graph for pruning. |
| `graph_degree` | `size_t` | Degree of output graph. |
| `build_algo` | `enum cuvsCagraGraphBuildAlgo` | ANN algorithm to build knn graph. |
| `nn_descent_niter` | `size_t` | Number of Iterations to run if building with NN_DESCENT |
| `compression` | `cuvsCagraCompressionParams_t` | Optional: specify compression parameters if compression is desired. NOTE: this is experimental new API, consider it unsafe. |
| `graph_build_params` | `void*` | Optional: specify graph build params based on build_algo<br />- IVF_PQ: cuvsIvfPqParams_t<br />- ACE: cuvsAceParams_t<br />- Others: nullptr |

_Source: `c/include/cuvs/neighbors/cagra.h:201`_

### cuvsCagraIndexParamsCreate

Allocate CAGRA Index params, and populate with default values

```c
cuvsError_t cuvsCagraIndexParamsCreate(cuvsCagraIndexParams_t* params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | `cuvsCagraIndexParams_t*` | cuvsCagraIndexParams_t to allocate |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/cagra.h:235`_

### cuvsCagraIndexParamsDestroy

De-allocate CAGRA Index params

```c
cuvsError_t cuvsCagraIndexParamsDestroy(cuvsCagraIndexParams_t params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | `cuvsCagraIndexParams_t` |  |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/cagra.h:243`_

### cuvsCagraCompressionParamsCreate

Allocate CAGRA Compression params, and populate with default values

```c
cuvsError_t cuvsCagraCompressionParamsCreate(cuvsCagraCompressionParams_t* params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | `cuvsCagraCompressionParams_t*` | cuvsCagraCompressionParams_t to allocate |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/cagra.h:251`_

### cuvsCagraCompressionParamsDestroy

De-allocate CAGRA Compression params

```c
cuvsError_t cuvsCagraCompressionParamsDestroy(cuvsCagraCompressionParams_t params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | `cuvsCagraCompressionParams_t` |  |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/cagra.h:259`_

### cuvsAceParamsCreate

Allocate ACE params, and populate with default values

```c
cuvsError_t cuvsAceParamsCreate(cuvsAceParams_t* params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | `cuvsAceParams_t*` | cuvsAceParams_t to allocate |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/cagra.h:267`_

### cuvsAceParamsDestroy

De-allocate ACE params

```c
cuvsError_t cuvsAceParamsDestroy(cuvsAceParams_t params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | `cuvsAceParams_t` |  |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/cagra.h:275`_

### cuvsCagraIndexParamsFromHnswParams

Create CAGRA index parameters similar to an HNSW index

```c
cuvsError_t cuvsCagraIndexParamsFromHnswParams(cuvsCagraIndexParams_t params,
int64_t n_rows,
int64_t dim,
int M,
int ef_construction,
enum cuvsCagraHnswHeuristicType heuristic,
cuvsDistanceType metric);
```

This factory function creates CAGRA parameters that yield a graph compatible with an HNSW graph with the given parameters.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | out | `cuvsCagraIndexParams_t` | The CAGRA index params to populate |
| `n_rows` | in | `int64_t` | Number of rows in the dataset |
| `dim` | in | `int64_t` | Number of dimensions in the dataset |
| `M` | in | `int` | HNSW index parameter M |
| `ef_construction` | in | `int` | HNSW index parameter ef_construction |
| `heuristic` | in | `enum cuvsCagraHnswHeuristicType` | Strategy for parameter selection |
| `metric` | in | `cuvsDistanceType` | Distance metric to use |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/cagra.h:292`_

## C API for CUDA ANN Graph-based nearest neighbor search

_Doxygen group: `cagra_c_extend_params`_

### cuvsCagraExtendParams

Supplemental parameters to extend CAGRA Index

```c
struct cuvsCagraExtendParams { ... } ;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `max_chunk_size` | `uint32_t` | The additional dataset is divided into chunks and added to the graph. This is the knob to adjust the tradeoff between the recall and operation throughput. Large chunk sizes can result in high throughput, but use more working memory (O(max_chunk_size*degree^2)). This can also degrade recall because no edges are added between the nodes in the same chunk. Auto select when 0. |

_Source: `c/include/cuvs/neighbors/cagra.h:312`_

### cuvsCagraExtendParamsCreate

Allocate CAGRA Extend params, and populate with default values

```c
cuvsError_t cuvsCagraExtendParamsCreate(cuvsCagraExtendParams_t* params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | `cuvsCagraExtendParams_t*` | cuvsCagraExtendParams_t to allocate |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/cagra.h:329`_

### cuvsCagraExtendParamsDestroy

De-allocate CAGRA Extend params

```c
cuvsError_t cuvsCagraExtendParamsDestroy(cuvsCagraExtendParams_t params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | `cuvsCagraExtendParams_t` |  |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/cagra.h:337`_

### cuvsCagraExtend

Extend a CAGRA index with a `DLManagedTensor` which has underlying

```c
cuvsError_t cuvsCagraExtend(cuvsResources_t res,
cuvsCagraExtendParams_t params,
DLManagedTensor* additional_dataset,
cuvsCagraIndex_t index);
```

`DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`, or `kDLCPU`. Also, acceptable underlying types are:

1. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
2. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 16`
3. `kDLDataType.code == kDLInt` and `kDLDataType.bits = 8`
4. `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 8`

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `params` | in | `cuvsCagraExtendParams_t` | cuvsCagraExtendParams_t used to extend CAGRA index |
| `additional_dataset` | in | `DLManagedTensor*` | DLManagedTensor* additional dataset |
| `index` | in,out | `cuvsCagraIndex_t` | cuvsCagraIndex_t CAGRA index |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/cagra.h:644`_

## C API for CUDA ANN Graph-based nearest neighbor search

_Doxygen group: `cagra_c_search_params`_

### cuvsCagraSearchAlgo

Enum to denote algorithm used to search CAGRA Index

```c
enum cuvsCagraSearchAlgo { ... } ;
```

**Values**

| Name | Value |
| --- | --- |
| `SINGLE_CTA` | `0` |
| `MULTI_CTA` | `1` |
| `MULTI_KERNEL` | `2` |
| `AUTO` | `100` |

_Source: `c/include/cuvs/neighbors/cagra.h:352`_

### cuvsCagraHashMode

Enum to denote Hash Mode used while searching CAGRA index

```c
enum cuvsCagraHashMode { ... } ;
```

**Values**

| Name | Value |
| --- | --- |
| `HASH` | `0` |
| `SMALL` | `1` |
| `AUTO_HASH` | `100` |

_Source: `c/include/cuvs/neighbors/cagra.h:365`_

### cuvsCagraSearchParams

Supplemental parameters to search CAGRA index

```c
struct cuvsCagraSearchParams { ... } ;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `max_queries` | `size_t` | Maximum number of queries to search at the same time (batch size). Auto select when 0. |
| `itopk_size` | `size_t` | Number of intermediate search results retained during the search. This is the main knob to adjust trade off between accuracy and search speed. Higher values improve the search accuracy. |
| `max_iterations` | `size_t` | Upper limit of search iterations. Auto select when 0. |
| `algo` | `enum cuvsCagraSearchAlgo` | Which search implementation to use. |
| `team_size` | `size_t` | Number of threads used to calculate a single distance. 4, 8, 16, or 32. |
| `search_width` | `size_t` | Number of graph nodes to select as the starting point for the search in each iteration. aka search width? |
| `min_iterations` | `size_t` | Lower limit of search iterations. |
| `thread_block_size` | `size_t` | Thread block size. 0, 64, 128, 256, 512, 1024. Auto selection when 0. |
| `hashmap_mode` | `enum cuvsCagraHashMode` | Hashmap type. Auto selection when AUTO. |
| `hashmap_min_bitlen` | `size_t` | Lower limit of hashmap bit length. More than 8. |
| `hashmap_max_fill_rate` | `float` | Upper limit of hashmap fill rate. More than 0.1, less than 0.9. |
| `num_random_samplings` | `uint32_t` | Number of iterations of initial random seed node selection. 1 or more. |
| `rand_xor_mask` | `uint64_t` | Bit mask used for initial random seed node selection. |
| `persistent` | `bool` | Whether to use the persistent version of the kernel (only SINGLE_CTA is supported a.t.m.) |
| `persistent_lifetime` | `float` | Persistent kernel: time in seconds before the kernel stops if no requests received. |
| `persistent_device_usage` | `float` | Set the fraction of maximum grid size used by persistent kernel. Value 1.0 means the kernel grid size is maximum possible for the selected device. The value must be greater than 0.0 and not greater than 1.0. One may need to run other kernels alongside this persistent kernel. This parameter can be used to reduce the grid size of the persistent kernel to leave a few SMs idle. Note: running any other work on GPU alongside with the persistent kernel makes the setup fragile.<br />- Running another kernel in another thread usually works, but no progress guaranteed<br />- Any CUDA allocations block the context (this issue may be obscured by using pools)<br />- Memory copies to not-pinned host memory may block the context Even when we know there are no other kernels working at the same time, setting kDeviceUsage to 1.0 surprisingly sometimes hurts performance. Proceed with care. If you suspect this is an issue, you can reduce this number to ~0.9 without a significant impact on the throughput. |

_Source: `c/include/cuvs/neighbors/cagra.h:371`_

### cuvsCagraSearchParamsCreate

Allocate CAGRA search params, and populate with default values

```c
cuvsError_t cuvsCagraSearchParamsCreate(cuvsCagraSearchParams_t* params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | `cuvsCagraSearchParams_t*` | cuvsCagraSearchParams_t to allocate |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/cagra.h:447`_

### cuvsCagraSearchParamsDestroy

De-allocate CAGRA search params

```c
cuvsError_t cuvsCagraSearchParamsDestroy(cuvsCagraSearchParams_t params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | `cuvsCagraSearchParams_t` |  |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/cagra.h:455`_

## C API for CUDA ANN Graph-based nearest neighbor search

_Doxygen group: `cagra_c_index`_

### cuvsCagraIndexCreate

Allocate CAGRA index

```c
cuvsError_t cuvsCagraIndexCreate(cuvsCagraIndex_t* index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | `cuvsCagraIndex_t*` | cuvsCagraIndex_t to allocate |

**Returns**

`cuvsError_t`

cagraError_t

_Source: `c/include/cuvs/neighbors/cagra.h:484`_

### cuvsCagraIndexDestroy

De-allocate CAGRA index

```c
cuvsError_t cuvsCagraIndexDestroy(cuvsCagraIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | `cuvsCagraIndex_t` | cuvsCagraIndex_t to de-allocate |

**Returns**

`cuvsError_t`

_Source: `c/include/cuvs/neighbors/cagra.h:491`_

### cuvsCagraIndexGetDims

Get dimension of the CAGRA index

```c
cuvsError_t cuvsCagraIndexGetDims(cuvsCagraIndex_t index, int64_t* dim);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | `cuvsCagraIndex_t` | CAGRA index |
| `dim` | out | `int64_t*` | return dimension of the index |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/cagra.h:500`_

### cuvsCagraIndexGetSize

Get size of the CAGRA index

```c
cuvsError_t cuvsCagraIndexGetSize(cuvsCagraIndex_t index, int64_t* size);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | `cuvsCagraIndex_t` | CAGRA index |
| `size` | out | `int64_t*` | return number of vectors in the index |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/cagra.h:509`_

### cuvsCagraIndexGetGraphDegree

Get graph degree of the CAGRA index

```c
cuvsError_t cuvsCagraIndexGetGraphDegree(cuvsCagraIndex_t index, int64_t* graph_degree);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | `cuvsCagraIndex_t` | CAGRA index |
| `graph_degree` | out | `int64_t*` | return graph degree |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/cagra.h:518`_

### cuvsCagraIndexGetDataset

Returns a view of the CAGRA dataset

```c
cuvsError_t cuvsCagraIndexGetDataset(cuvsCagraIndex_t index, DLManagedTensor* dataset);
```

This function returns a non-owning view of the CAGRA dataset. The output will be referencing device memory that is directly used in CAGRA, without copying the dataset at all. This means that the output is only valid as long as the CAGRA index is alive, and once cuvsCagraIndexDestroy is called on the cagra index - the returned dataset view will be invalid.

Note that the DLManagedTensor dataset returned will have an associated 'deleter' function that must be called when the dataset is no longer needed. This will free up host memory that stores the shape of the dataset view.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | `cuvsCagraIndex_t` | CAGRA index |
| `dataset` | out | `DLManagedTensor*` | the dataset used in cagra |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/cagra.h:539`_

### cuvsCagraIndexGetGraph

Returns a view of the CAGRA graph

```c
cuvsError_t cuvsCagraIndexGetGraph(cuvsCagraIndex_t index, DLManagedTensor* graph);
```

This function returns a non-owning view of the CAGRA graph. The output will be referencing device memory that is directly used in CAGRA, without copying the graph at all. This means that the output is only valid as long as the CAGRA index is alive, and once cuvsCagraIndexDestroy is called on the cagra index - the returned graph view will be invalid.

Note that the DLManagedTensor graph returned will have an associated 'deleter' function that must be called when the graph is no longer needed. This will free up host memory that stores the metadata for the graph view.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | `cuvsCagraIndex_t` | CAGRA index |
| `graph` | out | `DLManagedTensor*` | the output knn graph. |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/cagra.h:560`_

## C API for CUDA ANN Graph-based nearest neighbor search

_Doxygen group: `cagra_c_index_build`_

### cuvsCagraBuild

Build a CAGRA index with a `DLManagedTensor` which has underlying

```c
cuvsError_t cuvsCagraBuild(cuvsResources_t res,
cuvsCagraIndexParams_t params,
DLManagedTensor* dataset,
cuvsCagraIndex_t index);
```

`DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`, or `kDLCPU`. Also, acceptable underlying types are:

1. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
2. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 16`
3. `kDLDataType.code == kDLInt` and `kDLDataType.bits = 8`
4. `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 8`

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `params` | in | `cuvsCagraIndexParams_t` | cuvsCagraIndexParams_t used to build CAGRA index |
| `dataset` | in | `DLManagedTensor*` | DLManagedTensor* training dataset |
| `index` | inout | `cuvsCagraIndex_t` | cuvsCagraIndex_t Newly built CAGRA index. This index needs to be already created with cuvsCagraIndexCreate. |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/cagra.h:615`_

## C API for CUDA ANN Graph-based nearest neighbor search

_Doxygen group: `cagra_c_index_search`_

### cuvsCagraSearch

Search a CAGRA index with a `DLManagedTensor` which has underlying

```c
cuvsError_t cuvsCagraSearch(cuvsResources_t res,
cuvsCagraSearchParams_t params,
cuvsCagraIndex_t index,
DLManagedTensor* queries,
DLManagedTensor* neighbors,
DLManagedTensor* distances,
cuvsFilter filter);
```

`DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`. It is also important to note that the CAGRA Index must have been built with the same type of `queries`, such that `index.dtype.code == queries.dl_tensor.dtype.code` Types for input are:

1. `queries`: a. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32` b. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 16` c. `kDLDataType.code == kDLInt` and `kDLDataType.bits = 8` d. `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 8`
2. `neighbors`: `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 32` or `kDLDataType.code == kDLInt`  and `kDLDataType.bits = 64`
3. `distances`: `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `params` | in | `cuvsCagraSearchParams_t` | cuvsCagraSearchParams_t used to search CAGRA index |
| `index` | in | `cuvsCagraIndex_t` | cuvsCagraIndex which has been returned by `cuvsCagraBuild` |
| `queries` | in | `DLManagedTensor*` | DLManagedTensor* queries dataset to search |
| `neighbors` | out | `DLManagedTensor*` | DLManagedTensor* output `k` neighbors for queries |
| `distances` | out | `DLManagedTensor*` | DLManagedTensor* output `k` distances for queries |
| `filter` | in | `cuvsFilter` | cuvsFilter input filter that can be used to filter queries and neighbors based on the given bitset. |

**Returns**

`cuvsError_t`

_Source: `c/include/cuvs/neighbors/cagra.h:707`_

## CAGRA C-API serialize functions

_Doxygen group: `cagra_c_index_serialize`_

### cuvsCagraSerialize

Save the index to file.

```c
cuvsError_t cuvsCagraSerialize(cuvsResources_t res,
const char* filename,
cuvsCagraIndex_t index,
bool include_dataset);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `filename` | in | `const char*` | the file name for saving the index |
| `index` | in | `cuvsCagraIndex_t` | CAGRA index |
| `include_dataset` | in | `bool` | Whether or not to write out the dataset to the file. |

**Returns**

`cuvsError_t`

_Source: `c/include/cuvs/neighbors/cagra.h:745`_

### cuvsCagraSerializeToHnswlib

Save the CAGRA index to file in hnswlib format.

```c
cuvsError_t cuvsCagraSerializeToHnswlib(cuvsResources_t res,
const char* filename,
cuvsCagraIndex_t index);
```

NOTE: The saved index can only be read by the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib.

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `filename` | in | `const char*` | the file name for saving the index |
| `index` | in | `cuvsCagraIndex_t` | CAGRA index |

**Returns**

`cuvsError_t`

_Source: `c/include/cuvs/neighbors/cagra.h:774`_

### cuvsCagraDeserialize

Load index from file.

```c
cuvsError_t cuvsCagraDeserialize(cuvsResources_t res, const char* filename, cuvsCagraIndex_t index);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `filename` | in | `const char*` | the name of the file that stores the index |
| `index` | inout | `cuvsCagraIndex_t` | cuvsCagraIndex_t CAGRA index loaded from disk. This index needs to be already created with cuvsCagraIndexCreate. |

**Returns**

`cuvsError_t`

_Source: `c/include/cuvs/neighbors/cagra.h:788`_

### cuvsCagraIndexFromArgs

Load index from a dataset and graph

```c
cuvsError_t cuvsCagraIndexFromArgs(cuvsResources_t res,
cuvsDistanceType metric,
DLManagedTensor* graph,
DLManagedTensor* dataset,
cuvsCagraIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `metric` | in | `cuvsDistanceType` | cuvsDistanceType to use in the index |
| `graph` | in | `DLManagedTensor*` | the knn graph to use, shape (size, graph_degree) |
| `dataset` | in | `DLManagedTensor*` | the dataset to use, shape (size, dim) |
| `index` | inout | `cuvsCagraIndex_t` | cuvsCagraIndex_t CAGRA index populated with the graph and dataset. This index needs to be already created with cuvsCagraIndexCreate. |

**Returns**

`cuvsError_t`

_Source: `c/include/cuvs/neighbors/cagra.h:824`_

## CAGRA C-API merge functions

_Doxygen group: `cagra_c_index_merge`_

### cuvsCagraMerge

Merge multiple CAGRA indices into a single CAGRA index.

```c
cuvsError_t cuvsCagraMerge(cuvsResources_t res,
cuvsCagraIndexParams_t params,
cuvsCagraIndex_t* indices,
size_t num_indices,
cuvsFilter filter,
cuvsCagraIndex_t output_index);
```

All input indices must have been built with the same data type (`index.dtype`) and have the same dimensionality (`index.dims`). The merged index uses the output parameters specified in `cuvsCagraIndexParams`.

Input indices must have:

- `index.dtype.code` and `index.dtype.bits` matching across all indices.
- Supported data types for indices: a. `kDLFloat` with `bits = 32` b. `kDLFloat` with `bits = 16` c. `kDLInt` with `bits = 8` d. `kDLUInt` with `bits = 8`

The resulting output index will have the same data type as the input indices.

Example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `params` | in | `cuvsCagraIndexParams_t` | cuvsCagraIndexParams_t parameters controlling merge behavior |
| `indices` | in | `cuvsCagraIndex_t*` | Array of input cuvsCagraIndex_t handles to merge |
| `num_indices` | in | `size_t` | Number of input indices |
| `filter` | in | `cuvsFilter` | Filter that can be used to filter out vectors from the merged index |
| `output_index` | out | `cuvsCagraIndex_t` | Output handle that will store the merged index. Must be initialized using `cuvsCagraIndexCreate` before use. |

**Returns**

`cuvsError_t`

_Source: `c/include/cuvs/neighbors/cagra.h:891`_
