---
slug: api-reference/c-api-neighbors-hnsw
---

# HNSW

_Source header: `c/include/cuvs/neighbors/hnsw.h`_

## C API for HNSW index params

_Doxygen group: `hnsw_c_index_params`_

### cuvsHnswHierarchy

Hierarchy for HNSW index when converting from CAGRA index

NOTE: When the value is `NONE`, the HNSW index is built as a base-layer-only index.

```c
enum cuvsHnswHierarchy { ... } ;
```

**Values**

| Name | Value |
| --- | --- |
| `CPU` | `1` |
| `GPU` | `2` |

_Source: `c/include/cuvs/neighbors/hnsw.h:30`_

### cuvsHnswAceParams

Parameters for ACE (Augmented Core Extraction) graph build for HNSW.

ACE enables building indexes for datasets too large to fit in GPU memory by:

1. Partitioning the dataset in core and augmented partitions using balanced k-means
2. Building sub-indexes for each partition independently
3. Concatenating sub-graphs into a final unified index

```c
struct cuvsHnswAceParams { ... } ;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `npartitions` | `size_t` | Number of partitions for ACE partitioned build. When set to 0 (default), the number of partitions is automatically derived based on available host and GPU memory to maximize partition size while ensuring the build fits in memory. Small values might improve recall but potentially degrade performance and increase memory usage. The partition size is on average 2 * (n_rows / npartitions) * dim * sizeof(T). 2 is because of the core and augmented vectors. Please account for imbalance in the partition sizes (up to 3x in our tests). If the specified number of partitions results in partitions that exceed available memory, the value will be automatically increased to fit memory constraints and a warning will be issued. |
| `build_dir` | `const char*` | Directory to store ACE build artifacts (e.g., KNN graph, optimized graph). Used when `use_disk` is true or when the graph does not fit in memory. |
| `use_disk` | `bool` | Whether to use disk-based storage for ACE build. When true, enables disk-based operations for memory-efficient graph construction. |
| `max_host_memory_gb` | `double` | Maximum host memory to use for ACE build in GiB. When set to 0 (default), uses available host memory. Useful for testing or when running alongside other memory-intensive processes. |
| `max_gpu_memory_gb` | `double` | Maximum GPU memory to use for ACE build in GiB. When set to 0 (default), uses available GPU memory. Useful for testing or when running alongside other memory-intensive processes. |

_Source: `c/include/cuvs/neighbors/hnsw.h:46`_

### cuvsHnswAceParamsCreate

Allocate HNSW ACE params, and populate with default values

```c
cuvsError_t cuvsHnswAceParamsCreate(cuvsHnswAceParams_t* params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | `cuvsHnswAceParams_t*` | cuvsHnswAceParams_t to allocate |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/hnsw.h:97`_

### cuvsHnswAceParamsDestroy

De-allocate HNSW ACE params

```c
cuvsError_t cuvsHnswAceParamsDestroy(cuvsHnswAceParams_t params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | `cuvsHnswAceParams_t` |  |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/hnsw.h:105`_

### cuvsHnswIndexParamsCreate

Allocate HNSW Index params, and populate with default values

```c
cuvsError_t cuvsHnswIndexParamsCreate(cuvsHnswIndexParams_t* params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | `cuvsHnswIndexParams_t*` | cuvsHnswIndexParams_t to allocate |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/hnsw.h:141`_

### cuvsHnswIndexParamsDestroy

De-allocate HNSW Index params

```c
cuvsError_t cuvsHnswIndexParamsDestroy(cuvsHnswIndexParams_t params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | `cuvsHnswIndexParams_t` |  |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/hnsw.h:149`_

## C API for hnswlib wrapper index

_Doxygen group: `hnsw_c_index`_

### cuvsHnswIndexCreate

Allocate HNSW index

```c
cuvsError_t cuvsHnswIndexCreate(cuvsHnswIndex_t* index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | `cuvsHnswIndex_t*` | cuvsHnswIndex_t to allocate |

**Returns**

`cuvsError_t`

HnswError_t

_Source: `c/include/cuvs/neighbors/hnsw.h:178`_

### cuvsHnswIndexDestroy

De-allocate HNSW index

```c
cuvsError_t cuvsHnswIndexDestroy(cuvsHnswIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | `cuvsHnswIndex_t` | cuvsHnswIndex_t to de-allocate |

**Returns**

`cuvsError_t`

_Source: `c/include/cuvs/neighbors/hnsw.h:185`_

## Parameters for extending HNSW index

_Doxygen group: `hnsw_c_extend_params`_

### cuvsHnswExtendParams

Parameters for extending HNSW index

```c
struct cuvsHnswExtendParams { ... } ;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `num_threads` | `int` | Number of CPU threads used to extend additional vectors |

_Source: `c/include/cuvs/neighbors/hnsw.h:196`_

### cuvsHnswExtendParamsCreate

Allocate HNSW extend params, and populate with default values

```c
cuvsError_t cuvsHnswExtendParamsCreate(cuvsHnswExtendParams_t* params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | `cuvsHnswExtendParams_t*` | cuvsHnswExtendParams_t to allocate |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/hnsw.h:209`_

### cuvsHnswExtendParamsDestroy

De-allocate HNSW extend params

```c
cuvsError_t cuvsHnswExtendParamsDestroy(cuvsHnswExtendParams_t params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | `cuvsHnswExtendParams_t` | cuvsHnswExtendParams_t to de-allocate |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/hnsw.h:218`_

## Load CAGRA index as hnswlib index

_Doxygen group: `hnsw_c_index_load`_

### cuvsHnswFromCagra

Convert a CAGRA Index to an HNSW index.

```c
cuvsError_t cuvsHnswFromCagra(cuvsResources_t res,
cuvsHnswIndexParams_t params,
cuvsCagraIndex_t cagra_index,
cuvsHnswIndex_t hnsw_index);
```

NOTE: When hierarchy is:

1. `NONE`: This method uses the filesystem to write the CAGRA index in `/tmp/&lt;random_number&gt;.bin` before reading it as an hnswlib index, then deleting the temporary file. The returned index is immutable and can only be searched by the hnswlib wrapper in cuVS, as the format is not compatible with the original hnswlib.
2. `CPU`: The returned index is mutable and can be extended with additional vectors. The serialized index is also compatible with the original hnswlib library.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `params` | in | `cuvsHnswIndexParams_t` | cuvsHnswIndexParams_t used to load Hnsw index |
| `cagra_index` | in | `cuvsCagraIndex_t` | cuvsCagraIndex_t to convert to HNSW index |
| `hnsw_index` | out | `cuvsHnswIndex_t` | cuvsHnswIndex_t to return the HNSW index |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/hnsw.h:270`_

## Build HNSW index using ACE algorithm

_Doxygen group: `hnsw_c_index_build`_

### cuvsHnswBuild

Build an HNSW index using ACE (Augmented Core Extraction) algorithm.

```c
cuvsError_t cuvsHnswBuild(cuvsResources_t res,
cuvsHnswIndexParams_t params,
DLManagedTensor* dataset,
cuvsHnswIndex_t index);
```

ACE enables building HNSW indexes for datasets too large to fit in GPU memory by:

1. Partitioning the dataset using balanced k-means into core and augmented partitions
2. Building sub-indexes for each partition independently
3. Concatenating sub-graphs into a final unified index

NOTE: This function requires CUDA to be available at runtime.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `params` | in | `cuvsHnswIndexParams_t` | cuvsHnswIndexParams_t with ACE parameters configured |
| `dataset` | in | `DLManagedTensor*` | DLManagedTensor* host dataset to build index from |
| `index` | out | `cuvsHnswIndex_t` | cuvsHnswIndex_t to return the built HNSW index |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/hnsw.h:347`_

## Extend HNSW index with additional vectors

_Doxygen group: `hnsw_c_index_extend`_

### cuvsHnswExtend

Add new vectors to an HNSW index

```c
cuvsError_t cuvsHnswExtend(cuvsResources_t res,
cuvsHnswExtendParams_t params,
DLManagedTensor* additional_dataset,
cuvsHnswIndex_t index);
```

NOTE: The HNSW index can only be extended when the hierarchy is `CPU` when converting from a CAGRA index.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `params` | in | `cuvsHnswExtendParams_t` | cuvsHnswExtendParams_t used to extend Hnsw index |
| `additional_dataset` | in | `DLManagedTensor*` | DLManagedTensor* additional dataset to extend the index |
| `index` | inout | `cuvsHnswIndex_t` | cuvsHnswIndex_t to extend |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/hnsw.h:405`_

## C API for hnswlib wrapper search params

_Doxygen group: `hnsw_c_search_params`_

### cuvsHnswSearchParams

C API for hnswlib wrapper search params

```c
struct cuvsHnswSearchParams { ... } ;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `ef` | `int32_t` |  |
| `num_threads` | `int32_t` |  |

_Source: `c/include/cuvs/neighbors/hnsw.h:419`_

### cuvsHnswSearchParamsCreate

Allocate HNSW search params, and populate with default values

```c
cuvsError_t cuvsHnswSearchParamsCreate(cuvsHnswSearchParams_t* params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | `cuvsHnswSearchParams_t*` | cuvsHnswSearchParams_t to allocate |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/hnsw.h:432`_

### cuvsHnswSearchParamsDestroy

De-allocate HNSW search params

```c
cuvsError_t cuvsHnswSearchParamsDestroy(cuvsHnswSearchParams_t params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | `cuvsHnswSearchParams_t` | cuvsHnswSearchParams_t to de-allocate |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/hnsw.h:440`_

## C API for CUDA ANN Graph-based nearest neighbor search

_Doxygen group: `hnsw_c_index_search`_

### cuvsHnswSearch

Search a HNSW index with a `DLManagedTensor` which has underlying

```c
cuvsError_t cuvsHnswSearch(cuvsResources_t res,
cuvsHnswSearchParams_t params,
cuvsHnswIndex_t index,
DLManagedTensor* queries,
DLManagedTensor* neighbors,
DLManagedTensor* distances);
```

`DLDeviceType` equal to `kDLCPU`, `kDLCUDAHost`, or `kDLCUDAManaged`. It is also important to note that the HNSW Index must have been built with the same type of `queries`, such that `index.dtype.code == queries.dl_tensor.dtype.code` Supported types for input are:

1. `queries`: a. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32` b. `kDLDataType.code == kDLInt` and `kDLDataType.bits = 8` c. `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 8`
2. `neighbors`: `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 64`
3. `distances`: `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32` NOTE: When hierarchy is `NONE`, the HNSW index can only be searched by the hnswlib wrapper in cuVS, as the format is not compatible with the original hnswlib.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `params` | in | `cuvsHnswSearchParams_t` | cuvsHnswSearchParams_t used to search Hnsw index |
| `index` | in | `cuvsHnswIndex_t` | cuvsHnswIndex which has been returned by `cuvsHnswFromCagra` |
| `queries` | in | `DLManagedTensor*` | DLManagedTensor* queries dataset to search |
| `neighbors` | out | `DLManagedTensor*` | DLManagedTensor* output `k` neighbors for queries |
| `distances` | out | `DLManagedTensor*` | DLManagedTensor* output `k` distances for queries |

**Returns**

`cuvsError_t`

_Source: `c/include/cuvs/neighbors/hnsw.h:499`_

## HNSW C-API serialize functions

_Doxygen group: `hnsw_c_index_serialize`_

### cuvsHnswSerialize

Serialize a CAGRA index to a file as an hnswlib index

```c
cuvsError_t cuvsHnswSerialize(cuvsResources_t res, const char* filename, cuvsHnswIndex_t index);
```

NOTE: When hierarchy is `NONE`, the saved hnswlib index is immutable and can only be read by the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib. However, when hierarchy is `CPU`, the saved hnswlib index is compatible with the original hnswlib library.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `filename` | in | `const char*` | the name of the file to save the index |
| `index` | in | `cuvsHnswIndex_t` | cuvsHnswIndex_t to serialize |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/hnsw.h:554`_

### cuvsHnswDeserialize

Load hnswlib index from file which was serialized from a HNSW index.

```c
cuvsError_t cuvsHnswDeserialize(cuvsResources_t res,
cuvsHnswIndexParams_t params,
const char* filename,
int dim,
cuvsDistanceType metric,
cuvsHnswIndex_t index);
```

NOTE: When hierarchy is `NONE`, the loaded hnswlib index is immutable, and only be read by the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib. Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `params` | in | `cuvsHnswIndexParams_t` | cuvsHnswIndexParams_t used to load Hnsw index |
| `filename` | in | `const char*` | the name of the file that stores the index |
| `dim` | in | `int` | the dimension of the vectors in the index |
| `metric` | in | `cuvsDistanceType` | the distance metric used to build the index |
| `index` | out | `cuvsHnswIndex_t` | HNSW index loaded disk |

**Returns**

`cuvsError_t`

_Source: `c/include/cuvs/neighbors/hnsw.h:592`_
