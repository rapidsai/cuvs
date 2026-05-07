---
slug: api-reference/c-api-neighbors-mg-ivf-pq
---

# Multi-GPU IVF PQ

_Source header: `c/include/cuvs/neighbors/mg_ivf_pq.h`_

## Multi-GPU IVF-PQ index build parameters

_Doxygen group: `mg_ivf_pq_c_index_params`_

### cuvsMultiGpuIvfPqIndexParams

Multi-GPU parameters to build IVF-PQ Index

This structure extends the base IVF-PQ index parameters with multi-GPU specific settings.

```c
struct cuvsMultiGpuIvfPqIndexParams { ... } ;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `base_params` | `cuvsIvfPqIndexParams_t` | Base IVF-PQ index parameters |
| `mode` | `cuvsMultiGpuDistributionMode` | Distribution mode for multi-GPU setup |

_Source: `c/include/cuvs/neighbors/mg_ivf_pq.h:28`_

### cuvsMultiGpuIvfPqIndexParamsCreate

Allocate Multi-GPU IVF-PQ Index params, and populate with default values

```c
cuvsError_t cuvsMultiGpuIvfPqIndexParamsCreate(cuvsMultiGpuIvfPqIndexParams_t* index_params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index_params` | in | `cuvsMultiGpuIvfPqIndexParams_t*` | cuvsMultiGpuIvfPqIndexParams_t to allocate |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/mg_ivf_pq.h:43`_

### cuvsMultiGpuIvfPqIndexParamsDestroy

De-allocate Multi-GPU IVF-PQ Index params

```c
cuvsError_t cuvsMultiGpuIvfPqIndexParamsDestroy(cuvsMultiGpuIvfPqIndexParams_t index_params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index_params` | in | `cuvsMultiGpuIvfPqIndexParams_t` |  |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/mg_ivf_pq.h:51`_

## Multi-GPU IVF-PQ index search parameters

_Doxygen group: `mg_ivf_pq_c_search_params`_

### cuvsMultiGpuIvfPqSearchParams

Multi-GPU parameters to search IVF-PQ index

This structure extends the base IVF-PQ search parameters with multi-GPU specific settings.

```c
struct cuvsMultiGpuIvfPqSearchParams { ... } ;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `base_params` | `cuvsIvfPqSearchParams_t` | Base IVF-PQ search parameters |
| `search_mode` | `cuvsMultiGpuReplicatedSearchMode` | Replicated search mode |
| `merge_mode` | `cuvsMultiGpuShardedMergeMode` | Sharded merge mode |
| `n_rows_per_batch` | `int64_t` | Number of rows per batch |

_Source: `c/include/cuvs/neighbors/mg_ivf_pq.h:67`_

### cuvsMultiGpuIvfPqSearchParamsCreate

Allocate Multi-GPU IVF-PQ search params, and populate with default values

```c
cuvsError_t cuvsMultiGpuIvfPqSearchParamsCreate(cuvsMultiGpuIvfPqSearchParams_t* params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | `cuvsMultiGpuIvfPqSearchParams_t*` | cuvsMultiGpuIvfPqSearchParams_t to allocate |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/mg_ivf_pq.h:86`_

### cuvsMultiGpuIvfPqSearchParamsDestroy

De-allocate Multi-GPU IVF-PQ search params

```c
cuvsError_t cuvsMultiGpuIvfPqSearchParamsDestroy(cuvsMultiGpuIvfPqSearchParams_t params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | `cuvsMultiGpuIvfPqSearchParams_t` |  |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/mg_ivf_pq.h:94`_

## Multi-GPU IVF-PQ index

_Doxygen group: `mg_ivf_pq_c_index`_

### cuvsMultiGpuIvfPqIndexCreate

Allocate Multi-GPU IVF-PQ index

```c
cuvsError_t cuvsMultiGpuIvfPqIndexCreate(cuvsMultiGpuIvfPqIndex_t* index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | `cuvsMultiGpuIvfPqIndex_t*` | cuvsMultiGpuIvfPqIndex_t to allocate |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/mg_ivf_pq.h:122`_

### cuvsMultiGpuIvfPqIndexDestroy

De-allocate Multi-GPU IVF-PQ index

```c
cuvsError_t cuvsMultiGpuIvfPqIndexDestroy(cuvsMultiGpuIvfPqIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | `cuvsMultiGpuIvfPqIndex_t` | cuvsMultiGpuIvfPqIndex_t to de-allocate |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/mg_ivf_pq.h:130`_

## Multi-GPU IVF-PQ index build

_Doxygen group: `mg_ivf_pq_c_index_build`_

### cuvsMultiGpuIvfPqBuild

Build a Multi-GPU IVF-PQ index

```c
cuvsError_t cuvsMultiGpuIvfPqBuild(cuvsResources_t res,
cuvsMultiGpuIvfPqIndexParams_t params,
DLManagedTensor* dataset_tensor,
cuvsMultiGpuIvfPqIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `params` | in | `cuvsMultiGpuIvfPqIndexParams_t` | Multi-GPU IVF-PQ index parameters |
| `dataset_tensor` | in | `DLManagedTensor*` | DLManagedTensor* training dataset |
| `index` | out | `cuvsMultiGpuIvfPqIndex_t` | Multi-GPU IVF-PQ index |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/mg_ivf_pq.h:150`_

## Multi-GPU IVF-PQ index search

_Doxygen group: `mg_ivf_pq_c_index_search`_

### cuvsMultiGpuIvfPqSearch

Search a Multi-GPU IVF-PQ index

```c
cuvsError_t cuvsMultiGpuIvfPqSearch(cuvsResources_t res,
cuvsMultiGpuIvfPqSearchParams_t params,
cuvsMultiGpuIvfPqIndex_t index,
DLManagedTensor* queries_tensor,
DLManagedTensor* neighbors_tensor,
DLManagedTensor* distances_tensor);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `params` | in | `cuvsMultiGpuIvfPqSearchParams_t` | Multi-GPU IVF-PQ search parameters |
| `index` | in | `cuvsMultiGpuIvfPqIndex_t` | Multi-GPU IVF-PQ index |
| `queries_tensor` | in | `DLManagedTensor*` | DLManagedTensor* queries dataset |
| `neighbors_tensor` | out | `DLManagedTensor*` | DLManagedTensor* output neighbors |
| `distances_tensor` | out | `DLManagedTensor*` | DLManagedTensor* output distances |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/mg_ivf_pq.h:175`_

## Multi-GPU IVF-PQ index extend

_Doxygen group: `mg_ivf_pq_c_index_extend`_

### cuvsMultiGpuIvfPqExtend

Extend a Multi-GPU IVF-PQ index

```c
cuvsError_t cuvsMultiGpuIvfPqExtend(cuvsResources_t res,
cuvsMultiGpuIvfPqIndex_t index,
DLManagedTensor* new_vectors_tensor,
DLManagedTensor* new_indices_tensor);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `index` | in,out | `cuvsMultiGpuIvfPqIndex_t` | Multi-GPU IVF-PQ index to extend |
| `new_vectors_tensor` | in | `DLManagedTensor*` | DLManagedTensor* new vectors to add |
| `new_indices_tensor` | in | `DLManagedTensor*` | DLManagedTensor* new indices (optional, can be NULL) |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/mg_ivf_pq.h:200`_

## Multi-GPU IVF-PQ index serialize

_Doxygen group: `mg_ivf_pq_c_index_serialize`_

### cuvsMultiGpuIvfPqSerialize

Serialize a Multi-GPU IVF-PQ index to file

```c
cuvsError_t cuvsMultiGpuIvfPqSerialize(cuvsResources_t res,
cuvsMultiGpuIvfPqIndex_t index,
const char* filename);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `index` | in | `cuvsMultiGpuIvfPqIndex_t` | Multi-GPU IVF-PQ index to serialize |
| `filename` | in | `const char*` | Path to the output file |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/mg_ivf_pq.h:222`_

## Multi-GPU IVF-PQ index deserialize

_Doxygen group: `mg_ivf_pq_c_index_deserialize`_

### cuvsMultiGpuIvfPqDeserialize

Deserialize a Multi-GPU IVF-PQ index from file

```c
cuvsError_t cuvsMultiGpuIvfPqDeserialize(cuvsResources_t res,
const char* filename,
cuvsMultiGpuIvfPqIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `filename` | in | `const char*` | Path to the input file |
| `index` | out | `cuvsMultiGpuIvfPqIndex_t` | Multi-GPU IVF-PQ index |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/mg_ivf_pq.h:243`_

## Multi-GPU IVF-PQ index distribute

_Doxygen group: `mg_ivf_pq_c_index_distribute`_

### cuvsMultiGpuIvfPqDistribute

Distribute a local IVF-PQ index to create a Multi-GPU index

```c
cuvsError_t cuvsMultiGpuIvfPqDistribute(cuvsResources_t res,
const char* filename,
cuvsMultiGpuIvfPqIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `filename` | in | `const char*` | Path to the local index file |
| `index` | out | `cuvsMultiGpuIvfPqIndex_t` | Multi-GPU IVF-PQ index |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/mg_ivf_pq.h:264`_
