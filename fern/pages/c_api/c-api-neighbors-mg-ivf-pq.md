---
slug: api-reference/c-api-neighbors-mg-ivf-pq
---

# Multi-GPU IVF PQ

_Source header: `c/include/cuvs/neighbors/mg_ivf_pq.h`_

## Multi-GPU IVF-PQ index build parameters

<a id="cuvsmultigpuivfpqindexparams"></a>
### cuvsMultiGpuIvfPqIndexParams

Multi-GPU parameters to build IVF-PQ Index

This structure extends the base IVF-PQ index parameters with multi-GPU specific settings.

```c
struct cuvsMultiGpuIvfPqIndexParams { ... };
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `base_params` | [`cuvsIvfPqIndexParams_t`](/api-reference/c-api-neighbors-ivf-pq#cuvsivfpqindexparams) | Base IVF-PQ index parameters |
| `mode` | [`cuvsMultiGpuDistributionMode`](/api-reference/c-api-neighbors-mg-common#cuvsmultigpudistributionmode) | Distribution mode for multi-GPU setup |

_Source: `c/include/cuvs/neighbors/mg_ivf_pq.h:28`_

<a id="cuvsmultigpuivfpqindexparamscreate"></a>
### cuvsMultiGpuIvfPqIndexParamsCreate

Allocate Multi-GPU IVF-PQ Index params, and populate with default values

```c
cuvsError_t cuvsMultiGpuIvfPqIndexParamsCreate(cuvsMultiGpuIvfPqIndexParams_t* index_params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index_params` | in | [`cuvsMultiGpuIvfPqIndexParams_t*`](/api-reference/c-api-neighbors-mg-ivf-pq#cuvsmultigpuivfpqindexparams) | cuvsMultiGpuIvfPqIndexParams_t to allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

cuvsError_t

_Source: `c/include/cuvs/neighbors/mg_ivf_pq.h:43`_

<a id="cuvsmultigpuivfpqindexparamsdestroy"></a>
### cuvsMultiGpuIvfPqIndexParamsDestroy

De-allocate Multi-GPU IVF-PQ Index params

```c
cuvsError_t cuvsMultiGpuIvfPqIndexParamsDestroy(cuvsMultiGpuIvfPqIndexParams_t index_params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index_params` | in | [`cuvsMultiGpuIvfPqIndexParams_t`](/api-reference/c-api-neighbors-mg-ivf-pq#cuvsmultigpuivfpqindexparams) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

cuvsError_t

_Source: `c/include/cuvs/neighbors/mg_ivf_pq.h:51`_

## Multi-GPU IVF-PQ index search parameters

<a id="cuvsmultigpuivfpqsearchparams"></a>
### cuvsMultiGpuIvfPqSearchParams

Multi-GPU parameters to search IVF-PQ index

This structure extends the base IVF-PQ search parameters with multi-GPU specific settings.

```c
struct cuvsMultiGpuIvfPqSearchParams { ... };
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `base_params` | [`cuvsIvfPqSearchParams_t`](/api-reference/c-api-neighbors-ivf-pq#cuvsivfpqsearchparams) | Base IVF-PQ search parameters |
| `search_mode` | [`cuvsMultiGpuReplicatedSearchMode`](/api-reference/c-api-neighbors-mg-common#cuvsmultigpureplicatedsearchmode) | Replicated search mode |
| `merge_mode` | [`cuvsMultiGpuShardedMergeMode`](/api-reference/c-api-neighbors-mg-common#cuvsmultigpushardedmergemode) | Sharded merge mode |
| `n_rows_per_batch` | `int64_t` | Number of rows per batch |

_Source: `c/include/cuvs/neighbors/mg_ivf_pq.h:67`_

<a id="cuvsmultigpuivfpqsearchparamscreate"></a>
### cuvsMultiGpuIvfPqSearchParamsCreate

Allocate Multi-GPU IVF-PQ search params, and populate with default values

```c
cuvsError_t cuvsMultiGpuIvfPqSearchParamsCreate(cuvsMultiGpuIvfPqSearchParams_t* params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | [`cuvsMultiGpuIvfPqSearchParams_t*`](/api-reference/c-api-neighbors-mg-ivf-pq#cuvsmultigpuivfpqsearchparams) | cuvsMultiGpuIvfPqSearchParams_t to allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

cuvsError_t

_Source: `c/include/cuvs/neighbors/mg_ivf_pq.h:86`_

<a id="cuvsmultigpuivfpqsearchparamsdestroy"></a>
### cuvsMultiGpuIvfPqSearchParamsDestroy

De-allocate Multi-GPU IVF-PQ search params

```c
cuvsError_t cuvsMultiGpuIvfPqSearchParamsDestroy(cuvsMultiGpuIvfPqSearchParams_t params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | [`cuvsMultiGpuIvfPqSearchParams_t`](/api-reference/c-api-neighbors-mg-ivf-pq#cuvsmultigpuivfpqsearchparams) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

cuvsError_t

_Source: `c/include/cuvs/neighbors/mg_ivf_pq.h:94`_

## Multi-GPU IVF-PQ index

<a id="cuvsmultigpuivfpqindex"></a>
### cuvsMultiGpuIvfPqIndex

Struct to hold address of cuvs::neighbors::mg_index&lt;ivf_pq::index&gt; and its active trained

dtype

```c
typedef struct { ... } cuvsMultiGpuIvfPqIndex;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `addr` | `uintptr_t` |  |
| `dtype` | `DLDataType` |  |

_Source: `c/include/cuvs/neighbors/mg_ivf_pq.h:109`_

<a id="cuvsmultigpuivfpqindexcreate"></a>
### cuvsMultiGpuIvfPqIndexCreate

Allocate Multi-GPU IVF-PQ index

```c
cuvsError_t cuvsMultiGpuIvfPqIndexCreate(cuvsMultiGpuIvfPqIndex_t* index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | [`cuvsMultiGpuIvfPqIndex_t*`](/api-reference/c-api-neighbors-mg-ivf-pq#cuvsmultigpuivfpqindex) | cuvsMultiGpuIvfPqIndex_t to allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

cuvsError_t

_Source: `c/include/cuvs/neighbors/mg_ivf_pq.h:122`_

<a id="cuvsmultigpuivfpqindexdestroy"></a>
### cuvsMultiGpuIvfPqIndexDestroy

De-allocate Multi-GPU IVF-PQ index

```c
cuvsError_t cuvsMultiGpuIvfPqIndexDestroy(cuvsMultiGpuIvfPqIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | [`cuvsMultiGpuIvfPqIndex_t`](/api-reference/c-api-neighbors-mg-ivf-pq#cuvsmultigpuivfpqindex) | cuvsMultiGpuIvfPqIndex_t to de-allocate |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

cuvsError_t

_Source: `c/include/cuvs/neighbors/mg_ivf_pq.h:130`_

## Multi-GPU IVF-PQ index build

<a id="cuvsmultigpuivfpqbuild"></a>
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
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `params` | in | [`cuvsMultiGpuIvfPqIndexParams_t`](/api-reference/c-api-neighbors-mg-ivf-pq#cuvsmultigpuivfpqindexparams) | Multi-GPU IVF-PQ index parameters |
| `dataset_tensor` | in | `DLManagedTensor*` | DLManagedTensor* training dataset |
| `index` | out | [`cuvsMultiGpuIvfPqIndex_t`](/api-reference/c-api-neighbors-mg-ivf-pq#cuvsmultigpuivfpqindex) | Multi-GPU IVF-PQ index |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

cuvsError_t

_Source: `c/include/cuvs/neighbors/mg_ivf_pq.h:150`_

## Multi-GPU IVF-PQ index search

<a id="cuvsmultigpuivfpqsearch"></a>
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
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `params` | in | [`cuvsMultiGpuIvfPqSearchParams_t`](/api-reference/c-api-neighbors-mg-ivf-pq#cuvsmultigpuivfpqsearchparams) | Multi-GPU IVF-PQ search parameters |
| `index` | in | [`cuvsMultiGpuIvfPqIndex_t`](/api-reference/c-api-neighbors-mg-ivf-pq#cuvsmultigpuivfpqindex) | Multi-GPU IVF-PQ index |
| `queries_tensor` | in | `DLManagedTensor*` | DLManagedTensor* queries dataset |
| `neighbors_tensor` | out | `DLManagedTensor*` | DLManagedTensor* output neighbors |
| `distances_tensor` | out | `DLManagedTensor*` | DLManagedTensor* output distances |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

cuvsError_t

_Source: `c/include/cuvs/neighbors/mg_ivf_pq.h:175`_

## Multi-GPU IVF-PQ index extend

<a id="cuvsmultigpuivfpqextend"></a>
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
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `index` | in,out | [`cuvsMultiGpuIvfPqIndex_t`](/api-reference/c-api-neighbors-mg-ivf-pq#cuvsmultigpuivfpqindex) | Multi-GPU IVF-PQ index to extend |
| `new_vectors_tensor` | in | `DLManagedTensor*` | DLManagedTensor* new vectors to add |
| `new_indices_tensor` | in | `DLManagedTensor*` | DLManagedTensor* new indices (optional, can be NULL) |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

cuvsError_t

_Source: `c/include/cuvs/neighbors/mg_ivf_pq.h:200`_

## Multi-GPU IVF-PQ index serialize

<a id="cuvsmultigpuivfpqserialize"></a>
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
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `index` | in | [`cuvsMultiGpuIvfPqIndex_t`](/api-reference/c-api-neighbors-mg-ivf-pq#cuvsmultigpuivfpqindex) | Multi-GPU IVF-PQ index to serialize |
| `filename` | in | `const char*` | Path to the output file |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

cuvsError_t

_Source: `c/include/cuvs/neighbors/mg_ivf_pq.h:222`_

## Multi-GPU IVF-PQ index deserialize

<a id="cuvsmultigpuivfpqdeserialize"></a>
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
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `filename` | in | `const char*` | Path to the input file |
| `index` | out | [`cuvsMultiGpuIvfPqIndex_t`](/api-reference/c-api-neighbors-mg-ivf-pq#cuvsmultigpuivfpqindex) | Multi-GPU IVF-PQ index |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

cuvsError_t

_Source: `c/include/cuvs/neighbors/mg_ivf_pq.h:243`_

## Multi-GPU IVF-PQ index distribute

<a id="cuvsmultigpuivfpqdistribute"></a>
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
| `res` | in | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) | cuvsResources_t opaque C handle |
| `filename` | in | `const char*` | Path to the local index file |
| `index` | out | [`cuvsMultiGpuIvfPqIndex_t`](/api-reference/c-api-neighbors-mg-ivf-pq#cuvsmultigpuivfpqindex) | Multi-GPU IVF-PQ index |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

cuvsError_t

_Source: `c/include/cuvs/neighbors/mg_ivf_pq.h:264`_
