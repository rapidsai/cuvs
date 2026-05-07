---
slug: api-reference/c-api-neighbors-mg-ivf-flat
---

# Multi-GPU IVF Flat

_Source header: `c/include/cuvs/neighbors/mg_ivf_flat.h`_

## Multi-GPU IVF-Flat index build parameters

_Doxygen group: `mg_ivf_flat_c_index_params`_

### cuvsMultiGpuIvfFlatIndexParams

Multi-GPU parameters to build IVF-Flat Index

This structure extends the base IVF-Flat index parameters with multi-GPU specific settings.

```c
struct cuvsMultiGpuIvfFlatIndexParams { ... } ;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `base_params` | `cuvsIvfFlatIndexParams_t` | Base IVF-Flat index parameters |
| `mode` | `cuvsMultiGpuDistributionMode` | Distribution mode for multi-GPU setup |

_Source: `c/include/cuvs/neighbors/mg_ivf_flat.h:28`_

### cuvsMultiGpuIvfFlatIndexParamsCreate

Allocate Multi-GPU IVF-Flat Index params, and populate with default values

```c
cuvsError_t cuvsMultiGpuIvfFlatIndexParamsCreate(cuvsMultiGpuIvfFlatIndexParams_t* index_params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index_params` | in | `cuvsMultiGpuIvfFlatIndexParams_t*` | cuvsMultiGpuIvfFlatIndexParams_t to allocate |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/mg_ivf_flat.h:43`_

### cuvsMultiGpuIvfFlatIndexParamsDestroy

De-allocate Multi-GPU IVF-Flat Index params

```c
cuvsError_t cuvsMultiGpuIvfFlatIndexParamsDestroy(cuvsMultiGpuIvfFlatIndexParams_t index_params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index_params` | in | `cuvsMultiGpuIvfFlatIndexParams_t` |  |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/mg_ivf_flat.h:51`_

## Multi-GPU IVF-Flat index search parameters

_Doxygen group: `mg_ivf_flat_c_search_params`_

### cuvsMultiGpuIvfFlatSearchParams

Multi-GPU parameters to search IVF-Flat index

This structure extends the base IVF-Flat search parameters with multi-GPU specific settings.

```c
struct cuvsMultiGpuIvfFlatSearchParams { ... } ;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `base_params` | `cuvsIvfFlatSearchParams_t` | Base IVF-Flat search parameters |
| `search_mode` | `cuvsMultiGpuReplicatedSearchMode` | Replicated search mode |
| `merge_mode` | `cuvsMultiGpuShardedMergeMode` | Sharded merge mode |
| `n_rows_per_batch` | `int64_t` | Number of rows per batch |

_Source: `c/include/cuvs/neighbors/mg_ivf_flat.h:67`_

### cuvsMultiGpuIvfFlatSearchParamsCreate

Allocate Multi-GPU IVF-Flat search params, and populate with default values

```c
cuvsError_t cuvsMultiGpuIvfFlatSearchParamsCreate(cuvsMultiGpuIvfFlatSearchParams_t* params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | `cuvsMultiGpuIvfFlatSearchParams_t*` | cuvsMultiGpuIvfFlatSearchParams_t to allocate |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/mg_ivf_flat.h:86`_

### cuvsMultiGpuIvfFlatSearchParamsDestroy

De-allocate Multi-GPU IVF-Flat search params

```c
cuvsError_t cuvsMultiGpuIvfFlatSearchParamsDestroy(cuvsMultiGpuIvfFlatSearchParams_t params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | `cuvsMultiGpuIvfFlatSearchParams_t` |  |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/mg_ivf_flat.h:94`_

## Multi-GPU IVF-Flat index

_Doxygen group: `mg_ivf_flat_c_index`_

### cuvsMultiGpuIvfFlatIndexCreate

Allocate Multi-GPU IVF-Flat index

```c
cuvsError_t cuvsMultiGpuIvfFlatIndexCreate(cuvsMultiGpuIvfFlatIndex_t* index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | `cuvsMultiGpuIvfFlatIndex_t*` | cuvsMultiGpuIvfFlatIndex_t to allocate |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/mg_ivf_flat.h:122`_

### cuvsMultiGpuIvfFlatIndexDestroy

De-allocate Multi-GPU IVF-Flat index

```c
cuvsError_t cuvsMultiGpuIvfFlatIndexDestroy(cuvsMultiGpuIvfFlatIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | `cuvsMultiGpuIvfFlatIndex_t` | cuvsMultiGpuIvfFlatIndex_t to de-allocate |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/mg_ivf_flat.h:130`_

## Multi-GPU IVF-Flat index build

_Doxygen group: `mg_ivf_flat_c_index_build`_

### cuvsMultiGpuIvfFlatBuild

Build a Multi-GPU IVF-Flat index

```c
cuvsError_t cuvsMultiGpuIvfFlatBuild(cuvsResources_t res,
cuvsMultiGpuIvfFlatIndexParams_t params,
DLManagedTensor* dataset_tensor,
cuvsMultiGpuIvfFlatIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `params` | in | `cuvsMultiGpuIvfFlatIndexParams_t` | Multi-GPU IVF-Flat index parameters |
| `dataset_tensor` | in | `DLManagedTensor*` | DLManagedTensor* training dataset |
| `index` | out | `cuvsMultiGpuIvfFlatIndex_t` | Multi-GPU IVF-Flat index |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/mg_ivf_flat.h:150`_

## Multi-GPU IVF-Flat index search

_Doxygen group: `mg_ivf_flat_c_index_search`_

### cuvsMultiGpuIvfFlatSearch

Search a Multi-GPU IVF-Flat index

```c
cuvsError_t cuvsMultiGpuIvfFlatSearch(cuvsResources_t res,
cuvsMultiGpuIvfFlatSearchParams_t params,
cuvsMultiGpuIvfFlatIndex_t index,
DLManagedTensor* queries_tensor,
DLManagedTensor* neighbors_tensor,
DLManagedTensor* distances_tensor);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `params` | in | `cuvsMultiGpuIvfFlatSearchParams_t` | Multi-GPU IVF-Flat search parameters |
| `index` | in | `cuvsMultiGpuIvfFlatIndex_t` | Multi-GPU IVF-Flat index |
| `queries_tensor` | in | `DLManagedTensor*` | DLManagedTensor* queries dataset |
| `neighbors_tensor` | out | `DLManagedTensor*` | DLManagedTensor* output neighbors |
| `distances_tensor` | out | `DLManagedTensor*` | DLManagedTensor* output distances |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/mg_ivf_flat.h:175`_

## Multi-GPU IVF-Flat index extend

_Doxygen group: `mg_ivf_flat_c_index_extend`_

### cuvsMultiGpuIvfFlatExtend

Extend a Multi-GPU IVF-Flat index

```c
cuvsError_t cuvsMultiGpuIvfFlatExtend(cuvsResources_t res,
cuvsMultiGpuIvfFlatIndex_t index,
DLManagedTensor* new_vectors_tensor,
DLManagedTensor* new_indices_tensor);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `index` | in,out | `cuvsMultiGpuIvfFlatIndex_t` | Multi-GPU IVF-Flat index to extend |
| `new_vectors_tensor` | in | `DLManagedTensor*` | DLManagedTensor* new vectors to add |
| `new_indices_tensor` | in | `DLManagedTensor*` | DLManagedTensor* new indices (optional, can be NULL) |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/mg_ivf_flat.h:200`_

## Multi-GPU IVF-Flat index serialize

_Doxygen group: `mg_ivf_flat_c_index_serialize`_

### cuvsMultiGpuIvfFlatSerialize

Serialize a Multi-GPU IVF-Flat index to file

```c
cuvsError_t cuvsMultiGpuIvfFlatSerialize(cuvsResources_t res,
cuvsMultiGpuIvfFlatIndex_t index,
const char* filename);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `index` | in | `cuvsMultiGpuIvfFlatIndex_t` | Multi-GPU IVF-Flat index to serialize |
| `filename` | in | `const char*` | Path to the output file |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/mg_ivf_flat.h:222`_

## Multi-GPU IVF-Flat index deserialize

_Doxygen group: `mg_ivf_flat_c_index_deserialize`_

### cuvsMultiGpuIvfFlatDeserialize

Deserialize a Multi-GPU IVF-Flat index from file

```c
cuvsError_t cuvsMultiGpuIvfFlatDeserialize(cuvsResources_t res,
const char* filename,
cuvsMultiGpuIvfFlatIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `filename` | in | `const char*` | Path to the input file |
| `index` | out | `cuvsMultiGpuIvfFlatIndex_t` | Multi-GPU IVF-Flat index |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/mg_ivf_flat.h:243`_

## Multi-GPU IVF-Flat index distribute

_Doxygen group: `mg_ivf_flat_c_index_distribute`_

### cuvsMultiGpuIvfFlatDistribute

Distribute a local IVF-Flat index to create a Multi-GPU index

```c
cuvsError_t cuvsMultiGpuIvfFlatDistribute(cuvsResources_t res,
const char* filename,
cuvsMultiGpuIvfFlatIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `filename` | in | `const char*` | Path to the local index file |
| `index` | out | `cuvsMultiGpuIvfFlatIndex_t` | Multi-GPU IVF-Flat index |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/mg_ivf_flat.h:264`_
