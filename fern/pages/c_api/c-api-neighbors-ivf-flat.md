---
slug: api-reference/c-api-neighbors-ivf-flat
---

# IVF Flat

_Source header: `c/include/cuvs/neighbors/ivf_flat.h`_

## IVF-Flat index build parameters

_Doxygen group: `ivf_flat_c_index_params`_

### cuvsIvfFlatIndexParams

Supplemental parameters to build IVF-Flat Index

```c
struct cuvsIvfFlatIndexParams { ... } ;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `metric` | `cuvsDistanceType` | Distance type. |
| `metric_arg` | `float` | The argument used by some distance metrics. |
| `add_data_on_build` | `bool` | Whether to add the dataset content to the index, i.e.: |
| `n_lists` | `uint32_t` | The number of inverted lists (clusters) |
| `kmeans_n_iters` | `uint32_t` | The number of iterations searching for kmeans centers (index building). |
| `kmeans_trainset_fraction` | `double` | The fraction of data to use during iterative kmeans building. |
| `adaptive_centers` | `bool` | By default (adaptive_centers = false), the cluster centers are trained in `ivf_flat::build`, |
| `conservative_memory_allocation` | `bool` | By default, the algorithm allocates more space than necessary for individual clusters |

_Source: `c/include/cuvs/neighbors/ivf_flat.h:27`_

### cuvsIvfFlatIndexParamsCreate

Allocate IVF-Flat Index params, and populate with default values

```c
cuvsError_t cuvsIvfFlatIndexParamsCreate(cuvsIvfFlatIndexParams_t* index_params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index_params` | in | `cuvsIvfFlatIndexParams_t*` | cuvsIvfFlatIndexParams_t to allocate |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/ivf_flat.h:80`_

### cuvsIvfFlatIndexParamsDestroy

De-allocate IVF-Flat Index params

```c
cuvsError_t cuvsIvfFlatIndexParamsDestroy(cuvsIvfFlatIndexParams_t index_params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index_params` | in | `cuvsIvfFlatIndexParams_t` |  |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/ivf_flat.h:88`_

## IVF-Flat index search parameters

_Doxygen group: `ivf_flat_c_search_params`_

### cuvsIvfFlatSearchParams

Supplemental parameters to search IVF-Flat index

```c
struct cuvsIvfFlatSearchParams { ... } ;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `n_probes` | `uint32_t` | The number of clusters to search. |

_Source: `c/include/cuvs/neighbors/ivf_flat.h:101`_

### cuvsIvfFlatSearchParamsCreate

Allocate IVF-Flat search params, and populate with default values

```c
cuvsError_t cuvsIvfFlatSearchParamsCreate(cuvsIvfFlatSearchParams_t* params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | `cuvsIvfFlatSearchParams_t*` | cuvsIvfFlatSearchParams_t to allocate |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/ivf_flat.h:114`_

### cuvsIvfFlatSearchParamsDestroy

De-allocate IVF-Flat search params

```c
cuvsError_t cuvsIvfFlatSearchParamsDestroy(cuvsIvfFlatSearchParams_t params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | `cuvsIvfFlatSearchParams_t` |  |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/ivf_flat.h:122`_

## IVF-Flat index

_Doxygen group: `ivf_flat_c_index`_

### cuvsIvfFlatIndexCreate

Allocate IVF-Flat index

```c
cuvsError_t cuvsIvfFlatIndexCreate(cuvsIvfFlatIndex_t* index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | `cuvsIvfFlatIndex_t*` | cuvsIvfFlatIndex_t to allocate |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/ivf_flat.h:148`_

### cuvsIvfFlatIndexDestroy

De-allocate IVF-Flat index

```c
cuvsError_t cuvsIvfFlatIndexDestroy(cuvsIvfFlatIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | `cuvsIvfFlatIndex_t` | cuvsIvfFlatIndex_t to de-allocate |

**Returns**

`cuvsError_t`

_Source: `c/include/cuvs/neighbors/ivf_flat.h:155`_

### cuvsIvfFlatIndexGetNLists

Get the number of clusters/inverted lists

```c
cuvsError_t cuvsIvfFlatIndexGetNLists(cuvsIvfFlatIndex_t index, int64_t* n_lists);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` |  | `cuvsIvfFlatIndex_t` |  |
| `n_lists` |  | `int64_t*` |  |

**Returns**

`cuvsError_t`

_Source: `c/include/cuvs/neighbors/ivf_flat.h:158`_

### cuvsIvfFlatIndexGetDim

Get the dimensionality of the data

```c
cuvsError_t cuvsIvfFlatIndexGetDim(cuvsIvfFlatIndex_t index, int64_t* dim);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` |  | `cuvsIvfFlatIndex_t` |  |
| `dim` |  | `int64_t*` |  |

**Returns**

`cuvsError_t`

_Source: `c/include/cuvs/neighbors/ivf_flat.h:161`_

### cuvsIvfFlatIndexGetCenters

Get the cluster centers corresponding to the lists [n_lists, dim]

```c
cuvsError_t cuvsIvfFlatIndexGetCenters(cuvsIvfFlatIndex_t index, DLManagedTensor* centers);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | `cuvsIvfFlatIndex_t` | cuvsIvfFlatIndex_t Built Ivf-Flat Index |
| `centers` | out | `DLManagedTensor*` | Preallocated array on host or device memory to store output, [n_lists, dim] |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/ivf_flat.h:170`_

## IVF-Flat index build

_Doxygen group: `ivf_flat_c_index_build`_

### cuvsIvfFlatBuild

Build a IVF-Flat index with a `DLManagedTensor` which has underlying

```c
cuvsError_t cuvsIvfFlatBuild(cuvsResources_t res,
cuvsIvfFlatIndexParams_t index_params,
DLManagedTensor* dataset,
cuvsIvfFlatIndex_t index);
```

`DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`, or `kDLCPU`. Also, acceptable underlying types are: 1. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32` 2. `kDLDataType.code == kDLInt` and `kDLDataType.bits = 8` 3. `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 8`

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `index_params` | in | `cuvsIvfFlatIndexParams_t` | cuvsIvfFlatIndexParams_t used to build IVF-Flat index |
| `dataset` | in | `DLManagedTensor*` | DLManagedTensor* training dataset |
| `index` | out | `cuvsIvfFlatIndex_t` | cuvsIvfFlatIndex_t Newly built IVF-Flat index |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/ivf_flat.h:222`_

## IVF-Flat index search

_Doxygen group: `ivf_flat_c_index_search`_

### cuvsIvfFlatSearch

Search a IVF-Flat index with a `DLManagedTensor` which has underlying

```c
cuvsError_t cuvsIvfFlatSearch(cuvsResources_t res,
cuvsIvfFlatSearchParams_t search_params,
cuvsIvfFlatIndex_t index,
DLManagedTensor* queries,
DLManagedTensor* neighbors,
DLManagedTensor* distances,
cuvsFilter filter);
```

`DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`. It is also important to note that the IVF-Flat Index must have been built with the same type of `queries`, such that `index.dtype.code == queries.dl_tensor.dtype.code` Types for input are: 1. `queries`: `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32` 2. `neighbors`: `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 32` 3. `distances`: `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `search_params` | in | `cuvsIvfFlatSearchParams_t` | cuvsIvfFlatSearchParams_t used to search IVF-Flat index |
| `index` | in | `cuvsIvfFlatIndex_t` | ivfFlatIndex which has been returned by `ivfFlatBuild` |
| `queries` | in | `DLManagedTensor*` | DLManagedTensor* queries dataset to search |
| `neighbors` | out | `DLManagedTensor*` | DLManagedTensor* output `k` neighbors for queries |
| `distances` | out | `DLManagedTensor*` | DLManagedTensor* output `k` distances for queries |
| `filter` | in | `cuvsFilter` | cuvsFilter input filter that can be used to filter queries and neighbors based on the given bitset. |

**Returns**

`cuvsError_t`

_Source: `c/include/cuvs/neighbors/ivf_flat.h:279`_

## IVF-Flat C-API serialize functions

_Doxygen group: `ivf_flat_c_index_serialize`_

### cuvsIvfFlatSerialize

Save the index to file.

```c
cuvsError_t cuvsIvfFlatSerialize(cuvsResources_t res,
const char* filename,
cuvsIvfFlatIndex_t index);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `filename` | in | `const char*` | the file name for saving the index |
| `index` | in | `cuvsIvfFlatIndex_t` | IVF-Flat index |

**Returns**

`cuvsError_t`

_Source: `c/include/cuvs/neighbors/ivf_flat.h:315`_

### cuvsIvfFlatDeserialize

Load index from file.

```c
cuvsError_t cuvsIvfFlatDeserialize(cuvsResources_t res,
const char* filename,
cuvsIvfFlatIndex_t index);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `filename` | in | `const char*` | the name of the file that stores the index |
| `index` | out | `cuvsIvfFlatIndex_t` | IVF-Flat index loaded disk |

**Returns**

`cuvsError_t`

_Source: `c/include/cuvs/neighbors/ivf_flat.h:328`_

## IVF-Flat index extend

_Doxygen group: `ivf_flat_c_index_extend`_

### cuvsIvfFlatExtend

Extend the index with the new data.

```c
cuvsError_t cuvsIvfFlatExtend(cuvsResources_t res,
DLManagedTensor* new_vectors,
DLManagedTensor* new_indices,
cuvsIvfFlatIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `new_vectors` | in | `DLManagedTensor*` | DLManagedTensor* the new vectors to add to the index |
| `new_indices` | in | `DLManagedTensor*` | DLManagedTensor* vector of new indices for the new vectors |
| `index` | inout | `cuvsIvfFlatIndex_t` | IVF-Flat index to be extended |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/ivf_flat.h:348`_
