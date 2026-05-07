---
slug: api-reference/c-api-neighbors-tiered-index
---

# Tiered Index

_Source header: `c/include/cuvs/neighbors/tiered_index.h`_

## Tiered Index

_Doxygen group: `tiered_index_c_index`_

### cuvsTieredIndexCreate

Allocate Tiered Index

```c
cuvsError_t cuvsTieredIndexCreate(cuvsTieredIndex_t* index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | `cuvsTieredIndex_t*` | cuvsTieredIndex_t to allocate |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/tiered_index.h:53`_

### cuvsTieredIndexDestroy

De-allocate Tiered index

```c
cuvsError_t cuvsTieredIndexDestroy(cuvsTieredIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | `cuvsTieredIndex_t` | cuvsTieredIndex_t to de-allocate |

**Returns**

`cuvsError_t`

_Source: `c/include/cuvs/neighbors/tiered_index.h:60`_

## Tiered Index build parameters

_Doxygen group: `tiered_c_index_params`_

### cuvsTieredIndexParams

Supplemental parameters to build a TieredIndex

```c
struct cuvsTieredIndexParams { ... } ;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `metric` | `cuvsDistanceType` | Distance type. |
| `algo` | `cuvsTieredIndexANNAlgo` | The type of ANN algorithm we are using |
| `min_ann_rows` | `int64_t` | The minimum number of rows necessary in the index to create an |
| `create_ann_index_on_extend` | `bool` | Whether or not to create a new ann index on extend, if the number |
| `cagra_params` | `cuvsCagraIndexParams_t` | Optional parameters for building a cagra index |
| `ivf_flat_params` | `cuvsIvfFlatIndexParams_t` | Optional parameters for building a ivf_flat index |
| `ivf_pq_params` | `cuvsIvfPqIndexParams_t` | Optional parameters for building a ivf-pq index |

_Source: `c/include/cuvs/neighbors/tiered_index.h:72`_

### cuvsTieredIndexParamsCreate

Allocate Tiered Index Params and populate with default values

```c
cuvsError_t cuvsTieredIndexParamsCreate(cuvsTieredIndexParams_t* index_params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index_params` | in | `cuvsTieredIndexParams_t*` | cuvsTieredIndexParams_t to allocate |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/tiered_index.h:105`_

### cuvsTieredIndexParamsDestroy

De-allocate Tiered Index params

```c
cuvsError_t cuvsTieredIndexParamsDestroy(cuvsTieredIndexParams_t index_params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index_params` | in | `cuvsTieredIndexParams_t` |  |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/tiered_index.h:113`_

## Tiered index build

_Doxygen group: `tieredindex_c_index_build`_

### cuvsTieredIndexBuild

Build a TieredIndex index with a `DLManagedTensor` which has underlying

```c
cuvsError_t cuvsTieredIndexBuild(cuvsResources_t res,
cuvsTieredIndexParams_t index_params,
DLManagedTensor* dataset,
cuvsTieredIndex_t index);
```

`DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`, or `kDLCPU`. Also, acceptable underlying types are: 1. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32` 2. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 16`

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `index_params` | in | `cuvsTieredIndexParams_t` | Index parameters to use when building the index |
| `dataset` | in | `DLManagedTensor*` | DLManagedTensor* training dataset |
| `index` | out | `cuvsTieredIndex_t` | cuvsTieredIndex_t Newly built TieredIndex index |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/tiered_index.h:162`_

## Tiered index search

_Doxygen group: `tieredindex_c_index_search`_

### cuvsTieredIndexSearch

Search a TieredIndex index with a `DLManagedTensor`

```c
cuvsError_t cuvsTieredIndexSearch(cuvsResources_t res,
void* search_params,
cuvsTieredIndex_t index,
DLManagedTensor* queries,
DLManagedTensor* neighbors,
DLManagedTensor* distances,
cuvsFilter prefilter);
```

cuvsCagraSearchParams_t, cuvsIvfFlatSearchParams_t, cuvsIvfPqSearchParams_t depending on the type of the tiered index used

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `search_params` | in | `void*` | params used to the ANN index, should be one of |
| `index` | in | `cuvsTieredIndex_t` | cuvsTieredIndex which has been returned by `cuvsTieredIndexBuild` |
| `queries` | in | `DLManagedTensor*` | DLManagedTensor* queries dataset to search |
| `neighbors` | out | `DLManagedTensor*` | DLManagedTensor* output `k` neighbors for queries |
| `distances` | out | `DLManagedTensor*` | DLManagedTensor* output `k` distances for queries |
| `prefilter` | in | `cuvsFilter` | cuvsFilter input prefilter that can be used to filter queries and neighbors based on the given bitmap. |

**Returns**

`cuvsError_t`

_Source: `c/include/cuvs/neighbors/tiered_index.h:212`_

## Tiered index extend

_Doxygen group: `tiered_c_index_extend`_

### cuvsTieredIndexExtend

Extend the index with the new data.

```c
cuvsError_t cuvsTieredIndexExtend(cuvsResources_t res,
DLManagedTensor* new_vectors,
cuvsTieredIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `new_vectors` | in | `DLManagedTensor*` | DLManagedTensor* the new vectors to add to the index |
| `index` | inout | `cuvsTieredIndex_t` | Tiered index to be extended |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/tiered_index.h:235`_

## Tiered index merge

_Doxygen group: `tiered_c_index_merge`_

### cuvsTieredIndexMerge

Merge multiple indices together into a single index

```c
cuvsError_t cuvsTieredIndexMerge(cuvsResources_t res,
cuvsTieredIndexParams_t index_params,
cuvsTieredIndex_t* indices,
size_t num_indices,
cuvsTieredIndex_t output_index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `index_params` | in | `cuvsTieredIndexParams_t` | Index parameters to use when merging |
| `indices` | in | `cuvsTieredIndex_t*` | pointers to indices to merge together |
| `num_indices` | in | `size_t` | the number of indices to merge |
| `output_index` | out | `cuvsTieredIndex_t` | the merged index |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/tiered_index.h:256`_
