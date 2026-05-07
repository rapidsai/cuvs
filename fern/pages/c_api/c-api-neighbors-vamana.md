---
slug: api-reference/c-api-neighbors-vamana
---

# Vamana

_Source header: `c/include/cuvs/neighbors/vamana.h`_

## C API for Vamana index build

_Doxygen group: `vamana_c_index_params`_

### cuvsVamanaIndexParams

Supplemental parameters to build Vamana Index

`graph_degree`: Maximum degree of graph; corresponds to the R parameter of Vamana algorithm in the literature. `visited_size`: Maximum number of visited nodes per search during Vamana algorithm. Loosely corresponds to the L parameter in the literature. `vamana_iters`: The number of times all vectors are inserted into the graph. If &gt; 1, all vectors are re-inserted to improve graph quality. `max_fraction`: The maximum batch size is this fraction of the total dataset size. Larger gives faster build but lower graph quality. `alpha`: Used to determine how aggressive the pruning will be.

```c
struct cuvsVamanaIndexParams { ... } ;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `metric` | `cuvsDistanceType` | Distance type. |
| `graph_degree` | `uint32_t` | Maximum degree of output graph corresponds to the R parameter in the original Vamana |
| `visited_size` | `uint32_t` | Maximum number of visited nodes per search corresponds to the L parameter in the Vamana |
| `vamana_iters` | `float` | Number of Vamana vector insertion iterations (each iteration inserts all vectors). |
| `alpha` | `float` | Alpha for pruning parameter |
| `max_fraction` | `float` | Maximum fraction of dataset inserted per batch.              * |
| `batch_base` | `float` | Base of growth rate of batch sizes * |
| `queue_size` | `uint32_t` | Size of candidate queue structure - should be (2^x)-1 |
| `reverse_batchsize` | `uint32_t` | Max batchsize of reverse edge processing (reduces memory footprint) |

_Source: `c/include/cuvs/neighbors/vamana.h:37`_

### cuvsVamanaIndexParamsCreate

Allocate Vamana Index params, and populate with default values

```c
cuvsError_t cuvsVamanaIndexParamsCreate(cuvsVamanaIndexParams_t* params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | `cuvsVamanaIndexParams_t*` | cuvsVamanaIndexParams_t to allocate |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/vamana.h:69`_

### cuvsVamanaIndexParamsDestroy

De-allocate Vamana Index params

```c
cuvsError_t cuvsVamanaIndexParamsDestroy(cuvsVamanaIndexParams_t params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | `cuvsVamanaIndexParams_t` | cuvsVamanaIndexParams_t to de-allocate |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/vamana.h:77`_

## Vamana index

_Doxygen group: `vamana_c_index`_

### cuvsVamanaIndexCreate

Allocate Vamana index

```c
cuvsError_t cuvsVamanaIndexCreate(cuvsVamanaIndex_t* index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | `cuvsVamanaIndex_t*` | cuvsVamanaIndex_t to allocate |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/vamana.h:106`_

### cuvsVamanaIndexDestroy

De-allocate Vamana index

```c
cuvsError_t cuvsVamanaIndexDestroy(cuvsVamanaIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | `cuvsVamanaIndex_t` | cuvsVamanaIndex_t to de-allocate |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/vamana.h:114`_

### cuvsVamanaIndexGetDims

Get the dimension of the index

```c
cuvsError_t cuvsVamanaIndexGetDims(cuvsVamanaIndex_t index, int* dim);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | `cuvsVamanaIndex_t` | cuvsVamanaIndex_t to get dimension of |
| `dim` | out | `int*` | pointer to dimension to set |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/vamana.h:123`_

## Vamana index build

_Doxygen group: `vamana_c_index_build`_

### cuvsVamanaBuild

Build Vamana index

```c
cuvsError_t cuvsVamanaBuild(cuvsResources_t res,
cuvsVamanaIndexParams_t params,
DLManagedTensor* dataset,
cuvsVamanaIndex_t index);
```

Build the index from the dataset for efficient DiskANN search. The build uses the Vamana insertion-based algorithm to create the graph. The algorithm starts with an empty graph and iteratively inserts batches of nodes. Each batch involves performing a greedy search for each vector to be inserted, and inserting it with edges to all nodes traversed during the search. Reverse edges are also inserted and robustPrune is applied to improve graph quality. The index_params struct controls the degree of the final graph. The following distance metrics are supported: - L2 Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `params` | in | `cuvsVamanaIndexParams_t` | cuvsVamanaIndexParams_t used to build Vamana index |
| `dataset` | in | `DLManagedTensor*` | DLManagedTensor* training dataset |
| `index` | out | `cuvsVamanaIndex_t` | cuvsVamanaIndex_t Vamana index |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/vamana.h:169`_

## Vamana index serialize

_Doxygen group: `vamana_c_index_serialize`_

### cuvsVamanaSerialize

Save Vamana index to file

```c
cuvsError_t cuvsVamanaSerialize(cuvsResources_t res,
const char* filename,
cuvsVamanaIndex_t index,
bool include_dataset);
```

Matches the file format used by the DiskANN open-source repository, allowing cross-compatibility. Serialized Index is to be used by the DiskANN open-source repository for graph search.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `filename` | in | `const char*` | the file prefix for where the index is saved |
| `index` | in | `cuvsVamanaIndex_t` | cuvsVamanaIndex_t to serialize |
| `include_dataset` | in | `bool` | whether to include the dataset in the serialized index |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/vamana.h:205`_
