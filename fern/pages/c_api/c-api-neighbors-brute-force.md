---
slug: api-reference/c-api-neighbors-brute-force
---

# Brute Force

_Source header: `c/include/cuvs/neighbors/brute_force.h`_

## Bruteforce index

_Doxygen group: `bruteforce_c_index`_

### cuvsBruteForceIndexCreate

Allocate BRUTEFORCE index

```c
cuvsError_t cuvsBruteForceIndexCreate(cuvsBruteForceIndex_t* index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | `cuvsBruteForceIndex_t*` | cuvsBruteForceIndex_t to allocate |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/brute_force.h:39`_

### cuvsBruteForceIndexDestroy

De-allocate BRUTEFORCE index

```c
cuvsError_t cuvsBruteForceIndexDestroy(cuvsBruteForceIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | `cuvsBruteForceIndex_t` | cuvsBruteForceIndex_t to de-allocate |

**Returns**

`cuvsError_t`

_Source: `c/include/cuvs/neighbors/brute_force.h:46`_

## Bruteforce index build

_Doxygen group: `bruteforce_c_index_build`_

### cuvsBruteForceBuild

Build a BRUTEFORCE index with a `DLManagedTensor` which has underlying

```c
cuvsError_t cuvsBruteForceBuild(cuvsResources_t res,
DLManagedTensor* dataset,
cuvsDistanceType metric,
float metric_arg,
cuvsBruteForceIndex_t index);
```

`DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`, or `kDLCPU`. Also, acceptable underlying types are: 1. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32` 2. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 16`

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `dataset` | in | `DLManagedTensor*` | DLManagedTensor* training dataset |
| `metric` | in | `cuvsDistanceType` | metric |
| `metric_arg` | in | `float` | metric_arg |
| `index` | out | `cuvsBruteForceIndex_t` | cuvsBruteForceIndex_t Newly built BRUTEFORCE index |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/brute_force.h:92`_

## Bruteforce index search

_Doxygen group: `bruteforce_c_index_search`_

### cuvsBruteForceSearch

Search a BRUTEFORCE index with a `DLManagedTensor` which has underlying

```c
cuvsError_t cuvsBruteForceSearch(cuvsResources_t res,
cuvsBruteForceIndex_t index,
DLManagedTensor* queries,
DLManagedTensor* neighbors,
DLManagedTensor* distances,
cuvsFilter prefilter);
```

`DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`. It is also important to note that the BRUTEFORCE index must have been built with the same type of `queries`, such that `index.dtype.code == queries.dl_tensor.dtype.code` Types for input are: 1. `queries`: `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32` or `kDLDataType.bits = 16` 2. `neighbors`: `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 32` 3. `distances`: `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `index` | in | `cuvsBruteForceIndex_t` | cuvsBruteForceIndex which has been returned by `cuvsBruteForceBuild` |
| `queries` | in | `DLManagedTensor*` | DLManagedTensor* queries dataset to search |
| `neighbors` | out | `DLManagedTensor*` | DLManagedTensor* output `k` neighbors for queries |
| `distances` | out | `DLManagedTensor*` | DLManagedTensor* output `k` distances for queries |
| `prefilter` | in | `cuvsFilter` | cuvsFilter input prefilter that can be used to filter queries and neighbors based on the given bitmap. |

**Returns**

`cuvsError_t`

_Source: `c/include/cuvs/neighbors/brute_force.h:148`_

## BRUTEFORCE C-API serialize functions

_Doxygen group: `bruteforce_c_index_serialize`_

### cuvsBruteForceSerialize

Save the index to file.

```c
cuvsError_t cuvsBruteForceSerialize(cuvsResources_t res,
const char* filename,
cuvsBruteForceIndex_t index);
```

The serialization format can be subject to changes, therefore loading an index saved with a previous version of cuvs is not guaranteed to work.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `filename` | in | `const char*` | the file name for saving the index |
| `index` | in | `cuvsBruteForceIndex_t` | BRUTEFORCE index |

**Returns**

`cuvsError_t`

_Source: `c/include/cuvs/neighbors/brute_force.h:184`_

### cuvsBruteForceDeserialize

Load index from file.

```c
cuvsError_t cuvsBruteForceDeserialize(cuvsResources_t res,
const char* filename,
cuvsBruteForceIndex_t index);
```

The serialization format can be subject to changes, therefore loading an index saved with a previous version of cuvs is not guaranteed to work.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `filename` | in | `const char*` | the name of the file that stores the index |
| `index` | out | `cuvsBruteForceIndex_t` | BRUTEFORCE index loaded disk |

**Returns**

`cuvsError_t`

_Source: `c/include/cuvs/neighbors/brute_force.h:211`_
