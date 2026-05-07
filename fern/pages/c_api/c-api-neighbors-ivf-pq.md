---
slug: api-reference/c-api-neighbors-ivf-pq
---

# IVF PQ

_Source header: `c/include/cuvs/neighbors/ivf_pq.h`_

## IVF-PQ index build parameters

_Doxygen group: `ivf_pq_c_index_params`_

### cuvsIvfPqCodebookGen

A type for specifying how PQ codebooks are created

```c
enum cuvsIvfPqCodebookGen { ... } ;
```

**Values**

| Name | Value |
| --- | --- |
| `CUVS_IVF_PQ_CODEBOOK_GEN_PER_SUBSPACE` | `0` |
| `CUVS_IVF_PQ_CODEBOOK_GEN_PER_CLUSTER` | `1` |

_Source: `c/include/cuvs/neighbors/ivf_pq.h:26`_

### cuvsIvfPqListLayout

A type for specifying the memory layout of IVF-PQ list data

```c
enum cuvsIvfPqListLayout { ... } ;
```

**Values**

| Name | Value |
| --- | --- |
| `CUVS_IVF_PQ_LIST_LAYOUT_FLAT` | `0` |
| `CUVS_IVF_PQ_LIST_LAYOUT_INTERLEAVED` | `1` |

_Source: `c/include/cuvs/neighbors/ivf_pq.h:35`_

### cuvsIvfPqIndexParams

Supplemental parameters to build IVF-PQ Index

```c
struct cuvsIvfPqIndexParams { ... } ;
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
| `pq_bits` | `uint32_t` | The bit length of the vector element after compression by PQ. |
| `pq_dim` | `uint32_t` | The dimensionality of the vector after compression by PQ. When zero, an optimal value is |
| `codebook_kind` | `enum cuvsIvfPqCodebookGen` | How PQ codebooks are created. |
| `force_random_rotation` | `bool` | Apply a random rotation matrix on the input data and queries even if `dim % pq_dim == 0`. |
| `conservative_memory_allocation` | `bool` | By default, the algorithm allocates more space than necessary for individual clusters |
| `max_train_points_per_pq_code` | `uint32_t` | The max number of data points to use per PQ code during PQ codebook training. Using more data |
| `codes_layout` | `enum cuvsIvfPqListLayout` | Memory layout of the IVF-PQ list data. |

_Source: `c/include/cuvs/neighbors/ivf_pq.h:44`_

### cuvsIvfPqIndexParamsCreate

Allocate IVF-PQ Index params, and populate with default values

```c
cuvsError_t cuvsIvfPqIndexParamsCreate(cuvsIvfPqIndexParams_t* index_params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index_params` | in | `cuvsIvfPqIndexParams_t*` | cuvsIvfPqIndexParams_t to allocate |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/ivf_pq.h:144`_

### cuvsIvfPqIndexParamsDestroy

De-allocate IVF-PQ Index params

```c
cuvsError_t cuvsIvfPqIndexParamsDestroy(cuvsIvfPqIndexParams_t index_params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index_params` | in | `cuvsIvfPqIndexParams_t` |  |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/ivf_pq.h:152`_

## IVF-PQ index search parameters

_Doxygen group: `ivf_pq_c_search_params`_

### cuvsIvfPqSearchParams

Supplemental parameters to search IVF-PQ index

```c
struct cuvsIvfPqSearchParams { ... } ;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `n_probes` | `uint32_t` | The number of clusters to search. |
| `lut_dtype` | `cudaDataType_t` | Data type of look up table to be created dynamically at search time. |
| `internal_distance_dtype` | `cudaDataType_t` | Storage data type for distance/similarity computed at search time. |
| `coarse_search_dtype` | `cudaDataType_t` | The data type to use as the GEMM element type when searching the clusters to probe. |
| `max_internal_batch_size` | `uint32_t` | Set the internal batch size to improve GPU utilization at the cost of larger memory footprint. |
| `preferred_shmem_carveout` | `double` | Preferred fraction of SM's unified memory / L1 cache to be used as shared memory. |

_Source: `c/include/cuvs/neighbors/ivf_pq.h:165`_

### cuvsIvfPqSearchParamsCreate

Allocate IVF-PQ search params, and populate with default values

```c
cuvsError_t cuvsIvfPqSearchParamsCreate(cuvsIvfPqSearchParams_t* params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | `cuvsIvfPqSearchParams_t*` | cuvsIvfPqSearchParams_t to allocate |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/ivf_pq.h:227`_

### cuvsIvfPqSearchParamsDestroy

De-allocate IVF-PQ search params

```c
cuvsError_t cuvsIvfPqSearchParamsDestroy(cuvsIvfPqSearchParams_t params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | `cuvsIvfPqSearchParams_t` |  |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/ivf_pq.h:235`_

## IVF-PQ index

_Doxygen group: `ivf_pq_c_index`_

### cuvsIvfPqIndexCreate

Allocate IVF-PQ index

```c
cuvsError_t cuvsIvfPqIndexCreate(cuvsIvfPqIndex_t* index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | `cuvsIvfPqIndex_t*` | cuvsIvfPqIndex_t to allocate |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/ivf_pq.h:261`_

### cuvsIvfPqIndexDestroy

De-allocate IVF-PQ index

```c
cuvsError_t cuvsIvfPqIndexDestroy(cuvsIvfPqIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | `cuvsIvfPqIndex_t` | cuvsIvfPqIndex_t to de-allocate |

**Returns**

`cuvsError_t`

_Source: `c/include/cuvs/neighbors/ivf_pq.h:268`_

### cuvsIvfPqIndexGetNLists

Get the number of clusters/inverted lists

```c
cuvsError_t cuvsIvfPqIndexGetNLists(cuvsIvfPqIndex_t index, int64_t* n_lists);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` |  | `cuvsIvfPqIndex_t` |  |
| `n_lists` |  | `int64_t*` |  |

**Returns**

`cuvsError_t`

_Source: `c/include/cuvs/neighbors/ivf_pq.h:271`_

### cuvsIvfPqIndexGetDim

Get the dimensionality

```c
cuvsError_t cuvsIvfPqIndexGetDim(cuvsIvfPqIndex_t index, int64_t* dim);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` |  | `cuvsIvfPqIndex_t` |  |
| `dim` |  | `int64_t*` |  |

**Returns**

`cuvsError_t`

_Source: `c/include/cuvs/neighbors/ivf_pq.h:274`_

### cuvsIvfPqIndexGetSize

Get the size of the index

```c
cuvsError_t cuvsIvfPqIndexGetSize(cuvsIvfPqIndex_t index, int64_t* size);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` |  | `cuvsIvfPqIndex_t` |  |
| `size` |  | `int64_t*` |  |

**Returns**

`cuvsError_t`

_Source: `c/include/cuvs/neighbors/ivf_pq.h:277`_

### cuvsIvfPqIndexGetPqDim

Get the dimensionality of an encoded vector after compression by PQ.

```c
cuvsError_t cuvsIvfPqIndexGetPqDim(cuvsIvfPqIndex_t index, int64_t* pq_dim);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` |  | `cuvsIvfPqIndex_t` |  |
| `pq_dim` |  | `int64_t*` |  |

**Returns**

`cuvsError_t`

_Source: `c/include/cuvs/neighbors/ivf_pq.h:280`_

### cuvsIvfPqIndexGetPqBits

Get the bit length of an encoded vector element after compression by PQ.

```c
cuvsError_t cuvsIvfPqIndexGetPqBits(cuvsIvfPqIndex_t index, int64_t* pq_bits);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` |  | `cuvsIvfPqIndex_t` |  |
| `pq_bits` |  | `int64_t*` |  |

**Returns**

`cuvsError_t`

_Source: `c/include/cuvs/neighbors/ivf_pq.h:283`_

### cuvsIvfPqIndexGetPqLen

Get the Dimensionality of a subspace, i.e. the number of vector

```c
cuvsError_t cuvsIvfPqIndexGetPqLen(cuvsIvfPqIndex_t index, int64_t* pq_len);
```

components mapped to a subspace

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` |  | `cuvsIvfPqIndex_t` |  |
| `pq_len` |  | `int64_t*` |  |

**Returns**

`cuvsError_t`

_Source: `c/include/cuvs/neighbors/ivf_pq.h:287`_

### cuvsIvfPqIndexGetCenters

Get the cluster centers corresponding to the lists in the original space

```c
cuvsError_t cuvsIvfPqIndexGetCenters(cuvsIvfPqIndex_t index, DLManagedTensor* centers);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | `cuvsIvfPqIndex_t` | cuvsIvfPqIndex_t Built Ivf-Pq index |
| `centers` | out | `DLManagedTensor*` | Output tensor that will be populated with a non-owning view of the data |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/ivf_pq.h:296`_

### cuvsIvfPqIndexGetCentersPadded

Get the padded cluster centers [n_lists, dim_ext]

```c
cuvsError_t cuvsIvfPqIndexGetCentersPadded(cuvsIvfPqIndex_t index, DLManagedTensor* centers);
```

where dim_ext = round_up(dim + 1, 8) This returns the full padded centers as a contiguous array, suitable for use with cuvsIvfPqBuildPrecomputed.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | `cuvsIvfPqIndex_t` | cuvsIvfPqIndex_t Built Ivf-Pq index |
| `centers` | out | `DLManagedTensor*` | Output tensor that will be populated with a non-owning view of the data |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/ivf_pq.h:309`_

### cuvsIvfPqIndexGetPqCenters

Get the PQ cluster centers

```c
cuvsError_t cuvsIvfPqIndexGetPqCenters(cuvsIvfPqIndex_t index, DLManagedTensor* pq_centers);
```

- CUVS_IVF_PQ_CODEBOOK_GEN_PER_SUBSPACE: [pq_dim , pq_len, pq_book_size] - CUVS_IVF_PQ_CODEBOOK_GEN_PER_CLUSTER:  [n_lists, pq_len, pq_book_size]

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | `cuvsIvfPqIndex_t` | cuvsIvfPqIndex_t Built Ivf-Pq index |
| `pq_centers` | out | `DLManagedTensor*` | Output tensor that will be populated with a non-owning view of the data |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/ivf_pq.h:321`_

### cuvsIvfPqIndexGetCentersRot

Get the rotated cluster centers [n_lists, rot_dim]

```c
cuvsError_t cuvsIvfPqIndexGetCentersRot(cuvsIvfPqIndex_t index, DLManagedTensor* centers_rot);
```

where rot_dim = pq_len * pq_dim

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | `cuvsIvfPqIndex_t` | cuvsIvfPqIndex_t Built Ivf-Pq index |
| `centers_rot` | out | `DLManagedTensor*` | Output tensor that will be populated with a non-owning view of the data |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/ivf_pq.h:331`_

### cuvsIvfPqIndexGetRotationMatrix

Get the rotation matrix [rot_dim, dim]

```c
cuvsError_t cuvsIvfPqIndexGetRotationMatrix(cuvsIvfPqIndex_t index,
DLManagedTensor* rotation_matrix);
```

Transform matrix (original space -&gt; rotated padded space) data

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | `cuvsIvfPqIndex_t` | cuvsIvfPqIndex_t Built Ivf-Pq index |
| `rotation_matrix` | out | `DLManagedTensor*` | Output tensor that will be populated with a non-owning view of the |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/ivf_pq.h:342`_

### cuvsIvfPqIndexGetListSizes

Get the sizes of each list

```c
cuvsError_t cuvsIvfPqIndexGetListSizes(cuvsIvfPqIndex_t index, DLManagedTensor* list_sizes);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | `cuvsIvfPqIndex_t` | cuvsIvfPqIndex_t Built Ivf-Pq index |
| `list_sizes` | out | `DLManagedTensor*` | Output tensor that will be populated with a non-owning view of the data |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/ivf_pq.h:352`_

### cuvsIvfPqIndexUnpackContiguousListData

Unpack `n_rows` consecutive PQ encoded vectors of a single list (cluster) in the

```c
cuvsError_t cuvsIvfPqIndexUnpackContiguousListData(cuvsResources_t res,
cuvsIvfPqIndex_t index,
DLManagedTensor* out_codes,
uint32_t label,
uint32_t offset);
```

compressed index starting at given `offset`, not expanded to one code per byte. Each code in the output buffer occupies ceildiv(index.pq_dim() * index.pq_bits(), 8) bytes.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | raft resource |
| `index` | in | `cuvsIvfPqIndex_t` | cuvsIvfPqIndex_t Built Ivf-Pq index |
| `out_codes` | out | `DLManagedTensor*` | the destination buffer [n_rows, ceildiv(index.pq_dim() * index.pq_bits(), 8)]. The length `n_rows` defines how many records to unpack, offset + n_rows must be smaller than or equal to the list size. This DLManagedTensor must already point to allocated device memory |
| `label` | in | `uint32_t` | The id of the list (cluster) to decode. |
| `offset` | in | `uint32_t` | How many records in the list to skip. |

**Returns**

`cuvsError_t`

_Source: `c/include/cuvs/neighbors/ivf_pq.h:371`_

### cuvsIvfPqIndexGetListIndices

Get the indices of each vector in a ivf-pq list

```c
cuvsError_t cuvsIvfPqIndexGetListIndices(cuvsIvfPqIndex_t index,
uint32_t label,
DLManagedTensor* out_labels);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` | in | `cuvsIvfPqIndex_t` | cuvsIvfPqIndex_t Built Ivf-Pq index |
| `label` | in | `uint32_t` | The id of the list (cluster) to decode. |
| `out_labels` | out | `DLManagedTensor*` | output tensor that will be populated with a non-owning view of the data |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/ivf_pq.h:386`_

## IVF-PQ index build

_Doxygen group: `ivf_pq_c_index_build`_

### cuvsIvfPqBuild

Build a IVF-PQ index with a `DLManagedTensor` which has underlying

```c
cuvsError_t cuvsIvfPqBuild(cuvsResources_t res,
cuvsIvfPqIndexParams_t params,
DLManagedTensor* dataset,
cuvsIvfPqIndex_t index);
```

`DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`, or `kDLCPU`. Also, acceptable underlying types are: 1. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32` 2. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 16` 3. `kDLDataType.code == kDLInt` and `kDLDataType.bits = 8` 4. `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 8`

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `params` | in | `cuvsIvfPqIndexParams_t` | cuvsIvfPqIndexParams_t used to build IVF-PQ index |
| `dataset` | in | `DLManagedTensor*` | DLManagedTensor* training dataset |
| `index` | out | `cuvsIvfPqIndex_t` | cuvsIvfPqIndex_t Newly built IVF-PQ index |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/ivf_pq.h:440`_

### cuvsIvfPqBuildPrecomputed

Build a view-type IVF-PQ index from device memory precomputed centroids and codebook.

```c
cuvsError_t cuvsIvfPqBuildPrecomputed(cuvsResources_t res,
cuvsIvfPqIndexParams_t params,
uint32_t dim,
DLManagedTensor* pq_centers,
DLManagedTensor* centers,
DLManagedTensor* centers_rot,
DLManagedTensor* rotation_matrix,
cuvsIvfPqIndex_t index);
```

This function creates a non-owning index that stores a reference to the provided device data. All parameters must be provided with correct extents. The caller is responsible for ensuring the lifetime of the input data exceeds the lifetime of the returned index. The index_params must be consistent with the provided matrices. Specifically: - index_params.codebook_kind determines the expected shape of pq_centers - index_params.metric will be stored in the index - index_params.conservative_memory_allocation will be stored in the index The function will verify consistency between index_params, dim, and the matrix extents. matrices) dim]

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `params` | in | `cuvsIvfPqIndexParams_t` | cuvsIvfPqIndexParams_t used to configure the index (must be consistent with |
| `dim` | in | `uint32_t` | dimensionality of the input data |
| `pq_centers` | in | `DLManagedTensor*` | PQ codebook on device memory with required shape: - codebook_kind CUVS_IVF_PQ_CODEBOOK_GEN_PER_SUBSPACE: [pq_dim, pq_len, pq_book_size] - codebook_kind CUVS_IVF_PQ_CODEBOOK_GEN_PER_CLUSTER:  [n_lists, pq_len, pq_book_size] |
| `centers` | in | `DLManagedTensor*` | Cluster centers in the original space [n_lists, dim_ext] where dim_ext = round_up(dim + 1, 8) |
| `centers_rot` | in | `DLManagedTensor*` | Rotated cluster centers [n_lists, rot_dim] where rot_dim = pq_len * pq_dim |
| `rotation_matrix` | in | `DLManagedTensor*` | Transform matrix (original space -&gt; rotated padded space) [rot_dim, |
| `index` | out | `cuvsIvfPqIndex_t` | cuvsIvfPqIndex_t Newly built view-type IVF-PQ index |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/ivf_pq.h:474`_

## IVF-PQ index search

_Doxygen group: `ivf_pq_c_index_search`_

### cuvsIvfPqSearch

Search a IVF-PQ index with a `DLManagedTensor` which has underlying

```c
cuvsError_t cuvsIvfPqSearch(cuvsResources_t res,
cuvsIvfPqSearchParams_t search_params,
cuvsIvfPqIndex_t index,
DLManagedTensor* queries,
DLManagedTensor* neighbors,
DLManagedTensor* distances);
```

`DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`. It is also important to note that the IVF-PQ Index must have been built with the same type of `queries`, such that `index.dtype.code == queries.dl_tensor.dtype.code` Types for input are: 1. `queries`: `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32` or `kDLDataType.bits = 16` 2. `neighbors`: `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 32` 3. `distances`: `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `search_params` | in | `cuvsIvfPqSearchParams_t` | cuvsIvfPqSearchParams_t used to search IVF-PQ index |
| `index` | in | `cuvsIvfPqIndex_t` | cuvsIvfPqIndex which has been returned by `cuvsIvfPqBuild` |
| `queries` | in | `DLManagedTensor*` | DLManagedTensor* queries dataset to search |
| `neighbors` | out | `DLManagedTensor*` | DLManagedTensor* output `k` neighbors for queries |
| `distances` | out | `DLManagedTensor*` | DLManagedTensor* output `k` distances for queries |

**Returns**

`cuvsError_t`

_Source: `c/include/cuvs/neighbors/ivf_pq.h:534`_

## IVF-PQ C-API serialize functions

_Doxygen group: `ivf_pq_c_index_serialize`_

### cuvsIvfPqSerialize

Save the index to file.

```c
cuvsError_t cuvsIvfPqSerialize(cuvsResources_t res, const char* filename, cuvsIvfPqIndex_t index);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `filename` | in | `const char*` | the file name for saving the index |
| `index` | in | `cuvsIvfPqIndex_t` | IVF-PQ index |

**Returns**

`cuvsError_t`

_Source: `c/include/cuvs/neighbors/ivf_pq.h:568`_

### cuvsIvfPqDeserialize

Load index from file.

```c
cuvsError_t cuvsIvfPqDeserialize(cuvsResources_t res, const char* filename, cuvsIvfPqIndex_t index);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `filename` | in | `const char*` | the name of the file that stores the index |
| `index` | out | `cuvsIvfPqIndex_t` | IVF-PQ index loaded disk |

**Returns**

`cuvsError_t`

_Source: `c/include/cuvs/neighbors/ivf_pq.h:579`_

## IVF-PQ index extend

_Doxygen group: `ivf_pq_c_index_extend`_

### cuvsIvfPqExtend

Extend the index with the new data.

```c
cuvsError_t cuvsIvfPqExtend(cuvsResources_t res,
DLManagedTensor* new_vectors,
DLManagedTensor* new_indices,
cuvsIvfPqIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `new_vectors` | in | `DLManagedTensor*` | DLManagedTensor* the new vectors to add to the index |
| `new_indices` | in | `DLManagedTensor*` | DLManagedTensor* vector of new indices for the new vectors |
| `index` | inout | `cuvsIvfPqIndex_t` | IVF-PQ index to be extended |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/ivf_pq.h:597`_

## IVF-PQ index transform

_Doxygen group: `ivf_pq_c_index_transform`_

### cuvsIvfPqTransform

Transform the input data by applying pq-encoding

```c
cuvsError_t cuvsIvfPqTransform(cuvsResources_t res,
cuvsIvfPqIndex_t index,
DLManagedTensor* input_dataset,
DLManagedTensor* output_labels,
DLManagedTensor* output_dataset);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `cuvsResources_t` | cuvsResources_t opaque C handle |
| `index` | in | `cuvsIvfPqIndex_t` | IVF-PQ index |
| `input_dataset` | in | `DLManagedTensor*` | DLManagedTensor* vectors to transform |
| `output_labels` | out | `DLManagedTensor*` | DLManagedTensor* Vector of cluster labels for each vector in the input |
| `output_dataset` | out | `DLManagedTensor*` | DLManagedTensor* input vectors after pq-encoding |

**Returns**

`cuvsError_t`

cuvsError_t

_Source: `c/include/cuvs/neighbors/ivf_pq.h:619`_
